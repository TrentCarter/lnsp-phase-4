# PRD: Hierarchical Health Monitoring and Retry System (HHMRS)
## aka TRON - Centralized Health Monitoring

**Version**: 1.0
**Date**: 2025-11-12
**Status**: Draft
**Owner**: LNSP Core Team

---

## Executive Summary

**HHMRS** is a comprehensive health monitoring and fault recovery system that ensures no task runs forever and provides graceful degradation when agents fail. It implements hierarchical responsibility where parents are accountable for children, while **TRON** (aka HeartbeatMonitor) handles centralized monitoring and reporting.

**What HHMRS Is**:
- Centralized heartbeat monitoring via **TRON** (pure Python, no LLM)
- Children send heartbeats every 30s, TRON detects timeouts at 60s (2 missed)
- Automatic retry with escalation (child restart → LLM change → permanent failure)
- Configurable failure thresholds (default: 3 restarts, 3 LLM retries)
- Comprehensive metrics collection per Prime Directive, Agent, LLM, Task, Project

**What HHMRS Is NOT**:
- A replacement for the existing PAS architecture
- An LLM-based system (TRON is pure Python heuristics)
- A duplicate communication system (uses existing comms_logger)
- A new database system (extends existing registry.db)

**Key Architecture**:
- **TRON** (Port 6109): Centralized monitoring agent (pure Python)
- **Parent Responsibility**: Parents decide retry/escalate/fail when alerted
- **Hierarchical Escalation**: 3 restarts → grandparent → try different LLM → fail

**Core Innovation**: Multi-layered defense against runaway tasks with centralized monitoring, hierarchical responsibility, automatic recovery, and system-wide oversight to prevent the "stale run" issue (9c2c9284) that plagued integration tests.

---

## 1. Problem Statement

### Current State
- Agents can fail silently, leaving tasks stuck in "executing" state forever
- Integration test hit a stale run (9c2c9284) that never completed
- Directors couldn't report back (HTTP 422 errors), parent never knew
- No automatic retry or recovery mechanism
- No system-wide health monitoring

### Pain Points
1. **Silent failures**: Child agents can crash/hang without parent knowing
2. **Runaway tasks**: Tasks run indefinitely when communication breaks
3. **Manual intervention**: Human must detect and kill stuck processes
4. **No retry logic**: Single failure means complete task failure
5. **No escalation**: Parent has no way to ask for help when child repeatedly fails
6. **No metrics**: Can't analyze failure patterns or optimize retry strategies

### Root Cause Analysis
The integration test failure (9c2c9284) exposed a critical gap:
- Director tried to report completion → HTTP 422 error (validation failure)
- PAS Root never received completion notification
- Task remained in "executing" state forever
- No timeout, no health check, no recovery

### Why Now?
- P0 stack is production-ready but lacks fault tolerance
- Integration tests uncovered critical architectural gap
- Cannot ship to production without ensuring tasks complete or fail gracefully
- Need system-wide monitoring before expanding to more agents/tasks

---

## 2. Goals & Success Criteria

### Goals
1. **Prevent runaway tasks**: No task runs longer than configured max duration
2. **Automatic recovery**: Child failures trigger automatic retry (up to limit)
3. **Hierarchical escalation**: When child repeatedly fails, escalate to grandparent
4. **System-wide monitoring**: Second-layer agent watches all heartbeats
5. **Comprehensive metrics**: Track failures, retries, escalations per task/agent/LLM
6. **Configurable thresholds**: HMI settings for heartbeat interval, max restarts, max failures

### Success Criteria (V1 MVP)
- [ ] Heartbeat protocol working for all parent-child relationships
- [ ] Parent detects missing heartbeat within 2x heartbeat interval
- [ ] Automatic child restart on heartbeat timeout (up to max restarts)
- [ ] Escalation to grandparent after max restarts exceeded
- [ ] Grandparent can retry with different LLM/agent
- [ ] System fails gracefully after max failures exceeded
- [ ] HMI "Tasks" settings menu for configuration
- [ ] Monitoring agent alerts on errors/delays/max limits hit
- [ ] Integration test (9c2c9284 scenario) completes or fails gracefully in <5 minutes
- [ ] Metrics collection working for all dimensions (Prime Directive, Agent, LLM, Task, Project)

### Success Criteria (V2 Future)
- [ ] Predictive failure detection (ML on metrics)
- [ ] Auto-tuning of heartbeat intervals based on task type
- [ ] Cascading failure prevention (don't retry during system-wide outage)
- [ ] Cost-aware retry (use cheaper LLM for retries)

---

## 3. Architecture & Design

### 3.1 System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                 HHMRS (Health Monitoring Layer)                   │
│                                                                    │
│  ┌────────────────┐      ┌────────────────┐      ┌─────────────┐ │
│  │  Heartbeat     │      │  Retry Logic   │      │  Monitoring │ │
│  │  Protocol      │─────▶│  & Escalation  │◀─────│  Agent      │ │
│  └────────────────┘      └────────────────┘      └─────────────┘ │
│         │                         │                       │        │
│         │ status/heartbeat        │ restart/escalate      │ alert  │
│         ▼                         ▼                       ▼        │
└────────────────────────────────────────────────────────────────────┘
         │                          │                        │
         ▼                          ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Existing PAS Architecture                     │
│                                                                   │
│  PAS Root ──▶ Architect ──▶ Director ──▶ Manager ──▶ LCO       │
│     │             │             │            │           │        │
│     └─────────────┴─────────────┴────────────┴───────────┘       │
│              (parent monitors child)                              │
└─────────────────────────────────────────────────────────────────┘
         │                          │                        │
         ▼                          ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                    │
│                                                                   │
│  registry.db (SQLite)          comms logs           metrics.db   │
│  - heartbeat_status table      - parent/child      - failures    │
│  - retry_history table         - timestamps        - retries     │
│  - failure_metrics table       - status changes    - escalations │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Heartbeat Protocol

**Design Principles** (Centralized Monitoring):
- **Centralized TRON (aka HeartbeatMonitor, Port 6109)** tracks all agent heartbeats
- Children send heartbeat messages every 30 seconds to TRON
- TRON polls every 30 seconds to check for timeouts
- TRON alerts parent when child exceeds timeout (2 missed heartbeats = 60s)
- Parents are invoked on-demand to handle retry/escalation (not "always awake")

**Why Centralized?**
- Parents are LLMs invoked on-demand (expensive to keep "always awake")
- Large context windows make parent polling costly
- Lightweight TRON (pure Python, no LLM) is fast and cost-free
- Single source of truth for all agent health

**Heartbeat Message Format**:
```json
{
  "type": "heartbeat",
  "agent": "director-code-abc123",
  "run_id": "9c2c9284-abc-def",
  "timestamp": "2025-11-12T10:30:45.123Z",
  "state": "executing",
  "message": "running tests",
  "progress": 0.45,
  "llm_model": "claude-sonnet-4-5",
  "parent_agent": "architect-xyz789",
  "metadata": {
    "cpu_pct": 23.5,
    "memory_mb": 512,
    "active_threads": 4
  }
}
```

**State Transitions**:
```
Normal Flow (Every 30 seconds):
  Child ─────[heartbeat]────▶ TRON (HeartbeatMonitor)
                             (stores in _heartbeats dict)

Monitor Polling (Every 30 seconds):
  TRON checks all agents:
    time_since_last = now - last_heartbeat
    if time_since_last > 60s (2 missed):
      ─────[timeout_alert]────▶ Parent Agent

Parent Handles Timeout:
  Parent receives alert from TRON
  Parent decides action:
    - Restart child (if restart_count < max_restarts)
    - Escalate to grandparent (if restart_count >= max_restarts)
    - Report permanent failure (if failure_count >= max_failures)
```

**Existing Implementation**:
- TRON (aka HeartbeatMonitor) already exists: `services/common/heartbeat.py`
- Background checker thread runs every 30s: `_check_health_loop()`
- Current timeout: `MISS_TIMEOUT_S = 150s` (will update to 60s = 2 * 30s)
- Health API: `get_health()`, `get_unhealthy_agents()`

### 3.3 Retry Logic & Escalation

**Retry Levels**:

**Level 1: Child Restart (0 → Max Restarts)**
- Parent detects child failure (missed heartbeat + poll timeout)
- Parent kills child process (if still running)
- Parent clears child state
- Parent restarts child with same configuration
- Parent reports "child_restarted" up the chain
- Increment restart_count in registry.db

**Level 2: Grandparent Escalation (Max Restarts → Max Failures)**
- restart_count reaches max_task_restarts (default: 3)
- Parent reports "child_max_restarts_exceeded" to grandparent
- Grandparent takes over:
  - Try different LLM (e.g., Anthropic → Ollama, or vice versa)
  - Try different agent type (if applicable)
  - Adjust timeout/resource limits
- Grandparent increments failure_count
- If failure_count < max_failed_tasks (default: 3), retry
- If failure_count >= max_failed_tasks, escalate to next level

**Level 3: System Failure (Max Failures Exceeded)**
- failure_count reaches max_failed_tasks (default: 3)
- Grandparent reports "task_failed_permanently" up to PAS Root
- PAS Root marks task as "failed"
- PAS Root notifies Gateway
- Gateway returns failure to user with diagnostic information
- Monitoring Agent alerts HMI

**Configuration (stored in settings.json)**:
```json
{
  "hhmrs": {
    "heartbeat_interval_sec": 30,
    "heartbeat_timeout_multiplier": 2.0,
    "max_task_restarts": 3,
    "max_failed_tasks": 3,
    "poll_child_on_missed_heartbeat": true,
    "kill_timeout_sec": 10,
    "escalation_strategies": [
      "try_different_llm",
      "try_different_agent",
      "increase_timeout"
    ]
  }
}
```

### 3.4 TRON (aka HeartbeatMonitor) - Centralized Monitoring

**Service**: TRON (Port 6109) - Already exists in `services/common/heartbeat.py`

**Name**: TRON (like the movie - system monitoring agent watching the grid)
**Alias**: HeartbeatMonitor (code retains this name for backward compatibility)

**Architecture**: Pure Python heuristic code (NO LLM)
- Fast: Timeout detection <1ms
- Deterministic: Simple arithmetic (`now - last_heartbeat > 60s`)
- Cost-free: No API calls
- Reliable: No hallucinations or LLM failures

**Purpose**: Centralized health monitoring and timeout detection for all agents

**Current Responsibilities** (Already Implemented):
1. ✅ Track all agent heartbeats in-memory (`_heartbeats` dict)
2. ✅ Poll every 30 seconds via background thread (`_check_health_loop()`)
3. ✅ Detect stale agents (current timeout = 150s, will update to 60s = 2 missed heartbeats)
4. ✅ Provide health API (`get_health()`, `get_unhealthy_agents()`)
5. ✅ Track agent hierarchy and state

**New Responsibilities** (To Be Added):
1. ❌ Alert parent when child times out (send RPC message to parent)
2. ❌ Track retry history (write to retry_history table)
3. ❌ Detect patterns: repeated failures, cascading failures, system-wide outages
4. ❌ Alert HMI on anomalies
5. ❌ Collect failure metrics per Prime Directive, Agent, LLM, Task, Project
6. ❌ Prevent unnecessary retries during system-wide issues

**Architecture**:
```
┌──────────────────────────────────────────────────────────────────┐
│         HeartbeatMonitor (Port 6109) - services/common/heartbeat.py │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Heartbeat  │  │   Timeout    │  │    Metrics   │           │
│  │  Tracking   │─▶│   Detection  │─▶│   Collector  │           │
│  └─────────────┘  └──────────────┘  └──────────────┘           │
│         │                │                   │                   │
│         │ record         │ alert             │ store             │
│         ▼                ▼                   ▼                   │
│  _heartbeats dict   Parent RPC         registry.db              │
│                     (restart/escalate)                           │
└──────────────────────────────────────────────────────────────────┘
```

**How Timeout Detection Works**:
```python
# Every 30 seconds (_check_health_loop):
for agent in self._heartbeats:
    time_since_last = now - self._heartbeats[agent].timestamp
    missed_count = time_since_last / HEARTBEAT_INTERVAL_S  # 30s

    if missed_count >= MISS_THRESHOLD:  # 2 missed = 60s+
        # TODO (Line 350): Currently just tracks, need to alert parent
        parent = self._heartbeats[agent].parent_agent
        await self._alert_parent(parent, agent, "timeout")
```

**Alert Conditions**:
- Agent missed 3+ consecutive heartbeats
- restart_count approaching max_task_restarts
- failure_count approaching max_failed_tasks
- Same agent/LLM failing repeatedly across different tasks
- Cascading failures (>30% of agents failing simultaneously)
- Unexpected delays (task taking >3x estimated time)

**Metrics Collected**:
```python
class HealthMetrics:
    # Per Prime Directive
    prime_directive_id: str
    total_tasks: int
    failed_tasks: int
    retried_tasks: int
    escalated_tasks: int
    avg_completion_time_sec: float

    # Per Agent Type
    agent_type: str  # "architect", "director", "manager", "lco"
    agent_failures: int
    agent_restarts: int
    avg_heartbeat_latency_ms: float

    # Per LLM
    llm_provider: str  # "anthropic", "ollama"
    llm_model: str     # "claude-sonnet-4-5", "llama3.1:8b"
    llm_failures: int
    llm_success_rate_pct: float

    # Per Task Type
    task_type: str  # "code_generation", "debugging", "refactoring"
    task_failures: int
    avg_retries_per_task: float

    # Per Project
    project_id: str
    project_health_score: float  # 0-100
    at_risk: bool
```

### 3.5 HMI Integration

**TRON Tree View Visualization**

TRON appears at the top of the agent hierarchy as a collapsed or super-thin bar. When timeouts occur, TRON ORANGE lines show the alert flow.

**Normal State** (No alerts):
```
┌─────────────────────────────────────────────────────────────┐
│  [TRON] ██████████████████████████ (collapsed/thin bar)     │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────┐
        │  PAS Root   │ (green)
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  Architect  │ (green)
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │ Director-   │ (green)
        │   Code      │
        └─────────────┘
```

**Alert State** (Director-Code timed out):
```
┌─────────────────────────────────────────────────────────────┐
│  [TRON] ██████████████████████████ (TRON ORANGE highlight)  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ (TRON ORANGE line)
                        │ "Child timeout: Director-Code"
                        ↓
        ┌─────────────┐
        │  PAS Root   │ (green)
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  Architect  │ (yellow - alerted)
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │ Director-   │ (TRON ORANGE - failed/timeout)
        │   Code      │ "No heartbeat for 60s"
        └─────────────┘
```

**Key Visual Elements**:
- **TRON**: Collapsed thin bar at top (no incoming lines)
- **TRON ORANGE**: `#FF6B35` (distinctive orange, not error red)
- **Alert Line**: TRON → Parent (shows "TRON alerts Architect about Director-Code")
- **Failed Agent**: Highlighted in TRON ORANGE (shows "This is the failed agent")
- **Alerted Parent**: Yellow/amber (shows "Parent received alert, taking action")

---

**New UI: Tasks Settings Menu**

Location: Left sidebar, between "Projects" and "Services"

```
┌─────────────────────────────────────┐
│  Tasks Settings                      │
├─────────────────────────────────────┤
│                                      │
│  Health Monitoring                   │
│  ├─ Heartbeat Interval: [30] sec    │
│  ├─ Timeout Multiplier: [2.0]x      │
│  └─ Enable Polling: [✓]             │
│                                      │
│  Retry Configuration                 │
│  ├─ Max Task Restarts: [3]          │
│  ├─ Max Failed Tasks:  [3]          │
│  └─ Kill Timeout:      [10] sec     │
│                                      │
│  Escalation Strategies               │
│  ├─ [✓] Try Different LLM           │
│  ├─ [✓] Try Different Agent         │
│  └─ [✓] Increase Timeout            │
│                                      │
│  Monitoring                          │
│  ├─ [✓] Enable System Monitor       │
│  ├─ [✓] Alert on Failures           │
│  └─ [✓] Alert on Delays             │
│                                      │
│  [Save Configuration]                │
└─────────────────────────────────────┘
```

**HMI Alerts**:
```javascript
// Alert types
{
  type: "agent_failure",
  severity: "warning",  // or "error", "critical"
  agent_id: "director-code-abc123",
  message: "Agent missed 3 consecutive heartbeats",
  action: "Restarting agent (attempt 2 of 3)",
  timestamp: "2025-11-12T10:30:45Z"
}
```

**Health Indicators in Existing Views**:
- Tree View: Show heartbeat icon next to each agent (green/yellow/red)
- Sequencer: Show retry count badge on failed steps
- Project Detail: Show health score (0-100) based on failure rate

---

## 4. Technical Specifications

### 4.1 Database Schema Extensions

**New Table: heartbeat_status**
```sql
CREATE TABLE heartbeat_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    parent_agent_id TEXT,
    run_id TEXT NOT NULL,
    last_heartbeat_timestamp TEXT NOT NULL,  -- ISO 8601
    status TEXT NOT NULL,  -- "executing", "waiting", "completed", "failed"
    progress_pct REAL DEFAULT 0.0,
    current_subtask TEXT,
    health_cpu_pct REAL,
    health_memory_mb INTEGER,
    health_active_threads INTEGER,
    missed_heartbeat_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_heartbeat_agent_run ON heartbeat_status(agent_id, run_id);
CREATE INDEX idx_heartbeat_timestamp ON heartbeat_status(last_heartbeat_timestamp);
```

**New Table: retry_history**
```sql
CREATE TABLE retry_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    parent_agent_id TEXT,
    retry_type TEXT NOT NULL,  -- "child_restart", "llm_change", "agent_change", "timeout_increase"
    retry_count INTEGER NOT NULL,
    reason TEXT NOT NULL,
    old_config JSON,
    new_config JSON,
    timestamp TEXT NOT NULL,
    outcome TEXT,  -- "success", "failure", "pending"
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_retry_run_task ON retry_history(run_id, task_id);
CREATE INDEX idx_retry_agent ON retry_history(agent_id);
```

**New Table: failure_metrics**
```sql
CREATE TABLE failure_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,  -- "prime_directive", "agent", "llm", "task", "project"
    metric_key TEXT NOT NULL,  -- e.g., "pd-001", "director-code", "anthropic-claude", etc.
    total_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    restart_count INTEGER DEFAULT 0,
    escalation_count INTEGER DEFAULT 0,
    avg_completion_time_sec REAL,
    avg_retries_per_task REAL,
    health_score REAL,  -- 0-100
    period_start TEXT NOT NULL,  -- ISO 8601
    period_end TEXT NOT NULL,    -- ISO 8601
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_metrics_type_key ON failure_metrics(metric_type, metric_key);
CREATE INDEX idx_metrics_period ON failure_metrics(period_start, period_end);
```

### 4.2 Child Agent Implementation (Sending Heartbeats)

**How Children Send Heartbeats** (Simple - No Parent Monitoring Loops):

```python
# Example: Director-Code agent sends heartbeats
# services/pas/director_code/app.py

import asyncio
from services.common.heartbeat import get_monitor, AgentState

@app.on_event("startup")
async def startup():
    """Register agent and start heartbeat loop"""
    monitor = get_monitor()

    # Register this agent
    monitor.register_agent(
        agent="Director-Code",
        parent="Architect",
        llm_model="claude-sonnet-4-5",
        role="director",
        tier="coordinator"
    )

    # Start heartbeat background task
    asyncio.create_task(send_heartbeats())

async def send_heartbeats():
    """Send heartbeat every 30 seconds"""
    monitor = get_monitor()

    while True:
        try:
            # Send heartbeat with current state
            monitor.heartbeat(
                agent="Director-Code",
                run_id=current_run_id,
                state=current_state,  # AgentState.EXECUTING, etc.
                message="Processing code lane tasks",
                llm_model="claude-sonnet-4-5",
                parent_agent="Architect",
                children_agents=["Manager-Code-01", "Manager-Code-02"],
                progress=0.65,
                metadata={"active_tasks": 3}
            )

            await asyncio.sleep(30)  # Send every 30 seconds

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(30)  # Continue even on error
```

**Key Points**:
- Children simply call `monitor.heartbeat()` every 30 seconds
- No parent monitoring loops (parents are not "always awake")
- TRON (HeartbeatMonitor) singleton tracks all heartbeats centrally

---

### 4.3 HeartbeatMonitor Enhancement (Timeout Detection & Parent Alerting)

**Current State**: HeartbeatMonitor exists in `services/common/heartbeat.py` with basic tracking

**Enhancements Needed**: Add parent alerting and retry tracking

```python
# services/common/heartbeat.py (Enhanced)

class HeartbeatMonitor:
    """Enhanced with parent alerting and retry tracking"""

    def __init__(self):
        # ... existing initialization ...
        self._retry_counts: Dict[str, int] = {}  # agent_id -> restart_count
        self._failure_counts: Dict[str, int] = {}  # agent_id -> failure_count

    def _check_health_loop(self) -> None:
        """Background thread to check agent health periodically"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds

                unhealthy = self.get_unhealthy_agents()
                for agent_id in unhealthy:
                    health = self.get_health(agent_id)

                    if health.missed_count >= self.MISS_THRESHOLD:
                        # Agent timed out (2+ missed heartbeats)
                        self._handle_timeout(agent_id, health)

            except Exception as e:
                logger.error(f"Health check error: {e}")
                pass  # Continue monitoring

    def _handle_timeout(self, agent_id: str, health: AgentHealth) -> None:
        """Handle agent timeout - alert parent"""
        record = self._heartbeats[agent_id]
        parent_id = record.parent_agent

        if not parent_id:
            logger.warning(f"Agent {agent_id} has no parent, cannot escalate")
            return

        # Get retry count
        restart_count = self._retry_counts.get(agent_id, 0)

        # Alert parent via RPC
        try:
            self._alert_parent(
                parent_id=parent_id,
                child_id=agent_id,
                reason="timeout",
                restart_count=restart_count,
                last_seen=health.last_heartbeat
            )

            # Write to retry_history table
            self._record_timeout(agent_id, parent_id, restart_count)

        except Exception as e:
            logger.error(f"Failed to alert parent {parent_id}: {e}")

    def _alert_parent(self, parent_id: str, child_id: str, reason: str,
                      restart_count: int, last_seen: float) -> None:
        """Send timeout alert to parent via RPC"""
        parent_url = self._get_agent_url(parent_id)

        alert = {
            "type": "child_timeout",
            "child_id": child_id,
            "reason": reason,
            "restart_count": restart_count,
            "last_seen_timestamp": last_seen,
            "timeout_duration_s": time.time() - last_seen
        }

        # POST to parent's /handle_child_timeout endpoint
        response = requests.post(
            f"{parent_url}/handle_child_timeout",
            json=alert,
            timeout=10.0
        )

        if response.status_code == 200:
            logger.info(f"Alerted parent {parent_id} about {child_id} timeout")
        else:
            logger.error(f"Parent {parent_id} rejected timeout alert: {response.text}")
```

---

### 4.4 Parent Agent Implementation (Retry/Escalation Handlers)

**How Parents Handle Timeout Alerts** (Invoked On-Demand by HeartbeatMonitor):

```python
# Example: Architect handles Director-Code timeout
# services/pas/architect/app.py

from fastapi import FastAPI, HTTPException
from services.common.heartbeat import get_monitor

app = FastAPI()

@app.post("/handle_child_timeout")
async def handle_child_timeout(alert: dict):
    """
    Called by HeartbeatMonitor when child times out

    Retry Logic:
    - restart_count < 3: Restart child with same config
    - restart_count >= 3: Escalate to grandparent (PAS Root)
    """
    child_id = alert["child_id"]
    restart_count = alert["restart_count"]

    logger.warning(f"Child {child_id} timed out (restart_count={restart_count})")

    # Check max restarts limit
    if restart_count >= MAX_TASK_RESTARTS:  # 3
        # Escalate to grandparent
        return await escalate_to_grandparent(child_id, restart_count)

    # Attempt restart
    try:
        # 1. Kill child process
        await kill_child_process(child_id)

        # 2. Clear child state
        await clear_child_state(child_id)

        # 3. Restart with same config
        await start_child(child_id, same_config=True)

        # 4. Increment retry count
        monitor = get_monitor()
        monitor._retry_counts[child_id] = restart_count + 1

        # 5. Write to retry_history
        db.record_retry(
            agent_id=child_id,
            retry_type="child_restart",
            retry_count=restart_count + 1,
            reason="timeout"
        )

        logger.info(f"Restarted {child_id} (attempt {restart_count + 1})")
        return {"status": "restarted", "restart_count": restart_count + 1}

    except Exception as e:
        logger.error(f"Failed to restart {child_id}: {e}")
        return await escalate_to_grandparent(child_id, restart_count)

async def escalate_to_grandparent(child_id: str, restart_count: int):
    """Escalate to PAS Root after max restarts exceeded"""

    logger.error(f"Child {child_id} exceeded max restarts ({restart_count}), escalating to PAS Root")

    # Alert PAS Root
    try:
        response = requests.post(
            "http://localhost:6100/handle_grandchild_failure",
            json={
                "type": "max_restarts_exceeded",
                "grandchild_id": child_id,
                "parent_id": "Architect",
                "restart_count": restart_count,
                "request": "retry_with_different_llm"
            },
            timeout=10.0
        )

        return {"status": "escalated", "grandparent": "PAS Root"}

    except Exception as e:
        logger.error(f"Failed to escalate to PAS Root: {e}")
        return {"status": "failed", "error": str(e)}
```

---

### 4.5 Grandparent Implementation (LLM Change & Final Failure)

**How Grandparents Handle Escalation** (Try Different LLM, Then Fail):

```python
# Example: PAS Root handles Architect's escalation
# services/pas/root/app.py

@app.post("/handle_grandchild_failure")
async def handle_grandchild_failure(alert: dict):
    """
    Called by Architect when Director exhausted max restarts

    Escalation Strategy:
    - failure_count < 3: Try different LLM (Anthropic → Ollama)
    - failure_count >= 3: Mark task as permanently failed
    """
    grandchild_id = alert["grandchild_id"]
    parent_id = alert["parent_id"]

    # Get failure count for this task
    failure_count = db.get_failure_count(grandchild_id)

    logger.warning(f"Grandchild {grandchild_id} escalated (failure_count={failure_count})")

    # Check max failures limit
    if failure_count >= MAX_FAILED_TASKS:  # 3
        # Permanent failure
        return await mark_task_failed(grandchild_id)

    # Try different LLM
    try:
        old_llm = "claude-sonnet-4-5"
        new_llm = "llama3.1:8b"  # Switch to local Ollama

        logger.info(f"Retrying {grandchild_id} with different LLM: {old_llm} → {new_llm}")

        # 1. Kill grandchild
        await kill_agent(grandchild_id)

        # 2. Restart with different LLM
        await start_agent(grandchild_id, llm_model=new_llm)

        # 3. Increment failure count
        db.increment_failure_count(grandchild_id)

        # 4. Record retry
        db.record_retry(
            agent_id=grandchild_id,
            retry_type="llm_change",
            retry_count=failure_count + 1,
            reason="max_restarts_exceeded",
            old_config={"llm": old_llm},
            new_config={"llm": new_llm}
        )

        return {"status": "retrying_with_different_llm", "new_llm": new_llm}

    except Exception as e:
        logger.error(f"Failed to retry with different LLM: {e}")
        return await mark_task_failed(grandchild_id)

async def mark_task_failed(agent_id: str):
    """Mark task as permanently failed"""

    logger.error(f"Task {agent_id} permanently failed after {MAX_FAILED_TASKS} attempts")

    # Update run status
    db.update_run_status(agent_id, status="failed", reason="max_failures_exceeded")

    # Alert Gateway
    requests.post(
        "http://localhost:6120/notify_run_failed",
        json={"agent_id": agent_id, "reason": "max_failures_exceeded"}
    )

    # Alert HMI
    monitor = get_monitor()
    monitor._alert_hmi({
        "type": "task_permanently_failed",
        "agent_id": agent_id,
        "severity": "critical"
    })

    return {"status": "permanently_failed"}
```

---

## 5. Implementation Phases

### Phase 1: TRON Timeout Detection & Parent Alerting (Critical - Fixes 9c2c9284)
**Goal**: TRON detects timeouts and alerts parents for retry

**Duration**: 2-3 hours

**Tasks**:
- [ ] Enhance TRON (services/common/heartbeat.py) with:
  - [ ] Add `_retry_counts` and `_failure_counts` tracking dicts
  - [ ] Enhance `_check_health_loop()` to call `_handle_timeout()` on stale agents
  - [ ] Implement `_handle_timeout(agent_id, health)` method
  - [ ] Implement `_alert_parent(parent_id, child_id, ...)` method (RPC call)
  - [ ] Implement `_get_agent_url(agent_id)` method (query registry or service discovery)
- [ ] Add retry_history table to registry.db
- [ ] Add `/handle_child_timeout` endpoint to Architect (services/pas/architect/app.py):
  - [ ] Check restart_count < max_task_restarts (3)
  - [ ] Kill and restart child with same config
  - [ ] Increment retry count in TRON
  - [ ] Log to retry_history
  - [ ] If restart_count >= 3, escalate to PAS Root
- [ ] Add `/handle_child_timeout` endpoint to Director-Code (services/pas/director_code/app.py):
  - [ ] Same logic as Architect (restart or escalate)
- [ ] Unit tests:
  - [ ] Test TRON timeout detection (mock missed heartbeats)
  - [ ] Test parent alert RPC call
  - [ ] Test restart logic

**Success Criteria**:
- ✅ TRON detects timeout after 60s (2 missed 30s heartbeats)
- ✅ TRON alerts parent via RPC to `/handle_child_timeout`
- ✅ Parent restarts child automatically (up to 3 times)
- ✅ Retry history logged to database
- ✅ Simulated Director failure triggers restart within 90s

**Test Plan**:
```bash
# Simulate Director-Code timeout
1. Start all services (TRON, Architect, Director-Code)
2. Kill Director-Code process
3. Verify TRON detects timeout in 60s
4. Verify Architect receives alert and restarts Director-Code
5. Check retry_history table shows restart
```

---

### Phase 2: Grandparent Escalation & LLM Retry (Complete Retry System)
**Goal**: Escalate to grandparent after max restarts, try different LLM

**Duration**: 2 hours

**Tasks**:
- [ ] Add `/handle_grandchild_failure` endpoint to PAS Root (services/pas/root/app.py):
  - [ ] Check failure_count < max_failed_tasks (3)
  - [ ] Kill grandchild
  - [ ] Restart with different LLM (Anthropic → Ollama, or vice versa)
  - [ ] Increment failure_count in database
  - [ ] Log to retry_history with retry_type="llm_change"
  - [ ] If failure_count >= 3, call `mark_task_failed()`
- [ ] Implement `mark_task_failed()` in PAS Root:
  - [ ] Update run status to "failed"
  - [ ] Alert Gateway via `/notify_run_failed`
  - [ ] Alert HMI via TRON
- [ ] Update Architect and Directors to escalate to PAS Root when restart_count >= 3
- [ ] Add failure_metrics table to registry.db
- [ ] Integration tests:
  - [ ] Test full escalation flow (3 restarts → grandparent → LLM change)
  - [ ] Test permanent failure after 3 grandparent attempts

**Success Criteria**:
- ✅ After 3 child restarts, parent escalates to grandparent
- ✅ Grandparent tries different LLM (Anthropic ↔ Ollama)
- ✅ After 3 LLM retries, task marked as permanently failed
- ✅ Gateway receives failure notification
- ✅ Integration test (9c2c9284 scenario) completes or fails in <5 min

**Test Plan**:
```bash
# Test full escalation
1. Start all services
2. Submit task that always fails
3. Verify 3 Director restarts (same LLM)
4. Verify escalation to PAS Root
5. Verify LLM change (check logs)
6. Verify 3 LLM retries
7. Verify permanent failure marked
8. Check Gateway receives failure notification
```

---

### Phase 3: System Prompts Update (Critical - Prevent False Timeouts)
**Goal**: Update all agent system prompts to send heartbeats mid-process

**Duration**: 1 hour

**Tasks**:
- [ ] Update Architect system prompt (docs/contracts/ARCHITECT_SYSTEM_PROMPT.md):
  - [ ] Add heartbeat rule: "Send heartbeat every 30s during long operations"
  - [ ] Example: "When decomposing large PRDs (>1 min), send progress heartbeats"
- [ ] Update Director-Code system prompt (docs/contracts/DIRECTOR_CODE_SYSTEM_PROMPT.md)
- [ ] Update Director-Models system prompt
- [ ] Update Director-Data system prompt
- [ ] Update Director-DevSecOps system prompt
- [ ] Update Director-Docs system prompt
- [ ] Update Manager system prompt template
- [ ] Update Executor system prompt templates
- [ ] Add heartbeat helper to all agents:
  ```python
  async def send_progress_heartbeat(message: str, progress: float):
      """Send heartbeat during long-running operation"""
      monitor = get_monitor()
      monitor.heartbeat(
          agent=self.agent_id,
          run_id=self.current_run_id,
          state=AgentState.EXECUTING,
          message=message,
          progress=progress
      )
  ```

**Success Criteria**:
- ✅ All agent system prompts include heartbeat rules
- ✅ Agents send heartbeats every 30s during long tasks (>1 min)
- ✅ No false timeouts during legitimate long operations

**Rule to Add to System Prompts**:
```
## Heartbeat Protocol

You MUST send heartbeat messages to TRON (HeartbeatMonitor) every 30 seconds during any operation lasting longer than 30 seconds.

**When to send heartbeats**:
- Before starting a long operation (e.g., "Starting task decomposition")
- Every 30s during the operation (e.g., "Processing subtask 3/10, 30% complete")
- After completing the operation (e.g., "Task decomposition complete")

**How to send heartbeats**:
Use the `send_progress_heartbeat()` helper:
```python
await send_progress_heartbeat("Processing code lane", progress=0.45)
```

**Critical**: If you fail to send heartbeats for 60 seconds (2 missed), TRON will assume you have crashed and alert your parent to restart you. This will interrupt your work and waste tokens/time.
```

---

### Phase 4: HMI Settings Menu & Real-Time Config
**Goal**: User-configurable heartbeat settings via HMI

**Duration**: 2 hours

**Tasks**:
- [ ] Create `config/hhmrs_settings.json`:
  ```json
  {
    "heartbeat_interval_sec": 30,
    "heartbeat_timeout_multiplier": 2.0,
    "max_task_restarts": 3,
    "max_failed_tasks": 3,
    "escalation_strategies": ["try_different_llm"]
  }
  ```
- [ ] Add "Tasks" menu to HMI left sidebar (between "Projects" and "Services")
- [ ] Create settings form component:
  - [ ] Heartbeat Interval slider (10-300s)
  - [ ] Timeout Multiplier input (1.5-5.0x)
  - [ ] Max Task Restarts input (0-10)
  - [ ] Max Failed Tasks input (0-10)
  - [ ] Escalation strategies checkboxes
- [ ] Implement settings save (POST /api/settings/hhmrs)
- [ ] Implement settings load (GET /api/settings/hhmrs)
- [ ] Add settings reload endpoint (POST /api/settings/hhmrs/reload)
  - [ ] Notify TRON to reload settings
  - [ ] No service restart required
- [ ] UI tests for settings menu

**Success Criteria**:
- ✅ Settings menu visible in HMI
- ✅ Configuration changes persist to hhmrs_settings.json
- ✅ TRON picks up new settings within 60s (no restart)
- ✅ Settings validation (e.g., heartbeat_interval >= 10s)

---

### Phase 5: Metrics Collection & HMI Alerts
**Goal**: System-wide metrics and alerts for failures/delays

**Duration**: 2-3 hours

**Tasks**:
- [ ] Add metrics collection to TRON:
  - [ ] Track per Prime Directive (total tasks, failures, retries, avg completion time)
  - [ ] Track per Agent Type (failures, restarts, avg heartbeat latency)
  - [ ] Track per LLM (failures, success rate)
  - [ ] Track per Task Type (failures, avg retries)
  - [ ] Store in failure_metrics table (aggregated every 1 min)
- [ ] Add anomaly detection to TRON:
  - [ ] Cascading failure (>30% agents failing simultaneously)
  - [ ] Repeated failure (same agent/LLM failing >5 times)
  - [ ] Unexpected delay (task taking >3x estimate)
- [ ] Add `/api/tron/alerts` endpoint (GET):
  - [ ] Return recent alerts (last 100)
- [ ] Add `/api/tron/metrics/{type}/{key}` endpoint (GET):
  - [ ] Return metrics for given type/key
- [ ] HMI integration:
  - [ ] Add alerts panel to HMI (show recent alerts)
  - [ ] Add health indicators to Tree view (green/yellow/red)
  - [ ] Add retry count badges to Sequencer
  - [ ] Add health score to Project Detail
  - [ ] **TRON Visualization** (Tree View):
    - [ ] Display TRON at top of hierarchy (collapsed or super-thin agent bar)
    - [ ] NO lines drawn TO TRON (children don't visually connect)
    - [ ] When TRON alerts parent about child timeout:
      - [ ] Draw **TRON ORANGE** line from TRON → Parent agent
      - [ ] Highlight failed/errored child agent in **TRON ORANGE**
      - [ ] Shows: "TRON detected Child failed → TRON alerts Parent"
    - [ ] TRON ORANGE color: `#FF6B35` (distinctive, not error red)

**Success Criteria**:
- ✅ Metrics collected every 1 min
- ✅ Anomaly detection working (cascading failures, repeated failures)
- ✅ HMI shows alerts in real-time
- ✅ Health indicators visible in Tree/Sequencer views
- ✅ TRON visible at top of Tree view (collapsed/thin)
- ✅ TRON ORANGE alert lines drawn when timeout detected
- ✅ Failed agents highlighted in TRON ORANGE

---

### Phase 6: Integration Testing & Documentation
**Goal**: End-to-end verification and user documentation

**Duration**: 2 hours

**Tasks**:
- [ ] Integration test: Stale run scenario (9c2c9284)
  - [ ] Simulate Director failure (kill process)
  - [ ] Verify TRON detects timeout in 60s
  - [ ] Verify parent restarts child
  - [ ] Verify task completes or fails gracefully in <5 min
- [ ] Integration test: Max restarts exceeded
  - [ ] Simulate repeated child failures
  - [ ] Verify 3 restarts
  - [ ] Verify escalation to grandparent
  - [ ] Verify LLM change
- [ ] Integration test: Max failures exceeded
  - [ ] Simulate persistent failures
  - [ ] Verify final failure after 3 LLM retries
  - [ ] Verify Gateway receives failure notification
- [ ] Integration test: Cascading failure
  - [ ] Kill 10 agents simultaneously
  - [ ] Verify TRON detects cascading failure
  - [ ] Verify alert to HMI
- [ ] Documentation:
  - [ ] Update CLAUDE.md with HHMRS summary
  - [ ] Create docs/TRON_HEARTBEAT_SYSTEM.md (user guide)
  - [ ] Add troubleshooting guide for common issues
  - [ ] Update service ports doc (TRON port 6109)

**Success Criteria**:
- ✅ All integration tests pass
- ✅ Documentation complete and reviewed
- ✅ HHMRS ready for production
- Metrics visible in HMI

### Phase 7: Integration Testing (Week 4)
**Goal**: Verify end-to-end system with real scenarios

**Tasks**:
- [ ] Test Case 1: Stale run scenario (9c2c9284)
  - Simulate Director failure (kill process)
  - Verify parent detects missed heartbeat
  - Verify automatic restart
  - Verify task completes or fails gracefully
- [ ] Test Case 2: Max restarts exceeded
  - Simulate repeated child failures
  - Verify escalation to grandparent
  - Verify LLM change
- [ ] Test Case 3: Max failures exceeded
  - Simulate persistent failures
  - Verify final failure reported to Gateway
- [ ] Test Case 4: Cascading failure
  - Simulate system-wide outage (kill Ollama)
  - Verify monitoring agent detects pattern
  - Verify excessive retries prevented

**Success Criteria**:
- All test cases pass
- Integration test (9c2c9284 scenario) completes in <5 minutes
- No runaway tasks
- Metrics collected correctly

---

## 6. Testing & Validation

### 6.1 Unit Tests

**heartbeat_mixin_test.py**:
- Test heartbeat sending (interval, format)
- Test heartbeat receiving (update last_seen)
- Test missed heartbeat detection (timeout logic)
- Test child polling (success/failure)
- Test restart logic (increment count, enforce max)

**monitoring_agent_test.py**:
- Test heartbeat tracking
- Test stale agent detection
- Test anomaly detection (cascading, repeated)
- Test metrics collection

### 6.2 Integration Tests

**test_heartbeat_protocol.py**:
```python
async def test_heartbeat_end_to_end():
    """Test heartbeat from child to parent to database."""
    # Start parent and child
    parent = await start_architect()
    child = await parent.start_director()

    # Wait for heartbeat
    await asyncio.sleep(5)

    # Check database
    heartbeats = db.query_heartbeats(child.agent_id)
    assert len(heartbeats) >= 1
    assert heartbeats[0].status == "executing"
```

**test_restart_on_failure.py**:
```python
async def test_child_restart():
    """Test automatic child restart on failure."""
    parent = await start_architect()
    child = await parent.start_director()

    # Kill child
    await child.kill()

    # Wait for parent to detect and restart
    await asyncio.sleep(90)  # 2x heartbeat interval + restart time

    # Check retry history
    retries = db.query_retry_history(child.task_id)
    assert len(retries) == 1
    assert retries[0].retry_type == "child_restart"
```

**test_escalation.py**:
```python
async def test_escalation_to_grandparent():
    """Test escalation after max restarts."""
    pas_root = await start_pas_root()
    architect = await pas_root.start_architect()

    # Configure low max_restarts for faster test
    MAX_TASK_RESTARTS = 2

    # Start director that always fails
    director = await architect.start_director(config={"always_fail": True})

    # Wait for restarts and escalation
    await asyncio.sleep(300)  # 5 minutes

    # Check retry history
    retries = db.query_retry_history(director.task_id)
    assert len([r for r in retries if r.retry_type == "child_restart"]) == 2
    assert len([r for r in retries if r.retry_type == "llm_change"]) >= 1
```

### 6.3 Load Tests

**test_many_agents.py**:
- Start 50 agents with heartbeats
- Verify monitoring agent tracks all
- Verify database performance (INSERT rate)
- Verify no missed heartbeats

**test_cascading_failure.py**:
- Start 20 agents
- Kill 10 simultaneously
- Verify monitoring agent detects cascading failure
- Verify excessive retries prevented

---

## 7. Metrics & Monitoring

### 7.1 Key Performance Indicators (KPIs)

**Reliability**:
- Task completion rate (% tasks that complete without escalation)
- Mean time to detection (MTTD) - time from failure to parent detection
- Mean time to recovery (MTTR) - time from detection to successful restart
- False positive rate - unnecessary restarts when child was healthy

**Efficiency**:
- Average retries per task
- Escalation rate (% tasks that escalate to grandparent)
- Final failure rate (% tasks that fail permanently)
- Cost per retry (token cost of retries)

**Performance**:
- Heartbeat latency (p50, p95, p99)
- Database write throughput (heartbeats/sec)
- Monitoring agent latency (detection lag)

### 7.2 Dashboards

**HMI Health Dashboard**:
- Real-time heartbeat map (all agents, green/yellow/red)
- Active retries count
- Escalations in progress
- Recent alerts (last 10)
- System health score (0-100)

**HMI Metrics Dashboard**:
- Failure rate by agent type (chart)
- Failure rate by LLM (chart)
- Retry count histogram
- Escalation rate over time (chart)
- Cost of retries (cumulative)

### 7.3 Alerts

**Critical**:
- Task failed permanently (after max failures)
- Cascading failure detected (>30% agents failing)
- Monitoring agent down

**Warning**:
- Child restarted (restart_count >= 2)
- Grandparent escalation (max restarts exceeded)
- Repeated failures (same agent/LLM failing >5 times)

**Info**:
- Child restarted (restart_count = 1)
- Unexpected delay (task taking >3x estimate)

---

## 8. Risks & Mitigations

### 8.1 Technical Risks

**Risk**: Heartbeat overhead slows down system
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Use lightweight heartbeat messages (<1KB)
  - Batch database writes (every 10 heartbeats)
  - Configurable interval (users can increase if needed)

**Risk**: False positives (unnecessary restarts of healthy children)
- **Likelihood**: Medium
- **Impact**: High (wasted tokens/time)
- **Mitigation**:
  - Poll child before restart (give it a chance to respond)
  - Use 2x heartbeat interval for timeout (60s default)
  - Track false positive rate in metrics, tune thresholds

**Risk**: Database contention (many agents writing heartbeats)
- **Likelihood**: Low (SQLite handles 50k writes/sec)
- **Impact**: Medium
- **Mitigation**:
  - Use WAL mode for concurrent writes
  - Batch writes where possible
  - Monitor database performance

**Risk**: Monitoring agent becomes single point of failure
- **Likelihood**: Low
- **Impact**: High
- **Mitigation**:
  - Monitoring agent is optional (parents still monitor children)
  - Auto-restart monitoring agent on failure
  - HMI shows "monitoring agent down" warning

### 8.2 Operational Risks

**Risk**: Configuration errors (heartbeat interval too short, max restarts too high)
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Sensible defaults (30s, 3 restarts, 3 failures)
  - Validation in settings UI (min/max values)
  - Documentation with examples

**Risk**: Runaway retry costs (many retries with expensive LLM)
- **Likelihood**: Low
- **Impact**: High
- **Mitigation**:
  - Max failures limit prevents infinite retries
  - Escalation can switch to cheaper LLM (Anthropic → Ollama)
  - Cost tracking in metrics

---

## 9. Future Enhancements (V2)

### Predictive Failure Detection
Use ML on failure_metrics to predict failures before they happen:
- Train classifier on features: agent type, LLM, task type, time of day, resource usage
- Predict P(failure) for each active task
- Proactively restart or switch LLM when P(failure) > threshold

### Auto-Tuning
Optimize configuration based on observed metrics:
- Decrease heartbeat interval for critical tasks
- Increase max_restarts for flaky but valuable tasks
- Adjust timeout multiplier based on agent type

### Cascading Failure Prevention
Smart retry logic during system-wide issues:
- Detect correlated failures (same root cause)
- Pause retries during outage
- Resume retries after root cause resolved

### Cost-Aware Retry
Factor in token costs when deciding retry strategy:
- Use cheaper LLM for retries (if acceptable quality)
- Skip retries for low-value tasks
- Aggregate retry budget per Prime Directive

---

## 10. Appendix

### A. Glossary

- **Heartbeat**: Periodic status message from child to parent
- **Timeout**: Time threshold for declaring child as failed (2x heartbeat interval)
- **Restart**: Kill and restart child with same configuration
- **Escalation**: Hand off failed task to grandparent after max restarts
- **Cascading Failure**: Multiple agents failing simultaneously due to shared root cause
- **False Positive**: Unnecessary restart of healthy child (mistaken for failed)

### B. Related Documents

- `docs/P0_END_TO_END_INTEGRATION.md` - PAS architecture
- `docs/COMMS_LOGGING_GUIDE.md` - Communication logging
- `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` - PLMS integration
- `docs/contracts/DIRENG_SYSTEM_PROMPT.md` - DirEng role

### C. References

- Heartbeat pattern: https://martinfowler.com/articles/patterns-of-distributed-systems/heartbeat.html
- Circuit breaker: https://martinfowler.com/bliki/CircuitBreaker.html
- Exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff

---

**End of PRD**
