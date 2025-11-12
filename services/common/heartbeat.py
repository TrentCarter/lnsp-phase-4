#!/usr/bin/env python3
"""
Heartbeat & Monitoring System for PAS Multi-Tier Architecture

Responsibilities:
- Track agent heartbeats (60s intervals)
- Detect missed heartbeats (2-miss rule)
- Escalate failures to parent agents
- Aggregate status across agent hierarchy
- Health dashboard data

Usage:
    from services.common.heartbeat import HeartbeatMonitor

    monitor = HeartbeatMonitor()
    monitor.register_agent("Architect", parent="PAS Root")
    monitor.heartbeat("Architect", state="planning", message="Decomposing Prime Directive")

    # Check health
    health = monitor.get_health("Architect")
    if not health["healthy"]:
        print(f"Agent unhealthy: {health['reason']}")
"""
import time
import threading
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import sqlite3
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Standard agent states"""
    IDLE = "idle"
    PLANNING = "planning"
    DELEGATING = "delegating"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    VALIDATING = "validating"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    HIBERNATED = "hibernated"


@dataclass
class HeartbeatRecord:
    """Single heartbeat record from an agent"""
    agent: str
    run_id: Optional[str]
    timestamp: float
    state: AgentState
    message: str
    llm_model: Optional[str] = None
    parent_agent: Optional[str] = None
    children_agents: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHealth:
    """Agent health status"""
    agent: str
    healthy: bool
    last_heartbeat: float
    missed_count: int
    state: AgentState
    message: str
    reason: Optional[str] = None  # Why unhealthy (if applicable)


class HeartbeatMonitor:
    """
    Global heartbeat monitor for all PAS agents

    Features:
    - 60s heartbeat interval tracking
    - 2-miss escalation (agent considered failed after 2 missed heartbeats)
    - Parent-child hierarchy tracking
    - Status aggregation
    - Thread-safe operations
    """

    # Singleton instance
    _instance: Optional['HeartbeatMonitor'] = None
    _lock = threading.Lock()

    # Default constants (can be overridden by settings)
    DEFAULT_HEARTBEAT_INTERVAL_S = 30
    DEFAULT_TIMEOUT_THRESHOLD_S = 60
    DEFAULT_MAX_RESTARTS = 3
    DEFAULT_MAX_LLM_RETRIES = 3
    MISS_THRESHOLD = 2  # Number of missed heartbeats before escalation

    def __new__(cls):
        """Singleton pattern - only one monitor instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    # Agent port mapping (for HHMRS Phase 1 parent alerting)
    AGENT_PORT_MAP = {
        "PAS Root": 6100,
        "Architect": 6110,
        "Director-Code": 6111,
        "Director-Models": 6112,
        "Director-Data": 6113,
        "Director-DevSecOps": 6114,
        "Director-Docs": 6115,
    }

    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from artifacts/pas_settings.json
        Falls back to defaults if file doesn't exist or is invalid
        """
        settings_file = Path("artifacts/pas_settings.json")
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    logger.info(f"TRON loaded settings from {settings_file}")
                    return settings.get("hhmrs", {})
            except Exception as e:
                logger.warning(f"Failed to load settings from {settings_file}: {e}, using defaults")
        else:
            logger.info(f"Settings file not found at {settings_file}, using defaults")
        return {}

    def __init__(self):
        """Initialize heartbeat monitor (only once)"""
        if self._initialized:
            return

        self._initialized = True
        self._heartbeats: Dict[str, HeartbeatRecord] = {}  # agent -> latest heartbeat
        self._registry: Dict[str, Dict[str, Any]] = {}  # agent -> metadata
        self._lock = threading.RLock()  # Reentrant lock for nested access

        # HHMRS Phase 1: Retry tracking
        self._retry_counts: Dict[str, int] = {}  # agent_id -> restart_count
        self._failure_counts: Dict[str, int] = {}  # agent_id -> failure_count

        # Load settings and apply to instance
        hhmrs_settings = self._load_settings()
        self.heartbeat_interval_s = hhmrs_settings.get("heartbeat_interval_s", self.DEFAULT_HEARTBEAT_INTERVAL_S)
        self.timeout_threshold_s = hhmrs_settings.get("timeout_threshold_s", self.DEFAULT_TIMEOUT_THRESHOLD_S)
        self.max_restarts = hhmrs_settings.get("max_restarts", self.DEFAULT_MAX_RESTARTS)
        self.max_llm_retries = hhmrs_settings.get("max_llm_retries", self.DEFAULT_MAX_LLM_RETRIES)
        self.enable_auto_restart = hhmrs_settings.get("enable_auto_restart", True)
        self.enable_llm_switching = hhmrs_settings.get("enable_llm_switching", True)

        logger.info(
            f"TRON initialized with: heartbeat_interval={self.heartbeat_interval_s}s, "
            f"timeout_threshold={self.timeout_threshold_s}s, "
            f"max_restarts={self.max_restarts}, max_llm_retries={self.max_llm_retries}"
        )

        # Start background checker thread
        self._checker_thread = threading.Thread(target=self._check_health_loop, daemon=True)
        self._checker_thread.start()

    def register_agent(
        self,
        agent: str,
        parent: Optional[str] = None,
        llm_model: Optional[str] = None,
        role: Optional[str] = None,
        tier: Optional[str] = None
    ) -> None:
        """
        Register an agent in the monitoring system

        Args:
            agent: Agent ID (e.g., "Architect", "Dir-Code", "Mgr-Code-01")
            parent: Parent agent ID (e.g., "PAS Root")
            llm_model: LLM model used by this agent
            role: Agent role (architect, director, manager, programmer)
            tier: Agent tier (coordinator, executor)
        """
        with self._lock:
            self._registry[agent] = {
                "parent": parent,
                "llm_model": llm_model,
                "role": role,
                "tier": tier,
                "registered_at": time.time()
            }

    def heartbeat(
        self,
        agent: str,
        run_id: Optional[str] = None,
        state: AgentState = AgentState.IDLE,
        message: str = "",
        llm_model: Optional[str] = None,
        parent_agent: Optional[str] = None,
        children_agents: Optional[List[str]] = None,
        progress: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a heartbeat from an agent

        Args:
            agent: Agent ID
            run_id: Current run ID (if active)
            state: Current agent state
            message: Status message
            llm_model: LLM model being used
            parent_agent: Parent agent ID
            children_agents: List of child agent IDs
            progress: Progress (0.0 to 1.0)
            metadata: Additional metadata
        """
        with self._lock:
            record = HeartbeatRecord(
                agent=agent,
                run_id=run_id,
                timestamp=time.time(),
                state=state,
                message=message,
                llm_model=llm_model,
                parent_agent=parent_agent,
                children_agents=children_agents or [],
                progress=progress,
                metadata=metadata or {}
            )
            self._heartbeats[agent] = record

    def get_health(self, agent: str) -> AgentHealth:
        """
        Get health status for an agent

        Returns:
            AgentHealth object with healthy flag and details
        """
        with self._lock:
            if agent not in self._heartbeats:
                return AgentHealth(
                    agent=agent,
                    healthy=False,
                    last_heartbeat=0,
                    missed_count=0,
                    state=AgentState.IDLE,
                    message="No heartbeat recorded",
                    reason="Agent not initialized"
                )

            record = self._heartbeats[agent]
            now = time.time()
            time_since_last = now - record.timestamp
            missed_count = int(time_since_last / self.heartbeat_interval_s)

            # Healthy if last heartbeat within timeout (use settings value)
            healthy = time_since_last < self.timeout_threshold_s
            reason = None if healthy else f"No heartbeat for {time_since_last:.0f}s ({missed_count} missed)"

            return AgentHealth(
                agent=agent,
                healthy=healthy,
                last_heartbeat=record.timestamp,
                missed_count=missed_count,
                state=record.state,
                message=record.message,
                reason=reason
            )

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """Get health status for all registered agents"""
        with self._lock:
            return {agent: self.get_health(agent) for agent in self._heartbeats.keys()}

    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agent IDs"""
        with self._lock:
            return [
                agent
                for agent, health in self.get_all_health().items()
                if not health.healthy
            ]

    def get_hierarchy(self, root: str = "PAS Root") -> Dict[str, Any]:
        """
        Get agent hierarchy tree starting from root

        Returns:
            Nested dict structure showing parent-child relationships
        """
        with self._lock:
            def build_tree(agent: str) -> Dict[str, Any]:
                children = [
                    hb.agent
                    for hb in self._heartbeats.values()
                    if hb.parent_agent == agent
                ]

                health = self.get_health(agent)

                return {
                    "agent": agent,
                    "healthy": health.healthy,
                    "state": health.state.value,
                    "message": health.message,
                    "children": [build_tree(child) for child in children]
                }

            return build_tree(root)

    def get_status_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get executive status summary

        Args:
            run_id: Filter by run ID (if specified)

        Returns:
            Summary dict with overall health, agent counts, progress
        """
        with self._lock:
            # Filter by run_id if specified
            if run_id:
                relevant_agents = [
                    agent
                    for agent, hb in self._heartbeats.items()
                    if hb.run_id == run_id
                ]
            else:
                relevant_agents = list(self._heartbeats.keys())

            all_health = self.get_all_health()

            healthy_count = sum(1 for a in relevant_agents if all_health[a].healthy)
            unhealthy_count = len(relevant_agents) - healthy_count

            # Aggregate progress
            total_progress = sum(
                self._heartbeats[a].progress
                for a in relevant_agents
                if a in self._heartbeats
            )
            avg_progress = total_progress / len(relevant_agents) if relevant_agents else 0.0

            # Count by state
            state_counts = {}
            for agent in relevant_agents:
                state = self._heartbeats[agent].state.value
                state_counts[state] = state_counts.get(state, 0) + 1

            return {
                "total_agents": len(relevant_agents),
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "overall_healthy": unhealthy_count == 0,
                "avg_progress": round(avg_progress, 2),
                "state_counts": state_counts,
                "timestamp": time.time()
            }

    def save_snapshot(self, path: Path) -> None:
        """Save current heartbeat state to JSON file"""
        with self._lock:
            snapshot = {
                "timestamp": time.time(),
                "heartbeats": {
                    agent: asdict(record)
                    for agent, record in self._heartbeats.items()
                },
                "registry": self._registry
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2))

    def load_snapshot(self, path: Path) -> None:
        """Load heartbeat state from JSON file"""
        with self._lock:
            snapshot = json.loads(path.read_text())

            self._heartbeats = {
                agent: HeartbeatRecord(**data)
                for agent, data in snapshot["heartbeats"].items()
            }
            self._registry = snapshot["registry"]

    def _get_agent_url(self, agent_id: str) -> str:
        """
        Resolve agent ID to URL

        Args:
            agent_id: Agent identifier (e.g., "Architect", "Director-Code")

        Returns:
            Full URL for the agent (e.g., "http://127.0.0.1:6110")
        """
        port = self.AGENT_PORT_MAP.get(agent_id)
        if not port:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        return f"http://127.0.0.1:{port}"

    def _record_timeout(self, agent_id: str, parent_id: str, restart_count: int) -> None:
        """
        Record timeout event in retry_history table

        Args:
            agent_id: Child agent that timed out
            parent_id: Parent agent being alerted
            restart_count: Current restart count for this agent
        """
        try:
            db_path = Path("artifacts/registry/registry.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get run_id from heartbeat record if available
            record = self._heartbeats.get(agent_id)
            run_id = record.run_id if record else None

            cursor.execute("""
                INSERT INTO retry_history (
                    run_id, task_id, agent_id, retry_type, retry_count,
                    reason, old_config, new_config, timestamp, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                None,  # task_id - not available at this level
                agent_id,
                "child_timeout",
                restart_count,
                "heartbeat_timeout",
                None,  # old_config
                None,  # new_config
                datetime.now().isoformat(),
                "pending"  # Will be updated by parent handler
            ))

            conn.commit()
            conn.close()

            logger.info(f"Recorded timeout for {agent_id} (restart_count={restart_count})")

        except Exception as e:
            logger.error(f"Failed to record timeout in database: {e}")

    def _alert_parent(
        self,
        parent_id: str,
        child_id: str,
        reason: str,
        restart_count: int,
        last_seen: float
    ) -> None:
        """
        Send timeout alert to parent via RPC

        Args:
            parent_id: Parent agent ID
            child_id: Child agent that timed out
            reason: Reason for timeout
            restart_count: Current restart count
            last_seen: Timestamp of last heartbeat
        """
        try:
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

        except Exception as e:
            logger.error(f"Failed to alert parent {parent_id}: {e}")

    def _handle_timeout(self, agent_id: str, health: AgentHealth) -> None:
        """
        Handle agent timeout - alert parent

        Args:
            agent_id: Agent that timed out
            health: Health status of the agent
        """
        with self._lock:
            record = self._heartbeats.get(agent_id)
            if not record:
                logger.warning(f"No heartbeat record for {agent_id}")
                return

            parent_id = record.parent_agent
            if not parent_id:
                logger.warning(f"Agent {agent_id} has no parent, cannot escalate")
                return

            # Get retry count
            restart_count = self._retry_counts.get(agent_id, 0)

            logger.warning(
                f"Agent {agent_id} timeout detected "
                f"(last_seen={health.last_heartbeat:.1f}s ago, restart_count={restart_count})"
            )

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
                logger.error(f"Failed to handle timeout for {agent_id}: {e}")

    def _check_health_loop(self) -> None:
        """Background thread to check agent health periodically"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds

                unhealthy = self.get_unhealthy_agents()
                if unhealthy:
                    # HHMRS Phase 1: Handle timeouts for unhealthy agents
                    for agent_id in unhealthy:
                        health = self.get_health(agent_id)

                        # Only handle if agent has exceeded miss threshold
                        if health.missed_count >= self.MISS_THRESHOLD:
                            logger.warning(
                                f"TRON detected timeout: {agent_id} "
                                f"(missed={health.missed_count}, threshold={self.MISS_THRESHOLD})"
                            )
                            self._handle_timeout(agent_id, health)

            except Exception as e:
                # Log errors but continue monitoring (thread should never crash)
                logger.error(f"Health check loop error: {e}")
                pass


# Global singleton instance
_monitor = HeartbeatMonitor()


def get_monitor() -> HeartbeatMonitor:
    """Get global heartbeat monitor instance"""
    return _monitor


def send_progress_heartbeat(agent: str, message: str) -> None:
    """
    Send a progress heartbeat from an agent during long operations.

    This helper function is designed for agents to send heartbeats every 30s
    during long-running operations (LLM calls, waiting for responses, data processing, etc.)
    to prevent TRON timeout detection.

    Args:
        agent: Agent ID (e.g., "Architect", "Dir-Code", "Dir-Models")
        message: Progress message describing current operation
                (e.g., "Step 3/5: Allocating resources", "Epoch 5/10, loss=0.45")

    Example:
        from services.common.heartbeat import send_progress_heartbeat

        # During long operation
        send_progress_heartbeat(
            agent="Architect",
            message="Waiting for Dir-Code response (3/5 Directors)"
        )

    Note:
        - Call this every 30s during operations longer than 30s
        - TRON detects timeout after 60s (2 missed heartbeats)
        - Automatic heartbeats sent every 30s when agent registered
        - Only use for operations that might take >30s

    See: docs/PRDs/PRD_Hierarchical_Health_Monitoring_Retry_System.md
    """
    from services.common.comms_logger import get_logger, MessageType

    monitor = get_monitor()
    logger = get_logger()

    # Send heartbeat to TRON
    monitor.heartbeat(agent)

    # Log progress message for debugging/audit
    logger.log(
        msg_type=MessageType.INFO,
        sender=agent,
        receiver="TRON",
        content=f"Progress heartbeat: {message}"
    )
