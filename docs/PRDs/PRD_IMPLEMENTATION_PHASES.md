# PRD — Implementation Phases for Polyglot Agent Swarm

**Owner:** Trent Carter
**Date:** 2025-11-06
**Status:** Active Implementation Plan
**Related PRDs:** PRD_Polyglot_Agent_Swarm.md, PRD_Human_Machine_Interface_HMI.md, HYBRID_AGENT_ARCHITECTURE.md

---

## Implementation Philosophy

**Principles:**
1. **Management First** - Build orchestration/monitoring before execution agents
2. **Test Each Phase** - Get approval before moving forward
3. **Real-Time Visibility** - HMI dashboard shows everything live
4. **Progressive Enhancement** - Each phase adds capability, doesn't break previous

**Approval Gates:**
- ✅ End of Phase 0: Core infrastructure working
- ✅ End of Phase 1: Management agents functional
- ✅ End of Phase 2: HMI dashboard live
- ✅ End of Phase 3: Gateway routing proven
- ✅ End of Phase 4+: Agent tiers operational

---

## Phase 0: Core Infrastructure (Days 1-2)

### Goal
Build the **foundation services** that all agents depend on:
- Service Registry (registration, discovery, TTL)
- Heartbeat Monitor (liveness detection)
- Central database (SQLite for state)
- Contract schemas (JSON validation)

### Services to Build

#### 1. Service Registry (Port 6121)
**File:** `services/registry/registry_service.py`

**Responsibilities:**
- Service registration (`POST /register`)
- Service discovery (`GET /discover`)
- Heartbeat tracking (`PUT /heartbeat`)
- TTL-based eviction (missed 2 beats → mark down)
- Promotion/demotion (`POST /promote`)

**Database:** SQLite @ `artifacts/registry/registry.db`

**Tables:**
```sql
CREATE TABLE services (
    service_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'model' | 'tool' | 'agent'
    role TEXT NOT NULL,  -- 'production' | 'staging' | 'canary' | 'experimental'
    url TEXT NOT NULL,
    caps TEXT NOT NULL,  -- JSON array
    labels TEXT,         -- JSON object
    ctx_limit INTEGER,
    cost_hint_usd_per_1k REAL,
    heartbeat_interval_s INTEGER DEFAULT 60,
    ttl_s INTEGER DEFAULT 90,
    status TEXT DEFAULT 'ok',  -- 'ok' | 'degraded' | 'down'
    last_heartbeat_ts TEXT,
    registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    p95_ms REAL,
    queue_depth INTEGER,
    load REAL
);

CREATE INDEX idx_services_name_role ON services(name, role);
CREATE INDEX idx_services_type_role ON services(type, role);
CREATE INDEX idx_services_status ON services(status);
```

**API Endpoints:**
- `POST /register` - Register new service
- `PUT /heartbeat` - Update service health
- `GET /discover?type=model&role=production&cap=infer` - Find services
- `POST /promote` - Change service role
- `POST /deregister` - Remove service
- `GET /health` - Registry health check

#### 2. Heartbeat Monitor (Port 6109)
**File:** `services/heartbeat_monitor/heartbeat_monitor.py`

**Responsibilities:**
- Background task: Check Registry every 30s
- Identify services with `last_heartbeat_ts` > TTL
- Mark services as `down` if 2 beats missed
- Deregister services after 3 beats missed
- Emit alerts to Event Stream (6102)

**Database:** Shares Registry DB

**Alerts:**
```json
{
  "alert_type": "heartbeat_miss",
  "service_id": "uuid",
  "service_name": "llama-3.1-8b",
  "missed_beats": 2,
  "last_seen": "2025-11-06T10:30:00Z",
  "action": "marked_down",
  "ts": "2025-11-06T10:31:00Z"
}
```

#### 3. Contract Schemas
**Directory:** `contracts/`

**Files to Create:**
- `service_registration.schema.json`
- `service_heartbeat.schema.json`
- `service_discovery_query.schema.json`
- `heartbeat_alert.schema.json`
- `job_card.schema.json`
- `resource_request.schema.json`
- `agent_config.schema.json`

**Validation:**
- FastAPI Pydantic models for type checking
- pytest contract tests in `tests/contracts/`

### Acceptance Criteria (Phase 0)

- [ ] Registry service starts on port 6121
- [ ] Can register a mock service via `POST /register`
- [ ] Can discover services via `GET /discover`
- [ ] Heartbeat Monitor detects missed beats (30s intervals)
- [ ] Services marked `down` after 2 misses, deregistered after 3
- [ ] SQLite database persists across restarts
- [ ] Contract schemas validate with pytest
- [ ] Health endpoints return 200 OK

**Test Command:**
```bash
# Start Registry
./.venv/bin/uvicorn services.registry.registry_service:app --host 127.0.0.1 --port 6121 &

# Start Heartbeat Monitor
./.venv/bin/uvicorn services.heartbeat_monitor.heartbeat_monitor:app --host 127.0.0.1 --port 6109 &

# Register a mock service
curl -X POST http://localhost:6121/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-service",
    "type": "model",
    "role": "experimental",
    "url": "http://127.0.0.1:8888",
    "caps": ["infer"],
    "ctx_limit": 32768
  }'

# Discover it
curl "http://localhost:6121/discover?role=experimental"

# Wait 3 minutes (90s TTL + 60s interval), check it's marked down
curl "http://localhost:6121/discover?status=down"
```

**Approval Gate:** Run tests, review logs, confirm TTL eviction working

---

## Phase 1: Management Agents (Days 3-4)

### Goal
Build the **management layer** that allocates resources and governs token budgets.

### Services to Build

#### 1. Resource Manager (Port 6104)
**File:** `services/resource_manager/resource_manager.py`

**Responsibilities:**
- Track system resources (CPU, memory, GPU, ports)
- Accept reservation requests from agents
- Grant/deny based on quotas
- Release resources on job completion
- Kill jobs on hard timeout

**Database:** SQLite @ `artifacts/resource_manager/resources.db`

**Tables:**
```sql
CREATE TABLE quotas (
    resource_type TEXT PRIMARY KEY,  -- 'cpu' | 'mem_mb' | 'gpu' | 'gpu_mem_mb' | 'ports'
    total_capacity REAL NOT NULL,
    allocated REAL DEFAULT 0,
    reserved REAL DEFAULT 0
);

CREATE TABLE reservations (
    reservation_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    cpu REAL,
    mem_mb INTEGER,
    gpu INTEGER,
    gpu_mem_mb INTEGER,
    ports TEXT,  -- JSON array
    status TEXT DEFAULT 'active',  -- 'active' | 'released' | 'expired'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

CREATE INDEX idx_reservations_job_id ON reservations(job_id);
CREATE INDEX idx_reservations_status ON reservations(status);
```

**API Endpoints:**
- `POST /reserve` - Request resources
- `POST /release` - Release resources
- `GET /quotas` - View capacity and allocation
- `GET /reservations` - List active reservations
- `POST /kill` - Force-kill a job and release resources

**Example Reservation:**
```json
{
  "job_id": "J-2025-11-06-001",
  "agent": "Q-Tower-Trainer",
  "cpu": 4.0,
  "mem_mb": 8192,
  "gpu": 1,
  "gpu_mem_mb": 8192,
  "ports": [9050]
}
```

#### 2. Token Governor (Port 6105)
**File:** `services/token_governor/token_governor.py`

**Responsibilities:**
- Track context usage per agent
- Enforce target (0.50) and hard max (0.75) ratios
- Trigger Save-State → Clear → Resume on breach
- Generate summary artifacts in `docs/runs/<run_id>_summary.md`

**Database:** SQLite @ `artifacts/token_governor/tokens.db`

**Tables:**
```sql
CREATE TABLE agent_contexts (
    agent TEXT PRIMARY KEY,
    run_id TEXT,
    ctx_used INTEGER DEFAULT 0,
    ctx_limit INTEGER NOT NULL,
    ctx_ratio REAL GENERATED ALWAYS AS (CAST(ctx_used AS REAL) / ctx_limit) STORED,
    target_ratio REAL DEFAULT 0.50,
    hard_max_ratio REAL DEFAULT 0.75,
    status TEXT DEFAULT 'ok',  -- 'ok' | 'warning' | 'breach'
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE summarizations (
    summary_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    run_id TEXT NOT NULL,
    trigger_reason TEXT,  -- 'hard_max_breach' | 'manual'
    ctx_before INTEGER,
    ctx_after INTEGER,
    summary_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**API Endpoints:**
- `POST /track` - Update agent context usage
- `GET /status?agent=Architect` - Get context status
- `POST /summarize` - Trigger Save-State → Clear → Resume
- `GET /summaries` - List summarization events

**Save-State Flow:**
```
1. Detect breach (ctx_ratio >= 0.75)
2. Call local LLM (llama-3.1-8b) to summarize thread
3. Write summary to docs/runs/<run_id>_summary.md
4. Clear agent context
5. Resume with trimmed state (summary only)
```

### Acceptance Criteria (Phase 1)

- [ ] Resource Manager starts on port 6104
- [ ] Can reserve resources (CPU, memory, GPU, ports)
- [ ] Reservations denied when quota exceeded
- [ ] Resources released on job completion
- [ ] Token Governor starts on port 6105
- [ ] Context usage tracked per agent
- [ ] Breach triggers summarization (mock LLM OK for testing)
- [ ] Summary artifacts written to `docs/runs/`

**Test Command:**
```bash
# Start Resource Manager
./.venv/bin/uvicorn services.resource_manager.resource_manager:app --host 127.0.0.1 --port 6104 &

# Start Token Governor
./.venv/bin/uvicorn services.token_governor.token_governor:app --host 127.0.0.1 --port 6105 &

# Reserve resources
curl -X POST http://localhost:6104/reserve \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "J-TEST-001",
    "agent": "Q-Tower-Trainer",
    "cpu": 4.0,
    "mem_mb": 8192,
    "gpu": 1,
    "gpu_mem_mb": 8192,
    "ports": [9050]
  }'

# Check quotas
curl http://localhost:6104/quotas

# Track context usage
curl -X POST http://localhost:6105/track \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Architect",
    "ctx_used": 12000,
    "ctx_limit": 16000
  }'

# Check context status (should be warning at 0.75)
curl "http://localhost:6105/status?agent=Architect"
```

**Approval Gate:** Demonstrate resource reservation and token governance working

---

## Phase 2: Flask HMI Dashboard (Days 5-7)

### Goal
Build the **live web dashboard** with real-time updates, agent hierarchy tree, and resource/task visualization.

### Services to Build

#### 1. Event Stream (Port 6102)
**File:** `services/event_stream/event_stream.py`

**Responsibilities:**
- WebSocket server (Flask-SocketIO)
- Pub/sub for events (heartbeats, status updates, alerts)
- Broadcast to all connected HMI clients

**Events:**
```json
{
  "event_type": "heartbeat",
  "agent": "Architect",
  "run_id": "R-001",
  "ts": "2025-11-06T10:30:00Z",
  "progress": 0.35,
  "status": "running",
  "token_usage": {"ctx_used": 8000, "ctx_limit": 16000},
  "resources": {"cpu": 2.5, "mem_mb": 4096}
}
```

#### 2. Flask HMI Web UI (Port 6101)
**File:** `services/webui/hmi_app.py`

**Pages:**
1. **Dashboard** (`/`) - Global overview
2. **Agent Tree** (`/tree`) - Interactive hierarchy
3. **Sequencer** (`/sequencer`) - Timeline view
4. **Resources** (`/resources`) - Quotas and allocations
5. **Costs** (`/costs`) - Token and $ tracking
6. **Alerts** (`/alerts`) - Missed heartbeats, breaches

**Technology Stack:**
- **Backend:** Flask + Flask-SocketIO
- **Frontend:** HTML5 + JavaScript (Vanilla or Alpine.js for reactivity)
- **Tree Visualization:** D3.js (collapsible tree) or jsTree (simpler)
- **Charts:** Chart.js for resource/cost charts
- **WebSocket:** Socket.IO client for live updates

**Key Features:**

**A) Agent Tree Visualization**
```javascript
// D3.js collapsible tree
// Node size = ctx_used / ctx_limit (8px to 48px)
// Node color = status (queued=gray, running=blue, error=red, done=green)
// Edge pulse = message throughput
{
  "name": "Architect",
  "status": "running",
  "ctx_used": 8000,
  "ctx_limit": 16000,
  "children": [
    {
      "name": "Director-Code",
      "status": "running",
      "children": [
        {"name": "Manager-Code-North", "status": "running"},
        {"name": "Manager-Code-South", "status": "idle"}
      ]
    },
    {
      "name": "Director-Models",
      "status": "running",
      "children": [
        {"name": "Q-Tower-Trainer", "status": "running"},
        {"name": "Reranker-Trainer", "status": "queued"}
      ]
    }
  ]
}
```

**B) Real-Time Status Cards**
```html
<div class="agent-card" id="agent-architect">
  <h3>Architect</h3>
  <div class="status running">Running</div>
  <div class="progress">
    <div class="bar" style="width: 35%;"></div>
    <span>35%</span>
  </div>
  <div class="context">
    <span>Context: 8,000 / 16,000</span>
    <div class="bar" style="width: 50%;"></div>
  </div>
  <div class="resources">
    <span>CPU: 2.5 / 4.0</span>
    <span>Memory: 4,096 / 8,192 MB</span>
  </div>
  <div class="heartbeat">
    <span>Last beat: 2s ago</span>
    <span class="status-indicator green"></span>
  </div>
</div>
```

**C) Global Controls**
```html
<div class="global-controls">
  <button onclick="pauseAll()">Pause All</button>
  <button onclick="resumeAll()">Resume All</button>
  <div class="settings">
    <label>Update Frequency:
      <select id="update-freq">
        <option value="1000">1s (High)</option>
        <option value="5000" selected>5s (Normal)</option>
        <option value="10000">10s (Low)</option>
      </select>
    </label>
    <label>
      <input type="checkbox" id="enable-sounds"> Enable Sounds
    </label>
    <label>
      <input type="checkbox" id="enable-narration"> Enable Narration
    </label>
  </div>
</div>
```

**D) WebSocket Integration**
```javascript
const socket = io('http://localhost:6102');

socket.on('heartbeat', (data) => {
  updateAgentCard(data.agent, {
    status: data.status,
    progress: data.progress,
    ctx_used: data.token_usage.ctx_used,
    ctx_limit: data.token_usage.ctx_limit,
    cpu: data.resources.cpu,
    mem_mb: data.resources.mem_mb,
    last_heartbeat: new Date(data.ts)
  });
});

socket.on('status_update', (data) => {
  if (data.event === 'completed') {
    playSound('complete');
    narrate(`${data.agent} completed task`);
  }
});

socket.on('alert', (data) => {
  if (data.alert_type === 'heartbeat_miss') {
    showAlert(`${data.service_name} missed ${data.missed_beats} heartbeats`, 'error');
    playSound('alert');
  }
});
```

### Database Integration

**HMI Backend Queries Registry DB:**
```python
# services/webui/hmi_app.py
import sqlite3

def get_agent_tree():
    """Build agent hierarchy from Registry DB"""
    conn = sqlite3.connect('artifacts/registry/registry.db')
    cursor = conn.cursor()

    # Get all agents with their parent relationships
    # (Assume agents register with 'parent' label)
    cursor.execute("""
        SELECT service_id, name, type, role, status,
               json_extract(labels, '$.parent') as parent
        FROM services
        WHERE type = 'agent'
        ORDER BY name
    """)

    agents = cursor.fetchall()
    conn.close()

    # Build tree structure
    return build_tree(agents)
```

### Acceptance Criteria (Phase 2)

- [ ] Event Stream starts on port 6102 (WebSocket)
- [ ] Flask HMI starts on port 6101
- [ ] Can view agent tree at `http://localhost:6101/tree`
- [ ] Tree updates in real-time (<1s lag) when services register
- [ ] Agent cards show status, progress, context, resources
- [ ] Heartbeat indicator shows green (recent) or red (missed)
- [ ] Global controls (update freq, sounds, narration) working
- [ ] WebSocket events received from Registry/Heartbeat Monitor
- [ ] Can pause/resume agents (mock OK for testing)

**Test Command:**
```bash
# Start Event Stream
./.venv/bin/python services/event_stream/event_stream.py &

# Start HMI Web UI
./.venv/bin/python services/webui/hmi_app.py &

# Open browser
open http://localhost:6101

# Register mock agents with parent relationships
curl -X POST http://localhost:6121/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Architect",
    "type": "agent",
    "role": "coordinator",
    "url": "http://127.0.0.1:8000",
    "caps": ["plan"],
    "labels": {"tier": "coordinator", "parent": null}
  }'

curl -X POST http://localhost:6121/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Director-Code",
    "type": "agent",
    "role": "coordinator",
    "url": "http://127.0.0.1:8001",
    "caps": ["manage"],
    "labels": {"tier": "coordinator", "parent": "Architect"}
  }'

# Watch tree update in browser
```

**Approval Gate:** Demonstrate live HMI with real-time agent tree and status updates

---

## Phase 3: Gateway & Routing (Days 8-9)

### Goal
Build the **routing layer** that directs requests to appropriate agents based on capability, cost, and SLA.

### Services to Build

#### 1. Provider Router (Port 6103)
**File:** `services/router/provider_router.py`

**Responsibilities:**
- Load provider matrix (`config/providers.matrix.json`)
- Match capability to available services
- Apply routing policy (cheapest, fastest, etc.)
- Return candidate list to Gateway

#### 2. Gateway (Port 6120)
**File:** `services/gateway/gateway_service.py`

**Responsibilities:**
- Single client entrypoint
- Discover services via Registry
- Route requests via Provider Router
- Forward to upstream service
- Generate routing receipts
- Save receipts to `artifacts/costs/<run_id>.json`

**API Endpoints:**
- `POST /invoke` - Route and execute request
- `GET /health` - Gateway health

**Example Invocation:**
```bash
curl -X POST http://localhost:6120/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "target": {
      "type": "model",
      "role": "production",
      "require_caps": ["classify"]
    },
    "payload": {
      "text": "quantum entanglement physics"
    },
    "policy": {
      "prefer": "local",
      "timeout_s": 30
    }
  }'
```

**Routing Receipt:**
```json
{
  "run_id": "gw-2025-11-06-10-30-00-123456",
  "target_request": {
    "type": "model",
    "role": "production",
    "require_caps": ["classify"]
  },
  "resolved": {
    "service_id": "uuid-123",
    "name": "llama-3.1-8b",
    "url": "http://127.0.0.1:8050"
  },
  "timings_ms": {
    "discovery": 2,
    "upstream": 512,
    "total": 514
  },
  "status": "ok",
  "cost_estimate": {"usd": 0.0},
  "ts": "2025-11-06T10:30:00Z"
}
```

### Acceptance Criteria (Phase 3)

- [ ] Provider Router loads matrix and matches capabilities
- [ ] Gateway starts on port 6120
- [ ] Can invoke via Gateway with capability filters
- [ ] Gateway discovers services from Registry
- [ ] Gateway routes to correct upstream service
- [ ] Routing receipts saved to `artifacts/costs/`
- [ ] HMI shows routing receipts in real-time

**Approval Gate:** Demonstrate end-to-end routing (Gateway → Registry → Upstream → Receipt)

---

## Phase 4: Claude Sub-Agents (Day 10)

### Goal
Create all **42 Claude Code sub-agents** in `.claude/agents/`

**Agents to Create:**
- 23 Coordinator/System agents
- 19 Execution agents

**Format:** Markdown with YAML frontmatter

**Example:** `.claude/agents/corpus-auditor.md`
```markdown
---
name: corpus-auditor
description: Validate data sources, licensing, and dataset statistics
tools: Read, Glob, Grep, Bash, Write
model: inherit
---

You are the Corpus Auditor agent. Your responsibilities:

1. Check file licensing and attribution
2. Validate encoding and format consistency
3. Generate dataset statistics reports
4. Flag PII or sensitive content

Constraints:
- Token budget: 0.30/0.50 (target/hard)
- Heartbeat every 60s to Registry (6121)
- Write reports to artifacts/corpus_reports/
- Request approval before deletions

Parent: Manager-Data
Children: Cleaner/Normalizer

Registry URL: http://localhost:6121
```

### Acceptance Criteria (Phase 4)

- [ ] All 42 agents created in `.claude/agents/`
- [ ] Each agent has proper YAML frontmatter
- [ ] Each agent specifies parent/children relationships
- [ ] Each agent includes Registry URL and heartbeat instructions
- [ ] Can invoke agents via Task tool
- [ ] Agents appear in HMI tree after first invocation

---

## Phase 5: Local LLM Services (Days 11-12)

### Goal
Wrap existing local LLMs (llama-3.1-8b, TinyLlama) as FastAPI services with Registry integration

**Services to Create:**
- llama-3.1-8b wrapper @ :8050
- TinyLlama wrapper @ :8051
- TLC Domain Classifier @ :8052

### Acceptance Criteria (Phase 5)

- [ ] Local LLM services register with Registry on startup
- [ ] Services send heartbeats every 60s
- [ ] Can invoke via Gateway with capability='classify_light'
- [ ] Gateway routes to local services (prefer local policy)
- [ ] HMI shows local services in tree

---

## Phase 6: External API Adapters (Days 13-14)

### Goal
Create FastAPI adapters for external LLM APIs (OpenAI, Anthropic, Gemini, Grok)

**Services to Create:**
- OpenAI adapter @ :8100
- Anthropic adapter @ :8101
- Gemini adapter @ :8102
- Grok adapter @ :8103

**Credentials:** Load from `.env`

### Acceptance Criteria (Phase 6)

- [ ] External adapters register with Registry
- [ ] Credentials loaded from `.env` correctly
- [ ] Can invoke via Gateway with capability='plan'
- [ ] Gateway routes to external services when local unavailable
- [ ] Cost estimates included in routing receipts
- [ ] HMI shows cost breakdown per service

---

## Directory Structure

```
lnsp-phase-4/
  .claude/
    agents/               # Claude sub-agents (Phase 4)
      architect.md
      director-code.md
      corpus-auditor.md
      ...
  services/
    registry/             # Phase 0
      registry_service.py
    heartbeat_monitor/    # Phase 0
      heartbeat_monitor.py
    resource_manager/     # Phase 1
      resource_manager.py
    token_governor/       # Phase 1
      token_governor.py
    event_stream/         # Phase 2
      event_stream.py
    webui/                # Phase 2
      hmi_app.py
      templates/
        dashboard.html
        tree.html
        sequencer.html
      static/
        js/
          tree.js
          websocket.js
        css/
          hmi.css
    router/               # Phase 3
      provider_router.py
    gateway/              # Phase 3
      gateway_service.py
    local_llm/            # Phase 5
      llama_service.py
      tinyllama_service.py
      tlc_classifier.py
    external_llm/         # Phase 6
      openai_adapter.py
      anthropic_adapter.py
      gemini_adapter.py
      grok_adapter.py
  contracts/              # JSON schemas
    service_registration.schema.json
    heartbeat.schema.json
    job_card.schema.json
    resource_request.schema.json
    agent_config.schema.json
  config/
    providers.matrix.json # Routing policy
  artifacts/
    registry/
      registry.db         # SQLite
    resource_manager/
      resources.db
    token_governor/
      tokens.db
    costs/                # Routing receipts
      gw-*.json
    hmi/
      events/             # Event log
  tests/
    contracts/            # Contract tests
    services/             # Service tests
```

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/contracts/ -v          # Schema validation
pytest tests/services/registry/ -v  # Registry CRUD
pytest tests/services/gateway/ -v   # Routing logic
```

### Integration Tests
```bash
# Phase 0: Registry + Heartbeat Monitor
pytest tests/integration/test_phase0.py -v

# Phase 1: Add Resource Manager + Token Governor
pytest tests/integration/test_phase1.py -v

# Phase 2: Add HMI WebSocket events
pytest tests/integration/test_phase2.py -v

# Phase 3: Add Gateway routing
pytest tests/integration/test_phase3.py -v
```

### E2E Smoke Test
```bash
# Start all services
./scripts/start_pas_services.sh

# Run smoke test
python tests/e2e/test_full_workflow.py

# Expected: Architect → Gateway → Local LLM → Receipt → HMI update
```

---

## Monitoring & Observability

### Logs
```bash
# Centralized logging
tail -f /tmp/pas_logs/*.log

# Per-service logs
tail -f /tmp/pas_logs/registry.log
tail -f /tmp/pas_logs/gateway.log
```

### Metrics (Future)
- Prometheus @ :9090
- Grafana @ :3000

---

## Rollback Plan

**If any phase fails:**
1. Stop new services
2. Revert to previous phase commit
3. Restore database from backup
4. Review logs and fix issues
5. Re-test and retry

**Database Backups:**
```bash
# Backup before each phase
cp artifacts/registry/registry.db artifacts/registry/registry.db.phase1.bak
```

---

## Success Metrics

### Phase 0
- Registry uptime: 100% over 24h
- Heartbeat detection: <30s lag
- TTL eviction: 100% accuracy

### Phase 1
- Resource reservation success rate: >99%
- Token governance breaches detected: 100%
- Summary generation time: <5s

### Phase 2
- HMI event lag: <1s P95
- WebSocket connection stability: >99.9%
- Tree rendering: <250ms P95

### Phase 3
- Gateway routing success rate: >99%
- Cost receipt generation: 100%
- Routing latency: <10ms P95

---

## Timeline Summary

| Phase | Days | Deliverables |
|-------|------|--------------|
| 0     | 1-2  | Registry + Heartbeat Monitor |
| 1     | 3-4  | Resource Manager + Token Governor |
| 2     | 5-7  | Flask HMI + WebSocket + Tree viz |
| 3     | 8-9  | Gateway + Router + Receipts |
| 4     | 10   | 42 Claude sub-agents |
| 5     | 11-12| Local LLM wrappers |
| 6     | 13-14| External API adapters |
| **Total** | **14 days** | **Full PAS operational** |

---

## Open Questions

1. SQLite vs PostgreSQL for Registry database?
   - **Recommendation:** Start with SQLite, migrate to PostgreSQL if needed

2. WebSocket library: Flask-SocketIO vs native WebSockets?
   - **Recommendation:** Flask-SocketIO (easier integration with Flask)

3. Tree visualization: D3.js vs jsTree?
   - **Recommendation:** D3.js (more flexible, better for dynamic updates)

4. Audio library for sonification?
   - **Recommendation:** Tone.js (Web Audio API wrapper)

5. TTS backend for narration?
   - **Recommendation:** Start with browser Web Speech API, migrate to local TTS later

---

**END OF PHASED IMPLEMENTATION PLAN**
