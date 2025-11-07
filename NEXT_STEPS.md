# Next Steps — Resume After /clear

**Date:** 2025-11-06
**Current Status:** ✅ Phase 0, Phase 1, Phase 2 & Phase 3 Complete
**Next Phase:** Phase 4 - Claude Sub-Agents

---

## Quick Resume Guide

### 1. Start Services (if not running)
```bash
./scripts/start_all_pas_services.sh
```

### 2. Verify Services Healthy
```bash
curl http://localhost:6121/health  # Registry
curl http://localhost:6109/health  # Heartbeat Monitor
curl http://localhost:6104/health  # Resource Manager
curl http://localhost:6105/health  # Token Governor
curl http://localhost:6102/health  # Event Stream
curl http://localhost:6101/health  # Flask HMI
curl http://localhost:6103/health  # Provider Router
curl http://localhost:6120/health  # Gateway
```

### 3. Review What's Been Done
```bash
cat docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE01_COMPLETE.md
cat docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE02_COMPLETE.md
cat docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE03_COMPLETE.md
```

### 4. Access HMI Dashboard
```bash
open http://localhost:6101
```

---

## What's Working

✅ **Phase 0 Complete** (Ports 6121, 6109)
- Registry: Service registration, discovery, heartbeats, TTL
- Heartbeat Monitor: Health checks, alerts, auto-recovery

✅ **Phase 1 Complete** (Ports 6104, 6105)
- Resource Manager: CPU/memory/GPU allocation, quotas
- Token Governor: Context tracking, breach detection, summarization

✅ **Phase 2 Complete** (Ports 6101, 6102)
- Event Stream: WebSocket server, event broadcasting, buffering
- Flask HMI: Dashboard, D3.js tree, real-time updates, API endpoints

✅ **Phase 3 Complete** (Ports 6103, 6120)
- Provider Router: Provider registration, capability matching, selection
- Gateway: Central routing hub, cost tracking, LDJSON receipts
- Cost Tracking: Budget management, rolling windows, event broadcasting

---

## Phase 4 Overview

**Goal:** Define and register 42 Claude sub-agents for PAS swarm intelligence

**Components to Build:**
1. **Agent Definitions** - 42 agent specifications in `.claude/agents/`
2. **Agent Hierarchy** - Parent-child relationships and routing rules
3. **Registry Integration** - Auto-register agents with Registry on startup
4. **Test Invocations** - Verify agents can be invoked and respond correctly

**Key Features:**
- 23 Coordinator/System agents (top-level planning, coordination)
- 19 Execution agents (specialized tasks, code generation, testing)
- Hierarchical structure (Tree Supervisor → Domain Coordinators → Workers)
- Automatic capability-based routing
- Integration with Gateway for cost tracking

**Timeline:** 1-2 days

---

## Phase 4 Implementation Steps

### Day 1: Agent Definitions & Hierarchy

**1. Create Agent Definition Schema**
```bash
# File: contracts/agent_definition.schema.json
# - Agent metadata (name, role, capabilities)
# - Routing rules (parent, children, specialization)
# - Resource requirements (CPU, memory, tokens)
```

**2. Define Coordinator Agents (23 agents)**
```bash
# File: .claude/agents/coordinator/*.md
# Examples:
# - tree_supervisor.md (top-level orchestrator)
# - cpe_coordinator.md (CPE/hypothesis domain)
# - code_coordinator.md (code generation domain)
# - test_coordinator.md (testing domain)
# - data_coordinator.md (data processing domain)
```

**3. Define Execution Agents (19 agents)**
```bash
# File: .claude/agents/execution/*.md
# Examples:
# - cpe_generator.md (generate CPE quadruplets)
# - code_writer.md (write production code)
# - test_writer.md (write test cases)
# - data_ingester.md (ingest data pipelines)
# - debugger.md (debug code issues)
```

### Day 2: Registry Integration & Testing

**4. Create Agent Registration Script**
```bash
# File: tools/register_agents.py
# - Read agent definitions from .claude/agents/
# - Parse agent metadata (name, role, capabilities)
# - Register with Registry service
# - Verify registration successful
```

**5. Integration Tests**
```bash
# File: scripts/test_phase4.sh
# - Verify all 42 agents registered
# - Test agent discovery by capability
# - Test hierarchical routing (parent → child)
# - Validate resource requirements
```

**6. Agent Invocation Tests**
```bash
# File: tests/test_agent_invocations.py
# - Invoke Tree Supervisor with sample task
# - Verify delegation to appropriate coordinator
# - Verify coordinator delegates to worker
# - Validate response chain
```

---

## Files to Create (Phase 4)

```
.claude/
  agents/
    coordinator/
      tree_supervisor.md           (Top-level orchestrator)
      cpe_coordinator.md            (CPE domain)
      code_coordinator.md           (Code generation domain)
      test_coordinator.md           (Testing domain)
      data_coordinator.md           (Data processing domain)
      ... (18 more coordinator agents)
    execution/
      cpe_generator.md              (CPE quadruplet generation)
      code_writer.md                (Code generation)
      test_writer.md                (Test generation)
      data_ingester.md              (Data ingestion)
      debugger.md                   (Debugging)
      ... (14 more execution agents)

contracts/
  agent_definition.schema.json      (Agent metadata schema)

tools/
  register_agents.py                (Agent registration script)

scripts/
  test_phase4.sh                    (Integration tests)

tests/
  test_agent_invocations.py         (Agent invocation tests)
```

---

## Technology Stack (Phase 4)

### Agent Definitions
- **Markdown:** Agent specifications (`.md` files)
- **YAML frontmatter:** Structured metadata
- **JSON Schema:** Validation

### Registration
- **Python:** Agent registration script
- **httpx:** HTTP client for Registry API
- **Registry API:** Service registration

### Testing
- **pytest:** Unit tests for agent logic
- **Integration tests:** End-to-end agent invocations
- **Mock responses:** Simulate agent outputs

---

## Agent Hierarchy Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Tree Supervisor                          │
│  - Top-level orchestration                                  │
│  - Route requests to domain coordinators                    │
│  - Aggregate results from sub-trees                         │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  CPE Coord      │    │  Code Coord     │    ... (5 more coordinators)
│  - CPE domain   │    │  - Code domain  │
│  - Delegate CPE │    │  - Delegate code│
└────────┬────────┘    └────────┬────────┘
         │                      │
    ┌────┴────┐            ┌────┴────┐
    ▼         ▼            ▼         ▼
┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐
│CPE Gen│ │CPE Val│  │Code   │ │Test   │
│       │ │       │  │Writer │ │Writer │
└───────┘ └───────┘  └───────┘ └───────┘
```

---

## Example Agent Definition

**File:** `.claude/agents/coordinator/tree_supervisor.md`

```markdown
---
name: Tree Supervisor
role: coordinator
parent: null
children:
  - cpe_coordinator
  - code_coordinator
  - test_coordinator
  - data_coordinator
capabilities:
  - task_decomposition
  - sub_agent_routing
  - result_aggregation
resources:
  max_tokens: 200000
  cpu_cores: 2
  memory_mb: 4096
---

# Tree Supervisor

## Role
Top-level orchestrator for the Polyglot Agent Swarm.

## Responsibilities
- Receive high-level tasks from users
- Decompose tasks into domain-specific sub-tasks
- Route sub-tasks to appropriate domain coordinators
- Aggregate results from sub-trees
- Return unified response to user

## Routing Strategy
- **CPE/Hypothesis tasks** → CPE Coordinator
- **Code generation tasks** → Code Coordinator
- **Testing tasks** → Test Coordinator
- **Data processing tasks** → Data Coordinator

## Example Invocation
**Input:** "Generate CPE quadruplets for the concept 'diabetes'"
**Action:** Route to CPE Coordinator
**Output:** Aggregated CPE quadruplets with metadata
```

---

## Phase 4 Acceptance Criteria

### P0 (Must Have)
- [ ] 42 agent definition files created in `.claude/agents/`
- [ ] Agent hierarchy structure defined (parent-child relationships)
- [ ] Agent registration script (`tools/register_agents.py`)
- [ ] All agents successfully registered with Registry
- [ ] Integration tests verify agent discovery
- [ ] Agent invocation tests pass (Tree Supervisor → Worker)
- [ ] Documentation for agent definitions

### P1 (Should Have)
- [ ] Agent capability-based routing
- [ ] Dynamic agent discovery (query by capability)
- [ ] Agent health monitoring
- [ ] Agent usage analytics (invocations, latency, cost)
- [ ] Agent versioning support

### P2 (Nice to Have)
- [ ] Agent hot-reloading (update definitions without restart)
- [ ] Agent A/B testing
- [ ] Agent performance profiling
- [ ] Agent skill learning (update capabilities based on success)

---

## Testing Approach (Phase 4)

### Unit Tests
```bash
# Test agent metadata parsing
pytest tests/test_agent_metadata.py

# Test agent registration logic
pytest tests/test_agent_registration.py
```

### Integration Tests
```bash
# scripts/test_phase4.sh
1. Start all services (Phase 0+1+2+3)
2. Run agent registration script
3. Verify all 42 agents registered in Registry
4. Query agents by capability
5. Verify hierarchical relationships
6. Test agent invocation chain
```

### Manual Testing
```bash
# Register agents
./.venv/bin/python tools/register_agents.py

# Verify registration
curl http://localhost:6121/list | jq '.services[] | select(.type=="agent")'

# Query by capability
curl "http://localhost:6121/discover?capability=code_generation" | jq .

# Invoke Tree Supervisor (via Claude Code)
claude sub-agents tree_supervisor "Generate CPE for diabetes"
```

---

## Integration with Phase 3 (Gateway)

### Agent → Gateway Flow

1. **User Request** → Tree Supervisor
2. **Tree Supervisor** decomposes task → Route to CPE Coordinator
3. **CPE Coordinator** → Route to CPE Generator (via Gateway)
4. **Gateway** selects provider (Claude Sonnet 3.5) via Provider Router
5. **Gateway** tracks cost and generates receipt
6. **CPE Generator** returns result → CPE Coordinator → Tree Supervisor → User

### Cost Tracking for Agents

- Each agent invocation tracked as separate request
- Run ID groups related agent invocations
- Budget applies across entire agent swarm
- Cost receipts include agent name in metadata

**Example Receipt:**
```json
{
  "request_id": "req-001",
  "run_id": "R-diabetes-001",
  "agent": "cpe_generator",
  "provider": "anthropic-sonnet",
  "model": "claude-sonnet-3.5",
  "input_tokens": 1200,
  "output_tokens": 800,
  "cost_usd": 0.042,
  "latency_ms": 3400,
  "status": "success",
  "metadata": {
    "parent_agent": "cpe_coordinator",
    "task_type": "cpe_generation",
    "domain": "medical"
  }
}
```

---

## Optional: HMI Cost Dashboard (P1 from Phase 3)

**Before starting Phase 4**, you may want to complete the Phase 3 P1 feature:

### HMI Cost Dashboard Features

1. **Real-time Cost Metrics**
   - $/min and tokens/min display
   - Live cost chart (last 60 minutes)
   - Cost breakdown by provider
   - Cost breakdown by agent (once Phase 4 complete)

2. **Budget Alerts**
   - Visual alerts at 75%, 90%, 100% thresholds
   - Caution (yellow), Warning (orange), Critical (red)
   - Budget runway display (time until budget exhausted)

3. **Cost Analytics**
   - Top 10 most expensive agents
   - Cost trends over time
   - Cost vs performance (latency) scatter plot

**Files to Update:**
- `services/webui/hmi_app.py` - Add `/api/costs` endpoint
- `services/webui/templates/dashboard.html` - Add cost card
- `services/webui/templates/costs.html` - New cost analytics page
- `services/webui/static/css/hmi.css` - Cost dashboard styling

**Estimated Time:** 2-3 hours

---

## Interactive API Docs (Currently Available)

- http://localhost:6121/docs - Registry
- http://localhost:6109/docs - Heartbeat Monitor
- http://localhost:6104/docs - Resource Manager
- http://localhost:6105/docs - Token Governor
- http://localhost:6101 - Flask HMI Dashboard (Web UI)
- http://localhost:6103/docs - Provider Router
- http://localhost:6120/docs - Gateway

**After Phase 4:**
- Agent definitions will be queryable via Registry API
- No new API docs (agents are data, not services)

---

## Useful Commands

### Start Everything
```bash
./scripts/start_all_pas_services.sh
```

### Check All Services
```bash
for port in 6121 6109 6104 6105 6102 6101 6103 6120; do
  echo "Port $port: $(curl -s http://localhost:$port/health | jq -r .status)"
done
```

### View Cost Receipts
```bash
cat artifacts/costs/*.jsonl | jq .
```

### Register Test Provider
```bash
curl -X POST http://localhost:6103/register -H "Content-Type: application/json" -d '{
  "name": "test-claude",
  "model": "claude-sonnet-3.5",
  "context_window": 200000,
  "cost_per_input_token": 0.000003,
  "cost_per_output_token": 0.000015,
  "endpoint": "http://localhost:8101",
  "features": ["function_calling", "streaming"]
}'
```

### Route Test Request
```bash
curl -X POST http://localhost:6120/route -H "Content-Type: application/json" -d '{
  "request_id": "test-001",
  "run_id": "test-run",
  "agent": "test-agent",
  "requirements": {
    "model": "claude-sonnet-3.5",
    "context_window": 10000
  },
  "optimization": "cost"
}'
```

### Monitor Logs
```bash
tail -f /tmp/pas_logs/*.log
```

### Clean Restart
```bash
./scripts/stop_all_pas_services.sh
rm -rf artifacts/registry/*.db
rm -rf artifacts/resource_manager/*.db
rm -rf artifacts/token_governor/*.db
rm -rf artifacts/provider_router/*.db
rm -rf artifacts/costs/*.jsonl
./scripts/start_all_pas_services.sh
```

---

## References

- **Phase 0+1 Summary:** `docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE01_COMPLETE.md`
- **Phase 2 Summary:** `docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE02_COMPLETE.md`
- **Phase 3 Summary:** `docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE03_COMPLETE.md`
- **Implementation Plan:** `docs/PRDs/PRD_IMPLEMENTATION_PHASES.md`
- **HMI Requirements:** `docs/PRDs/PRD_Human_Machine_Interface_HMI.md`
- **PAS Requirements:** `docs/PRDs/PRD_Polyglot_Agent_Swarm.md`
- **Architecture:** `docs/HYBRID_AGENT_ARCHITECTURE.md`
- **Progress:** `PROGRESS.md`

---

**Ready to continue with Phase 4 or add HMI Cost Dashboard!** 🚀
