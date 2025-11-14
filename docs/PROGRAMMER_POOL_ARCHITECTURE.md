# Programmer Pool Architecture

**Status**: ✅ Production Ready
**Date**: 2025-11-13
**Version**: 2.0

## Overview

The Programmer Pool is a scalable, load-balanced pool of **10 programmer services** (Prog-001 through Prog-010) that execute code changes via Aider CLI. Each programmer has configurable primary/backup LLM assignments with automatic failover.

## Key Features

### 1. **10 Programmers with LLM Diversity**
- **Prog-001 to Prog-005**: Fast local models (Qwen 2.5 Coder 7B → 14B backup)
- **Prog-006 to Prog-007**: Premium Claude (Sonnet 4 → 3.7 backup)
- **Prog-008**: Premium OpenAI (GPT-4o → GPT-4o-mini backup)
- **Prog-009 to Prog-010**: DeepSeek Coder V3 (14B → 7B backup)

### 2. **Automatic LLM Failover**
- **Circuit breaker**: 3 failures opens circuit for 5 minutes
- **Retry logic**: Try primary → backup → fail
- **Health tracking**: Real-time monitoring of LLM availability

### 3. **Load Balancing**
- **Strategies**: least_loaded, round_robin, capability_match
- **Queue depth tracking**: Dispatch to least busy programmer
- **Capability-based routing**: Match tasks to appropriate LLMs

### 4. **Cost Optimization**
- **Prefer free local models** over paid APIs
- **Daily budget tracking**: $50/day across all programmers
- **Automatic fallback**: Premium models only when needed

## Architecture

```
Manager-Code (Port 6141)
    ↓
ProgrammerPool (Load Balancer)
    ↓
┌──────────────────────────────────────────────────┐
│  Prog-001 (6151) - Qwen 7B → 14B                 │
│  Prog-002 (6152) - Qwen 7B → 14B                 │
│  Prog-003 (6153) - Qwen 7B → 14B                 │
│  Prog-004 (6154) - Qwen 7B → 14B                 │
│  Prog-005 (6155) - Qwen 7B → 14B                 │
│  Prog-006 (6156) - Claude Sonnet 4 → 3.7         │
│  Prog-007 (6157) - Claude Sonnet 4 → 3.7         │
│  Prog-008 (6158) - GPT-4o → GPT-4o-mini          │
│  Prog-009 (6159) - DeepSeek V3 14B → 7B          │
│  Prog-010 (6160) - DeepSeek V3 14B → 7B          │
└──────────────────────────────────────────────────┘
```

## Configuration

### Programmer Pool Config

**File**: `configs/pas/programmer_pool.yaml`

```yaml
pool_size: 10

programmers:
  - id: "001"
    port: 6151
    primary_llm: "ollama/qwen2.5-coder:7b-instruct"
    backup_llm: "ollama/qwen2.5-coder:14b-instruct"
    capabilities: ["fast", "local", "free"]
  # ... (see full config file)

load_balancing:
  strategy: "least_loaded"  # least_loaded | round_robin | capability_match
  prefer_free_models: true
  max_queue_per_programmer: 5
  health_check_interval_s: 30

failover:
  enabled: true
  max_retries: 2
  retry_delay_s: 5
  circuit_breaker:
    enabled: true
    failure_threshold: 3
    window_minutes: 10
    cooldown_minutes: 5

capability_routing:
  fast: ["001", "002", "003", "004", "005"]
  premium: ["006", "007", "008"]
  reasoning: ["006", "007", "008"]
  free: ["001", "002", "003", "004", "005", "009", "010"]
  paid: ["006", "007", "008"]

cost_optimization:
  enabled: true
  prefer_free_over_paid: true
  max_daily_spend_usd: 50.00
```

## Usage

### Starting the Programmer Pool

```bash
# Start all 10 programmers
./scripts/start_programmers.sh start

# Check status
./scripts/start_programmers.sh status

# Stop all
./scripts/start_programmers.sh stop

# Restart all
./scripts/start_programmers.sh restart
```

### Manager-Code Integration

Manager-Code automatically dispatches tasks to the programmer pool:

```python
from services.common.programmer_pool import get_programmer_pool

pool = get_programmer_pool()

# Dispatch task with capability preferences
result = await pool.dispatch_task(
    task_description="Add docstrings to all functions",
    files=["src/module.py"],
    run_id="run-123",
    capabilities=["fast"],  # Prefer fast local models
    prefer_free=True        # Prefer free over paid
)

# Result includes programmer_id, status, and Aider output
print(f"Task completed by {result['programmer_id']}")
```

### Health Monitoring

```bash
# Check individual programmer health
curl http://localhost:6151/health | jq

# Output:
{
  "status": "ok",
  "agent": "Prog-001",
  "programmer_id": "001",
  "port": 6151,
  "llm": {
    "current": "ollama/qwen2.5-coder:7b-instruct",
    "primary": "ollama/qwen2.5-coder:7b-instruct",
    "backup": "ollama/qwen2.5-coder:14b-instruct",
    "using_backup": false,
    "circuit_open": false,
    "failures": 0
  },
  "capabilities": ["fast", "local", "free"]
}

# Check entire pool status via Manager-Code
curl http://localhost:6141/programmer_pool/status | jq

# Output:
{
  "pool_size": 10,
  "available": 10,
  "unavailable": 0,
  "using_backup": 0,
  "total_queue_depth": 0,
  "programmers": [...]
}
```

## LLM Failover Flow

```
1. Task arrives → Manager-Code
2. Manager dispatches to ProgrammerPool
3. Pool selects Prog-003 (least loaded, fast capability)
4. Prog-003 executes with primary LLM (Qwen 7B)
   ├─ Success → Return result
   └─ Failure (timeout/error)
      ├─ Record failure (count: 1)
      ├─ Retry with backup LLM (Qwen 14B)
      │  ├─ Success → Return result
      │  └─ Failure
      │     ├─ Record failure (count: 2)
      │     └─ Retry with backup again
      │        ├─ Success → Return result
      │        └─ Failure
      │           ├─ Record failure (count: 3)
      │           ├─ Open circuit breaker
      │           ├─ Switch all future tasks to backup
      │           └─ Return error to Manager
```

## Circuit Breaker Logic

When a programmer's primary LLM fails 3 times within 10 minutes:

1. **Circuit Opens**: All traffic switched to backup LLM
2. **Cooldown**: 5 minutes before retrying primary
3. **Auto-Recovery**: Circuit closes if primary succeeds after cooldown

Example:
```
Time 00:00 - Prog-006 primary fails (Claude Sonnet 4 timeout)
Time 00:02 - Prog-006 primary fails (2nd failure)
Time 00:05 - Prog-006 primary fails (3rd failure)
Time 00:05 - ⚠️ Circuit OPEN - All tasks use Claude 3.7 backup
Time 00:10 - Cooldown expires, retry primary
Time 00:10 - Primary succeeds ✅ Circuit CLOSED
```

## Load Balancing Strategies

### 1. Least Loaded (Default)
Selects programmer with lowest queue depth.

**Use case**: Evenly distribute load across pool

### 2. Round Robin
Cycles through programmers sequentially.

**Use case**: Fair distribution when all programmers equal

### 3. Capability Match
Selects first programmer matching capabilities.

**Use case**: Route complex tasks to premium models

## Testing

```bash
# Run programmer pool tests
PYTHONPATH=. ./.venv/bin/python tests/test_programmer_pool.py

# Output:
✅ test_pool_initialization passed
✅ test_capability_routing passed
✅ test_available_programmers passed
✅ test_programmer_selection_strategies passed
✅ test_pool_status passed
🎉 All tests passed!
```

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `configs/pas/programmer_pool.yaml` | ✅ Created | Pool configuration with 10 programmers |
| `services/tools/aider_rpc/app.py` | ✅ Updated | Multi-instance support, LLM failover |
| `services/common/programmer_pool.py` | ✅ Created | Load balancer and pool manager |
| `services/pas/manager_code/app.py` | ✅ Updated | Pool integration |
| `scripts/start_programmers.sh` | ✅ Created | Startup script for all programmers |
| `docs/SERVICE_PORTS.md` | ✅ Updated | Programmer pool documentation |
| `tests/test_programmer_pool.py` | ✅ Created | Unit tests |

## Environment Variables

Each programmer reads these environment variables:

```bash
# Programmer ID (001-010)
export PROGRAMMER_ID=001

# Port is auto-configured from pool config
# LLMs are auto-configured from pool config
```

## API Endpoints

### Programmer (e.g., Prog-001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with LLM status |
| `/aider/edit` | POST | Execute code changes (direct RPC) |
| `/agent_chat/receive` | POST | Execute via agent chat (recommended) |

### Manager-Code

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with pool summary |
| `/programmer_pool/status` | GET | Detailed pool status |
| `/agent_chat/receive` | POST | Receive tasks from Dir-Code |

## Monitoring

### Key Metrics

1. **Pool availability**: How many programmers are healthy
2. **Backup usage**: How many using backup LLMs (circuit breaker active)
3. **Queue depth**: Total tasks queued across pool
4. **Cost tracking**: Daily spend across all programmers

### Alerts

- **Low availability**: < 5 programmers available
- **High backup usage**: > 3 programmers using backup (suggests primary LLM issues)
- **Budget threshold**: Daily spend > $40 (80% of $50 limit)

## Future Enhancements

1. **Dynamic pool sizing**: Scale from 10 → 49 programmers based on load
2. **Cost-aware routing**: Route expensive tasks to cheaper models first
3. **Performance profiling**: Track task duration by LLM to optimize routing
4. **Multi-region support**: Deploy programmers across regions for redundancy

## See Also

- [SERVICE_PORTS.md](SERVICE_PORTS.md) - Complete port mapping
- [Manager-Code Contract](contracts/MANAGER_CODE_SYSTEM_PROMPT.md) - Manager responsibilities
- [Aider RPC Config](../configs/pas/aider.yaml) - Legacy single-programmer config (deprecated)
