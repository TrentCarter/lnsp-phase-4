# PAS Service Port Mapping

## Multi-Tier PAS Architecture (Current)

### === Starting Services ===
[1/8] Starting Architect (port 6110)...
[2/8] Starting Director-Code (port 6111)...
[3/8] Starting Director-Models (port 6112)...
[4/8] Starting Director-Data (port 6113)...
[5/8] Starting Director-DevSecOps (port 6114)...
[6/8] Starting Director-Docs (port 6115)...
[7/8] Starting PAS Root (port 6100)...
[8/8] Starting Gateway (port 6120)...

### === Health Checks ===
✓ Architect (port 6110)
✓ Director-Code (port 6111)
✓ Director-Models (port 6112)
✓ Director-Data (port 6113)
✓ Director-DevSecOps (port 6114)
✓ Director-Docs (port 6115)
✓ PAS Root (port 6100)
✓ Gateway (port 6120)

### === Multi-Tier PAS Started Successfully ===

## Complete Service Port Mapping

| **Service** | **Port** | **URL** | **Description** | **Tier** |
|-------------|----------|---------|-----------------|----------|
| **Gateway** | 6120 | http://127.0.0.1:6120 | Single entrypoint for all Prime Directive submissions | 0 |
| **PAS Root** | 6100 | http://127.0.0.1:6100 | Orchestrator API for job management | 1 |
| **Architect** | 6110 | http://127.0.0.1:6110 | LLM-powered task decomposition coordinator | 2 |
| **Director-Code** | 6111 | http://127.0.0.1:6111 | Code lane coordinator | 3 |
| **Director-Models** | 6112 | http://127.0.0.1:6112 | Models/training lane coordinator | 3 |
| **Director-Data** | 6113 | http://127.0.0.1:6113 | Data processing lane coordinator | 3 |
| **Director-DevSecOps** | 6114 | http://127.0.0.1:6114 | DevSecOps lane coordinator | 3 |
| **Director-Docs** | 6115 | http://127.0.0.1:6115 | Documentation lane coordinator | 3 |
| **Manager-Code-01** | 6141 | http://127.0.0.1:6141 | Code lane task breakdown | 4 |
| **Manager-Code-02** | 6142 | http://127.0.0.1:6142 | Code lane task breakdown | 4 |
| **Manager-Code-03** | 6143 | http://127.0.0.1:6143 | Code lane task breakdown | 4 |
| **Manager-Models-01** | 6144 | http://127.0.0.1:6144 | Models lane task breakdown | 4 |
| **Manager-Data-01** | 6145 | http://127.0.0.1:6145 | Data lane task breakdown | 4 |
| **Manager-DevSecOps-01** | 6146 | http://127.0.0.1:6146 | DevSecOps lane task breakdown | 4 |
| **Manager-Docs-01** | 6147 | http://127.0.0.1:6147 | Docs lane task breakdown | 4 |
| **Prog-001** | 6151 | http://127.0.0.1:6151 | Code execution (Primary: Qwen 7B, Backup: Qwen 14B) | 5 |
| **Prog-002** | 6152 | http://127.0.0.1:6152 | Code execution (Primary: Qwen 7B, Backup: Qwen 14B) | 5 |
| **Prog-003** | 6153 | http://127.0.0.1:6153 | Code execution (Primary: Qwen 7B, Backup: Qwen 14B) | 5 |
| **Prog-004** | 6154 | http://127.0.0.1:6154 | Code execution (Primary: Qwen 7B, Backup: Qwen 14B) | 5 |
| **Prog-005** | 6155 | http://127.0.0.1:6155 | Code execution (Primary: Qwen 7B, Backup: Qwen 14B) | 5 |
| **Prog-006** | 6156 | http://127.0.0.1:6156 | Code execution (Primary: Claude Sonnet 4, Backup: Claude 3.7) | 5 |
| **Prog-007** | 6157 | http://127.0.0.1:6157 | Code execution (Primary: Claude Sonnet 4, Backup: Claude 3.7) | 5 |
| **Prog-008** | 6158 | http://127.0.0.1:6158 | Code execution (Primary: GPT-4o, Backup: GPT-4o-mini) | 5 |
| **Prog-009** | 6159 | http://127.0.0.1:6159 | Code execution (Primary: DeepSeek V3 14B, Backup: DeepSeek V3 7B) | 5 |
| **Prog-010** | 6160 | http://127.0.0.1:6160 | Code execution (Primary: DeepSeek V3 14B, Backup: DeepSeek V3 7B) | 5 |

## Core Infrastructure Services

| **Service** | **Port** | **URL** | **Description** |
|-------------|----------|---------|-----------------|
| **HMI Dashboard** | 6101 | http://127.0.0.1:6101 | Web UI for monitoring and approvals |
| **Event Stream** | 6102 | http://127.0.0.1:6102 | Pub/sub for real-time updates |
| **Provider Router** | 6103 | http://127.0.0.1:6103 | Model selection and cost optimization |
| **Resource Manager** | 6104 | http://127.0.0.1:6104 | Resource quotas and reservations |
| **Token Governor** | 6105 | http://127.0.0.1:6105 | Context budget enforcement |
| **Contract Tester** | 6106 | http://127.0.0.1:6106 | Schema validation and testing |
| **Experiment Ledger** | 6107 | http://127.0.0.1:6107 | Run tracking and reproducibility |
| **Peer Review Coordinator** | 6108 | http://127.0.0.1:6108 | Cross-vendor PR enforcement |
| **Heartbeat Monitor** | 6109 | http://127.0.0.1:6109 | Service health monitoring |
| **File Queue Watcher** | 6110 | http://127.0.0.1:6110 | Atomic JSONL processing |
| **Agent Router** | 6119 | http://127.0.0.1:6119 | Agent routing and discovery |
| **Service Registry** | 6121 | http://127.0.0.1:6121 | Service registration and discovery |

## LLM Services

| **Service** | **Port** | **URL** | **Description** |
|-------------|----------|---------|-----------------|
| **Aider-LCO** | 6130 | http://127.0.0.1:6130 | Aider RPC wrapper for code editing |
| **Model Pool** | 8050 | http://127.0.0.1:8050 | Llama 3.1 8B FastAPI service |
| **Model: TinyLlama** | 8051 | http://127.0.0.1:8051 | TinyLlama 1.1B FastAPI service |
| **Model: TLC Classifier** | 8052 | http://127.0.0.1:8052 | TLC Domain Classifier service |
| **Ollama** | 11434 | http://127.0.0.1:11434 | Local Ollama instance (Qwen2.5-coder, DeepSeek, etc.) |

## Legacy Services

| **Service** | **Port** | **URL** | **Description** |
|-------------|----------|---------|-----------------|
| **LVM Dashboard** | 8999 | http://127.0.0.1:8999 | LVM evaluation dashboard |
| **GTR-T5 Encoder** | 7001 | http://127.0.0.1:7001 | Text to vector encoding |
| **Vec2Text Decoder** | 7002 | http://127.0.0.1:7002 | Vector to text decoding |

## Port Ranges

- **6100-6119**: Core PAS services (Tier 0-2: Gateway, PAS Root, Architect)
- **6110**: Architect (Tier 2)
- **6111-6115**: Directors (Tier 3, 5 lanes)
- **6120-6129**: Gateway and Registry (Tier 0)
- **6130-6139**: Legacy/tooling (deprecated single Aider RPC)
- **6141-6150**: Managers (Tier 4, up to 10 managers)
- **6151-6199**: Programmers (Tier 5, up to 49 programmers for parallelization)
- **7001-7002**: Encoding/decoding services
- **8050-8099**: FastAPI LLM services (Model Pool)
- **8999**: LVM evaluation (legacy)

**NEW PORT ALLOCATION:**
- Managers: 6141-6150 (10 slots, 7 allocated)
- Programmers: 6151-6199 (49 slots, 10 allocated initially)

## Environment Variables

```bash
# Multi-Tier PAS
ARCHITECT_URL=http://127.0.0.1:6110
DIR_CODE_URL=http://127.0.0.1:6111
DIR_MODELS_URL=http://127.0.0.1:6112
DIR_DATA_URL=http://127.0.0.1:6113
DIR_DEVSECOPS_URL=http://127.0.0.1:6114
DIR_DOCS_URL=http://127.0.0.1:6115
PAS_ROOT_URL=http://127.0.0.1:6100
GATEWAY_URL=http://127.0.0.1:6120

# Core Infrastructure
REGISTRY_URL=http://127.0.0.1:6121
AIDER_RPC_URL=http://127.0.0.1:6130
```

## Programmer Pool Architecture

**10 Programmers** with configurable primary/backup LLM assignments:

- **Programmers 001-005**: Fast local models (Qwen 2.5 Coder 7B → 14B backup)
- **Programmers 006-007**: Premium Claude (Sonnet 4 → 3.7 backup)
- **Programmer 008**: Premium OpenAI (GPT-4o → GPT-4o-mini backup)
- **Programmers 009-010**: DeepSeek Coder V3 (14B → 7B backup)

**Features:**
- Automatic LLM failover (primary → backup on error)
- Circuit breaker (3 failures opens circuit for 5 minutes)
- Load balancing by Manager-Code
- Capability-based routing (fast, premium, reasoning, free, paid)
- Cost optimization (prefer free local models)

**Configuration:** `configs/pas/programmer_pool.yaml`

**Startup:**
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

## Quick Health Check Script

```bash
#!/bin/bash
echo "=== PAS Service Health Check ==="

# Core services
for port in 6110 6111 6112 6113 6114 6115 6100 6120 6121; do
    response=$(curl -s http://127.0.0.1:$port/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        service=$(echo $response | jq -r '.service' 2>/dev/null || echo "Unknown")
        echo "✓ Port $port: $service"
    else
        echo "✗ Port $port: Not responding"
    fi
done

# Programmer pool
echo ""
echo "=== Programmer Pool ==="
for port in {6151..6160}; do
    response=$(curl -s http://127.0.0.1:$port/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        prog_id=$(echo $response | jq -r '.programmer_id' 2>/dev/null || echo "?")
        current_llm=$(echo $response | jq -r '.llm.current' 2>/dev/null || echo "?")
        using_backup=$(echo $response | jq -r '.llm.using_backup' 2>/dev/null || echo "false")

        if [ "$using_backup" = "true" ]; then
            echo "⚠️  Port $port: Prog-$prog_id (USING BACKUP: $current_llm)"
        else
            echo "✓ Port $port: Prog-$prog_id ($current_llm)"
        fi
    else
        echo "✗ Port $port: Not responding"
    fi
done
```
