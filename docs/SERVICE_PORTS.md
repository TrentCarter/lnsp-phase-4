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

- **6100-6119**: Core PAS services
- **6120-6129**: Gateway and Registry
- **6130-6139**: Aider and tooling
- **8050-8059**: FastAPI LLM services
- **7001-7002**: Encoding/decoding services
- **8999**: LVM evaluation (legacy)

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

## Quick Health Check Script

```bash
#!/bin/bash
echo "=== PAS Service Health Check ==="
for port in 6110 6111 6112 6113 6114 6115 6100 6120 6121 6130; do
    response=$(curl -s http://127.0.0.1:$port/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        service=$(echo $response | jq -r '.service' 2>/dev/null || echo "Unknown")
        echo "✓ Port $port: $service"
    else
        echo "✗ Port $port: Not responding"
    fi
done
```
