# Phase 5: Local LLM Services Architecture

**Status:** 🚧 In Progress
**Created:** 2025-11-06
**Owner:** Trent Carter

---

## Overview

Phase 5 wraps Ollama-hosted local LLMs as standardized FastAPI services that integrate with the Polyglot Agent Swarm (PAS) infrastructure. These services enable:

1. **Model abstraction** - Hide Ollama implementation details behind standard API
2. **Registry integration** - Automatic service discovery via Agent Registry
3. **Provider routing** - Route tasks to optimal local models based on capability
4. **Cost tracking** - Zero-cost attribution for local models
5. **Unified interface** - OpenAI-compatible API for consistency

---

## Service Specifications

### 1. Llama-3.1-8b Wrapper Service (Port 8050)

**Purpose:** General-purpose reasoning and complex task processing

**Capabilities:**
- `reasoning` - Complex multi-step reasoning
- `planning` - Task planning and decomposition
- `code_review` - Code analysis and suggestions
- `explanation` - Detailed explanations

**Model:**
- Ollama: `llama3.1:8b` (8B params, Q4_K_M quantization)
- Ollama Port: 11434 (default)
- Performance: ~73 tok/s (M4 Max)
- Context: 8k tokens

**Agent Definition:**
```json
{
  "agent_id": "llm_llama31_8b",
  "name": "Llama 3.1 8B Service",
  "tier": 2,
  "role": "execution",
  "capabilities": ["reasoning", "planning", "code_review", "explanation"],
  "endpoint": "http://localhost:8050",
  "transport": "rest",
  "cost_per_1k_tokens": 0.0,
  "metadata": {
    "model": "llama3.1:8b",
    "provider": "ollama",
    "parameters": "8B",
    "quantization": "Q4_K_M",
    "avg_latency_ms": null,
    "throughput_tok_s": 73
  }
}
```

---

### 2. TinyLlama Wrapper Service (Port 8051)

**Purpose:** Fast, lightweight classification and simple tasks

**Capabilities:**
- `classification` - Fast text classification
- `tagging` - Domain/category tagging
- `extraction` - Simple entity extraction
- `filtering` - Content filtering

**Model:**
- Ollama: `tinyllama:1.1b` (1.1B params, Q4_0 quantization)
- Ollama Port: 11435 (dedicated instance)
- Performance: ~277 tok/s (M4 Max)
- Context: 2k tokens

**Agent Definition:**
```json
{
  "agent_id": "llm_tinyllama_1b",
  "name": "TinyLlama 1.1B Service",
  "tier": 2,
  "role": "execution",
  "capabilities": ["classification", "tagging", "extraction", "filtering"],
  "endpoint": "http://localhost:8051",
  "transport": "rest",
  "cost_per_1k_tokens": 0.0,
  "metadata": {
    "model": "tinyllama:1.1b",
    "provider": "ollama",
    "parameters": "1.1B",
    "quantization": "Q4_0",
    "avg_latency_ms": null,
    "throughput_tok_s": 277
  }
}
```

---

### 3. TLC Domain Classifier Service (Port 8052)

**Purpose:** Specialized TMD (Task-Method-Domain) domain classification

**Capabilities:**
- `domain_classification` - Classify query domain
- `tmd_extraction` - Extract Task-Method-Domain triplet
- `domain_validation` - Validate domain against taxonomy

**Model:**
- Backend: TinyLlama via port 8051
- Specialized prompts for domain classification
- Domain taxonomy from TMD-LS PRD
- Performance: Same as TinyLlama (~277 tok/s)

**Agent Definition:**
```json
{
  "agent_id": "tlc_domain_classifier",
  "name": "TLC Domain Classifier",
  "tier": 2,
  "role": "execution",
  "capabilities": ["domain_classification", "tmd_extraction", "domain_validation"],
  "endpoint": "http://localhost:8052",
  "transport": "rest",
  "cost_per_1k_tokens": 0.0,
  "metadata": {
    "model": "tinyllama:1.1b",
    "provider": "ollama",
    "specialized_for": "TMD classification",
    "domain_taxonomy_version": "1.0",
    "avg_latency_ms": null,
    "throughput_tok_s": 277
  }
}
```

**Domain Taxonomy (TMD-LS):**
```python
DOMAINS = [
    "FACTOIDWIKI",
    "ARTIFICIAL_INTELLIGENCE",
    "MEDICINE",
    "BIOLOGY",
    "PHYSICS",
    "CHEMISTRY",
    "MATHEMATICS",
    "COMPUTER_SCIENCE",
    "ENGINEERING",
    "HISTORY",
    "LITERATURE",
    "PHILOSOPHY",
    "LAW",
    "ECONOMICS",
    "BUSINESS",
    "PSYCHOLOGY",
    "SOCIOLOGY",
    "GEOGRAPHY",
    "SPORTS",
    "ENTERTAINMENT",
    "OTHER"
]
```

---

## API Contract

### Common Endpoints (All Services)

#### 1. Chat Completions (OpenAI-Compatible)

```http
POST /chat/completions
Content-Type: application/json

{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chat-abc123",
  "object": "chat.completion",
  "created": 1699564800,
  "model": "llama3.1:8b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Artificial intelligence (AI) refers to..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}
```

#### 2. Generate (Ollama-Compatible)

```http
POST /generate
Content-Type: application/json

{
  "model": "llama3.1:8b",
  "prompt": "What is AI?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 500
  }
}
```

**Response:**
```json
{
  "model": "llama3.1:8b",
  "created_at": "2025-11-06T12:00:00Z",
  "response": "Artificial intelligence (AI) refers to...",
  "done": true,
  "total_duration": 1500000000,
  "load_duration": 100000000,
  "prompt_eval_count": 15,
  "eval_count": 120
}
```

#### 3. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "llama31_8b_wrapper",
  "port": 8050,
  "ollama_backend": "http://localhost:11434",
  "model": "llama3.1:8b",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

#### 4. Service Info

```http
GET /info
```

**Response:**
```json
{
  "service_name": "Llama 3.1 8B Wrapper",
  "version": "1.0.0",
  "model": {
    "name": "llama3.1:8b",
    "parameters": "8B",
    "quantization": "Q4_K_M",
    "context_length": 8192,
    "embedding_dim": 4096
  },
  "capabilities": ["reasoning", "planning", "code_review", "explanation"],
  "performance": {
    "avg_throughput_tok_s": 73,
    "avg_latency_ms": null,
    "p95_latency_ms": null
  },
  "endpoints": [
    "/chat/completions",
    "/generate",
    "/health",
    "/info"
  ]
}
```

---

## TLC Domain Classifier - Specialized Endpoints

### 1. Classify Domain

```http
POST /classify_domain
Content-Type: application/json

{
  "query": "What is the role of glucose in diabetes?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "What is the role of glucose in diabetes?",
  "domains": [
    {"domain": "MEDICINE", "confidence": 0.92},
    {"domain": "BIOLOGY", "confidence": 0.76},
    {"domain": "CHEMISTRY", "confidence": 0.45}
  ],
  "primary_domain": "MEDICINE"
}
```

### 2. Extract TMD

```http
POST /extract_tmd
Content-Type: application/json

{
  "query": "What is the role of glucose in diabetes?",
  "context_docs": [
    "Diabetes is a metabolic disorder...",
    "Glucose regulation is critical..."
  ]
}
```

**Response:**
```json
{
  "query": "What is the role of glucose in diabetes?",
  "tmd": {
    "task": "ANSWER",
    "method": "DENSE",
    "domain": "MEDICINE"
  },
  "confidence": {
    "task": 0.95,
    "method": 0.88,
    "domain": 0.92
  }
}
```

---

## Implementation Architecture

### Service Structure

```
services/llm/
├── llama31_8b_service.py      # Port 8050
├── tinyllama_service.py       # Port 8051
├── tlc_classifier_service.py  # Port 8052
└── common/
    ├── base_llm_service.py    # Shared FastAPI base
    ├── ollama_client.py       # Ollama API wrapper
    └── schemas.py             # Pydantic models
```

### Base LLM Service (Shared)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import time

class BaseLLMService:
    def __init__(self, model: str, ollama_url: str, port: int, capabilities: list):
        self.app = FastAPI(title=f"{model} Wrapper Service")
        self.model = model
        self.ollama_url = ollama_url
        self.port = port
        self.capabilities = capabilities
        self.start_time = time.time()

        # Setup routes
        self.app.post("/chat/completions")(self.chat_completions)
        self.app.post("/generate")(self.generate)
        self.app.get("/health")(self.health)
        self.app.get("/info")(self.info)

    async def chat_completions(self, request: ChatCompletionRequest):
        # Forward to Ollama, transform response to OpenAI format
        pass

    async def generate(self, request: GenerateRequest):
        # Forward to Ollama directly
        pass

    async def health(self):
        # Check Ollama backend health
        pass

    async def info(self):
        # Return service metadata
        pass
```

---

## Registry Integration

### Auto-Registration on Startup

Each LLM service registers itself with the Agent Registry (port 6121) on startup:

```python
async def register_with_registry():
    agent_def = {
        "agent_id": "llm_llama31_8b",
        "name": "Llama 3.1 8B Service",
        "tier": 2,
        "role": "execution",
        "capabilities": ["reasoning", "planning", "code_review", "explanation"],
        "endpoint": "http://localhost:8050",
        "transport": "rest",
        "cost_per_1k_tokens": 0.0,
        "metadata": {
            "model": "llama3.1:8b",
            "provider": "ollama",
            "parameters": "8B"
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:6121/register",
            json=agent_def
        )
        if response.status_code == 200:
            print(f"✅ Registered {agent_def['agent_id']} with registry")
        else:
            print(f"❌ Registration failed: {response.text}")
```

---

## Provider Router Integration

The Provider Router (port 6103) can route tasks to local LLMs based on capability:

**Example Routing Logic:**

```python
# In provider_router.py

CAPABILITY_MAP = {
    "reasoning": [
        {"provider": "local", "model": "llama3.1:8b", "endpoint": "http://localhost:8050"},
        {"provider": "anthropic", "model": "claude-sonnet-4-5"},
        {"provider": "openai", "model": "gpt-4"}
    ],
    "classification": [
        {"provider": "local", "model": "tinyllama:1.1b", "endpoint": "http://localhost:8051"},
        {"provider": "local", "model": "llama3.1:8b", "endpoint": "http://localhost:8050"}
    ],
    "domain_classification": [
        {"provider": "local", "model": "tlc_classifier", "endpoint": "http://localhost:8052"}
    ]
}

async def route_task(capability: str, prefer_local: bool = True):
    candidates = CAPABILITY_MAP.get(capability, [])
    if prefer_local:
        # Try local models first (zero cost)
        for candidate in candidates:
            if candidate["provider"] == "local":
                return candidate
    # Fall back to cloud providers
    return candidates[0] if candidates else None
```

---

## Startup/Shutdown Scripts

### Start All LLM Services

**File:** `scripts/start_phase5_llm_services.sh`

```bash
#!/bin/bash
# Start Phase 5 LLM Services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "====================================="
echo "Starting Phase 5 LLM Services"
echo "====================================="

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama not running on port 11434"
    echo "Start with: ollama serve"
    exit 1
fi
echo "✅ Ollama running on port 11434"

# Start Llama 3.1 8B Wrapper (Port 8050)
echo "Starting Llama 3.1 8B Wrapper (Port 8050)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.llama31_8b_service:app \
  --host 127.0.0.1 --port 8050 \
  > /tmp/llm_llama31_8b.log 2>&1 &
LLM_LLAMA31_PID=$!
echo "  PID: $LLM_LLAMA31_PID"

# Start TinyLlama Wrapper (Port 8051)
echo "Starting TinyLlama Wrapper (Port 8051)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.tinyllama_service:app \
  --host 127.0.0.1 --port 8051 \
  > /tmp/llm_tinyllama.log 2>&1 &
LLM_TINYLLAMA_PID=$!
echo "  PID: $LLM_TINYLLAMA_PID"

# Start TLC Domain Classifier (Port 8052)
echo "Starting TLC Domain Classifier (Port 8052)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.tlc_classifier_service:app \
  --host 127.0.0.1 --port 8052 \
  > /tmp/llm_tlc_classifier.log 2>&1 &
LLM_TLC_PID=$!
echo "  PID: $LLM_TLC_PID"

# Wait for services to start
echo ""
echo "Waiting for services to start..."
sleep 3

# Health check
echo ""
echo "Health Check:"
for port in 8050 8051 8052; do
    if curl -s http://localhost:$port/health >/dev/null 2>&1; then
        echo "  Port $port: ✅ Healthy"
    else
        echo "  Port $port: ❌ Not responding"
    fi
done

echo ""
echo "====================================="
echo "Phase 5 LLM Services Started"
echo "====================================="
echo ""
echo "Service URLs:"
echo "  Llama 3.1 8B:     http://localhost:8050"
echo "  TinyLlama:        http://localhost:8051"
echo "  TLC Classifier:   http://localhost:8052"
echo ""
echo "Logs:"
echo "  tail -f /tmp/llm_llama31_8b.log"
echo "  tail -f /tmp/llm_tinyllama.log"
echo "  tail -f /tmp/llm_tlc_classifier.log"
```

---

## Testing Strategy

### 1. Unit Tests (Per Service)

```python
# tests/test_llm_services.py

import pytest
from fastapi.testclient import TestClient
from services.llm.llama31_8b_service import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8050

def test_chat_completions(client):
    request = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "stream": False
    }
    response = client.post("/chat/completions", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
```

### 2. Integration Tests (Cross-Service)

```python
# tests/test_llm_integration.py

import httpx
import pytest

@pytest.mark.asyncio
async def test_registry_integration():
    """Test that LLM services register with registry"""
    async with httpx.AsyncClient() as client:
        # Query registry for LLM agents
        response = await client.post(
            "http://localhost:6121/discover",
            json={"capabilities": ["reasoning"]}
        )
        assert response.status_code == 200
        agents = response.json()["agents"]

        # Should find llama31_8b agent
        llm_agents = [a for a in agents if a["agent_id"] == "llm_llama31_8b"]
        assert len(llm_agents) == 1
        assert llm_agents[0]["endpoint"] == "http://localhost:8050"

@pytest.mark.asyncio
async def test_provider_router_local_preference():
    """Test that provider router prefers local LLMs for zero cost"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:6103/route",
            json={
                "capability": "reasoning",
                "prefer_local": True
            }
        )
        assert response.status_code == 200
        route = response.json()
        assert route["provider"] == "local"
        assert route["endpoint"] == "http://localhost:8050"
```

---

## Cost Tracking

All local LLM services report **zero cost** to the cost tracker:

```python
# After each request
await track_usage(
    agent_id="llm_llama31_8b",
    model="llama3.1:8b",
    prompt_tokens=request.usage.prompt_tokens,
    completion_tokens=request.usage.completion_tokens,
    cost_usd=0.0  # Local models are free!
)
```

**Gateway Dashboard** shows:
- Total local tokens processed
- Zero cost for local requests
- Latency distribution
- Fallback to cloud provider (if local fails)

---

## Success Criteria

- ✅ All 3 LLM services start successfully
- ✅ Health checks pass for all services
- ✅ Services auto-register with Agent Registry
- ✅ Provider Router can discover and route to local LLMs
- ✅ OpenAI-compatible API works (chat completions)
- ✅ Ollama-compatible API works (generate)
- ✅ TLC classifier returns valid TMD triplets
- ✅ Integration tests pass (7/7)
- ✅ Documentation complete
- ✅ Startup/shutdown scripts work

---

## Next Steps (Phase 6)

After Phase 5 completion:
- Phase 6: Air-Gapped Mode (offline operation)
- Distributed scheduling (multi-host)
- Advanced model caching strategies
- LLM model hot-swapping
- Multi-GPU support for larger models

---

## References

- **Ollama Setup**: `docs/howto/how_to_access_local_AI.md`
- **TMD-LS PRD**: `docs/PRDs/PRD_TMD-LS.md`
- **PAS PRD**: `docs/PRDs/PRD_Polyglot_Agent_Swarm.md`
- **Provider Router**: `services/provider_router/provider_router.py`
- **Agent Registry**: `services/registry/registry_service.py`
