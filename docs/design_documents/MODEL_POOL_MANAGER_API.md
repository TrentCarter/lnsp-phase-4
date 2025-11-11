# Model Pool Manager API Design

**Service:** Model Pool Manager
**Port:** 8050
**Purpose:** Dynamic LLM model lifecycle management with automatic load/unload

---

## Core Concepts

### Model Registration
- Each model gets a dedicated FastAPI service on ports 8051-8099
- Models auto-register with PAS Registry (6121) when loaded
- Models auto-deregister when unloaded (TTL expired)

### Keep-Alive TTL
- Default: 15 minutes of inactivity before unload
- Each request extends TTL by 15 minutes
- Warmup models (qwen, llama3.1) never unload
- Prevents memory bloat from unused models

### Dynamic Allocation
- Models allocated on-demand when first requested
- Port assignment automatic (next available in 8051-8099)
- FastAPI service spawned as subprocess
- Ollama model loaded in background

---

## REST API Endpoints

### 1. Model Registry Management

#### `POST /models/register`
Register a new model for dynamic loading.

**Request:**
```json
{
  "model_name": "qwen2.5-coder:7b",
  "display_name": "Qwen 2.5 Coder 7B",
  "model_type": "code",
  "warmup": true,
  "ttl_minutes": 15,
  "priority": "high"
}
```

**Response:**
```json
{
  "model_id": "qwen2.5-coder:7b",
  "port": 8051,
  "state": "WARMING",
  "estimated_load_time_sec": 45,
  "endpoint": "http://localhost:8051"
}
```

#### `DELETE /models/{model_id}`
Remove model from registry and unload if active.

**Response:**
```json
{
  "model_id": "qwen2.5-coder:7b",
  "state": "UNLOADING",
  "port_released": 8051
}
```

#### `GET /models`
List all registered models and their states.

**Response:**
```json
{
  "models": [
    {
      "model_id": "qwen2.5-coder:7b",
      "port": 8051,
      "state": "HOT",
      "ttl_remaining_minutes": 12,
      "last_request_ago_seconds": 180,
      "request_count": 247,
      "uptime_minutes": 120,
      "memory_mb": 4200,
      "warmup": true
    },
    {
      "model_id": "deepseek-coder-v2:16b",
      "port": 8053,
      "state": "COOLING",
      "ttl_remaining_minutes": 2,
      "last_request_ago_seconds": 780,
      "request_count": 12,
      "uptime_minutes": 35,
      "memory_mb": 9100,
      "warmup": false
    }
  ],
  "total_memory_mb": 13300,
  "available_ports": 47
}
```

---

### 2. Model Lifecycle Control

#### `POST /models/{model_id}/load`
Force-load a model (bypass lazy loading).

**Response:**
```json
{
  "model_id": "llama3.1:8b",
  "state": "WARMING",
  "port": 8052,
  "estimated_ready_at": "2025-11-11T14:23:45Z"
}
```

#### `POST /models/{model_id}/unload`
Force-unload a model (bypass TTL).

**Query Params:**
- `force=true` - Immediate unload (default: graceful)

**Response:**
```json
{
  "model_id": "llama3.1:8b",
  "state": "COLD",
  "port_released": 8052,
  "memory_freed_mb": 4800
}
```

#### `POST /models/{model_id}/extend-ttl`
Manually extend TTL (usually auto-extended on request).

**Request:**
```json
{
  "minutes": 30
}
```

**Response:**
```json
{
  "model_id": "qwen2.5-coder:7b",
  "ttl_remaining_minutes": 30,
  "expires_at": "2025-11-11T15:05:00Z"
}
```

---

### 3. Health & Monitoring

#### `GET /health`
Pool manager health check.

**Response:**
```json
{
  "status": "healthy",
  "active_models": 3,
  "total_memory_mb": 13300,
  "available_ports": 47,
  "ollama_status": "connected"
}
```

#### `GET /models/{model_id}/health`
Individual model health check.

**Response:**
```json
{
  "model_id": "qwen2.5-coder:7b",
  "state": "HOT",
  "endpoint": "http://localhost:8051",
  "responsive": true,
  "latency_ms": 12,
  "last_error": null
}
```

#### `GET /metrics`
Prometheus-compatible metrics.

**Response:**
```
# HELP model_pool_active_models Number of active models
# TYPE model_pool_active_models gauge
model_pool_active_models 3

# HELP model_pool_memory_mb Total memory used by models
# TYPE model_pool_memory_mb gauge
model_pool_memory_mb 13300

# HELP model_pool_requests_total Total requests per model
# TYPE model_pool_requests_total counter
model_pool_requests_total{model="qwen2.5-coder:7b"} 247
model_pool_requests_total{model="llama3.1:8b"} 89
```

---

### 4. Configuration

#### `GET /config`
Get current configuration.

**Response:**
```json
{
  "default_ttl_minutes": 15,
  "min_ttl_minutes": 5,
  "max_ttl_minutes": 60,
  "check_interval_seconds": 30,
  "warmup_models": ["qwen2.5-coder:7b", "llama3.1:8b"],
  "port_range": {
    "start": 8051,
    "end": 8099
  },
  "max_concurrent_models": 5,
  "memory_limit_mb": 32000
}
```

#### `PATCH /config`
Update configuration (runtime).

**Request:**
```json
{
  "default_ttl_minutes": 20,
  "max_concurrent_models": 4
}
```

**Response:**
```json
{
  "status": "updated",
  "config": { /* full config */ }
}
```

---

## Model Service Template API

Each model service (ports 8051-8099) exposes:

### `POST /v1/chat/completions`
OpenAI-compatible chat endpoint.

**Request:**
```json
{
  "model": "qwen2.5-coder:7b",
  "messages": [
    {"role": "user", "content": "Write a Python function"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### `POST /v1/completions`
OpenAI-compatible completion endpoint.

**Request:**
```json
{
  "model": "qwen2.5-coder:7b",
  "prompt": "def fibonacci(",
  "max_tokens": 500
}
```

### `GET /health`
Model-specific health check.

**Response:**
```json
{
  "model": "qwen2.5-coder:7b",
  "status": "ready",
  "uptime_seconds": 7200,
  "requests_served": 247
}
```

---

## Client Usage Example

### Python Client (Provider Router)

```python
import httpx
from typing import Optional

class ModelPoolClient:
    def __init__(self, pool_manager_url: str = "http://localhost:8050"):
        self.base_url = pool_manager_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def get_model_endpoint(self, model_id: str) -> str:
        """Get endpoint for model, loading if necessary."""
        # Check if model is loaded
        resp = await self.client.get(f"{self.base_url}/models")
        models = resp.json()["models"]

        for model in models:
            if model["model_id"] == model_id:
                if model["state"] == "HOT":
                    return model["endpoint"]
                elif model["state"] == "WARMING":
                    # Wait for model to load
                    await self._wait_for_hot(model_id)
                    return f"http://localhost:{model['port']}"

        # Model not registered, load it
        resp = await self.client.post(
            f"{self.base_url}/models/{model_id}/load"
        )
        data = resp.json()
        await self._wait_for_hot(model_id)
        return f"http://localhost:{data['port']}"

    async def chat(self, model_id: str, messages: list) -> dict:
        """Send chat request, auto-loading model if needed."""
        endpoint = await self.get_model_endpoint(model_id)

        # Chat request to model service
        resp = await self.client.post(
            f"{endpoint}/v1/chat/completions",
            json={"model": model_id, "messages": messages}
        )
        return resp.json()

    async def _wait_for_hot(self, model_id: str, timeout: int = 120):
        """Poll until model is HOT."""
        import asyncio
        for _ in range(timeout):
            resp = await self.client.get(f"{self.base_url}/models")
            models = resp.json()["models"]
            for model in models:
                if model["model_id"] == model_id and model["state"] == "HOT":
                    return
            await asyncio.sleep(1)
        raise TimeoutError(f"Model {model_id} did not load in {timeout}s")


# Usage in Provider Router
async def route_to_model(agent_class: str, prompt: str) -> str:
    pool = ModelPoolClient()

    # Get model for agent class
    model_id = get_model_for_agent(agent_class)  # from model_preferences.json

    # Send request (auto-loads if needed)
    response = await pool.chat(
        model_id=model_id,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]
```

---

## Configuration Files

### 1. Model Registry (`configs/pas/model_pool_registry.json`)

```json
{
  "models": [
    {
      "model_id": "qwen2.5-coder:7b",
      "display_name": "Qwen 2.5 Coder 7B",
      "model_type": "code",
      "warmup": true,
      "ttl_minutes": 0,
      "priority": "high",
      "tags": ["code", "python", "fast"]
    },
    {
      "model_id": "llama3.1:8b",
      "display_name": "Llama 3.1 8B",
      "model_type": "general",
      "warmup": true,
      "ttl_minutes": 0,
      "priority": "high",
      "tags": ["general", "reasoning"]
    },
    {
      "model_id": "deepseek-coder-v2:16b",
      "display_name": "DeepSeek Coder V2 16B",
      "model_type": "code",
      "warmup": false,
      "ttl_minutes": 15,
      "priority": "medium",
      "tags": ["code", "advanced", "large"]
    },
    {
      "model_id": "codellama:13b",
      "display_name": "Code Llama 13B",
      "model_type": "code",
      "warmup": false,
      "ttl_minutes": 15,
      "priority": "low",
      "tags": ["code", "fallback"]
    }
  ]
}
```

### 2. Pool Configuration (`configs/pas/model_pool_config.json`)

```json
{
  "default_ttl_minutes": 15,
  "min_ttl_minutes": 5,
  "max_ttl_minutes": 60,
  "check_interval_seconds": 30,
  "warmup_models": ["qwen2.5-coder:7b", "llama3.1:8b"],
  "port_range": {
    "start": 8051,
    "end": 8099
  },
  "max_concurrent_models": 5,
  "memory_limit_mb": 32000,
  "auto_unload": true,
  "graceful_shutdown_seconds": 30
}
```

---

## Advantages of This Design

1. **Zero Configuration for Clients**: Provider Router just calls pool manager, models auto-load
2. **Memory Efficient**: Models unload after TTL, prevents RAM bloat
3. **Low Latency**: Warmup models stay hot, no load delay
4. **Scalable**: 50 port slots, can run 5-10 concurrent models depending on RAM
5. **Observable**: Full metrics, health checks, state tracking
6. **OpenAI Compatible**: Model services use standard API format
7. **Dynamic**: Add/remove models at runtime without restart

---

## Next Steps

1. Implement `services/model_pool_manager/model_pool_manager.py`
2. Create `services/model_pool_manager/model_service_template.py` (FastAPI wrapper)
3. Add pool manager to `scripts/start_all_fastapi_services.sh`
4. Update Provider Router to use Model Pool Client
5. Create admin UI in HMI Settings for model management
