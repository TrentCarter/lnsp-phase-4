# Phase 6: Cloud LLM Provider Adapters — Implementation Plan

**Status:** 🟡 In Progress (0% complete)
**Started:** 2025-11-06
**Owner:** Claude Code
**Estimated Duration:** 1-2 days

---

## 🎯 Objectives

Enable Polyglot Agent Swarm to route requests to **4 major cloud LLM providers**:

1. **OpenAI** (GPT-4, GPT-5, Codex)
2. **Anthropic** (Claude Sonnet 4.5, Claude Haiku 4.5)
3. **Google Gemini** (Gemini 2.5 Pro, Gemini 2.5 Flash)
4. **xAI Grok** (Grok models)

Each adapter will:
- Wrap the cloud provider's native API
- Expose OpenAI-compatible endpoints (`/chat/completions`, `/generate`)
- Auto-register with **Provider Router** (port 6103)
- Handle credentials from `.env` securely
- Provide health checks and service info
- Track usage for cost accounting via **Gateway** (port 6120)

---

## 📦 Deliverables

### P0 (Must Have)

- [x] **Base Cloud Provider Adapter** (`services/cloud_providers/common/base_adapter.py`)
  - Abstract base class for all cloud providers
  - OpenAI-compatible API format
  - Auto-registration with Provider Router
  - Credential management from `.env`
  - Health checks and error handling

- [ ] **OpenAI Adapter** (Port 8100)
  - Models: `gpt-5-codex`, `gpt-4-turbo`, `gpt-3.5-turbo`
  - Context window: 200k (GPT-5), 128k (GPT-4), 16k (GPT-3.5)
  - Cost: Variable ($0.003-$0.015 per 1k tokens)
  - Capabilities: `planning`, `code_write`, `reasoning`, `function_calling`

- [ ] **Anthropic Adapter** (Port 8101)
  - Models: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5`
  - Context window: 200k (Sonnet), 100k (Haiku)
  - Cost: $0.003/$0.015 (Sonnet), $0.00025/$0.00125 (Haiku)
  - Capabilities: `planning`, `code_write`, `reasoning`, `long_context`

- [ ] **Gemini Adapter** (Port 8102)
  - Models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
  - Context window: 2M (Pro), 1M (Flash)
  - Cost: Variable ($0.001-$0.010 per 1k tokens)
  - Capabilities: `planning`, `code_write`, `multimodal`, `long_context`

- [ ] **Grok Adapter** (Port 8103)
  - Models: `grok-beta`, `grok-1`
  - Context window: 128k
  - Cost: TBD (xAI pricing)
  - Capabilities: `planning`, `reasoning`, `real_time`

- [ ] **Credential Management**
  - `.env.template` with all provider keys
  - Secure loading via `python-dotenv`
  - Missing key detection with helpful error messages

- [ ] **Provider Router Integration**
  - Auto-registration on startup (all 4 adapters)
  - Provider metadata (model, context window, cost, features)
  - Deactivation on shutdown (graceful cleanup)

- [ ] **Test Suite** (20+ tests)
  - Health checks (all 4 providers)
  - Service info endpoints (all 4 providers)
  - Chat completions (mock responses)
  - Provider registration (registry integration)
  - Credential validation
  - Error handling (missing keys, API failures)

- [ ] **Startup/Shutdown Scripts**
  - `scripts/start_phase6_cloud_providers.sh`
  - `scripts/stop_phase6_cloud_providers.sh`
  - `scripts/test_phase6.sh`

- [ ] **Documentation**
  - `docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE06_COMPLETE.md`
  - Update `PROGRESS.md` to reflect Phase 6 completion
  - Update `docs/PHASE6_CLOUD_PROVIDERS_PLAN.md` (this file)

### P1 (Should Have)

- [ ] **Cost Tracking Integration**
  - Track per-provider spend in Gateway
  - Budget alerts (75%, 90%, 100% thresholds)
  - Cost receipts → `artifacts/costs/<run_id>.jsonl`

- [ ] **Retry Logic & Fallbacks**
  - Exponential backoff for transient errors
  - Automatic fallback to cheaper models
  - Circuit breaker for failed providers

- [ ] **Rate Limiting**
  - Per-provider rate limits
  - Token-based throttling
  - Queue overflow handling

### P2 (Nice to Have)

- [ ] **Streaming Support**
  - Server-Sent Events (SSE) for all providers
  - Streaming endpoint `/chat/completions?stream=true`
  - Backpressure handling

- [ ] **Function Calling**
  - OpenAI function calling format
  - Anthropic tool use
  - Gemini function declarations

- [ ] **Multi-Modal Support**
  - Image inputs (Gemini, GPT-4V)
  - File uploads
  - Vision capabilities

---

## 🏗️ Architecture

### Directory Structure

```
services/
  cloud_providers/
    common/
      __init__.py
      base_adapter.py          # Base class for all cloud adapters
      schemas.py               # Pydantic models (OpenAI-compatible)
      credential_manager.py    # .env credential loading
    openai/
      __init__.py
      openai_adapter.py        # OpenAI wrapper (Port 8100)
    anthropic/
      __init__.py
      anthropic_adapter.py     # Anthropic wrapper (Port 8101)
    gemini/
      __init__.py
      gemini_adapter.py        # Gemini wrapper (Port 8102)
    grok/
      __init__.py
      grok_adapter.py          # Grok wrapper (Port 8103)
```

### Base Adapter Interface

```python
class BaseCloudAdapter:
    """
    Abstract base class for cloud LLM provider adapters

    Responsibilities:
    - Wrap cloud provider API
    - Provide OpenAI-compatible endpoints
    - Auto-register with Provider Router
    - Load credentials from .env
    - Health checks and error handling
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        port: int,
        api_key_env_var: str,
        capabilities: List[str],
        context_window: int,
        cost_per_input_token: float,
        cost_per_output_token: float
    ):
        pass

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """OpenAI-compatible chat completions endpoint"""
        pass

    async def register_with_router(self, router_url: str = "http://localhost:6103"):
        """Register provider with Provider Router"""
        pass

    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        pass
```

### Registration Flow

```
┌──────────────────┐
│  Cloud Adapter   │
│  (Port 8100-8103)│
└────────┬─────────┘
         │
         │ 1. Startup
         │ Load credentials from .env
         │
         ▼
┌──────────────────┐
│  Provider Router │  2. Register provider metadata
│  (Port 6103)     │     - name: "openai-gpt5-codex"
└────────┬─────────┘     - model: "gpt-5-codex"
         │                - context_window: 200000
         │                - cost_per_input_token: 0.000003
         │                - cost_per_output_token: 0.000015
         │                - endpoint: "http://localhost:8100"
         │                - features: ["function_calling", "streaming"]
         ▼
┌──────────────────┐
│  Gateway         │  3. Route requests
│  (Port 6120)     │     - Select provider via /select
└──────────────────┘     - Track costs
                         - Broadcast events
```

---

## 🔐 Credential Management

### .env Template

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-5-codex

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_NAME_HIGH=claude-sonnet-4-5-20250929
ANTHROPIC_MODEL_NAME_MEDIUM=claude-sonnet-4-5-20250929
ANTHROPIC_MODEL_NAME_LOW=claude-haiku-4-5

# Google Gemini
GEMINI_API_KEY=AIza...
GEMINI_PROJECT_ID=your-project-id
GEMINI_MODEL_NAME_LOW=gemini-2.5-flash-lite
GEMINI_MODEL_NAME_MEDIUM=gemini-2.5-flash
GEMINI_MODEL_NAME_HIGH=gemini-2.5-pro

# xAI Grok
GROK_API_KEY=...
GROK_MODEL_NAME=grok-beta
```

### Credential Loading

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env\n"
        "Please add it to .env file:\n"
        "  OPENAI_API_KEY=sk-..."
    )
```

---

## 📊 Provider Matrix

| Provider | Model | Context | Cost (in/out per 1k) | Capabilities |
|----------|-------|---------|----------------------|--------------|
| **OpenAI** | gpt-5-codex | 200k | $0.003 / $0.015 | planning, code_write, function_calling |
| **OpenAI** | gpt-4-turbo | 128k | $0.010 / $0.030 | planning, reasoning, vision |
| **Anthropic** | claude-sonnet-4-5 | 200k | $0.003 / $0.015 | planning, code_write, long_context |
| **Anthropic** | claude-haiku-4-5 | 100k | $0.00025 / $0.00125 | classification, extraction, fast_tasks |
| **Gemini** | gemini-2.5-pro | 2M | $0.010 / $0.030 | planning, multimodal, long_context |
| **Gemini** | gemini-2.5-flash | 1M | $0.001 / $0.003 | fast_tasks, code_write |
| **Grok** | grok-beta | 128k | TBD | planning, reasoning, real_time |

---

## 🧪 Testing Strategy

### Unit Tests

```bash
# Test credential loading
pytest tests/test_credentials.py

# Test adapter initialization
pytest tests/test_cloud_adapters.py::test_openai_init
pytest tests/test_cloud_adapters.py::test_anthropic_init
pytest tests/test_cloud_adapters.py::test_gemini_init
pytest tests/test_cloud_adapters.py::test_grok_init
```

### Integration Tests

```bash
# Test provider registration
./scripts/test_phase6.sh

# Expected:
# ✅ OpenAI adapter registered with router
# ✅ Anthropic adapter registered with router
# ✅ Gemini adapter registered with router
# ✅ Grok adapter registered with router
```

### Health Checks

```bash
# All 4 adapters should be healthy
curl http://localhost:8100/health  # OpenAI
curl http://localhost:8101/health  # Anthropic
curl http://localhost:8102/health  # Gemini
curl http://localhost:8103/health  # Grok
```

---

## 📈 Success Metrics

- ✅ All 4 cloud providers registered in Provider Router
- ✅ 20+ tests passing (100% pass rate)
- ✅ Health checks passing for all 4 adapters
- ✅ Credentials loaded securely from .env
- ✅ OpenAI-compatible API endpoints working
- ✅ Provider selection via Gateway working
- ✅ Cost tracking receipts generated
- ✅ Startup/shutdown scripts functional
- ✅ Documentation complete

---

## 🚀 Implementation Steps

### Day 1: Core Infrastructure (6-8 hours)

1. **Create Base Adapter** (1-2 hours)
   - `services/cloud_providers/common/base_adapter.py`
   - `services/cloud_providers/common/schemas.py`
   - `services/cloud_providers/common/credential_manager.py`

2. **OpenAI Adapter** (1 hour)
   - `services/cloud_providers/openai/openai_adapter.py`
   - Test health check and registration

3. **Anthropic Adapter** (1 hour)
   - `services/cloud_providers/anthropic/anthropic_adapter.py`
   - Test health check and registration

4. **Gemini Adapter** (1.5 hours)
   - `services/cloud_providers/gemini/gemini_adapter.py`
   - Google-specific auth handling

5. **Grok Adapter** (1 hour)
   - `services/cloud_providers/grok/grok_adapter.py`
   - xAI-specific auth handling

6. **.env Template** (0.5 hours)
   - Create `.env.template`
   - Document credential setup

### Day 2: Testing & Integration (4-6 hours)

7. **Test Suite** (2-3 hours)
   - `scripts/test_phase6.sh`
   - Unit tests for all adapters
   - Integration tests with Provider Router

8. **Startup/Shutdown Scripts** (1 hour)
   - `scripts/start_phase6_cloud_providers.sh`
   - `scripts/stop_phase6_cloud_providers.sh`

9. **Documentation** (1-2 hours)
   - Session summary
   - Update PROGRESS.md
   - API usage examples

---

## 🔗 Dependencies

**Required Services:**
- ✅ Provider Router (Port 6103) - Phase 3
- ✅ Gateway (Port 6120) - Phase 3
- ✅ Agent Registry (Port 6121) - Phase 0

**Required Files:**
- `.env` with cloud provider API keys
- `python-dotenv` package (for credential loading)
- Cloud provider SDKs:
  - `openai` (OpenAI Python SDK)
  - `anthropic` (Anthropic Python SDK)
  - `google-generativeai` (Gemini SDK)
  - `xai` (Grok SDK, if available)

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Missing API keys | Provide `.env.template` and helpful error messages |
| Provider API changes | Use official SDKs, version pin |
| Cost overruns | Budget alerts via Gateway (75%, 90%, 100%) |
| Provider outages | Fallback to cheaper models or local LLMs |
| Rate limits | Exponential backoff, circuit breaker pattern |

---

## 📝 Notes

- **Phase 5 Integration**: Cloud adapters follow same pattern as local LLM services (Phase 5)
- **OpenAI Compatibility**: All adapters expose OpenAI-compatible endpoints for consistency
- **Cost Tracking**: Gateway automatically tracks costs via Provider Router metadata
- **Security**: API keys stored in `.env` (never committed to git)
- **Extensibility**: Easy to add new providers (e.g., Cohere, AI21) using base adapter

---

**Last Updated:** 2025-11-06
**Status:** 🟡 In Progress
**Next:** Create base adapter class and credential management
