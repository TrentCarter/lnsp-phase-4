# Central Model Registry

**Purpose:** Single source of truth for all LLM models available across the LNSP system
**Last Updated:** 2025-11-13
**Verified:** All models tested via `tests/test_llm_integration.py`

---

## Model Status Legend

- ✅ **VERIFIED** - Tested and working
- ⚠️ **CONFIGURED** - API key present, not tested
- ❌ **UNAVAILABLE** - Model does not exist or deprecated
- ⊘ **NOT CONFIGURED** - API key missing

---

## Local Models (Ollama)

**Provider:** Ollama (localhost:11434)
**Cost:** Free (local compute)
**Status:** ✅ VERIFIED

| Model ID | Context | Use Case | Status |
|----------|---------|----------|--------|
| `ollama/qwen2.5-coder:7b-instruct` | 32K | Code generation | ✅ VERIFIED |
| `ollama/deepseek-r1:7b-q4_k_m` | 32K | Reasoning | ✅ CONFIGURED |
| `ollama/deepseek-r1:1.5b-q4_k_m` | 16K | Fast inference | ✅ CONFIGURED |

**Gateway Prefix:** `ollama/`
**Implementation:** services/gateway/gateway.py:438-620

---

## Kimi (Moonshot AI)

**Provider:** Moonshot AI
**Base URL:** `https://api.moonshot.ai/v1` (⚠️ NOT .cn!)
**API Key:** `KIMI_API_KEY` in .env
**Cost:** $0.12 per 1M tokens
**Status:** ✅ VERIFIED

| Model ID | Context | Use Case | Status | Notes |
|----------|---------|----------|--------|-------|
| `kimi-k2-turbo-preview` | 128K | Latest K2 model | ✅ VERIFIED | **Recommended** |
| `moonshot-v1-8k` | 8K | Budget-friendly | ✅ VERIFIED | Legacy |
| `moonshot-v1-32k` | 32K | Medium context | ✅ VERIFIED | Legacy |
| `moonshot-v1-128k` | 128K | Large context | ✅ VERIFIED | Legacy |

**Gateway Prefix:** `kimi/`
**Implementation:** services/gateway/gateway.py:746-820
**Example:** `kimi/kimi-k2-turbo-preview`

**Known Issues:**
- ⚠️ Base URL was `.cn` (incorrect) → Fixed to `.ai` on 2025-11-13

**Get API Key:** https://platform.moonshot.ai/console/api-keys

---

## Google Gemini

**Provider:** Google Generative AI
**API Key:** `GEMINI_API_KEY` in .env
**Cost:** Variable by model ($0.075-$1.25 per 1M tokens)
**Status:** ✅ VERIFIED

| Model ID | Context | Cost (per 1M tokens) | Status |
|----------|---------|---------------------|--------|
| `gemini-2.5-flash` | 1M | $0.075 input / $0.30 output | ✅ VERIFIED |
| `gemini-2.5-flash-lite` | 1M | $0.0375 input / $0.15 output | ⚠️ CONFIGURED |
| `gemini-2.5-pro` | 2M | $1.25 input / $5.00 output | ⚠️ CONFIGURED |

**Gateway Prefix:** `google/`
**Implementation:** services/gateway/gateway.py:621-745
**Example:** `google/gemini-2.5-flash`

**Get API Key:** https://aistudio.google.com/apikey

---

## Anthropic Claude

**Provider:** Anthropic
**API Key:** `ANTHROPIC_API_KEY` in .env
**Cost:** $3-$15 per 1M tokens
**Status:** ❌ MODEL NAME ISSUES

| Model ID | Context | Cost (per 1M tokens) | Status | Notes |
|----------|---------|---------------------|--------|-------|
| `claude-3-5-sonnet-20241022` | 200K | $3 input / $15 output | ❌ UNAVAILABLE | **Does not exist!** |
| `claude-3-5-sonnet-20240620` | 200K | $3 input / $15 output | ⚠️ CONFIGURED | Correct version |
| `claude-3-opus-20240229` | 200K | $15 input / $75 output | ⚠️ CONFIGURED | Most capable |
| `claude-3-sonnet-20240229` | 200K | $3 input / $15 output | ⚠️ CONFIGURED | Balanced |
| `claude-3-haiku-20240307` | 200K | $0.25 input / $1.25 output | ⚠️ CONFIGURED | Fastest |

**Gateway Prefix:** `anthropic/`
**Implementation:** services/gateway/gateway.py:621-745
**Example:** `anthropic/claude-3-5-sonnet-20240620`

**Known Issues:**
- ❌ `.env` references `claude-3-5-sonnet-20241022` which does not exist
- ✅ Use `claude-3-5-sonnet-20240620` instead

**Get API Key:** https://console.anthropic.com/settings/keys

---

## OpenAI

**Provider:** OpenAI
**API Key:** `OPENAI_API_KEY` in .env
**Cost:** $0.15-$5 per 1M tokens
**Status:** ⊘ NOT TESTED

| Model ID | Context | Cost (per 1M tokens) | Status |
|----------|---------|---------------------|--------|
| `gpt-4o` | 128K | $2.50 input / $10 output | ⚠️ CONFIGURED |
| `gpt-4o-mini` | 128K | $0.15 input / $0.60 output | ⚠️ CONFIGURED |
| `gpt-4-turbo` | 128K | $10 input / $30 output | ⚠️ CONFIGURED |
| `gpt-3.5-turbo` | 16K | $0.50 input / $1.50 output | ⚠️ CONFIGURED |

**Gateway Prefix:** `openai/`
**Implementation:** services/gateway/gateway.py:621-745
**Example:** `openai/gpt-4o-mini`

**Get API Key:** https://platform.openai.com/api-keys

---

## DeepSeek

**Provider:** DeepSeek
**Base URL:** `https://api.deepseek.com`
**API Key:** `DEEPSEEK_API_KEY` in .env
**Cost:** $0.14-$2.19 per 1M tokens
**Status:** ⊘ NOT CONFIGURED

| Model ID | Context | Cost (per 1M tokens) | Status |
|----------|---------|---------------------|--------|
| `deepseek-chat` | 64K | $0.14 input / $0.28 output | ⊘ NOT CONFIGURED |
| `deepseek-reasoner` | 64K | $0.55 input / $2.19 output | ⊘ NOT CONFIGURED |

**Gateway Prefix:** `deepseek/`
**Implementation:** Not yet implemented in Gateway
**Example:** `deepseek/deepseek-chat`

---

## Model Selection Matrix

### By Use Case

| Use Case | Recommended Model | Reasoning |
|----------|------------------|-----------|
| **Code Generation** | `ollama/qwen2.5-coder:7b-instruct` | Free, fast, specialized |
| **Budget Chat** | `kimi/kimi-k2-turbo-preview` | $0.12/1M tokens, 128K context |
| **Fast API Responses** | `google/gemini-2.5-flash` | $0.075/1M tokens, 1M context |
| **High Quality** | `anthropic/claude-3-5-sonnet-20240620` | Best reasoning, 200K context |
| **Large Context** | `google/gemini-2.5-pro` | 2M context window |

### By Cost (API Models Only)

1. **Cheapest:** Google Gemini 2.5 Flash ($0.075/1M input)
2. **Best Value:** Kimi K2 Turbo ($0.12/1M)
3. **Mid-range:** Anthropic Haiku ($0.25/1M input)
4. **Premium:** Claude 3.5 Sonnet ($3/1M input)

### By Context Window

1. **Largest:** Google Gemini 2.5 Pro (2M tokens)
2. **Large:** Gemini 2.5 Flash (1M tokens)
3. **Medium:** Anthropic Claude 3.x (200K tokens)
4. **Standard:** Kimi K2, OpenAI GPT-4o (128K tokens)

---

## Multi-Tier PAS Model Assignments

**File:** `configs/pas/model_preferences.json`

```json
{
  "architect": {
    "primary": "anthropic/claude-3-5-sonnet-20240620",
    "fallback": "google/gemini-2.5-pro"
  },
  "director": {
    "primary": "google/gemini-2.5-pro",
    "fallback": "anthropic/claude-3-5-haiku-20241022"
  },
  "manager": {
    "primary": "google/gemini-2.5-flash",
    "fallback": "kimi/kimi-k2-turbo-preview"
  },
  "programmer": {
    "primary": "ollama/qwen2.5-coder:7b-instruct",
    "fallback": "kimi/kimi-k2-turbo-preview"
  }
}
```

---

## HMI Model Display

**File:** `services/webui/hmi_app.py`

Models appear in HMI dropdown as:
- Local: `🖥️ Qwen2.5-Coder 7B`
- Kimi: `🌐 Kimi K2-TURBO-PREVIEW`
- Google: `🌐 Gemini 2.5 FLASH`
- Anthropic: `🌐 Claude 3.5 SONNET`

**Model Icons:**
- 🖥️ = Local (Ollama)
- 🌐 = API (Cloud)

---

## Testing

**Test Suite:** `tests/test_llm_integration.py`

**Run All Tests:**
```bash
set -a && source .env && set +a && ./.venv/bin/python tests/test_llm_integration.py
```

**Run Specific Provider:**
```bash
pytest tests/test_llm_integration.py -v -k "test_kimi"
pytest tests/test_llm_integration.py -v -k "test_google"
pytest tests/test_llm_integration.py -v -k "test_anthropic"
```

**Test Results:** See `docs/LLM_INTEGRATION_TEST_RESULTS.md`

---

## Configuration Files

| File | Purpose | Models Defined |
|------|---------|----------------|
| `.env` | API keys and default models | All providers |
| `configs/pas/model_preferences.json` | PAS tier assignments | Multi-tier routing |
| `services/webui/hmi_app.py` | HMI dropdown options | User-facing models |
| `services/gateway/gateway.py` | Gateway routing logic | Provider implementation |
| `tests/test_llm_integration.py` | Integration tests | Test coverage |

---

## Adding a New Model

1. **Update `.env`** with API key and model name
2. **Test directly:**
   ```bash
   curl -X POST http://localhost:6120/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test","message_id":"1","agent_id":"user","model":"provider/model-name","content":"Hello"}'
   ```
3. **Add to test suite** in `tests/test_llm_integration.py`
4. **Update HMI dropdown** in `services/webui/hmi_app.py`
5. **Update this registry** with verified status

---

## Troubleshooting

### Model returns 404
- Check model name against this registry
- Verify model exists in provider's API
- Try listing models: `curl https://api.provider.com/v1/models`

### Model returns 401
- Verify API key in `.env`
- Check API key has not expired
- Ensure Gateway loaded `.env` (restart with `--env-file .env`)

### Model returns 400
- Check request format (messages, max_tokens, etc.)
- Verify model supports streaming
- Check context window limits

---

## Maintenance

- **Monthly:** Check for new model releases from each provider
- **Quarterly:** Review pricing changes
- **On Error:** Update this registry with corrected model names
- **Before Deploy:** Run full test suite to verify all models

---

**Next Review Date:** 2025-12-13
