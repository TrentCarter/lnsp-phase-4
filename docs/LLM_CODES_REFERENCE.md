# LLM Codes Reference

**Last Updated**: 2025-11-12

Quick reference guide for 6-character LLM codes used in communication logs.

---

## Purpose

The LLM code provides a quick visual identifier in logs to see which AI model was used for each operation, without having to read the full model string. This makes it easy to scan logs and identify which LLM handled each task.

---

## Format

- **Length**: 6 characters
- **Position**: Between message type and from_agent in parsed logs
- **Position**: 5th field in raw pipe-delimited log files
- **Color**: Magenta (in colored output)
- **Default**: `------` (6 dashes) when no LLM is used

---

## Complete Code Reference

### Claude Models

| Code | Model | Full Name |
|------|-------|-----------|
| `CLD450` | Claude 4.5 Sonnet | `anthropic/claude-4.5-sonnet` |
| `CLD370` | Claude 3.7 Sonnet | `anthropic/claude-3.7-sonnet` |
| `CLD350` | Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` |
| `CLD30O` | Claude 3 Opus | `anthropic/claude-3-opus` |
| `CLD30S` | Claude 3 Sonnet | `anthropic/claude-3-sonnet` |
| `CLD30H` | Claude 3 Haiku | `anthropic/claude-3-haiku` |
| `CLD300` | Claude 3 (generic) | `anthropic/claude-3` |

### GPT Models

| Code | Model | Full Name |
|------|-------|-----------|
| `GPT500` | GPT-5 | `openai/gpt-5` |
| `GPT450` | GPT-4.5 | `openai/gpt-4.5-turbo` |
| `GPT400` | GPT-4 | `openai/gpt-4` |
| `GPT350` | GPT-3.5 | `openai/gpt-3.5-turbo` |
| `GPT300` | GPT-3 | `openai/gpt-3` |

### Gemini Models

| Code | Model | Full Name |
|------|-------|-----------|
| `GMI250` | Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| `GM250P` | Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| `GMI200` | Gemini 2.0 | `google/gemini-2.0-flash` |
| `GMI150` | Gemini 1.5 | `google/gemini-1.5-pro` |

### Qwen Models

| Code | Model | Full Name |
|------|-------|-----------|
| `QWE250` | Qwen 2.5 | `ollama/qwen2.5-coder:7b-instruct` |
| `QWE200` | Qwen 2.0 | `ollama/qwen2-coder` |

### Llama Models

| Code | Model | Full Name |
|------|-------|-----------|
| `LMA310` | Llama 3.1 | `ollama/llama3.1:8b` |
| `LMA300` | Llama 3 | `ollama/llama3` |
| `LMA200` | Llama 2 | `ollama/llama2` |

### Special Cases

| Code | Meaning |
|------|---------|
| `------` | No LLM used (e.g., Gateway commands, system messages) |

---

## Implementation

### Code Location

- **Function**: `get_llm_code()` in `services/common/comms_logger.py`
- **Parser**: `tools/parse_comms_log.py` (formats for display)
- **Tests**: `tests/test_llm_codes.py` (validates all codes)

### Adding New LLM Codes

To add support for a new LLM model:

1. Edit `services/common/comms_logger.py`
2. Add detection logic to `get_llm_code()` function
3. Follow naming pattern: `[VENDOR][VERSION][VARIANT]`
   - Vendor: 3 letters (CLD, GPT, GMI, QWE, LMA)
   - Version: 2-3 chars (450, 30O, 250, 310)
   - Total: 6 characters
4. Add test case to `tests/test_llm_codes.py`
5. Run tests: `./.venv/bin/python tests/test_llm_codes.py`
6. Update this reference document

### Example New Code

```python
# In get_llm_code() function
if "mistral" in llm_lower:
    if "7b" in llm_lower:
        return "MST70B"
    elif "8x7b" in llm_lower:
        return "MST8X7"
    return "MISTRL"
```

---

## Usage Examples

### Raw Log Format

```
2025-11-12T17:43:55.830-05:00|Mgr-Code-02|System|STATUS|GMI250|Manager service started|google/gemini-2.5-flash|test-run-001|started|-|-
```

### Parsed Display Format

```
2025-11-12 17:43:55 STATUS     GMI250  Mgr-Code-02     → System          Manager service started [started] [gemini-2.5-flash]
```

### Filtering by LLM

```bash
# View all logs with Claude 4.5
./tools/parse_comms_log.py | grep CLD450

# View all logs with Qwen 2.5
./tools/parse_comms_log.py | grep QWE250

# View all logs WITHOUT LLM (system messages)
./tools/parse_comms_log.py | grep "\-\-\-\-\-\-"
```

---

## Benefits

1. **Quick Scanning**: Instantly identify which LLM handled each task
2. **Performance Analysis**: Compare task completion times across different LLMs
3. **Cost Tracking**: Identify usage patterns for different LLM tiers
4. **Debugging**: Trace issues back to specific LLM versions
5. **Auditing**: Verify which LLM was used for compliance/quality checks

---

## Related Documentation

- **Full Logging Guide**: `docs/COMMS_LOGGING_GUIDE.md`
- **Log Parser Tool**: `tools/parse_comms_log.py`
- **Implementation**: `services/common/comms_logger.py`
- **Tests**: `tests/test_llm_codes.py`

---

**Note**: LLM codes are automatically extracted from the `llm_model` field. No manual configuration is needed - just provide the full model string and the code is generated automatically.
