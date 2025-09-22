# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL RULES
1. **No placeholder/sample data in production.** Avoid template lists and hardcoded examples.
2. **Never run training without explicit permission.**
3. **Default to 768D GTR-T5** (no STELLA 1024D unless requested).
4. **Vec2Text usage**: follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage.
5. **Devices**: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
6. **Steps**: Use `--steps 1` for vec2text by default; increase only when asked.

<!-- Audio notifications section removed to keep repo guidance focused and neutral. -->

## 📍 CURRENT STATUS (2025-09-19)
- **Vec2Text**: Use `app/vect_text_vect/vec_text_vect_isolated.py` with `--vec2text-backend isolated`.
- **n8n MCP**: Configured and tested. Use `claude mcp list` to verify connection.


## 📂 KEY COMMANDS

### n8n Integration Commands (NEW - 2025-09-19)
```bash
# Setup n8n MCP server in Claude Code
claude mcp add n8n-local -- npx -y n8n-mcp --n8n-url=http://localhost:5678

# Check MCP connection status
claude mcp list

# Start n8n server
N8N_SECURE_COOKIE=false n8n start

# Import workflows
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json

# Test webhook integration
python3 n8n_workflows/test_webhook_simple.py
python3 n8n_workflows/test_batch_via_webhook.py
```

### General Commands
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "What is AI?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "One day, a little" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Individual decoder checks (optional)
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "girl named Lily found" \
  --subscribers ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Key parameters
# --vec2text-backend isolated (required)
# --subscribers jxe,ielab to test both decoders
# --steps 1 for speed (use 5 for better quality when requested)
# Environment variables enforce CPU usage and project venv
```

## 🏗️ REPOSITORY POINTERS
- **Core runtime**: `app/`
  - Orchestrators: `app/agents/`
  - Models/training: `app/mamba/`, `app/nemotron_vmmoe/`
  - Vec2Text: `app/vect_text_vect/`
  - Utilities: `app/utils/`
- **CLIs and pipelines**: `app/cli/`, `app/pipeline/` (if present)
- **Tests**: `tests/`
- **Docs**: `docs/how_to_use_jxe_and_ielab.md`

## 💡 DEVELOPMENT GUIDELINES
- Python 3.11+ with venv (`python3 -m venv venv && source venv/bin/activate`)
- Install with `python -m pip install -r requirements.txt`
- Lint with `ruff check app tests scripts`
- Run smoke tests: `pytest tests/lnsp_vec2text_cli_main_test.py -k smoke`
- Keep changes aligned with vec2text isolated backend unless otherwise specified