# LNSP Phase 4 — Quickstart and Working Commands


This README is focused on the current Phase 4 repository. Legacy notes and command dumps from earlier projects have been archived to `docs/archive/readme_legacy_20250918.txt`.

# 9/27/2025

./venv/bin/python3 tests/data_generator.py
./venv/bin/python3 tests/data_snapshot.py


# 9/26/2025
make lnsp-status   #LNSP RAG — System Status

# 9/24/2025
PYTHONPATH=src \
LNSP_LLM_ENDPOINT=http://127.0.0.1:11434 \
LNSP_LLM_MODEL=llama3.1:8b-instruct \
python3 tools/make_cpesh_quadruplets.py --n 10 --embed


# 9/22/2025
tests/test_prompt_extraction.py


# 9/21/2025. 

./venv/bin/python3 scripts/data_processing/read_factoid_wiki.py



# 9/20/25
┌─────────────────────────────────────────────────────────────────┐
│ Concept (C): "Light-dependent reactions split water"                │
│ Probe Q (Q): "What process in photosynthesis splits water?"         │
│ Expected (A): "Photolysis of water"                                  │
└─────────────────────────────────────

##. Pre Requisites

export N8N_BASIC_AUTH_ACTIVE=true
export N8N_BASIC_AUTH_USER=trent@trentcarter.com
export N8N_BASIC_AUTH_PASSWORD=Karenm12!


lsof -nP -iTCP:5678 -sTCP:LISTEN -t | xargs -r kill -9

N8N_USER_MANAGEMENT_DISABLED=true N8N_SECURE_COOKIE=false n8n start
N8N_BASIC_AUTH_ACTIVE=true N8N_BASIC_AUTH_USER=trent@trentcarter.com N8N_BASIC_AUTH_PASSWORD=Karenm8n! n8n start

export N8N_PUBLIC_API_DISABLED=true
export N8N_PUBLIC_API_SWAGGERUI_DISABLED=true
N8N_USER_MANAGEMENT_DISABLED=true
N8N_SECURE_COOKIE=false
n8n start

⏺ n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json

  Or for multiple workflows:
  n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json
  n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json

  You can also use wildcards:
  n8n import:workflow --input=n8n_workflows/*.json

## Environment Setup

- Create and activate a virtual environment (Python 3.11+):
  
  ```bash
  python3 -m venv venv && source venv/bin/activate
  ```

- Install dependencies:
  
  ```bash
  python -m pip install -r requirements.txt
  # Add IELab adapters if needed
  # python -m pip install -r requirements_ielab.txt
  ```

## Lint and Tests

- Lint (fast):
  
  ```bash
  ruff check app tests scripts
  ```

- Smoke test (CLI regression):
  
  ```bash
  pytest tests/lnsp_vec2text_cli_main_test.py -k smoke
  ```

## Vec2Text Isolated Quick Tests (CPU)

Use the isolated vec2text orchestrator to validate decoders in a controlled environment. These commands are aligned with `CLAUDE.md` requirements.

```bash
# Test 1: "What is AI?" with both decoders
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "What is AI?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Test 2: "One day, a little" with both decoders
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "One day, a little" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Test 3: "girl named Lily found" with both decoders
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "girl named Lily found" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Optional: test decoders individually
# JXE only
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "girl named Lily found" \
  --subscribers jxe \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# IELab only
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "girl named Lily found" \
  --subscribers ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1
```

Notes:
- `--steps 1` for fast debugging; use `--steps 5` for better quality.
- `--subscribers jxe,ielab` runs both decoders.
- `--vec2text-backend isolated` follows the current project guidance.

## Optional: Pipeline and MLFlow

If present in your checkout, the following are the standard entry points:

- Pipeline dry-run:
  
  ```bash
  python app/pipeline/run_project.py --project-config docs/sample_project.json
  ```

- Launch MLFlow locally:
  
  ```bash
  bash scripts/start_mlflow.sh
  ```

If these files are not present, skip this section or consult project-specific docs.

## n8n Workflows

Test and automate the vec2text pipeline with n8n workflows:

```bash
# Start n8n
N8N_SECURE_COOKIE=false n8n start

# Import workflows (individual files)
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json

# Or use the import script
./n8n_workflows/import_workflows.sh

# Test webhook API (after activating in n8n UI)
curl -X POST http://localhost:5678/webhook/vec2text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is AI?"}'
```

Available workflows:
- `vec2text_test_workflow.json`: Batch testing with predefined texts
- `webhook_api_workflow.json`: REST API endpoint for vec2text
- See `n8n_workflows/README.md` for detailed usage

## Repository Pointers

- Core runtime in `app/`:
  - Orchestrators in `app/agents/`
  - VMMoE/Mamba components under `app/nemotron_vmmoe/` and `app/mamba/`
  - Vec2Text components under `app/vect_text_vect/`
  - Shared utilities in `app/utils/`
- Command-line interfaces under `app/cli/` and project pipelines in `app/pipeline/` (if present).
- Tests live in `tests/` and should mirror module paths.
- n8n workflows in `n8n_workflows/` for automation and API access

## Housekeeping

- This README intentionally removes legacy commands and logs from previous phases/projects.
- For historical reference, see: `docs/archive/readme_legacy_20250918.txt`.
