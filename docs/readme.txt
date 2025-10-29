# LNSP Phase 4 — Quickstart and Working Commands
source .venv/bin/activate
claude --dangerously-skip-permissions


docs/PRDs/PRD_FastAPI_Services.md
scripts/start_all_fastapi_services.sh
scripts/stop_lvm_services.sh
scripts/start_lvm_services.sh
./scripts/stop_lvm_services.sh && ./scripts/start_lvm_services.sh


# 10/29/2025

  🚀 How to Use

  1. Run Tests
  ./.venv/bin/pytest tests/test_p4_safeguards.py -v
  ./.venv/bin/pytest tests/test_p4_safeguards.py -v --tb=short

  2. Check Deployment Readiness
  bash scripts/deployment_gate.sh 9001

  3. View Metrics
  curl -s http://localhost:9001/metrics | jq

  4. Monitor Health
  curl -s http://localhost:9001/health | jq


  Should be loaded: artifacts/lvm/models/amn_20251023_204747/ (October 23, NEW - 0.5603 val cosine on 543k sequences)

# 10/28/2025 

  1. Start Core Services (GTR-T5 + Vec2Text):
  ./scripts/start_all_fastapi_services.sh
  sleep 10  # Wait for initialization

  2. Start LVM Chat Services:
  ./scripts/start_lvm_services.sh

  Created:
  - ✅ Symlink: artifacts/lvm/models/transformer_optimized_v0.pt → optimized model (val_cosine: 0.5865)
  - ✅ Updated scripts/start_lvm_services.sh to start port 9006
  - ✅ Updated scripts/stop_lvm_services.sh to stop port 9006

  Access the Master UI:
  http://localhost:9000

  Port Mapping Now:
  | Port | Model           | Type               | Val Cosine |
  |------|-----------------|--------------------|------------|
  | 9000 | MASTER AI CHAT  | ALL LVMs.          | N/A        |
  | 9001 | AMN             | Best OOD + Fastest | 0.5597     |
  | 9002 | Transformer     | Baseline           | 0.5774     |
  | 9003 | GRU             | Runner-up          | 0.5920     |
  | 9004 | LSTM            | Deprecated         | 0.4102 ⚠️  |
  | 9005 | Vec2Text Direct | Passthrough        | N/A        |
  | 9006 | Transformer     | Optimized ✅        | 0.5864     |

  2. Auto-Chunking Feature Added

  3 New Chunking Modes:

  1. Adaptive (≥5 sentences) [Default]
    - Auto-chunks only if message has ≥5 sentences
    - Otherwise uses retrieval-primed context (4 supports + 1 query)
  2. By Sentence (1:1) [NEW!]
    - 1 sentence = 1 chunk
    - 3 sentences → 3 vectors
    - 10 sentences → 10 vectors (last 5 used for LVM)
  3. Fixed (force 5 chunks)
    - Always chunks into 5 pieces
    - Groups sentences to make 5 chunks
  4. Off (1 vector)
    - No chunking, single vector with retrieval-primed context

API endpoints:
  POST /chat  - Chat-style inference (text → text)
  POST /infer - Low-level inference (vectors → vector)
  GET /info   - Model information
  (Vec2Text direct service skips /infer and LVM model loading)

  API endpoints:
    POST /chat  - Chat-style inference (text → text)
    POST /infer - Low-level inference (vectors → vector)
    GET /info   - Model information

  Logs: /tmp/lvm_api_logs/

  Quick Test

  # Test AMN Chat (fastest, best OOD)
  curl -X POST http://localhost:9001/chat \
    -H "Content-Type: application/json" \
    -d '{"messages": ["Hello, how are you?"], "temperature": 0.7}'

  # Or visit browser for beautiful purple UI
  open http://localhost:9001/chat
# ===. 10/10/2025. ===

# Recommended (no reload, venv Python 3.11)

./.venv/bin/python tools/launch_fastapis.py

# Enable reload for all services

./.venv/bin/python tools/launch_fastapis.py --reload

# Start a subset

./.venv/bin/python tools/launch_fastapis.py --services chunker,tmd_router,ingest

# Run a parallel stack with port offset

./.venv/bin/python tools/launch_fastapis.py --port-offset 100

# 10/8/2025. =====

# Chunker Test Web Interface 
http://127.0.0.1:8001/web



#===. MODELS. ====
10/7/2025
  - models/lvm_lstm_retrained.pt ← Use this for production
  - models/lvm_mamba_new.pt ← Keep for future comparison with larger dataset

# 6 Stage Process from 10/7/2025
  6-Stage Process:

  STAGE 1: Text → Concepts (LLM + TMD)
  - Input: "neural networks in AI"
  - LLM extracts concepts with TMD:
    - "neural network" → TMD: (15, 14, 9)
    - "artificial intelligence" → TMD: (15, 5, 9)

  STAGE 2: vecRAG Lookup (FAISS Search)
  - For each concept, search FAISS (8K concepts)
  - IF FOUND (cosine > 0.85): Use existing vector
  - IF NOT FOUND:
    - Generate GTR-T5 (768D) + LLM TMD (16D) → 784D
    - INSERT ATOMICALLY to PostgreSQL + Neo4j + FAISS
    - Assign parent_id/child_id for ontological order

  STAGE 3: Mamba Prediction
  - Load trained model: models/lvm_wordnet.pt
  - Forward pass: concept_vec (784D) → Mamba → next_vec (784D)

  STAGE 4: Next Concept Lookup
  - Search FAISS for next_vec
  - IF FOUND: Retrieve concept_text
  - IF NOT FOUND:
    - Fallback to vec2text (JXE + IELab decoders)
    - Generate text from vector (~2s latency)
    - INSERT new concept (atomic 3-way)

  STAGE 5: Multi-Concept Wait ⚠️ CRITICAL!
  - IF multiple concepts: Wait for ALL predictions
  - Collect: [(input₁, next₁), (input₂, next₂)]
  - DO NOT send partial results!

  STAGE 6: LLM Response Smoothing
  - Send to Llama 3.1:
    - Original query
    - Input concepts
    - Predicted next concepts
  - Generate smooth natural language response

  ---
  Performance:

  | Stage         | Operation            | Latency   |
  |---------------|----------------------|-----------|
  | 1             | Concept extraction   | 500ms     |
  | 2             | vecRAG lookup        | 0.1ms     |
  | 2b            | GTR-T5 + TMD (new)   | 50ms      |
  | 3             | Mamba forward        | 10ms      |
  | 4             | vecRAG lookup (next) | 0.1ms     |
  | 4b            | vec2text (if miss)   | 2000ms ⚠️ |
  | 6             | LLM smoothing        | 800ms     |
  | TOTAL (best)  | All found            | ~1.3s     |
  | TOTAL (worst) | New + vec2text       | ~3.5s     |

  

This README is focused on the current Phase 4 repository. Legacy notes and command dumps from earlier projects have been archived to `docs/archive/readme_legacy_20250918.txt`.

# 9/30/2025

  # Will process 5,007 remaining items (items 4,994-10,000)
  ./.venv/bin/python -m src.ingest_factoid \
    --file-path data/datasets/factoid-wiki-large/factoid_wiki.jsonl \
    --num-samples 10000 \
    --write-pg \
    --write-neo4j \
    --faiss-out artifacts/fw10k.npz

    
# 9/29/2025

./venv/bin/python3 tools/lnsprag_status.py --matrix

./venv/bin/python3 reports/scripts/generate_ingestion_report.py


# 9/28/2025
pytest tests/cli/test_ingestion_smoke.py


  - AUTO_RESUME=1: Automatically resumes from existing vectors without prompting
  - FRESH_START=1: Automatically starts fresh, removing existing vectors without prompting
  - Useful for automation and CI/CD pipelines

  3. Better Error Handling

  - Clear error messages when services aren't running
  - Provides helpful commands to start services manually
  - Won't start ingestion if critical services are missing

  Usage Examples:

  # Regular interactive mode
  make ingest-all

  # Auto-resume without prompts
  AUTO_RESUME=1 make ingest-all

  # Fresh start without prompts
  FRESH_START=1 make ingest-all

  # Custom batch size with fresh start
  FRESH_START=1 BATCH_SIZE=500 make ingest-all

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


⏺ Database Locations Summary

  I've created a complete reference at docs/DATABASE_LOCATIONS.md. Here's the quick overview:

  1. PostgreSQL (Primary Database)

  - Location: /opt/homebrew/var/postgresql@17/
  - Status: ✅ Running (port 5432)
  - Connection: host=localhost dbname=lnsp user=lnsp password=lnsp
  - Tables: cpe_entry, cpe_vectors

  2. Neo4j (Graph Database)

  - Location: /opt/homebrew/var/neo4j/
  - Status: ✅ Running (ports 7474/7687)
  - Web UI: http://localhost:7474/

  3. FAISS Indexes (Vector Search)

  - Location: ./artifacts/*.index
  - Active Files:
    - fw10k_ivf_flat_ip.index (6.1 MB - 10k vectors)
    - ontology_13k_ivf_flat_ip.index (6.1 MB - 13k vectors)
    - ontology_13k_ivf_flat_ip_rebuilt.index (6.4 MB - rebuilt)

  4. NPZ Vector Files (Training Data)

  - Location: ./artifacts/*.npz
  - Key Files:
    - fw10k_vectors.npz (150 MB - 10k concepts)
    - ontology_13k.npz (38 MB - 13k ontology)
    - lvm/wordnet_training_sequences.npz (LVM training)

  5. SQLite Caches

  - ./artifacts/cpesh_index.db (20 KB - CPESH cache)
  - ./app/utils/mlflow/mlflow.db (MLflow tracking)

  6. Ontology Chains (JSONL)

  - Location: ./artifacts/ontology_chains/
  - Files: swo_chains.jsonl, go_chains.jsonl, dbpedia_chains.jsonl, wordnet_chains.jsonl
