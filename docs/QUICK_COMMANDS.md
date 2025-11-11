# Quick Commands Reference

Consolidated command reference for common LNSP operations.

**Note**: For detailed usage and examples, see the linked documentation for each section.

---

## Table of Contents

- [Service Management](#service-management)
- [n8n Integration](#n8n-integration)
- [Vec2Text Testing](#vec2text-testing)
- [LLM (Ollama) Setup](#llm-ollama-setup)
- [Database Operations](#database-operations)
- [FAISS Index Operations](#faiss-index-operations)
- [Data Ingestion](#data-ingestion)
- [Testing & Validation](#testing--validation)
- [P0 Stack (Gateway/PAS/Aider)](#p0-stack-gatewaypasaider)
- [HMI (Web UI)](#hmi-web-ui)

---

## Service Management

### Start All Services
```bash
./scripts/start_all_fastapi_services.sh
```

### Stop All Services
```bash
./scripts/stop_all_fastapi_services.sh
```

### Check Service Health
```bash
curl -s http://localhost:8900/health  # Episode Chunker
curl -s http://localhost:8001/health  # Semantic Chunker
curl -s http://localhost:8767/health  # GTR-T5 Embeddings
curl -s http://localhost:8004/health  # Ingest API
curl -s http://localhost:7001/health  # Orchestrator Encoder (PRODUCTION)
curl -s http://localhost:7002/health  # Orchestrator Decoder (PRODUCTION)
```

### Restart Services Before Ingestion (Best Practice)
```bash
./scripts/stop_all_fastapi_services.sh && sleep 5 && \
./scripts/start_all_fastapi_services.sh && sleep 10
```

**See**: `docs/FASTAPI_CHUNKING_GUIDE.md` for detailed service documentation

---

## n8n Integration

### Setup n8n MCP Server
```bash
claude mcp add n8n-local -- npx -y n8n-mcp --n8n-url=http://localhost:5678
```

### Check MCP Connection
```bash
claude mcp list
```

### Start n8n Server
```bash
N8N_SECURE_COOKIE=false n8n start
```

### Import Workflows
```bash
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json
```

### Test Webhook Integration
```bash
python3 n8n_workflows/test_webhook_simple.py
python3 n8n_workflows/test_batch_via_webhook.py
```

**See**: `docs/n8n_integration_guide.md` for complete setup and examples

---

## Vec2Text Testing

### Test Encoding/Decoding (CORRECT Method)
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 \
VEC2TEXT_DEVICE=cpu \
TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "What is AI?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1
```

### Production Services (Ports 7001/7002)
```bash
# Start encoder (port 7001)
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app --host 127.0.0.1 --port 7001 &

# Start decoder (port 7002)
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app --host 127.0.0.1 --port 7002 &

# Test encoding
curl -X POST http://localhost:7001/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test text"]}'

# Test decoding
curl -X POST http://localhost:7002/decode \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1, 0.2, ...]], "subscriber": "ielab", "steps": 5, "original_texts": ["Test text"]}'
```

### Compare Encoders (Verify Compatibility)
```bash
./.venv/bin/python tools/compare_encoders.py
# Expected: CORRECT encoder = 0.89 cosine, WRONG encoder = 0.09 cosine
```

**See**: `docs/how_to_use_jxe_and_ielab.md` for detailed usage and examples

---

## LLM (Ollama) Setup

### Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Pull Model (Llama 3.1:8b)
```bash
ollama pull llama3.1:8b
```

### Start Ollama Server
```bash
ollama serve &
```

### Verify Ollama is Running
```bash
curl http://localhost:11434/api/tags
```

### Test LLM Chat
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### Set Environment Variables
```bash
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

**See**: `docs/howto/how_to_access_local_AI.md` for complete LLM setup

---

## Database Operations

### PostgreSQL

```bash
# Connect to database
psql lnsp

# Count concepts
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"

# Check vector dimensions
psql lnsp -c "SELECT jsonb_array_length(concept_vec) as dims FROM cpe_vectors LIMIT 1;"

# Verify CPE IDs
psql lnsp -c "SELECT id, concept_text FROM cpe_entry LIMIT 5;"

# Check for duplicates
psql lnsp -c "SELECT id, COUNT(*) FROM cpe_entry GROUP BY id HAVING COUNT(*) > 1;"
```

### Neo4j

```bash
# Check if Neo4j is running
cypher-shell -u neo4j -p password "RETURN 1"

# Count concepts
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN COUNT(c)"

# Show sample concepts
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN c.id, c.name LIMIT 5"

# Count relationships
cypher-shell -u neo4j -p password "MATCH ()-[r]->() RETURN COUNT(r)"
```

**See**: `docs/DATABASE_LOCATIONS.md` for complete database reference

---

## FAISS Index Operations

### Build FAISS Index (IVF Flat IP)
```bash
LNSP_EMBEDDER_PATH=./models/gtr-t5-base \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
make build-faiss ARGS="--type ivf_flat --metric ip --nlist 512 --nprobe 16"
```

### Verify FAISS Index
```bash
python -c "
import faiss
index = faiss.read_index('artifacts/fw10k_ivf_flat_ip.index')
print(f'Index size: {index.ntotal} vectors')
print(f'Index trained: {index.is_trained}')
"
```

### Test FAISS Search
```bash
python -c "
import faiss
import numpy as np

# Load index
index = faiss.read_index('artifacts/fw10k_ivf_flat_ip.index')

# Create query vector
query = np.random.randn(1, 768).astype('float32')

# Search
scores, indices = index.search(query, k=10)
print(f'Top 10 results: {indices[0]}')
print(f'Scores: {scores[0]}')
"
```

**See**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md` for optimal configuration

---

## Data Ingestion

### Wikipedia Pipeline (Full)
```bash
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 0 \
  --limit 1000
```

### Ontology Ingestion (No FactoidWiki)
```bash
# Set LLM config
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Ingest ontologies
./scripts/ingest_ontologies.sh

# Verify synchronization
./scripts/verify_data_sync.sh
```

### Generate 6-Degree Shortcuts (Optional)
```bash
./scripts/generate_6deg_shortcuts.sh
```

**See**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md` for complete ingestion guide

---

## Testing & Validation

### Run All Tests
```bash
LNSP_TEST_MODE=1 ./.venv/bin/pytest tests -m "not heavy" -v
```

### Run Specific Test
```bash
LNSP_TEST_MODE=1 ./.venv/bin/pytest tests/test_ingest.py::test_ingest_smoke -v
```

### Run vecRAG Evaluation
```bash
FAISS_NPZ_PATH=artifacts/fw9k_vectors.npz \
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,bm25,lex \
  --out RAG/results/evaluation.jsonl
```

### Verify Data Direction (Pre-Training Check)
```bash
./.venv/bin/python tools/diagnose_data_direction.py \
  artifacts/lvm/training_data.npz \
  --n-samples 5000
```

**See**: `docs/vecrag_test_suite.md` for complete testing guide

---

## P0 Stack (Gateway/PAS/Aider)

### Install Aider (One-Time)
```bash
pipx install aider-chat
export ANTHROPIC_API_KEY=your_key_here  # or OPENAI_API_KEY
```

### Start All P0 Services
```bash
bash scripts/run_stack.sh

# Expected output:
# [Aider-LCO] ✓ Started on http://127.0.0.1:6130
# [PAS Root]  ✓ Started on http://127.0.0.1:6100
# [Gateway]   ✓ Started on http://127.0.0.1:6120
```

### Test P0 Stack Health
```bash
./bin/verdict health
```

### Submit Prime Directive
```bash
./bin/verdict send \
  --title "Add docstrings" \
  --goal "Add docstrings to all functions in services/gateway/app.py" \
  --entry-file "services/gateway/app.py"
```

### Check Run Status
```bash
./bin/verdict status --run-id <uuid>
```

### View Run Artifacts
```bash
cat artifacts/runs/<uuid>/aider_stdout.txt
```

**See**: `docs/P0_END_TO_END_INTEGRATION.md` for complete architecture

---

## HMI (Web UI)

### Start HMI Server
```bash
./.venv/bin/uvicorn services.webui.hmi_app:app --host 127.0.0.1 --port 6101
```

### Check if HMI is Running
```bash
lsof -ti:6101 > /dev/null && echo "HMI running" || echo "HMI not running"
```

### Access HMI Pages
```
- Homepage: http://localhost:6101/
- Sequencer: http://localhost:6101/sequencer
- PLMS Dashboard: http://localhost:6101/plms (if enabled)
```

### View Communication Logs
```bash
# View all logs for today
./tools/parse_comms_log.py

# Filter by run ID
./tools/parse_comms_log.py --run-id abc123-def456

# Watch logs in real-time
./tools/parse_comms_log.py --tail

# Export to JSON
./tools/parse_comms_log.py --format json > logs.json
```

**See**: `docs/COMMS_LOGGING_GUIDE.md` for communication logging details

---

## System Status Check

### Quick Component Status
```bash
echo "=== LNSP Component Status ==="
echo "Ollama LLM:  " $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "PostgreSQL:  " $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Neo4j:       " $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "GTR-T5:      " $(curl -s http://localhost:8767/health >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Encoder:     " $(curl -s http://localhost:7001/health >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Decoder:     " $(curl -s http://localhost:7002/health >/dev/null 2>&1 && echo "✓" || echo "✗")
```

### Verify All Data Stores
```bash
# PostgreSQL
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"

# Neo4j
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN COUNT(c)"

# FAISS NPZ
python -c "import numpy as np; data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz'); print(f'Vectors: {len(data[\"vectors\"])}')"
```

**See**: `docs/DATABASE_LOCATIONS.md` for complete verification guide

---

## Common Environment Variables

```bash
# LLM Configuration
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# TMD Mode (hybrid = fallback + embedding)
export LNSP_TMD_MODE=hybrid

# Test Mode (uses test database)
export LNSP_TEST_MODE=1

# Vec2Text Configuration
export VEC2TEXT_FORCE_PROJECT_VENV=1
export VEC2TEXT_DEVICE=cpu
export TOKENIZERS_PARALLELISM=false

# macOS OpenMP Fix
export KMP_DUPLICATE_LIB_OK=TRUE

# FAISS Configuration
export LNSP_FAISS_INDEX=artifacts/fw10k_ivf_flat_ip.index
export FAISS_NPZ_PATH=artifacts/fw9k_vectors.npz
export FAISS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

---

## Troubleshooting Commands

### Kill Stuck Services
```bash
# Kill specific port
lsof -ti:8767 | xargs kill -9

# Kill all Python services
pkill -f "uvicorn"
```

### Clear Cache
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clear FAISS index cache (if needed)
rm -rf artifacts/*.index
```

### Rebuild FAISS Index
```bash
# Delete old index
rm artifacts/fw10k_ivf_flat_ip.index

# Rebuild
make build-faiss ARGS="--type ivf_flat --metric ip --nlist 512 --nprobe 16"
```

### Reset Database (DESTRUCTIVE)
```bash
# ⚠️ WARNING: This deletes ALL data!

# PostgreSQL
psql lnsp -c "TRUNCATE TABLE cpe_entry, cpe_vectors CASCADE;"

# Neo4j
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"
```

---

## See Also

- **Service Management**: `docs/FASTAPI_CHUNKING_GUIDE.md`
- **n8n Integration**: `docs/n8n_integration_guide.md`
- **Vec2Text Usage**: `docs/how_to_use_jxe_and_ielab.md`
- **LLM Setup**: `docs/howto/how_to_access_local_AI.md`
- **Database Reference**: `docs/DATABASE_LOCATIONS.md`
- **Data Flow**: `docs/DATA_FLOW_DIAGRAM.md`
- **vecRAG Configuration**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md`
- **P0 Architecture**: `docs/P0_END_TO_END_INTEGRATION.md`
- **Communication Logging**: `docs/COMMS_LOGGING_GUIDE.md`
- **Data Ingestion**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`
