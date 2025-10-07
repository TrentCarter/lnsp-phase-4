# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL: READ LONG-TERM MEMORY FIRST

**Before doing ANYTHING, read [LNSP_LONG_TERM_MEMORY.md](LNSP_LONG_TERM_MEMORY.md)**

That file contains the cardinal rules that must NEVER be violated:
1. Data Synchronization is Sacred (PostgreSQL + Neo4j + FAISS must stay synchronized)
2. NO FactoidWiki Data - Ontologies ONLY
3. Complete Data Pipeline: CPESH + TMD + Graph (all together, atomically)
4. LVM Architecture: Tokenless Vector-Native
5. Six Degrees of Separation + Shortcuts (0.5-3% shortcut edges)

---

## 🚨 CRITICAL RULES FOR DAILY OPERATIONS

1. **ALWAYS use REAL data** - Never use stub/placeholder data. Always use actual datasets from `data/` directory.
2. **NEVER use FactoidWiki** - Only ontologies (SWO, GO, ConceptNet, DBpedia). FactoidWiki is NOT ontological!
3. **ALWAYS verify dataset_source labels** - Ontology data must use `ontology-{source}` format (not `factoid-wiki-large`)
4. **ALWAYS call faiss_db.save()** - FAISS vectors must be persisted after ingestion (see Oct 4 fix)
5. **ALWAYS use REAL LLM** - Never fall back to stub extraction. Use Ollama with Llama 3.1:
   - Install: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull model: `ollama pull llama3.1:8b`
   - Start: `ollama serve` (keep running)
   - Verify: `curl http://localhost:11434/api/tags`
   - See `docs/howto/how_to_access_local_AI.md` for full setup
6. **🔴 CRITICAL: ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION** (Added Oct 7, 2025)
   - **Every concept MUST have a unique ID** (UUID/CPE ID) that links:
     - PostgreSQL `cpe_entry` table (concept text, CPESH negatives, metadata)
     - Neo4j `Concept` nodes (graph relationships)
     - FAISS NPZ file (768D/784D vectors at index position)
     - Training data chains (ordered sequences for LVM)
   - **NPZ files MUST include**:
     - `concept_texts`: Array of concept strings (for lookup)
     - `cpe_ids`: Array of UUIDs (for database correlation)
     - `vectors`: 768D or 784D arrays (for training/inference)
   - **Why this matters**:
     - vecRAG search: Query → FAISS index → CPE ID → concept text
     - LVM training: Chain concepts → match text → get vector index → training sequences
     - Inference: LVM output vector → FAISS nearest neighbor → CPE ID → final text
   - **Without IDs**: Cannot correlate data across stores → training/inference impossible!
3. **ALWAYS use REAL embeddings** - Use GTR-T5 for 768D vectors:
   - Model: `sentence-transformers/gtr-t5-base`
   - Install: `pip install sentence-transformers`
   - Usage: Via `EmbeddingBackend()` class in `src/vectorizer.py`
   - Generates true 768-dimensional dense vectors
   - See `models/` directory for cached model files
4. **Never run training without explicit permission.**
5. **Vec2Text usage**: follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage.
6. **Devices**: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
7. **Steps**: Use `--steps 1` for vec2text by default; increase only when asked.
8. **CPESH data**: Always generate complete CPESH (Concept-Probe-Expected-SoftNegatives-HardNegatives) using LLM, never empty arrays.

<!-- Audio notifications section removed to keep repo guidance focused and neutral. -->

## 📍 CURRENT STATUS (2025-10-04)
- **CPESH Integration**: Full CPESH (Concept-Probe-Expected-SoftNegatives-HardNegatives) implemented with real LLM generation
- **Vec2Text**: Use `app/vect_text_vect/vec_text_vect_isolated.py` with `--vec2text-backend isolated`.
- **n8n MCP**: Configured and tested. Use `claude mcp list` to verify connection.
- **Local LLM**: Ollama + Llama 3.1:8b running for real CPESH generation
- **Recent Fixes (Oct 4, 2025)**:
  - ✅ Fixed `dataset_source` labeling bug (ontology data now labeled correctly)
  - ✅ Fixed FAISS save() call (NPZ files now created automatically)
  - ✅ Updated validation script (checks content, not just labels)
  - See `docs/FIXES_Oct4_2025_FactoidWiki_Labeling.md` for details

## 🤖 REAL COMPONENT SETUP

### Local LLM Setup (Ollama + Llama 3.1)
```bash
# Quick setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve &

# Test LLM is working
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'

# Environment variables for LNSP integration
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

### Real Embeddings Setup (GTR-T5 768D)
```bash
# Install sentence-transformers
pip install sentence-transformers

# Verify GTR-T5 model download
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
print('GTR-T5 loaded successfully:', model.get_sentence_embedding_dimension(), 'dimensions')
"

# Test embedding generation
python -c "
from src.vectorizer import EmbeddingBackend
eb = EmbeddingBackend()
vec = eb.encode(['Hello world'])
print('Generated vector shape:', vec[0].shape)
"
```

### Ontology Data Ingestion (No FactoidWiki)
```bash
# CRITICAL: Do NOT use FactoidWiki. Use ontology sources only (SWO/GO/DBpedia/etc.)

# 1) Ensure local LLM is configured
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# 2) Ingest ontologies atomically (PostgreSQL + Neo4j + FAISS)
./scripts/ingest_ontologies.sh

# 3) Verify synchronization
./scripts/verify_data_sync.sh

# 4) (Optional) Add 6-degree shortcuts to Neo4j
./scripts/generate_6deg_shortcuts.sh
```


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

## 🔍 VERIFICATION COMMANDS

### Check All Real Components Are Working
```bash
# 1. Verify Ollama LLM is running
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | grep llama3.1

# 2. Verify GTR-T5 embeddings
python -c "from src.vectorizer import EmbeddingBackend; eb = EmbeddingBackend(); print('✓ GTR-T5 embeddings working')"

# 3. Verify CPESH data has real negatives (not empty arrays)
psql lnsp -c "SELECT count(*) as items_with_negatives FROM cpe_entry WHERE jsonb_array_length(soft_negatives) > 0 AND jsonb_array_length(hard_negatives) > 0;"

# 4. Verify real vector dimensions
psql lnsp -c "SELECT jsonb_array_length(concept_vec) as vector_dims FROM cpe_vectors LIMIT 1;"

# 5. Test complete CPESH extraction
python -c "
from src.prompt_extractor import extract_cpe_from_text
import os
os.environ['LNSP_LLM_ENDPOINT'] = 'http://localhost:11434'
os.environ['LNSP_LLM_MODEL'] = 'llama3.1:8b'
result = extract_cpe_from_text('The Eiffel Tower was built in 1889.')
print('✓ LLM extraction working')
print('Soft negatives:', len(result.get('soft_negatives', [])))
print('Hard negatives:', len(result.get('hard_negatives', [])))
"
```

### Component Status Check
```bash
# Complete system check
echo "=== LNSP Real Component Status ==="
echo "1. Ollama LLM:" $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running")
echo "2. PostgreSQL:" $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓ Connected" || echo "✗ Not connected")
echo "3. Neo4j:" $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓ Connected" || echo "✗ Not connected")
echo "4. GTR-T5:" $(python -c "from src.vectorizer import EmbeddingBackend; EmbeddingBackend()" >/dev/null 2>&1 && echo "✓ Available" || echo "✗ Not available")
```

## 💡 DEVELOPMENT GUIDELINES
- **ALWAYS verify real components before starting work** - Run status check above
- **NO STUB FUNCTIONS** - If LLM/embeddings fail, fix the service, don't fall back to stubs
- Python 3.11+ with venv (`python3 -m venv venv && source venv/bin/activate`)
- Install with `python -m pip install -r requirements.txt`
- Lint with `ruff check app tests scripts`
- Run smoke tests: `pytest tests/lnsp_vec2text_cli_main_test.py -k smoke`
- Keep changes aligned with vec2text isolated backend unless otherwise specified
- **Links to full documentation:**
  - LLM setup: `docs/howto/how_to_access_local_AI.md`
  - CPESH generation: `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`
  - Known-good procedures: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`