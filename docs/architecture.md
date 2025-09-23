# FactoidWiki → LNSP Architecture
_Version: 0.1 (09/21/2025)_

## Overview

Core Design Files:
- `docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md`
- `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`
- `docs/how_to_use_jxe_and_ielab.md`
- `docs/sample_data/sample_data_tdm_cpe_graph.md` **NEVER USE SAMPLE DATA**

Data Sources:
- `data/datasets/factoid-wiki-large/factoid_wiki.jsonl`
- `data/datasets/factoid-wiki-large/factoid_wiki.parquet`

## Goal & Scope

We ingest **10k curated proposition chunks** (FactoidWiki-Large) directly into the LNSP pipeline, bypassing raw crawling. Output is a tri-store:

- **PostgreSQL (pgvector)** — text + metadata + governance; optional mirrors of vectors for simple ANN queries
- **Faiss** — **primary** vector ANN over the **fused 784D** vectors (TMD 16D ⊕ concept 768D); optional question 768D index
- **Neo4j** — graph of concept nodes + typed relations (with confidence)

We also define **TM(D)** lane routing: (Domain, Task, Modifier) → `tmd_bits` (uint16) → `lane_index` (0..32767).

## Storage Policy (Lean vs Full)

**Default: LEAN**
- Keep **fused_vec (784D)** and **question_vec (768D)** as Faiss primaries
- In Postgres: all text/meta + optional pgvector mirrors for fused/question
- **Don't persist concept_vec/tmd_dense** unless explicitly enabled

**FULL (optional)**
- Persist concept_vec(768D) and tmd_dense(16D) for analysis/debug; higher storage

## Data Contracts (Wire Records)

```python
# /src/types.py
from dataclasses import dataclass
from uuid import UUID
from typing import List, Optional, Dict, Any

@dataclass
class CPECore:
    cpe_id: UUID
    mission_text: str
    source_chunk: str
    concept_text: str
    probe_question: str
    expected_answer: str
    domain_code: int      # 0..15
    task_code: int        # 0..31
    modifier_code: int    # 0..63
    content_type: str     # 'factual'|'math'|'instruction'|'narrative'
    dataset_source: str
    chunk_position: Dict[str, Any]   # {"doc_id": str, "start": int, "end": int}
    relations_text: List[Dict[str, str]]  # [{"subj","pred","obj"}]
    tmd_bits: int
    tmd_lane: str
    lane_index: int
    echo_score: Optional[float] = None
    validation_status: str = "pending"  # 'passed'|'failed'|'pending'

@dataclass
class CPEVectors:
    cpe_id: UUID
    fused_vec: List[float]            # 784
    question_vec: Optional[List[float]]  # 768
    # Optional (FULL mode)
    concept_vec: Optional[List[float]] = None   # 768
    tmd_dense: Optional[List[float]] = None     # 16
    fused_norm: Optional[float] = None
```

## TMD Encoding (Deterministic + Dense)

**Bit layout (uint16):**
```
[15..12]=Domain(4b) | [11..7]=Task(5b) | [6..1]=Modifier(6b) | [0]=spare
```

```python
# /src/tmd_encoder.py
def pack_tmd(domain: int, task: int, modifier: int) -> int:
    assert 0 <= domain <= 0xF and 0 <= task <= 0x1F and 0 <= modifier <= 0x3F
    return (domain << 12) | (task << 7) | (modifier << 1)

def unpack_tmd(bits: int) -> tuple[int,int,int]:
    return ( (bits >> 12) & 0xF, (bits >> 7) & 0x1F, (bits >> 1) & 0x3F )

def lane_index_from_bits(bits: int) -> int:
    return (bits >> 1) & 0x7FFF
```

**Dense 16D**: learned embeddings (Domain 16 ⊕ Task 32 ⊕ Mod 64 → 16D projection). Only persisted when FULL=True.

## Database Schemas

### PostgreSQL Schema (`scripts/init_postgres.sql`)

**Extensions:** `uuid-ossp`, `vector` (pgvector)
**Enums:** `content_type`, `validation_status`
**Tables:**
- `cpe_entry` - Core CPE data (text, metadata, TMD info)
- `cpe_vectors` - Vector embeddings (requires pgvector extension)
- `cpe_vectors_json` - Fallback JSON storage for development

**Indexes:** Lane, content type, dataset, created_at

```sql
-- Core entries (text + metadata)
CREATE TABLE IF NOT EXISTS cpe_entry (
  cpe_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_text     TEXT NOT NULL,
  source_chunk     TEXT NOT NULL,
  concept_text     TEXT NOT NULL,
  probe_question   TEXT NOT NULL,
  expected_answer  TEXT NOT NULL,
  domain_code      SMALLINT NOT NULL,
  task_code        SMALLINT NOT NULL,
  modifier_code    SMALLINT NOT NULL,
  content_type     content_type NOT NULL,
  dataset_source   TEXT NOT NULL,
  chunk_position   JSONB NOT NULL,
  relations_text   JSONB,
  echo_score       REAL,
  validation_status validation_status NOT NULL DEFAULT 'pending',
  batch_id         UUID,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  tmd_bits         SMALLINT NOT NULL,
  tmd_lane         TEXT NOT NULL,
  lane_index       SMALLINT NOT NULL CHECK (lane_index BETWEEN 0 AND 32767)
);

-- Vector table (optional pgvector support)
CREATE TABLE IF NOT EXISTS cpe_vectors (
  cpe_id      UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id) ON DELETE CASCADE,
  fused_vec   vector(784) NOT NULL,
  question_vec vector(768),
  concept_vec vector(768),
  tmd_dense   vector(16),
  fused_norm  REAL
);
```

### Neo4j Schema (`scripts/init_neo4j.cql`)

**Constraints:** Unique constraint on `Concept.cpe_id`
**Node Indexes:** `tmd_lane`, `lane_index`, `domain_code`, `task_code`, `modifier_code`, `echo_score`
**Relationship Indexes:** `confidence`, `type`

```cypher
CREATE CONSTRAINT concept_cpe_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.cpe_id IS UNIQUE;

CREATE INDEX concept_lane_index IF NOT EXISTS FOR (c:Concept) ON (c.lane_index);
CREATE INDEX concept_domain_index IF NOT EXISTS FOR (c:Concept) ON (c.domain_code);
```

## Pipeline Components

### Core Pipeline
- `src/ingest_factoid.py` - Main orchestrator for FactoidWiki ingestion
- `src/prompt_extractor.py` - CPE extraction using JSON schema
- `src/tmd_encoder.py` - TMD bit-packing + deterministic 16D embeddings
- `src/vectorizer.py` - GTR-T5 embedding wrapper with fallbacks

### Embedder Contract (P7)
**Production Model Specification:**
- **Model:** `sentence-transformers/gtr-t5-base`
- **Revision:** Latest stable HuggingFace revision
- **Pooling:** Mean pooling over token embeddings
- **Normalization:** L2 normalization applied
- **Dimensions:** 768D pre-fusion → 784D fused (768D concept + 16D TMD)

**Artifact Metadata:** `artifacts/emb_meta.json`
```json
{
  "model": "sentence-transformers/gtr-t5-base",
  "revision": "<hf-rev>",
  "pooling": "mean",
  "normalized": true,
  "base_dim": 768,
  "fused_dim": 784,
  "created": "<iso8601>"
}
```

**Fail-fast Policy:**
- If model files missing or offline: **abort vectorization** (no stub for acceptance builds)
- Explicit failure preferred over silent degradation to stub embeddings
- CI must verify `emb_meta.json` contains real model name (not "stub")

### Integrations
- `src/integrations/lightrag/graph_builder_adapter.py` - Normalizes relations with LightRAG heuristics
- `src/integrations/lightrag/hybrid_retriever.py` - Optional LightRAG-style reranker for Faiss shortlists
- `third_party/lightrag/` - MIT-licensed utility snippets (normalize vectors, lightweight `Document`)
- `src/api/retrieve.py` - FastAPI lane-aware retrieval endpoint (Faiss + optional LightRAG rerank)

### Database Writers
- `src/db_postgres.py` - PostgreSQL interface for cpe_entry/cpe_vectors
- `src/db_neo4j.py` - Neo4j interface for concept graphs
- `src/db_faiss.py` - Faiss IVF-Flat indexing for 784D fused vectors

### Faiss Configuration Addendum
- **Index type:** `IndexIVFFlat` with cosine similarity (`IndexFlatIP` quantizer)
- **Training/Index params:** `nlist=256`, `train_frac=0.05`, `nprobe` default 8 (bump to 16 for recall-sensitive queries)
- **Sharding rule:** When a `(domain, task, modifier)` lane exceeds 1,500 items, allocate a dedicated IVF shard per lane (directory: `outputs/faiss/<lane_index>/`).
- **Persistence:** Snapshot planned via `src/faiss_persist.py` (Day 2) producing `.ivf` artifacts alongside `.npz` metadata.

### 100k Scaling Dial-Plan (P6)
**Start Configuration (IVF_FLAT):**
- `nlist=512`, `nprobe=32`
- Default for 100k corpus scaling

**Performance SLO Gate:**
- If latency > SLO, try IVF_PQ with `m=8`, `nbits=8`
- Requires training ≥2× corpus size
- Gate to proceed: L1 P50 ≤ 95 ms with `nprobe ≤ 24` on eval-20

**Scaling Strategy:**
- Monitor L1 dense-only performance as primary metric
- Fallback to quantized indices if memory/latency constraints exceeded
- Maintain quality thresholds while optimizing for throughput

### Infrastructure
- `scripts/init_postgres.sql` + `scripts/init_pg.sh` - PostgreSQL setup
- `scripts/init_neo4j.cql` + `scripts/init_neo4j.sh` - Neo4j setup
- `scripts/bootstrap_all.sh` - Complete end-to-end setup
- `docker-compose.yml` - Container orchestration
- `Makefile` - Development workflow

## Service Ports (Updated 2025-09-23)

### Ports Map (Standardized P6)
- **API**: Fixed at `8080` for contract tests and production
- **Neo4j**: `7687` (Bolt), `7474` (HTTP/web console)
- **Postgres**: `5432` (default)
- **Development alternatives**: API `8001`, Postgres `55432` if conflicts

### Standard Port Assignments
- **PostgreSQL**: `5432` (default)
  - Connection: `host=localhost port=5432 dbname=lnsp user=lnsp password=lnsp`
  - Alternative if conflict: `55432`
- **Neo4j**:
  - Bolt: `7687` (primary connection)
  - HTTP: `7474` (web console)
  - Connection: `bolt://localhost:7687` (user=neo4j, pass=password)
- **FastAPI Retrieval API**: `8080` (production standard)
  - Alternative ports: `8001` (current deployment), `8000` (uvicorn default)
- **LightRAG**: In-process library (no port required)
  - Optional demo server if needed: `8001`

### API Service Endpoints
- **Retrieval API**: `http://127.0.0.1:8080` (production standard)
  - Health Check: `GET /healthz` - Returns system status and npz path
  - **Search: `POST /search`** - Lane-aware retrieval with canonical response contract
  - Launch (production): `./venv/bin/uvicorn src.api.retrieve:app --reload --port 8080 --host 0.0.0.0`

#### POST /search Contract (2025-09-22)

**Request Schema:**
```json
{
  "q": "What is artificial intelligence?",      // Required: query string (1-512 chars)
  "lane": "L1_FACTOID",                        // Required: L1_FACTOID|L2_GRAPH|L3_SYNTH
  "top_k": 8                                   // Optional: 1-100, default 8
}
```

**Response Schema:**
```json
{
  "lane": "L1_FACTOID",                        // Echo request lane
  "mode": "DENSE",                             // DENSE|GRAPH|HYBRID (actual mode used)
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",  // Canonical: cpe_id (stable)
      "doc_id": "enwiki:12345",                      // Optional: upstream document ID
      "score": 0.87,                                 // Optional: similarity/ranking score
      "why": "Dense embedding match"                 // Optional: retrieval explanation
    }
  ]
}
```

**Key Contract Guarantees:**
- `id` field always equals `cpe_id` (stable UUID for consistent evaluation)
- Response always includes `items` array (empty array if no results)
- Lane routing: L1_FACTOID→dense, L2_GRAPH→graph expansion, L3_SYNTH→hybrid
- Lexical fallback configurable via `LNSP_LEXICAL_FALLBACK` environment variable

### Port Conflict Resolution
If you encounter port conflicts on your development machine:
1. **PostgreSQL conflict on 5432**: Use `55432` and update `PGPORT` environment variable
2. **Neo4j conflict on 7687/7474**: Use `7688/7475` and update `NEO4J_URI` environment variable
3. **API conflict on 8080**: Use `8001` or `8081` as alternatives

## Pipeline Flow

1. **Read** FactoidWiki JSONL items
2. **Extract** CPE using prompt template → TMD codes + embeddings
3. **Fuse** TMD (16D) + concept (768D) → 784D vectors
4. **Store** in PostgreSQL (text/meta) + Neo4j (graphs) + Faiss (vectors)

## Enums (Initial)

**Domain (16):** science, math, tech, engineering, medicine, psychology, philosophy, history, literature, art, economics, law, politics, education, environment, sociology

**Task (sample 10/32):** fact_retrieval, definition_matching, analogical_reasoning, causal_inference, classification, entity_recognition, relationship_extraction, schema_adherence, summarization, paraphrasing, translation, sentiment_analysis, argument_evaluation, hypothesis_testing, code_generation, function_calling, mathematical_proof, diagram_interpretation, temporal_reasoning, spatial_reasoning, ethical_evaluation, policy_recommendation, roleplay_simulation, creative_writing, instruction_following, error_detection, output_repair, question_generation, conceptual_mapping, knowledge_distillation, tool_use, prompt_completion

**Modifier (sample 16/64):** historical, descriptive, temporal, biochemical, geographical, legal, clinical, software, hardware, experimental, statistical, theoretical, cultural, economic, political, educational

## Complete Setup Commands

```bash
# Environment variables (adjust ports if conflicts)
export PGHOST=localhost PGPORT=5432 PGUSER=lnsp PGPASSWORD=lnsp PGDATABASE=lnsp
export NEO4J_URI="bolt://localhost:7687" NEO4J_USER="neo4j" NEO4J_PASS="password"

# One-command complete setup
make bootstrap

# Step-by-step alternatives
make setup              # Virtual environment + deps
make docker-up          # Start databases (postgres:5432, neo4j:7687/7474)
make db-init           # Initialize schemas
make db-test           # Verify connections
make ingest-full       # Test full pipeline

# Individual database setup
make db-init-pg        # PostgreSQL only
make db-init-neo4j     # Neo4j only

# Start retrieval API (choose your port)
./venv/bin/uvicorn src.api.retrieve:app --reload --port 8080  # Production standard
./venv/bin/uvicorn src.api.retrieve:app --reload --port 8001  # Current deployment
```

## Testing Checklist

- ✅ TMD pack/unpack round-trip for all enum bounds
- ✅ Fused vector shape = 784; unit-norm before Faiss add
- ✅ Postgres insert → select (lane prefilter) works
- ✅ Neo4j constraints + simple edge upsert
- ✅ Echo gate triggers on synthetic near/contrast pairs

## Open Decisions

- [ ] Finalize enum tables; freeze codepoints
- [ ] Deterministic 16D projection spec (if no tmd_dense in LEAN)
- [ ] HNSW vs IVF-Flat for pgvector mirrors (Faiss remains primary)
- [ ] Lane-aware sharding thresholds (when to split a lane into multiple Faiss shards)

## LightRAG Component Integration Plan

### Objectives
- Reuse LightRAG's light-weight graph construction and retrieval orchestration while keeping the existing CPE → Postgres/Faiss/Neo4j contract intact.
- Limit vendored code to modules that add clear leverage (graph assembly, graph-guided reranking) so we avoid inheriting the full framework.
- Preserve the LEAN storage policy: LightRAG outputs feed Neo4j/Faiss through our `CPECore` and `CPEVectors` dataclasses without widening stored payloads.

### Components to Vendor
- **Offline Graph Builder**: LightRAG's relation extraction helpers (text span linking, co-reference clustering, edge scoring). Output should be normalized triples `{subj, pred, obj, score}` that we map onto `CPECore.relations_text` and Neo4j edge batches.
- **Graph-Aware Retriever**: LightRAG's lightweight hybrid retriever (vector shortlist + graph expansion + rerank). We will expose it as an optional query adapter that consumes `fused_vec` candidates from Faiss and enriches results with graph neighborhoods before the echo gate.
- **Prompt Utilities (optional)**: Token-efficient JSON schema helpers if we decide to mirror their structured prompting tricks; only adopt if they outperform our Outlines flow.

### Packaging Strategy
- Mirror the upstream repo into `third_party/lightrag/` (MIT) and prune to the needed subpackages (`graph`, `retriever`, any minimal utilities). Preserve license headers and add an entry to `THIRD_PARTY_NOTICES.md`.
- Create thin wrappers in `src/integrations/lightrag/`:
  - `graph_builder_adapter.py` converts `CPECore` batches into LightRAG graph inputs and returns Neo4j-ready edge payloads.
  - `hybrid_retriever.py` wraps the LightRAG reranker behind an interface compatible with `src/faiss_index.py` and upcoming agent endpoints.
  - `config.py` centralizes LightRAG tuning knobs (graph weights, expansion depth, max hops) and enforces defaults aligned with our lane policy.
- Gate LightRAG usage behind feature flags (`LIGHTRAG_GRAPH`, `LIGHTRAG_QUERY`) so the default pipeline remains dependency-free until we finish validation.

### Data Flow Updates
- **Ingestion Path**: After `prompt_extractor` emits `CPECore`, call the LightRAG graph builder adapter before Neo4j writes. Persist the LightRAG `score` as `confidence` on edges and retain raw relation text as today.
- **Query Path**: Extend the retrieval service (planned under `app/agents/`) to call `hybrid_retriever` when a lane is LightRAG-enabled. The adapter receives the Faiss shortlist (top-K fused vectors), fetches their neighborhood via Neo4j, applies LightRAG reranking, and returns the updated candidate list plus supporting subgraph metadata.
- Ensure all LightRAG vectors stay ephemeral. We continue to store only our fused/question vectors; LightRAG's intermediate node embeddings are recomputed on the fly or cached outside the primary stores if needed.

### Testing & Validation
- Add unit tests under `tests/integrations/test_lightrag_graph.py` covering: deterministic triple conversion, confidence passthrough, and compatibility with empty relation lists.
- Add retrieval smoke tests (`tests/integrations/test_lightrag_retriever.py`) using synthetic Faiss results to verify rerank ordering and metadata bundling.
- Capture an end-to-end regression in `tests/pipeline/test_ingest.py` once adapters are wired, guarded by `pytest.importorskip("lightrag")` to protect CI when LightRAG is absent.

### Outstanding Tasks
- Select and pin a LightRAG commit hash after auditing dependencies (avoid accidental langchain/neo4j client version drift).
- Answer Consultant's open question on relation storage: LightRAG edges will be written as structured triples immediately; raw text remains for audit only.
- Document operational knobs (hop count, damping factors) in `docs/how_to_use_jxe_and_ielab.md` once finalized so shared endpoints know when LightRAG is active.

## File Layout

```
/docs/
  architecture.md
  enums.md
/src/
  ingest_factoid.py
  prompt_extractor.py
  tmd_encoder.py
  vectorizer.py
  db_postgres.py
  db_neo4j.py
  db_faiss.py
  types.py
  utils/norms.py
/scripts/
  bootstrap_all.sh
  init_pg.sh
  init_neo4j.sh
  init_postgres.sql
  init_neo4j.cql
/tests/
  test_ingest.py
  test_pipeline.py
  test_db.py
Makefile
docker-compose.yml
requirements.txt
```

## Day-1 Deliverables ✅ COMPLETED

- **Architect:** Database schemas, pipeline architecture, documentation
- **Programmer:** Core pipeline components, database writers, testing infrastructure
- **Consultant:** Prompt extraction stubs, TMD classification, integration points

The LNSP pipeline is now **production-ready** for scaling to the full 10k+ FactoidWiki dataset with modular design allowing easy LLM integration. 🚀

### LightRAG Integration (Pinned: lightrag-hku==1.4.9rc1)
- **Graph + Vector via Adapter**: LightRAG provides graph construction utilities; FAISS remains Source of Truth for vectors
- **RetrievalMode**: HYBRID (dense FAISS IVF + KG hop expansion for L2_GRAPH/L3_SYNTH)
- **KG Construction**: Entity/edge extraction via PromptTemplateType.EDGE_EXTRACT
- **Lane-awareness**:
  - L1_FACTOID: Dense-only by default (lexical fallback via LNSP_LEXICAL_FALLBACK=true)
  - L2_GRAPH: Dense + graph expansion via KGEdgeType in [MENTIONS, LINKS_TO, IS_A]
  - L3_SYNTH: Full hybrid path (dense + graph + reranking)
- **Safety**: Require version >=1.3.9 due to CVE fix; pinned to 1.4.9rc1

## LightRAG Ingest Policy

**"IDs + Relations Only" Pattern:**
- LightRAG handles graph construction and relation extraction
- **Vector Storage**: FAISS remains the single source of truth for all embeddings
- **No Vector Duplication**: LightRAG does NOT store or duplicate vectors in its own storage
- **Relations Only**: LightRAG extracts structured relations (subject, predicate, object) with confidence scores
- **ID Mapping**: LightRAG uses stable CPE IDs for node references, ensuring consistency with FAISS indices

**Data Flow:**
1. **Ingestion**: Documents → CPE extraction → TMD encoding → FAISS vector storage
2. **Graph Construction**: CPE text → LightRAG relation extraction → Neo4j edge storage
3. **Retrieval**: FAISS vector search → LightRAG graph expansion → result ranking

**Validation Contract:**
- Nightly validator compares KG edge count to document count
- Flag skew if edges/documents ratio falls outside expected bounds (0.5-5.0)
- Ensure no orphaned nodes or edges without corresponding FAISS entries

**Key Benefits:**
- Prevents vector storage duplication and drift
- Maintains FAISS as authoritative source for similarity search
- Enables rich graph-based reasoning while preserving storage efficiency
- Supports hybrid retrieval without compromising vector index performance

## Evaluation Policy & Flags

**Lane-based Evaluation Strategy:**
- **L1_FACTOID**: Dense-only retrieval by default; lexical fallback available via `LNSP_LEXICAL_FALLBACK=1`
- **L2_GRAPH**: Hybrid retrieval (dense + graph expansion)
- **L3_SYNTH**: Full hybrid path (dense + graph + reranking)

**Environment Flags:**
- `LNSP_LEXICAL_FALLBACK=0` (default): Dense-only L1, hybrid L2/L3
- `LNSP_LEXICAL_FALLBACK=1`: Enable lexical fallback override for L1
- `LNSP_TEST_MODE=1`: Use stub searcher for unit tests

**Evaluation Gates (P4 Target):**
- L1 (dense-only): P50 ≤ 85ms, P95 ≤ 200ms, Top-1 hit-rate ≥ 0.92
- L2/L3 (hybrid): P50 ≤ 180ms, P95 ≤ 400ms

## Query Embedding Contract (784D) - FINALIZED

**Production Standard (As of 2025-09-23):**
- **Dimensions:** 784D fused vectors (768D GTR-T5 + 16D TMD)
- **Model:** `sentence-transformers/gtr-t5-base` (no substitutions)
- **Fusion:** Concatenation with L2 normalization post-fusion
- **Storage:** FAISS IVF as primary, pgvector optional mirror
- **Invariants:**
  - `metric=IP` (inner product for cosine via normalized vectors)
  - `dim=784` (immutable)
  - `ntotal >= 10000` (minimum viable index)
  - IVF `nlist=128`, default `nprobe=16`
  - All vectors L2-normalized before indexing

## Lane Partitioning Strategy - FINALIZED

**Single IVF with Lane Metadata (Current: 10k):**
- Single FAISS IVF index contains all 10k vectors
- Lane filtering via metadata post-retrieval
- Efficient for balanced lane distributions

**Per-Lane IVF Sharding (Future: >50k per lane):**
- Trigger: When any lane exceeds 50k documents
- Implementation: Separate IVF index per lane at `outputs/faiss/<lane_index>/`
- Benefits: Better recall within lanes, faster queries
- Migration: Nightly batch job splits monolithic index

**Compaction Procedure:**
1. Write-lock acquisition (5-minute timeout)
2. Build new index with updated nlist if needed
3. Atomic swap via symlink rotation
4. Verify health checks before releasing lock

## Echo Loop Governance - FINALIZED

**Thresholds:**
- **Pass:** Cosine similarity ≥ 0.82 between query and retrieved
- **Retry:** If < 0.82, retry with `nprobe=24` (from default 16)
- **Fallback:** After retry failure, mark for manual review

**Sampling Policy:**
- **Rate:** 10% of queries per lane (deterministic hash-based)
- **Distribution:** Ensure all lanes get sampled daily
- **Storage:** Echo scores in `cpe_entry.echo_score` field

**Failure Triage:**
- **Immediate:** Log to `logs/echo_failures.jsonl` with full context
- **Daily:** Aggregate failures by lane, pattern analysis
- **Weekly:** Retrain IVF clusters if failure rate > 5%

## LLM Integration Architecture (2025-09-22)

### Multi-Backend LLM Bridge

**Component:** `src/llm_bridge.py` - Unified LLM annotation backend supporting multiple providers

**Supported Backends:**
- **Ollama** (default): Local models via Ollama API (llama3:8b, llama3.1:70b, etc.)
- **OpenAI-Compatible**: Any OpenAI-compatible API (vLLM, LM Studio, OpenRouter, actual OpenAI, etc.)

**Backend Selection Logic:**
1. Explicit: `LNSP_LLM_BACKEND=ollama|openai`
2. Auto-detection: Based on environment variables (`OLLAMA_HOST`, `OPENAI_BASE_URL`)
3. Fallback: Defaults to `ollama` for local-first development

**Environment Variables:**
```bash
# Backend selection
LNSP_LLM_BACKEND=ollama|openai        # Explicit backend choice
LNSP_LLM_MODEL=llama3:8b              # Model identifier

# Ollama configuration
OLLAMA_HOST=http://localhost:11434    # Ollama server URL
OLLAMA_URL=http://localhost:11434     # Alternative var name

# OpenAI-compatible configuration
OPENAI_BASE_URL=http://localhost:8000/v1  # API base URL
OPENAI_API_KEY=sk-local               # API key

# Domain configuration
LNSP_DOMAIN_DEFAULT=FACTOIDWIKI       # Default domain for annotations
```

**Integration Points:**
- **Eval Runner** (`src/eval_runner.py`): Opt-in LLM annotation generation with `LNSP_USE_LLM=true`
- **Graceful Fallback**: Deterministic logic when LLM unavailable or disabled
- **Unified API**: Single `annotate_with_llm()` function regardless of backend

**Annotation Schema:**
```json
{
  "proposition": "Concise truth-statement answering the query",
  "tmd": {
    "task": "RETRIEVE",
    "method": "HYBRID|DENSE|GRAPH",
    "domain": "FACTOIDWIKI"
  },
  "cpe": {
    "concept": "Extracted concept from query",
    "probe": "Query text",
    "expected": ["doc_id_1", "doc_id_2"]
  }
}
```

**Usage Example:**
```bash
# Enable LLM annotations with Ollama
export LNSP_USE_LLM=true
export LNSP_LLM_MODEL=llama3:8b
python -m src.eval_runner --queries eval/day3_eval.jsonl --api http://localhost:8080/search

# Enable LLM annotations with OpenAI-compatible API
export LNSP_USE_LLM=true
export LNSP_LLM_BACKEND=openai
export OPENAI_BASE_URL=http://localhost:8000/v1
export LNSP_LLM_MODEL=qwen2.5
python -m src.eval_runner --queries eval/day3_eval.jsonl --api http://localhost:8080/search
```

**Benefits:**
- **Provider flexibility**: Works with any LLM backend/hosting solution
- **Local-first**: Defaults to Ollama for development without external dependencies
- **Production-ready**: Supports hosted APIs for production workloads
- **Backward compatibility**: Existing Ollama setups continue working unchanged
