# LNSP Phase 4: Tokenless Mamba LVM + vecRAG + GraphRAG

A vector-native latent variable model (LVM) with ontology-based retrieval and knowledge graph integration.

---

## 🚨 **READ FIRST: Critical Documentation**

Before doing ANYTHING, read these files in order:

1. **[LNSP_LONG_TERM_MEMORY.md](LNSP_LONG_TERM_MEMORY.md)** ← START HERE!
   - Cardinal rules that must NEVER be violated
   - NO FactoidWiki policy
   - Data synchronization requirements
   - 6-degrees shortcuts theory

2. **[CRITICAL_GraphRAG_Data_Synchronization.md](docs/CRITICAL_GraphRAG_Data_Synchronization.md)**
   - PostgreSQL + Neo4j + FAISS must stay synchronized
   - Atomic ingestion procedures

3. **[GraphRAG_Root_Cause_Analysis.md](docs/GraphRAG_Root_Cause_Analysis.md)**
   - Oct 2-3 incident: What went wrong and why

---

## Quick Start

### Prerequisites
```bash
# 1. Validate NO FactoidWiki data exists
./scripts/validate_no_factoidwiki.sh

# 2. Verify data synchronization
./scripts/verify_data_sync.sh

# If either fails, re-ingest ontologies (see below)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Optional: pip install fastapi uvicorn for API server
```

## Running the API

To start the FastAPI retrieval service:

```bash
uvicorn src.api.retrieve:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with endpoints:
- `GET /healthz` - Health check
- `GET /search?q=<query>&k=<num_results>` - Search with natural language query

## Runtime Matrix

| Python Version | Support Status | Notes |
|----------------|----------------|-------|
| 3.11.x | ✅ | Recommended and CI-tested |
| 3.13.x | 🚫 | FAISS/BLAS compatibility issues |

## Environment Variables

- `FAISS_NPZ_PATH`: Path to FAISS vectors file (default: artifacts/fw1k_vectors.npz)
