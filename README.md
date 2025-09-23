# LNSP Phase 4

A vector-based retrieval system with LightRAG integration.

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
