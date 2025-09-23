# Simple test API based on P8 specifications
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import os, json
from src.db_faiss import FaissDB

app = FastAPI()

FAISS_INDEX = os.getenv("FAISS_INDEX", "artifacts/fw10k_ivf.index")
FAISS_META  = os.getenv("FAISS_META_NPZ", "artifacts/fw10k_vectors.npz")
FAISS_NPROBE = int(os.getenv("FAISS_NPROBE", "16"))

db = None
emb_meta = {}

class SearchReq(BaseModel):
    q: str
    lane: str = "L1_FACTOID"
    top_k: int = 5

def encode_gtr768(text: str) -> np.ndarray:
    """Mock GTR-768 encoder - returns random 768D vector"""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(768).astype(np.float32)

def encode_tmd16(text: str) -> np.ndarray:
    """Mock TMD-16 encoder - returns random 16D vector"""
    np.random.seed((hash(text) + 1) % 2**32)
    return np.random.rand(16).astype(np.float32)

def _embed_784(text: str) -> np.ndarray:
    # 768D sentence embedding
    v768 = encode_gtr768(text)  # np.float32 (768,)
    # 16D TMD vector (Task-Modifier-Data)
    tmd16 = encode_tmd16(text)  # np.float32 (16,)
    v784 = np.concatenate([v768, tmd16], axis=0).astype(np.float32)
    # L2 normalize to match contract
    n = np.linalg.norm(v784) + 1e-12
    return (v784 / n).reshape(1, -1)

@app.on_event("startup")
def _startup():
    global db, emb_meta
    db = FaissDB(FAISS_INDEX, FAISS_META, FAISS_NPROBE).load()
    # read emb_meta.json for admin+health parity
    try:
        with open("artifacts/emb_meta.json", "r") as f:
            emb_meta = json.load(f)
    except Exception:
        emb_meta = {}

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "faiss": {
            "dim": db.dim if db else -1,
            "nprobe": FAISS_NPROBE,
            "vectors": int(db.index.ntotal) if (db and db.index) else 0,
        },
        "emb_meta": emb_meta.get("model"),
    }

@app.get("/admin/faiss")
def admin_faiss():
    import faiss
    idx = db.index
    # Try to read metric/nlist if present
    info = {"dim": db.dim, "vectors": int(idx.ntotal)}
    try:
        info["metric"] = "IP" if idx.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
    except Exception:
        pass
    try:
        # IVF params
        info["nlist"] = getattr(idx, "nlist", None)
        info["nprobe"] = getattr(idx, "nprobe", None)
    except Exception:
        pass
    return info

@app.post("/search")
def search(req: SearchReq):
    try:
        qv = _embed_784(req.q)  # (1,784)
        D, I = db.search(qv, req.top_k)
        # Resolve to app-level ids (map positions -> npz ids)
        raw_ids = []
        for idx in I[0].tolist():
            if idx == -1:
                continue
            # IVF returns positions; IDMap2 returns original ids directly
            raw_ids.append(int(idx))
        return {"top_k": req.top_k, "hits": [{"id": rid, "score": float(s)} for rid, s in zip(raw_ids, D[0].tolist())]}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Don't hide; make probe-visible
        raise HTTPException(status_code=503, detail=f"search_failed: {repr(e)}")