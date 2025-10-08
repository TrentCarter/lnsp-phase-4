#!/usr/bin/env python3
"""
Generate 10 demo cases (5 single-concept, 5 dual-concept) using vecRAG and LLM smoothing.
Outputs a Markdown report under RAG/results/demo_10cases_<timestamp>.md

Env overrides (optional):
- DEMO_NPZ   : path to NPZ (default: artifacts/ontology_13k.npz)
- DEMO_INDEX : path to FAISS index (default: artifacts/ontology_13k_ivf_flat_ip_rebuilt.index)
- LLM_URL    : chat endpoint (default: http://localhost:11434/api/chat)
- LLM_MODEL  : model name (default: llama3.1:8b)
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import faiss
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vectorizer import EmbeddingBackend
from src.utils.norms import l2_normalize

NPZ = os.getenv("DEMO_NPZ", "artifacts/ontology_13k.npz")
IDX = os.getenv("DEMO_INDEX", "artifacts/ontology_13k_ivf_flat_ip_rebuilt.index")
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
NO_LLM = os.getenv("DEMO_NO_LLM", "0") == "1"
try:
    MAX_CASES = int(os.getenv("DEMO_MAX_CASES", "10"))
except Exception:
    MAX_CASES = 10
try:
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45"))
except Exception:
    LLM_TIMEOUT = 45.0

if not Path(NPZ).exists():
    print(f"NPZ not found: {NPZ}")
    sys.exit(2)
if not Path(IDX).exists():
    print(f"FAISS index not found: {IDX}")
    sys.exit(2)

npz = np.load(NPZ, allow_pickle=True)
ids = npz['cpe_ids']
texts = npz['concept_texts']
index = faiss.read_index(str(IDX))
d = int(getattr(index, 'd', 768))
N = len(texts)

# Synchronous, CPU-only embedder (defaults enforced via env outside)
embedder = EmbeddingBackend()

def adapt(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    if d == 784:
        tmd = np.zeros((16,), dtype=np.float32)
        fused = np.concatenate([tmd, vec]).astype(np.float32)
        return l2_normalize(fused.reshape(1, -1)).astype(np.float32)
    elif d == 768:
        return l2_normalize(vec.reshape(1, -1)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported index dim {d}")

def search(q: str, k: int = 5):
    dv = embedder.encode([q])[0]
    qv = adapt(dv)
    D, I = index.search(qv, k)
    D = D[0]; I = I[0]
    items = []
    for s, i in zip(D, I):
        if 0 <= i < len(ids):
            items.append({
                'rank': len(items) + 1,
                'score': float(s),
                'id': str(ids[i]),
                'text': str(texts[i])
            })
    return items

def llm_smooth(query: str, concepts):
    concepts_text = "\n".join([f"- ({c['id']}): {c['text']}" for c in concepts])
    prompt = (
        "You are answering a scientific query using retrieved ontology concepts.\n\n"
        f"Query: {query}\n\nRetrieved Concepts:\n{concepts_text}\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. EVERY factual claim MUST cite a concept ID in (id:text) format.\n"
        "2. Use ONLY the IDs and texts provided above.\n"
        "3. Keep response concise: 2-3 sentences maximum.\n"
        "4. Do NOT invent facts not in the retrieved concepts.\n\n"
        "Response (with mandatory citations):"
    )
    if NO_LLM:
        return "<skipped: DEMO_NO_LLM=1>"
    try:
        r = requests.post(
            LLM_URL,
            json={'model': LLM_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False},
            timeout=LLM_TIMEOUT,
        )
        if r.status_code == 200:
            j = r.json()
            msg = j.get('message') or {}
            return (msg.get('content') or '').strip()
        return f"LLM Error: {r.status_code}"
    except Exception as e:
        return f"LLM Error: {e}"

# Select deterministically distributed examples
single_idx = [0, max(1, N // 5), max(2, N // 5), max(3, N // 5), max(4, N // 5)]
single_qs = [str(texts[i]) for i in single_idx]
pair_idx = [(i, (i + 7) % N) for i in single_idx]
dual_qs = [f"{texts[i]} and {texts[j]}" for i, j in pair_idx]

cases = [("single", q) for q in single_qs] + [("dual", q) for q in dual_qs]
if MAX_CASES < len(cases):
    cases = cases[:MAX_CASES]

out_dir = Path('RAG/results')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"demo_10cases_{time.strftime('%Y%m%d_%H%M%S')}.md"

with out_path.open('w') as f:
    f.write("# Demo Cases (10)\n\n")
    f.write(f"Index: {IDX} (dim={d}, ntotal={index.ntotal})\n\n")
    for n, (kind, q) in enumerate(cases, start=1):
        items = search(q, k=5)
        unsmoothed = items[0]['text'] if items else '<no result>'
        smoothed = llm_smooth(q, items[:5])
        f.write(f"## Case {n} — {kind}\n\n")
        f.write(f"- **Input**: {q}\n")
        f.write(f"- **Unsmoothed (vecRAG top1)**: {unsmoothed}\n")
        f.write(f"- **Smoothed (LLM)**:\n\n{smoothed}\n\n")

print(str(out_path))
