#!/bin/bash
# scripts/test_encode_gtr.sh
# Test encoding on first 10 chunks only

set -euo pipefail

echo "=== Testing GTR encoding on first 10 chunks ==="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

python3 - <<'PY'
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

root = Path(".")
chunks = root/"artifacts/fw10k_chunks.jsonl"
out = root/"artifacts/test_vectors.npz"

if not chunks.exists():
    raise FileNotFoundError(f"Chunks file not found: {chunks}")

model = SentenceTransformer("sentence-transformers/gtr-t5-base")
ids, vecs = [], []

print(f"Processing first 10 chunks from {chunks}...")
with chunks.open() as f:
    for i, line in enumerate(f):
        if i >= 10:  # Only process first 10
            break
        print(f"Processing chunk {i}...")
        rec = json.loads(line)
        did = rec.get("doc_id") or rec.get("id")
        txt = rec.get("concept") or rec.get("text") or rec.get("contents") or ""
        if not did or not txt.strip():
            print(f"  Skipping chunk {i} - no valid text")
            continue
        print(f"  Text preview: {txt[:50]}...")
        v = model.encode([txt], normalize_embeddings=True)[0]  # (768,), L2-normalized
        ids.append(did)  # Keep as string
        vecs.append(v.astype("float32"))
        print(f"  Encoded vector shape: {v.shape}")

if not vecs:
    raise ValueError("No vectors were encoded!")

emb = np.stack(vecs, axis=0)
ids = np.array(ids, dtype=object)  # Keep as strings

# Save the embeddings
np.savez(out, emb=emb, ids=ids)

# Report stats
zero_count = int((emb==0).all(axis=1).sum())
print(f"Wrote: {out}")
print(f"Shape: {emb.shape} (vectors), {ids.shape} (ids)")
print(f"Zero vectors: {zero_count}")
print(f"Non-zero vectors: {len(ids) - zero_count}")
print(f"Sample IDs: {ids[:3]}")
PY

echo "=== Test encoding complete ==="
