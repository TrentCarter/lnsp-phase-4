#!/bin/bash
# scripts/encode_real_gtr_100.sh
# Encode real GTR embeddings for first 100 chunks only (for testing)

set -euo pipefail

echo "=== Encoding real GTR embeddings (100 chunks for testing) ==="

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
out = root/"artifacts/fw100_vectors.npz"

if not chunks.exists():
    raise FileNotFoundError(f"Chunks file not found: {chunks}")

model = SentenceTransformer("sentence-transformers/gtr-t5-base")
ids = []
texts = []

print(f"Processing first 100 chunks from {chunks}...")
with chunks.open() as f:
    for i, line in enumerate(f):
        if i >= 100:  # Only process first 100
            break
        if i % 10 == 0:
            print(f"Processed {i} chunks...")
        rec = json.loads(line)
        did = rec.get("doc_id") or rec.get("id")
        txt = rec.get("concept") or rec.get("text") or rec.get("contents") or ""
        if not did or not txt.strip():
            continue
        ids.append(did)  # Keep as string
        texts.append(txt)

# Batch encode all texts
print(f"Batch encoding {len(texts)} texts...")
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
vecs = [v.astype("float32") for v in embeddings]

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
PY

echo "=== 100-chunk GTR encoding complete ==="
