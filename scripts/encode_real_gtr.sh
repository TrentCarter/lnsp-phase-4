#!/bin/bash
# scripts/encode_real_gtr.sh
# Encode real GTR embeddings (no stubs) for 10k FactoidWiki chunks
# Produces: artifacts/fw10k_vectors.npz with real 768D values

set -euo pipefail

echo "=== Encoding real GTR embeddings ==="

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
out = root/"artifacts/fw10k_vectors.npz"

if not chunks.exists():
    raise FileNotFoundError(f"Chunks file not found: {chunks}")

model = SentenceTransformer("sentence-transformers/gtr-t5-base")
ids = []
texts = []

print(f"Processing {chunks}...")
with chunks.open() as f:
    for i, line in enumerate(f):
        if i % 1000 == 0:
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

if zero_count > 0:
    print("WARNING: Found zero vectors - this should not happen with real encoding!")
PY

echo "=== Real GTR encoding complete ==="
