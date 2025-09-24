#!/bin/bash
# scripts/build_faiss_100_768.sh
# Build FAISS index for 100 chunks (768D, IP metric) - for testing
# Produces: artifacts/fw100_ivf.index and updates artifacts/faiss_meta.json

set -euo pipefail

echo "=== Building FAISS index (100 vectors, 768D, IP) ==="

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
import numpy as np, faiss, json
from pathlib import Path
from datetime import datetime

npz_path = "artifacts/fw100_vectors.npz"
index_path = "artifacts/fw100_ivf.index"
meta_path = "artifacts/faiss_meta.json"

# Load vectors
print(f"Loading vectors from {npz_path}...")
npz = np.load(npz_path, allow_pickle=True)
emb = npz["emb"].astype("float32")   # (N,768) normalized
N, D = emb.shape

print(f"Vectors shape: {emb.shape}")
assert D == 768 and N > 0, f"Expected 768D vectors, got {D}D with {N} vectors"

# Build FAISS index
nlist = 4  # Small nlist for 100 vectors
print(f"Building IVF index with nlist={nlist}, metric=IP...")
quant = faiss.IndexFlatIP(D)
index = faiss.IndexIVFFlat(quant, D, nlist, faiss.METRIC_INNER_PRODUCT)

print("Training index...")
index.train(emb)

print("Adding vectors...")
index.add(emb)

# Set search parameters
index.nprobe = 4

print(f"Saving index to {index_path}...")
faiss.write_index(index, index_path)

# Update metadata (maintain existing format for compatibility)
meta = {
    "num_vectors": int(index.ntotal),
    "index_type": "IndexIVFFlat",
    "nlist": nlist,
    "dimension": int(D),  # 768 for P10
    "npz_path": npz_path,
    "index_path": index_path,
    "last_updated": datetime.now().isoformat()
}

print(f"Writing metadata to {meta_path}...")
Path(meta_path).write_text(json.dumps(meta, indent=2))

print("FAISS index metadata:")
print(json.dumps(meta, indent=2))
PY

echo "=== FAISS index build complete ==="
