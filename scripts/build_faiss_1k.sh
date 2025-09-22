#!/usr/bin/env bash
# Build a Faiss IVF-Flat index from fused vectors NPZ
# Usage: scripts/build_faiss_1k.sh [path/to/fw1k_vectors.npz] [nlist]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

VEC_NPZ=${1:-artifacts/fw1k_vectors.npz}
NLIST=${2:-256}
OUT=${OUT:-artifacts/faiss_fw1k.ivf}
mkdir -p "$(dirname "$OUT")"

if [ ! -f "$VEC_NPZ" ]; then
  echo "❌ Missing vectors NPZ: $VEC_NPZ"
  exit 1
fi

python - <<PY
import numpy as np, faiss, os
npz = np.load("$VEC_NPZ")
# Use the stored vectors (fused)
X = npz["vectors"].astype('float32')
faiss.normalize_L2(X)
D = X.shape[1]
quant = faiss.IndexFlatIP(D)
index = faiss.IndexIVFFlat(quant, D, int("$NLIST"), faiss.METRIC_INNER_PRODUCT)
# train on 5% or up to 5000
N = X.shape[0]
train_k = max(1, min(5000, int(N*0.05)))
rs = np.random.default_rng(0).choice(N, size=train_k, replace=False)
index.train(X[rs])
index.add(X)
faiss.write_index(index, "$OUT")
print(f"✅ Saved IVF index to {os.path.abspath('$OUT')}  (N={N}, D={D}, nlist={int('$NLIST')})")
PY
