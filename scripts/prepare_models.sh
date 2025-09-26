#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/gtr-t5-base")
m.save("./models/gtr-t5-base")
print("Saved to ./models/gtr-t5-base")
PY
echo "Export: LNSP_EMBEDDER_PATH=./models/gtr-t5-base HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"