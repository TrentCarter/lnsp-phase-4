#!/usr/bin/env bash
set -euo pipefail
pip install "lightrag-hku==1.4.8.2"
python - <<'PY'
from lightrag import LightRAG, QueryParam
print("[vendor] LightRAG import OK:", LightRAG, QueryParam)
PY
echo "[vendor] done"
