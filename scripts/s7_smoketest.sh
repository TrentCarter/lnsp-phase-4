#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8092}"
API="http://127.0.0.1:${PORT}"
PASS=0; FAIL=0

step() { printf "\n==> %s\n" "$*"; }
ok()   { echo "[ok] $*"; PASS=$((PASS+1)); }
bad()  { echo "[fail] $*" >&2; FAIL=$((FAIL+1)); }

step "Check index meta"
if [[ -f artifacts/index_meta.json ]]; then
  # Check if we have nested structure with index files or flat structure
  if jq -e 'keys | length > 0' artifacts/index_meta.json >/dev/null 2>&1; then
    # Try nested structure first (current format)
    FIRST_KEY=$(jq -r 'keys[0]' artifacts/index_meta.json)
    if jq -e ".\"$FIRST_KEY\".nlist and .\"$FIRST_KEY\".max_safe_nlist and .\"$FIRST_KEY\".requested_nlist and .\"$FIRST_KEY\".count" artifacts/index_meta.json >/dev/null 2>&1; then
      ok "artifacts/index_meta.json has required keys (nested format)"
    elif jq -e '.nlist and .max_safe_nlist and .requested_nlist and .count' artifacts/index_meta.json >/dev/null 2>&1; then
      ok "artifacts/index_meta.json has required keys (flat format)"
    else
      bad "index_meta.json missing required keys"
    fi
  else
    bad "index_meta.json is empty"
  fi
else
  bad "artifacts/index_meta.json not found (run: make build-faiss)"
fi

step "Check gating decisions file (will appear after queries)"
if [[ -f artifacts/gating_decisions.jsonl ]]; then
  LINES=$(wc -l < artifacts/gating_decisions.jsonl || echo 0)
  [[ "$LINES" -gt 0 ]] && ok "gating_decisions.jsonl has ${LINES} lines" || bad "gating_decisions.jsonl empty"
else
  echo "[warn] artifacts/gating_decisions.jsonl not found yet"
fi

step "Ping API health/faiss"
if curl -sf "${API}/health/faiss" >/dev/null; then
  curl -sf "${API}/health/faiss" | tee artifacts/health_faiss.json >/dev/null
  ok "/health/faiss reachable (saved to artifacts/health_faiss.json)"
else
  bad "API not reachable on ${API} (start with: make api PORT=${PORT})"
fi

step "Check gating metrics"
if curl -sf "${API}/metrics/gating" >/dev/null; then
  curl -sf "${API}/metrics/gating" | tee artifacts/metrics_gating.json >/dev/null
  ok "/metrics/gating reachable (saved to artifacts/metrics_gating.json)"
else
  echo "[warn] /metrics/gating not available yet (no queries run?)"
fi

step "Optional: run a single /search smoke if embedder is local"
if [[ -n "${LNSP_EMBEDDER_PATH:-}" ]]; then
  # minimal body; adjust fields to match SearchRequest schema in your API
  BODY='{"q":"photosynthesis water split","lane":"L1_FACTOID","top_k":5,"compact":true}'
  if curl -sf -H 'Content-Type: application/json' -d "$BODY" "${API}/search" >/dev/null; then
    ok "POST /search succeeded (compact)"
  else
    bad "POST /search failed (check model path & index)"
  fi
else
  echo "[info] LNSP_EMBEDDER_PATH not set; skipping live /search smoke"
fi

step "Lock runtime"
echo "python=$(python -V 2>&1) faiss=$(python - <<'P'
import faiss;print(faiss.__version__)
P
) numpy=$(python - <<'P'
import numpy as n;print(n.__version__)
P
) date=$(date -u +%FT%TZ)" >> artifacts/runtime.lock
ok "Appended runtime to artifacts/runtime.lock"

printf "\nSummary: PASS=%d FAIL=%d\n" "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1