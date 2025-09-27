[Architect]
S7 “Cleanup & Test” Pack (commit-ready)
1) Quick repo hygiene (copy/paste)
Makefile: add a one-shot smoketest
.PHONY: smoketest
smoketest:
	@bash scripts/s7_smoketest.sh
New script: scripts/s7_smoketest.sh
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
  jq -e '.nlist and .max_safe_nlist and .requested_nlist and .count' artifacts/index_meta.json >/dev/null 2>&1 \
    && ok "artifacts/index_meta.json has required keys" \
    || bad "index_meta.json missing required keys"
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
echo "python=$(python -V 2>&1) faiss=$(python - <<'P';import faiss;print(faiss.__version__);P) numpy=$(python - <<'P';import numpy as n;print(n.__version__);P) date=$(date -u +%FT%TZ)" >> artifacts/runtime.lock
ok "Appended runtime to artifacts/runtime.lock"

printf "\nSummary: PASS=%d FAIL=%d\n" "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
Save & chmod +x scripts/s7_smoketest.sh. Run with make smoketest (API up), or PORT=8093 make smoketest.
2) One minimal pytest to guard the wiring
New file: tests/test_pipeline_smoke.py
import os, json, pytest

def require_file(path):
    assert os.path.exists(path), f"missing {path}"

def test_index_meta_keys():
    require_file("artifacts/index_meta.json")
    with open("artifacts/index_meta.json") as f:
        meta = json.load(f)
    for k in ("nlist","max_safe_nlist","requested_nlist","count"):
        assert k in meta
    assert meta["nlist"] <= meta["max_safe_nlist"]

@pytest.mark.skipif("S7_API_URL" not in os.environ, reason="set S7_API_URL to run API checks")
def test_gating_metrics_endpoint():
    import urllib.request
    url = os.environ["S7_API_URL"].rstrip("/") + "/metrics/gating"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert "total" in data and "used_cpesh" in data
Run: pytest -q (and optionally S7_API_URL=http://127.0.0.1:8092 pytest -q with API running).
3) Single-page “today wrap” you can commit
New doc: docs/S7_day_wrap_2025-09-25.md
# S7 Wrap — 2025-09-25 (FactoidWiki → LNSP)

## What’s working (live)
- **Dynamic nlist** with auto-downshift and telemetry (`artifacts/index_meta.json`).
- **CPESH two-stage gating** (quality/cos gates + lane overrides) active in `/search`.
- **Decision logging** → `artifacts/gating_decisions.jsonl` (non-empty).
- **Metrics endpoint** → `/metrics/gating` (used_cpesh / total).
- **Offline embedder** (local model; no network), Makefile targets (`api`, `gating-snapshot`, `smoketest`).

## 10k retrieval status
- Index type: IVF_FLAT, metric: IP (cosine).  
- Effective `nlist`: auto-calculated via 40× rule.  
- Recommended `nprobe`: 16 (CPESH path uses 8).  
- Hit@1 uplift with CPESH (S5): +4.2% (slice: CPESH-assisted vs fallback).

## CPESH datastore (PERMANENT)
- Active tier: JSONL append (training + inference).  
- Warm/Cold tiers: planned (GZ + Parquet) with SQLite/DuckDB index seam.  
- **Never delete CPESH**; organize by tier, not TTL.

## Pipeline map (P1–P17)
**Solid:** P1–P4 (ingest/chunk/label/mission), P6–P8 (TMD encode + fuse), P11 (FAISS index), P16 (lane-aware search path), API + metrics.  
**Operational but evolving:** P5 (teacher LLM orchestration & guards), P9 (graph triples), P13 (echo loop scoring gates), P14 (batching), P12 (graph DB adapters).  
**R&D tracks:** P15 (LNSP training loops, vec2text decoder), P17 (MoE inference wiring).

## Today’s artifacts
- `artifacts/index_meta.json` — nlist, max_safe_nlist, requested_nlist, count.  
- `artifacts/gating_decisions.jsonl` — decision trace (ts, lane, used_cpesh, chosen_nprobe, latency_ms).  
- `artifacts/metrics_gating.json` — snapshot via `make gating-snapshot`.  
- `artifacts/runtime.lock` — environment versions (append-only).  
- `eval/day_s5_report.md` — cold/warm sweep, CPESH usage, final nprobe & thresholds.

## Open risks / watch-items
- **Scale stepping:** at ≥20k vectors, allow `nlist` to rise (auto policy already in place).  
- **Tiering:** implement warm (GZ) + cold (Parquet) rotation in S8–S9; add ID→tier index.  
- **Prompt drift:** keep v2.0 frozen; record deltas when adjusting CPESH extraction.

## Next-day (S8) short targets
- Implement **active→warm rotation** (JSONL→GZ) with manifest + SQLite index.  
- Add **/metrics/slo** auto-fill from the eval harness (persist last run).  
- Optional: **IVF_PQ tryout** (M=8, nbits=8) vs IVF_FLAT at ≥25k vectors.
4) “Big picture” alignment (one-liner you can drop into PRs)
The system now matches the pipeline intent: CPESH is permanent, tiered training data; vector retrieval is lane-routed; index sizing is dynamic; CPESH is a confidence-gated accelerator, not a mandatory step. This keeps today’s 10k local run consistent with the 10M→10B scale path described in your PRD.
5) Runbook to close S7 (3 commands)
# 1) Ensure API is up (pick a free port)
make api PORT=8092

# 2) One-shot smoketest (validates meta, endpoints, decisions file, runtime lock)
make smoketest PORT=8092

# 3) (Optional) pytest with live API checks
S7_API_URL=http://127.0.0.1:8092 pytest -q