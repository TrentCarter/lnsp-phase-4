[Architect] Paste these in as-is.
Programmer: patches & new files
1) src/faiss_index.py — dynamic nlist + safer metadata
*** a/src/faiss_index.py
--- b/src/faiss_index.py
@@
+import math
+from typing import Optional
+
+def calculate_nlist(n_vectors: int, requested_nlist: Optional[int] = None) -> int:
+    """
+    Dynamic nlist per S5 policy:
+      - Enforce 40× training rule via max_safe_nlist = n_vectors // 40
+      - If requested_nlist is provided, downshift to safe bound if needed
+      - Otherwise choose by scale bands; >100k uses sqrt(N)
+    """
+    max_safe_nlist = max(1, n_vectors // 40)
+    if requested_nlist is not None:
+        if requested_nlist > max_safe_nlist:
+            print(f"[faiss_index] Warning: reducing nlist {requested_nlist} -> {max_safe_nlist} to satisfy 40× rule")
+        return min(requested_nlist, max_safe_nlist)
+    if n_vectors < 8000:
+        return max_safe_nlist
+    elif n_vectors < 20000:
+        return min(200, max_safe_nlist)
+    elif n_vectors < 40000:
+        return min(512, max_safe_nlist)
+    elif n_vectors <= 100000:
+        # hold at 1024 only if safe
+        return min(1024, max_safe_nlist)
+    else:
+        return min(int(math.sqrt(n_vectors)), max_safe_nlist)
@@
-    nlist = args.nlist
+    nlist = calculate_nlist(len(vectors), args.nlist)
@@
-    meta = {
-        "type": index_type, "metric": metric, "nlist": nlist, "nprobe": nprobe,
-        "count": len(vectors), "build_seconds": build_secs,
-    }
+    meta = {
+        "type": index_type,
+        "metric": metric,
+        "nlist": nlist,
+        "requested_nlist": args.nlist,
+        "max_safe_nlist": max(1, len(vectors)//40),
+        "nprobe": nprobe,
+        "count": len(vectors),
+        "build_seconds": build_secs,
+    }
     json.dump(meta, open("artifacts/index_meta.json","w"), indent=2)
     print(f"[faiss_index] Wrote index meta → artifacts/index_meta.json")
Nothing else in this file should need to change. If you currently enforce the 40× rule earlier, keep it—but do not exit; we now downshift automatically.
2) src/utils/gating.py — CPESH two-stage gating (new)
# src/utils/gating.py
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CPESHGateConfig:
    q_min: float = 0.82
    cos_min: float = 0.55
    nprobe_cpesh: int = 8
    nprobe_fallback: int = 16
    lane_overrides: Dict[str, Dict[str, float]] = None  # e.g. {"L1_FACTOID": {"q_min": 0.85}}

def apply_lane(cfg: CPESHGateConfig, lane: Optional[str]) -> CPESHGateConfig:
    if lane and cfg.lane_overrides and lane in cfg.lane_overrides:
        o = cfg.lane_overrides[lane]
        return CPESHGateConfig(
            q_min=o.get("q_min", cfg.q_min),
            cos_min=o.get("cos_min", cfg.cos_min),
            nprobe_cpesh=int(o.get("nprobe_cpesh", cfg.nprobe_cpesh)),
            nprobe_fallback=int(o.get("nprobe_fallback", cfg.nprobe_fallback)),
            lane_overrides=cfg.lane_overrides
        )
    return cfg

def should_use_cpesh(cpesh_entry: Optional[dict], cfg: CPESHGateConfig) -> bool:
    if not cpesh_entry:
        return False
    if cpesh_entry.get("insufficient_evidence"):
        return False
    return (cpesh_entry.get("quality", 0.0) >= cfg.q_min) and (cpesh_entry.get("cosine", 0.0) >= cfg.cos_min)
3) src/api/retrieve.py — wire in gating + decision logging + metrics
*** a/src/api/retrieve.py
--- b/src/api/retrieve.py
@@
+import json, time, os
+from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh
@@
-@app.post("/search")
+@app.post("/search")
 async def search(request: SearchRequest):
     """
     Lane-aware search with CPESH assist.
     """
-    # existing: build query_vec, access ctx, etc.
+    # existing: build query_vec, access ctx, etc.
+    gate_cfg = CPESHGateConfig(
+        q_min=float(os.getenv("LNSP_CPESH_Q_MIN", "0.82")),
+        cos_min=float(os.getenv("LNSP_CPESH_COS_MIN", "0.55")),
+        nprobe_cpesh=int(os.getenv("LNSP_NPROBE_CPESH", "8")),
+        nprobe_fallback=int(os.getenv("LNSP_NPROBE_DEFAULT", "16")),
+        lane_overrides={"L1_FACTOID": {"q_min": 0.85}}
+    )
+    gate = apply_lane(gate_cfg, getattr(request, "lane", None))
+    cpesh_entry = None
+    if hasattr(ctx, "cpesh_cache"):
+        # whatever keying you use (e.g., doc_id or normalized query hash)
+        key = getattr(request, "doc_id_hash", None) or getattr(request, "query_hash", None)
+        cpesh_entry = ctx.cpesh_cache.get(key) if key else None
+
+    t0 = time.time()
+    used_cpesh = False
+    chosen_nprobe = gate.nprobe_fallback
+    qv = cv = None
+    if should_use_cpesh(cpesh_entry, gate):
+        used_cpesh = True
+        chosen_nprobe = gate.nprobe_cpesh
+        qv = cpesh_entry.get("quality")
+        cv = cpesh_entry.get("cosine")
+        results = ctx.index.search(query_vec, nprobe=chosen_nprobe, boost_vectors=cpesh_entry.get("expected_vec"))
+    else:
+        results = ctx.index.search(query_vec, nprobe=chosen_nprobe)
+    latency_ms = (time.time() - t0) * 1000.0
+
+    # Decision log → artifacts/gating_decisions.jsonl
+    try:
+        os.makedirs("artifacts", exist_ok=True)
+        with open("artifacts/gating_decisions.jsonl", "a") as f:
+            f.write(json.dumps({
+                "request_id": getattr(request, "request_id", None),
+                "lane": getattr(request, "lane", None),
+                "used_cpesh": used_cpesh,
+                "quality": qv, "cos": cv,
+                "chosen_nprobe": chosen_nprobe,
+                "latency_ms": round(latency_ms, 3),
+                "ts": time.time()
+            }) + "\n")
+    except Exception:
+        pass
@@
     return formatted_response
@@
+@app.get("/metrics/gating")
+def gating_metrics():
+    """
+    Very light metrics for CPESH gating usage (for dashboards & SLO snapshots).
+    """
+    counts = {"total": 0, "used_cpesh": 0}
+    try:
+        with open("artifacts/gating_decisions.jsonl") as f:
+            for line in f:
+                counts["total"] += 1
+                try:
+                    if json.loads(line).get("used_cpesh"):
+                        counts["used_cpesh"] += 1
+                except Exception:
+                    continue
+    except FileNotFoundError:
+        pass
+    return counts
Keep your existing /health/faiss, /cache/stats, /metrics/slo endpoints. This just adds CPESH gating.
Env knobs (defaults match policy):
export LNSP_CPESH_Q_MIN=0.82
export LNSP_CPESH_COS_MIN=0.55
export LNSP_NPROBE_CPESH=8
export LNSP_NPROBE_DEFAULT=16
4) src/datastore/cpesh_store.py — thin, forward-compatible wrapper (new)
# src/datastore/cpesh_store.py
import json, os
from typing import Iterator, Dict, Any

class CPESHDataStore:
    """
    Permanent training datastore facade (Active tier for S5).
    Future tiers (Warm .jsonl.gz, Cold Parquet, SQLite index) can slot behind this.
    """
    def __init__(self, active_path: str = "artifacts/cpesh_active.jsonl"):
        self.active_path = active_path
        os.makedirs(os.path.dirname(active_path), exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        with open(self.active_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def iter_active(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.active_path):
            return iter(())
        with open(self.active_path) as f:
            for line in f:
                yield json.loads(line)
Replace direct JSONL writers with CPESHDataStore.append(...) over time. No runtime behavior change yet—this is a seam for S6 tiering.
Tests
5) tests/test_nlist_dynamic.py (new)
# tests/test_nlist_dynamic.py
import math
from src.faiss_index import calculate_nlist

def test_small_auto_maxsafe():
    assert calculate_nlist(5000, None) == 5000//40  # 125

def test_10k_default_band():
    # 10k → min(200, max_safe)
    assert calculate_nlist(10000, None) == 200

def test_25k_band():
    # 25k < 40k → min(512, max_safe) where max_safe=625
    assert calculate_nlist(25000, None) == 512

def test_120k_sqrt():
    n = 120000
    expected = min(int(math.sqrt(n)), n//40)
    assert calculate_nlist(n, None) == expected

def test_requested_downshift():
    # request 512 on 10k should downshift to max_safe=250
    assert calculate_nlist(10000, 512) == 250

def test_requested_safe():
    # request 200 on 10k is safe and equals policy's pick
    assert calculate_nlist(10000, 200) == 200
6) tests/test_gating.py (new)
# tests/test_gating.py
from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh

def test_gate_pass_basic():
    cfg = CPESHGateConfig()
    entry = {"quality": 0. Nine, "cosine": 0.56}  # typo caught: use 0.90
Oops—fixing before you paste:
from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh

def test_gate_pass_basic():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.90, "cosine": 0.56}
    assert should_use_cpesh(entry, cfg) is True

def test_gate_fail_quality():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.80, "cosine": 0.70}
    assert should_use_cpesh(entry, cfg) is False

def test_gate_fail_cos():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.90, "cosine": 0.50}
    assert should_use_cpesh(entry, cfg) is False

def test_gate_fail_insufficient():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.95, "cosine": 0.80, "insufficient_evidence": True}
    assert should_use_cpesh(entry, cfg) is False

def test_lane_override_factoid():
    base = CPESHGateConfig(q_min=0.82)
    lane_cfg = apply_lane(base, "L1_FACTOID")
    assert lane_cfg.q_min >= 0.85
7) tests/test_index_meta.py (augment)
*** a/tests/test_index_meta.py
--- b/tests/test_index_meta.py
@@
 def test_index_meta_keys(tmp_path):
     # ... your existing setup that writes artifacts/index_meta.json
     with open("artifacts/index_meta.json") as f:
         meta = json.load(f)
-    for k in ["type","metric","nlist","nprobe","count","build_seconds"]:
+    for k in ["type","metric","nlist","nprobe","count","build_seconds","requested_nlist","max_safe_nlist"]:
         assert k in meta
+    assert meta["nlist"] <= meta["max_safe_nlist"]
+    if meta["requested_nlist"] is not None:
+        assert meta["nlist"] <= max(meta["requested_nlist"], 0)
Makefile (optional niceties)
Add a tiny helper to snapshot gating usage in S5 reports:
.PHONY: gating-snapshot
gating-snapshot:
	@mkdir -p eval/snapshots
	@curl -sf http://127.0.0.1:8092/metrics/gating -o eval/snapshots/gating_$$(/bin/date -u +"%Y%m%dT%H%M%SZ").json
	@echo "[gating-snapshot] wrote eval/snapshots/…"
Consultant: runbook for S5 report
Build index at your 10k scale (request 512 to demonstrate auto-downshift):
make build-faiss ARGS="--type ivf_flat --metric ip --nlist 512 --nprobe 16"
cat artifacts/index_meta.json   # confirm nlist auto-reduced, requested_nlist=512
Evaluate (live API) at nprobe ∈ {8,16,24}; capture cold & warm:
# start API
make api

# cold sweeps
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 8  --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 24 --metric ip --queries eval/100q.jsonl

# warm (repeat best two)
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 8  --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl

make gating-snapshot
make slo-snapshot
Gating A/B (threshold grid):
# A/B: vary gates; record Hit@1/3 (overall + CPESH-assisted only), P50/P95
LNSP_CPESH_Q_MIN=0.78 LNSP_CPESH_COS_MIN=0.50 PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
LNSP_CPESH_Q_MIN=0.82 LNSP_CPESH_COS_MIN=0.55 PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
LNSP_CPESH_Q_MIN=0.86 LNSP_CPESH_COS_MIN=0.60 PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
Write eval/day_s5_report.md:
Table: requested_nlist vs effective nlist vs Hit@1/3 vs P50/P95.
Table: gating grid (q_min, cos_min) × metrics, plus % used_cpesh from /metrics/gating.
Final recommendation: nlist policy confirmation (auto), nprobe, and gating thresholds per lane.

[Consulatnt] 
  1. Finish the Programmer integration: enable gating logic in src/api/retrieve.py, emit artifacts/gating_decisions.jsonl, and expose /metrics/
  gating.
  2. Rebuild the 10k index (make build-faiss …) so artifacts/index_meta.json exists, bring up the API, then rerun the Consultant sweep grid and
  update eval/day_s5_report.md.

---

🔍 Consultant Execution Log — 2025-09-25T16:18Z
- Task 1 remains open: `src/api/retrieve.py` still imports gating helpers but does not apply them within the search flow; `/metrics/gating` endpoint is absent and no `artifacts/gating_decisions.jsonl` is produced.
- Task 2 blocked: without the gating integration and an updated index build (`artifacts/index_meta.json` still missing), the Consultant sweep grid cannot run; no new data for `eval/day_s5_report.md`.
- Awaiting Programmer updates before rerunning FAISS build, API startup, and evaluation commands listed above.

🔍 Consultant Execution Log — 2025-09-25T16:28Z
- Programmer follow-up landed: gating logic now executes inside `RetrievalContext.search()`, and `/metrics/gating` proxy exists. Rebuilt the index (auto-downshift confirmed) but subsequent steps still blocked by environment limits.
- Sandbox denies binding a local FastAPI port (`uvicorn` exits with “operation not permitted” / “address already in use”), so live HTTP sweeps and `gating-snapshot` cannot run.
- Direct context invocation stalls because the embedder needs HuggingFace assets and outbound network is disabled; without embeddings we can’t trigger gating decisions or generate CPESH usage metrics.
- Standing by for network/port approvals; once available we’ll rerun the evaluation and update `eval/day_s5_report.md` plus the Consultant deliverables.
  



[Extra Credit] 
[Architect]
1) Wire the gating path (final) + decision metrics
src/api/retrieve.py (patch)
*** a/src/api/retrieve.py
--- b/src/api/retrieve.py
@@
+import os, json, time
+from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh
@@
 async def search(request: SearchRequest):
     """
     Lane-aware search with CPESH assist.
     """
-    # (your existing: build query_vec, access ctx, etc.)
+    # (your existing: build query_vec, access ctx, etc.)
+    gate_cfg = CPESHGateConfig(
+        q_min=float(os.getenv("LNSP_CPESH_Q_MIN", "0.82")),
+        cos_min=float(os.getenv("LNSP_CPESH_COS_MIN", "0.55")),
+        nprobe_cpesh=int(os.getenv("LNSP_NPROBE_CPESH", "8")),
+        nprobe_fallback=int(os.getenv("LNSP_NPROBE_DEFAULT", "16")),
+        lane_overrides={"L1_FACTOID": {"q_min": 0.85}}
+    )
+    gate = apply_lane(gate_cfg, getattr(request, "lane", None))
+
+    # Pull CPESH entry if present (use your existing keying)
+    cpesh_entry = None
+    if hasattr(ctx, "cpesh_cache"):
+        key = getattr(request, "doc_id_hash", None) or getattr(request, "query_hash", None)
+        if key: cpesh_entry = ctx.cpesh_cache.get(key)
+
+    used_cpesh = False
+    chosen_nprobe = gate.nprobe_fallback
+    qv = cv = None
+    t0 = time.time()
+    if should_use_cpesh(cpesh_entry, gate):
+        used_cpesh = True
+        chosen_nprobe = gate.nprobe_cpesh
+        qv = cpesh_entry.get("quality")
+        cv = cpesh_entry.get("cosine")
+        # If your index.search() doesn’t support boost_vectors, just drop that kwarg.
+        try:
+            results = ctx.index.search(query_vec, nprobe=chosen_nprobe, boost_vectors=cpesh_entry.get("expected_vec"))
+        except TypeError:
+            results = ctx.index.search(query_vec, nprobe=chosen_nprobe)
+    else:
+        results = ctx.index.search(query_vec, nprobe=chosen_nprobe)
+    latency_ms = (time.time() - t0) * 1000.0
+
+    # Decision log → artifacts/gating_decisions.jsonl (best-effort)
+    try:
+        os.makedirs("artifacts", exist_ok=True)
+        with open("artifacts/gating_decisions.jsonl", "a") as f:
+            f.write(json.dumps({
+                "request_id": getattr(request, "request_id", None),
+                "lane": getattr(request, "lane", None),
+                "used_cpesh": used_cpesh,
+                "quality": qv, "cos": cv,
+                "chosen_nprobe": chosen_nprobe,
+                "latency_ms": round(latency_ms, 3),
+                "ts": time.time()
+            }) + "\n")
+    except Exception:
+        pass
@@
     return formatted_response
@@
+@app.get("/metrics/gating")
+def gating_metrics():
+    counts = {"total": 0, "used_cpesh": 0}
+    try:
+        with open("artifacts/gating_decisions.jsonl") as f:
+            for line in f:
+                counts["total"] += 1
+                try:
+                    if json.loads(line).get("used_cpesh"):
+                        counts["used_cpesh"] += 1
+                except Exception:
+                    continue
+    except FileNotFoundError:
+        pass
+    return counts
2) CPESH permanent datastore seam (active tier)
src/datastore/cpesh_store.py (new)
import json, os
from typing import Iterator, Dict, Any

class CPESHDataStore:
    """
    Permanent training datastore (Active tier only in S5).
    Future: warm .jsonl.gz, cold Parquet, and SQLite index.
    """
    def __init__(self, active_path: str = "artifacts/cpesh_active.jsonl"):
        self.active_path = active_path
        os.makedirs(os.path.dirname(active_path), exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        with open(self.active_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def iter_active(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.active_path):
            return iter(())
        with open(self.active_path) as f:
            for line in f:
                yield json.loads(line)
Minimal usage now: anywhere you currently write CPESH JSONL, call CPESHDataStore().append(entry) instead. No behavior change; just future-proofing for S6 tiering.
3) Dynamic nlist already in place — keep it enforced
(You’ve shipped calculate_nlist() and metadata keys; nothing to change here. Just make sure make build-faiss ARGS="--nlist 512" on a 10k set shows requested_nlist=512, nlist=250, max_safe_nlist=250 in artifacts/index_meta.json.)
4) Offline embedder — hard fix (no internet)
src/vectorizer.py (patch)
*** a/src/vectorizer.py
--- b/src/vectorizer.py
@@
-from sentence_transformers import SentenceTransformer
+import os
+from sentence_transformers import SentenceTransformer
@@
-def load_embedder():
-    return SentenceTransformer("sentence-transformers/gtr-t5-base")
+def load_embedder():
+    """
+    Enforces local-only model loading when offline.
+    Honor env:
+      - LNSP_EMBEDDER_PATH: local dir of the model (preferred)
+      - SENTENCE_TRANSFORMERS_HOME / HF_HOME: cache roots
+      - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: '1' to forbid downloads
+    """
+    local = os.getenv("LNSP_EMBEDDER_PATH")
+    if local and os.path.isdir(local):
+        return SentenceTransformer(local)
+    # If offline is requested, do not attempt network
+    if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
+        raise RuntimeError(
+            "Embedder is offline but LNSP_EMBEDDER_PATH not set. "
+            "Place the model at ./models/gtr-t5-base and export LNSP_EMBEDDER_PATH=./models/gtr-t5-base"
+        )
+    # Online path (allowed only if your environment permits)
+    return SentenceTransformer("sentence-transformers/gtr-t5-base")
Model prep script (optional helper, only when you do have network)
scripts/prepare_models.sh
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
5) Makefile targets (ensure these exist)
.PHONY: api
api:
	@PORT=$${PORT:-8092}; echo "[api] starting on $$PORT"
	@PYTHONPATH=src ./.venv/bin/uvicorn src.api.retrieve:app --host 127.0.0.1 --port $$PORT

.PHONY: gating-snapshot
gating-snapshot:
	@mkdir -p eval/snapshots
	@curl -sf http://127.0.0.1:$${PORT:-8092}/metrics/gating \
	  -o eval/snapshots/gating_$$(/bin/date -u +"%Y%m%dT%H%M%SZ").json
	@echo "[gating-snapshot] wrote eval/snapshots/…"
If your Makefile already has gating-snapshot, keep whichever version you prefer.
6) Tests (minimal but real)
tests/test_nlist_dynamic.py
(you already have this; keep as is)
tests/test_gating.py
from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh

def test_gate_pass_basic():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.90, "cosine": 0.56}
    assert should_use_cpesh(entry, cfg) is True

def test_gate_fail_quality():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.80, "cosine": 0.70}
    assert should_use_cpesh(entry, cfg) is False

def test_gate_fail_cos():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.90, "cosine": 0.50}
    assert should_use_cpesh(entry, cfg) is False

def test_gate_fail_insufficient():
    cfg = CPESHGateConfig()
    entry = {"quality": 0.95, "cosine": 0.80, "insufficient_evidence": True}
    assert should_use_cpesh(entry, cfg) is False

def test_lane_override_factoid():
    base = CPESHGateConfig(q_min=0.82)
    lane_cfg = apply_lane(base, "L1_FACTOID")
    assert lane_cfg.q_min >= 0.85
tests/test_index_meta.py (augment keys)
-    for k in ["type","metric","nlist","nprobe","count","build_seconds"]:
+    for k in ["type","metric","nlist","nprobe","count","build_seconds","requested_nlist","max_safe_nlist"]:
         assert k in meta
+    assert meta["nlist"] <= meta["max_safe_nlist"]
7) Port + process guard (the bind error you saw)
Don’t paste commented blocks into zsh. Use this when a port is stuck:
lsof -iTCP:8092 -sTCP:LISTEN -n -P | awk 'NR>1{print $$2}' | xargs -r kill -9
…or just run:
make api PORT=8093
8) S5 Runbook (to produce real metrics, offline-safe)
# One-time: ensure model is local (do this only when you have network)
# scripts/prepare_models.sh

# Offline env for all runs:
export LNSP_EMBEDDER_PATH=./models/gtr-t5-base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Rebuild index at 10k; request 512 to demonstrate downshift
make build-faiss ARGS="--type ivf_flat --metric ip --nlist 512 --nprobe 16"
cat artifacts/index_meta.json   # should show nlist == max_safe_nlist

# Start API (pick a free port if 8092 is busy)
make api PORT=8092
# In another terminal, do a quick smoke search to trigger gating
# (use your existing client or curl at /search)

# Consultant sweeps (live, cold then warm)
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 8  --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 24 --metric ip --queries eval/100q.jsonl

# Warm repeats (best two)
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 8  --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl

# Snapshots
make gating-snapshot PORT=8092
make slo-snapshot
S5 exit criteria (use this as your merge gate)
artifacts/index_meta.json shows requested_nlist and max_safe_nlist with nlist ≤ max_safe_nlist.
artifacts/gating_decisions.jsonl has > 0 entries from real queries.
/metrics/gating returns {"total":N,"used_cpesh":K} (N>0).
eval/day_s5_report.md updated with:
Hit@1/3 + P50/P95 vs nprobe (cold & warm),
% CPESH-assisted from /metrics/gating,
final recommended nprobe and lane-specific thresholds (L1_FACTOID q_min≥0.85).