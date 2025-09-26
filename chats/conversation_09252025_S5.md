S5 Additions (Organizer-approved)
1) Dynamic nlist (Architect → Programmer → Consultant)
What goes in docs (done by Architect; keep)
Add the dynamic policy verbatim in /docs/architecture.md under FAISS Dial-Plan.
Cross-link in /docs/runtime_env.md (“auto-downshift when train_points < 40×nlist”).
Code (Programmer)
Patch: src/faiss_index.py — auto-select & enforce safe nlist, emit telemetry.
*** a/src/faiss_index.py
--- b/src/faiss_index.py
@@
+import math
+from dataclasses import asdict, dataclass
+
+def calculate_nlist(n_vectors: int, requested_nlist: int | None = None) -> int:
+    max_safe_nlist = max(1, n_vectors // 40)  # 40× training rule
+    if requested_nlist:
+        if requested_nlist > max_safe_nlist:
+            print(f"[faiss_index] Warning: reducing nlist from {requested_nlist} -> {max_safe_nlist} to satisfy 40× rule")
+        return min(requested_nlist, max_safe_nlist)
+    # auto-select by scale
+    if n_vectors < 8000:
+        return max_safe_nlist
+    elif n_vectors < 20000:
+        return min(200, max_safe_nlist)
+    elif n_vectors < 40000:
+        return min(512, max_safe_nlist)
+    else:
+        return min(int(math.sqrt(n_vectors)), max_safe_nlist)
@@
-    nlist = args.nlist
+    nlist = calculate_nlist(len(vectors), args.nlist)
     nprobe = args.nprobe
@@
-    meta = {
-        "type": index_type, "metric": metric, "nlist": nlist, "nprobe": nprobe,
-        "count": len(vectors), "build_seconds": build_secs,
-    }
+    meta = {
+        "type": index_type, "metric": metric,
+        "nlist": nlist, "nprobe": nprobe, "count": len(vectors),
+        "build_seconds": build_secs, "requested_nlist": args.nlist,
+        "max_safe_nlist": max(1, len(vectors)//40),
+    }
     json.dump(meta, open("artifacts/index_meta.json","w"), indent=2)
     print(f"[faiss_index] Wrote index meta → artifacts/index_meta.json")
New tests
tests/test_nlist_dynamic.py: unit tests for calculate_nlist() (edges: 5k, 10k, 25k, 120k; with/without requested_nlist).
Extend tests/test_index_meta.py to assert requested_nlist and max_safe_nlist are present and consistent.
Make target
make build-faiss already exists; no change. Add ARGS="--nlist 512" still works—auto-downshifts safely.
Eval (Consultant)
In eval/day_s5_report.md, add a “Dynamic nlist effects” section: show requested_nlist=512 at 10k auto-reducing to 200, with Hit@1/3 and latency deltas vs fixed 200.
2) CPESH two-stage gating (Architect → Programmer → Consultant)
Policy (Architect; already documented)
CPESH is permanent training data, not a cache. Gating is assist-time only; no deletion.
Default gates: quality ≥ 0.82 and cosine ≥ 0.55. Per-lane overrides allowed (L1_FACTOID → quality ≥ 0.85).
Code (Programmer)
New module: src/utils/gating.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CPESHGateConfig:
    q_min: float = 0.82
    cos_min: float = 0.55
    nprobe_cpesh: int = 8
    nprobe_fallback: int = 16
    lane_overrides: Dict[str, Dict[str, float]] = None  # e.g., {"L1_FACTOID": {"q_min": 0.85}}

def apply_lane(cfg: CPESHGateConfig, lane: Optional[str]) -> CPESHGateConfig:
    if lane and cfg.lane_overrides and lane in cfg.lane_overrides:
        o = cfg.lane_overrides[lane]
        return CPESHGateConfig(
            q_min=o.get("q_min", cfg.q_min),
            cos_min=o.get("cos_min", cfg.cos_min),
            nprobe_cpesh=o.get("nprobe_cpesh", cfg.nprobe_cpesh),
            nprobe_fallback=o.get("nprobe_fallback", cfg.nprobe_fallback),
            lane_overrides=cfg.lane_overrides
        )
    return cfg

def should_use_cpesh(cpesh_entry: Optional[dict], cfg: CPESHGateConfig) -> bool:
    if not cpesh_entry: return False
    if cpesh_entry.get("insufficient_evidence"): return False
    return (cpesh_entry.get("quality", 0) >= cfg.q_min) and (cpesh_entry.get("cosine", 0) >= cfg.cos_min)
Wire into API: src/api/retrieve.py
*** a/src/api/retrieve.py
--- b/src/api/retrieve.py
@@
-from .… import …
+from src.utils.gating import CPESHGateConfig, apply_lane, should_use_cpesh
+import json, time
@@
-    # existing retrieval context ...
+    gate_cfg = CPESHGateConfig(
+        q_min=float(os.getenv("LNSP_CPESH_Q_MIN", "0.82")),
+        cos_min=float(os.getenv("LNSP_CPESH_COS_MIN", "0.55")),
+        nprobe_cpesh=int(os.getenv("LNSP_NPROBE_CPESH", "8")),
+        nprobe_fallback=int(os.getenv("LNSP_NPROBE_DEFAULT", "16")),
+        lane_overrides={"L1_FACTOID": {"q_min": 0.85}}
+    )
@@ def search(request: SearchRequest):
-    # prepare query_vec, etc.
+    gate = apply_lane(gate_cfg, request.lane)
+    cpesh_entry = ctx.cpesh_cache.get(request.doc_id_hash) if hasattr(ctx, "cpesh_cache") else None
+    t0 = time.time()
+    if should_use_cpesh(cpesh_entry, gate):
+        results = ctx.index.search(query_vec, nprobe=gate.nprobe_cpesh, boost_vectors=cpesh_entry.get("expected_vec"))
+        used_cpesh, chosen_nprobe, qv, cv = True, gate.nprobe_cpesh, cpesh_entry.get("quality"), cpesh_entry.get("cosine")
+    else:
+        results = ctx.index.search(query_vec, nprobe=gate.nprobe_fallback)
+        used_cpesh, chosen_nprobe = False, gate.nprobe_fallback
+        qv = cpesh_entry.get("quality") if cpesh_entry else None
+        cv = cpesh_entry.get("cosine") if cpesh_entry else None
+    latency_ms = (time.time() - t0) * 1000
+    # decision log (append JSONL)
+    try:
+        with open("artifacts/gating_decisions.jsonl","a") as f:
+            f.write(json.dumps({
+                "query_id": request.request_id,
+                "lane": request.lane,
+                "used_cpesh": used_cpesh,
+                "quality": qv, "cos": cv,
+                "chosen_nprobe": chosen_nprobe,
+                "latency_ms": round(latency_ms, 3),
+                "ts": time.time()
+            })+"\n")
+    except Exception:
+        pass
Expose metrics: add endpoint to retrieve.py
@app.get("/metrics/gating")
def gating_metrics():
    counts = {"total":0,"used_cpesh":0}
    try:
        with open("artifacts/gating_decisions.jsonl") as f:
            for line in f:
                counts["total"] += 1
                if json.loads(line).get("used_cpesh"):
                    counts["used_cpesh"] += 1
    except FileNotFoundError:
        pass
    return counts
Env knobs (default sane):
export LNSP_CPESH_Q_MIN=0.82
export LNSP_CPESH_COS_MIN=0.55
export LNSP_NPROBE_CPESH=8
export LNSP_NPROBE_DEFAULT=16
Tests
tests/test_gating.py: gate pass/fail cases; “insufficient_evidence” exclusion; lane override to 0.85.
Extend consultant harness to split metrics: CPESH-assisted vs fallback Hit@1/3 & latency.
3) CPESH Data Store (tiered) — minimal code hooks for S5
You already fixed the docs. In S5, add interfaces so we can evolve storage without breaking callers:
New: src/datastore/cpesh_store.py (thin wrapper; JSONL today, ready for warm/cold tiers later)
class CPESHDataStore:
    def __init__(self, active_path="artifacts/cpesh_active.jsonl"):
        self.active_path = active_path
    def append(self, entry: dict) -> None:
        with open(self.active_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    def iter_active(self):
        with open(self.active_path) as f:
            for line in f: yield json.loads(line)
Keep current JSONL behavior, but move all writers to go through this wrapper now. In S6 we can drop in gzip/parquet/SQLite index without touching API code.
Make targets
make cpesh-rotate (stub): for S5 just checks line count and prints a recommendation; implement real rotation in S6.
4) Consultant tasks for S5 (so we see impact)
Run A/B with LNSP_CPESH_Q_MIN ∈ {0.78, 0.82, 0.86} × LNSP_CPESH_COS_MIN ∈ {0.50, 0.55, 0.60}.
Report Hit@1/3 (overall + CPESH-assisted only) and P50/P95; include usage rate from /metrics/gating.
Add decision trace audit (sample 10 cases where gate failed but high-rank doc existed—tune thresholds accordingly).
Deliver as eval/day_s5_report.md with a final recommended gates row and the chosen nprobe.
TL;DR
✅ I agree: make nlist dynamic and safe automatically.
✅ I agree: treat CPESH as a confidence-gated accelerator, not a mandatory step; log every decision.
✅ I agree: CPESH is permanent training data; in S5 we add a thin store wrapper so tiering can land cleanly in S6.

---

🔍 Consultant Execution Log — 2025-09-25T15:48Z
Task 4.1 A/B sweeps — Blocked. Dynamic nlist + gating code paths are not implemented yet (missing src/utils/gating.py, API wiring, and telemetry). No live API with 10k index is running, so consultant_eval cannot exercise threshold grid.
Task 4.2 Decision trace audit — Blocked pending gating decision logs (artifacts/gating_decisions.jsonl) which will arrive once Programmer tasks land.
Task 4.3 Report updates — Blocked because no fresh measurements exist; eval/day_s5_report.md unchanged.

Notes:
- No commands executed beyond verification of missing modules; prerequisites from Architect/Programmer tracks must land before Consultant work can proceed.

🔍 Consultant Execution Log — 2025-09-25T16:05Z
- Rechecked repo post-Architect updates: dynamic nlist logic now lives in `src/faiss_index.py`, but gating wiring in `src/api/retrieve.py` still imports helpers without invoking them, so CPESH thresholds and logging remain inert.
- `src/utils/gating.py` and related tests exist, yet `/metrics/gating` endpoint and decision log writer are not active; no `artifacts/gating_decisions.jsonl` produced.
- `artifacts/index_meta.json` is absent until `make build-faiss` runs; consultant evaluation still blocked pending full Programmer integration and a running API.

🔍 Consultant Execution Log — 2025-09-25T16:28Z
- Rebuilt FAISS index with the new dial-plan knobs via `python src/faiss_index.py --npz artifacts/fw10k_vectors_768.npz --type ivf_flat --metric ip --nlist 512 --nprobe 16`; `artifacts/index_meta.json` now records `nlist=250`, `requested_nlist=512`, confirming auto-downshift.
- Attempted to start FastAPI (both `uvicorn` and `python3.11 -m uvicorn`) but the sandbox blocks binding to 127.0.0.1:8092 (`Operation not permitted` / `address already in use`).
- Tried to drive `RetrievalContext.search()` directly; embedding backend needs HuggingFace downloads, which are blocked by the restricted network policy, so no gating decisions were produced.
- Next opportunity once ports/network are available: bring up the API, run the nprobe sweeps and gating grid, then refresh `eval/day_s5_report.md` and snapshot `/metrics/gating`.
