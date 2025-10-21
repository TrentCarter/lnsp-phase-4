Awesome — here’s the **Hybrid Retrieval Spec (v0.1)** you can hand to the Architect and the API team. It’s tight, testable, and comes with an experiment grid + success gates.

# HYBRID RETRIEVAL SPEC (v0.1)

```
User Q → embed → FAISS(Top-K0) → LVM re-rank → TMD re-rank → Top-K
                       |              |               |
                  recall@scale     precision       semantic control
```

## 0) Goals

* Turn the Phase-3 LVM (1k-ctx) into a **production re-ranker** inside a two/three-stage pipeline.
* Align training/eval: Phase-3 remains a **small-candidate ranker**, FAISS handles **global recall**.
* Achieve **Hit@5 ≥ 78%** (vs 75.65%) at **sub-30ms** offline per query on test bench (tunable).

---

## 1) Pipeline Stages

### Stage-1: FAISS ANN (Recall Engine)

* **Input:** `query_vec: float32[D]` (L2-normalized)
* **Index:** IVF_FLAT (cosine/IP), `nlist≈512–1024`, `nprobe` lane-aware (default 16)
* **Output:** `candidates_K0`: list of `{id:int64, cos:float32, tmd_bits:uint16, lane:int16}` of size `K0 ∈ {100, 200, 500, 1000}`

### Stage-2: LVM Re-Ranker (Precision on small set)

* **Input:** `query_vec`, `candidates_K0`
* **Scoring:** Use Phase-3 model to score each candidate (the same geometry as training; cosine of reconstructed next-vector vs candidate)
* **Output:** `candidates_K1` top-K1 by **LVM score** (K1=50 by default)

### Stage-3: TMD Re-Ranker (Semantic Control)

* **Input:** `query_tmd_bits`, `candidates_K1` (each with `tmd_bits`, `lane`)
* **Alignment:**

  ```
  tmd_align = 
    1.00 if lane_match
    0.80 if domain_match_only
    0.40 if task_mismatch
    0.20 if domain_mismatch
  ```
* **Final score:** `final = λ * lvm_score + (1-λ) * tmd_align` (default λ=0.7; sweep 0.6–0.8)
* **Output:** final top-K (`K ∈ {5,10}`)

---

## 2) API Endpoints

### 2.1 `/search/hybrid`

**Request**

```json
{
  "query_vec": [..D..],
  "tmd_bits": 1234,
  "lane": 9,
  "k0": 500,
  "k1": 50,
  "k": 10,
  "lambda_tmd": 0.7,
  "faiss": {"nprobe": 16},
  "return_fields": ["id","score","cos","lvm","tmd_bits","lane"]
}
```

**Response**

```json
{
  "model_version": "phase3_lvm_1k_ctx",
  "index_version": "wikipedia_637k_ivf512",
  "k0": 500, "k1": 50, "k": 10,
  "results": [
    {"id": 123456, "score": 0.912, "lvm": 0.874, "cos": 0.861, "tmd_bits": 1234, "lane": 9},
    ...
  ],
  "telemetry": {"faiss_ms": 4.1, "lvm_ms": 7.6, "tmd_ms": 0.2, "total_ms": 11.9}
}
```

### 2.2 `/metrics/hitk_offline`

* Triggers offline evaluation on the held-out set with current settings.
* Returns Hit@1/5/10, MRR@n, latency slices, and lane-wise breakdown.

---

## 3) Data Contracts

### Vector Bank NPZ (one-time, already fixed)

* `vectors: float32[N,D]` (L2-normed)
* `bank_ids: int64[N]`
* `tmd_bits: uint16[N]`
* `lane_indices: int16[N]`

### Eval/Train NPZ (already patched)

* `val_curr_ids: int64[T]`
* `val_next_ids: int64[T]`
* (train_* equivalents for training diagnostics)

---

## 4) Scoring Details

### 4.1 LVM score (Phase-3 geometry)

* Predict Δ, reconstruct `ŷ = l2(x + Δ̂)`
* Score against candidate `c`: `lvm_score = cosine(ŷ, v_c)`
* (Must use the **same normalization** as in training; assert at runtime.)

### 4.2 TMD alignment (configurable)

* Start with the 1.0 / 0.8 / 0.4 / 0.2 scale above.
* Optional domain matrix if you have a hierarchy:

  * Same super-domain (e.g., LifeSci): 0.9
  * Sister domain: 0.7
  * Distant domain: 0.3

---

## 5) Pseudocode (server-side)

```python
def search_hybrid(query_vec, tmd_bits, lane, k0=500, k1=50, k=10, nprobe=16, lam=0.7):
    # Stage 1: FAISS
    cand = faiss_search(query_vec, topk=k0, nprobe=nprobe)  # [(id, cos, tmd_bits, lane), ...]
    # Stage 2: LVM
    lvm_scored = []
    for c in cand:
        score = lvm_score_phase3(query_vec, bank_vectors[c.id])
        lvm_scored.append((c, score))
    lvm_scored.sort(key=lambda x: x[1], reverse=True)
    top = lvm_scored[:k1]
    # Stage 3: TMD
    def tmd_align(q_tmd, q_lane, c_tmd, c_lane):
        if c_lane == q_lane: return 1.0
        if domain(c_tmd) == domain(q_tmd): return 0.8
        if task(c_tmd) != task(q_tmd): return 0.4
        return 0.2
    rescored = []
    for (c, s) in top:
        ta = tmd_align(tmd_bits, lane, c.tmd_bits, c.lane)
        final = lam*s + (1-lam)*ta
        rescored.append((c, final, s))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return format_results(rescored[:k])
```

---

## 6) Experiment Grid (offline A/B)

```
┌───────────────────────────────────────────────────────────────┐
│ Dataset: Phase-3 (1k-ctx, 637k bank); Eval: val_curr/next_ids │
└───────────────────────────────────────────────────────────────┘
Variables:
  K0 ∈ {100, 200, 500, 1000}
  λ ∈ {0.6, 0.7, 0.8}
  nprobe ∈ {8, 16}
  K1 = 50 (fixed), K ∈ {5,10}

Collect:
  Hit@1/5/10, MRR@10, P50/P95 latency, lane-wise Hit@5
Success gate (promote config):
  Hit@5 uplift ≥ +2.0% absolute vs LVM-only (K0 fixed)
  P95 latency +≤ 20% vs LVM-only
  No lane regressions > 3%
```

**Recommended starting point:** `K0=500, λ=0.7, nprobe=16, K1=50`

---

## 7) Telemetry & Logging

* Log per-request:

  ```json
  {
    "model":"phase3_1k",
    "index":"ivf512",
    "k0":500,"k1":50,"k":10,"lambda_tmd":0.7,"nprobe":16,
    "faiss_ms":4.0,"lvm_ms":7.5,"tmd_ms":0.2,"total_ms":11.7,
    "lane":9,"hit5":1,"rank_true":3
  }
  ```
* Expose rollups:

  * `/metrics/hitk_offline`: Hit@1/5/10, MRR@n (global + per lane)
  * `/metrics/latency`: P50/P95 by stage
  * `/metrics/routing`: nprobe distribution, K0 histograms

---

## 8) Makefile Targets (orchestration)

```makefile
# Evaluate hybrid grid and print a summary table
eval-hybrid:
	python tools/eval_hybrid.py \
	  --bank artifacts/wikipedia_637k_vectors.npz \
	  --val artifacts/lvm/phase3_eval/training_sequences_ctx100.npz \
	  --k0 100 200 500 1000 --lambda 0.6 0.7 0.8 --nprobe 8 16 \
	  --k1 50 --k 10 --out artifacts/evals/hybrid_grid.json

# Serve hybrid endpoint (dev)
serve-hybrid:
	uvicorn api.hybrid:app --port 8088 --reload
```

---

## 9) Risks & Mitigations

* **Eval mismatch:** Ensure `*_next_ids` and `bank_ids` are used; forbid vector matching by float.
* **Latency creep:** Set a guardrail: `P95_total_ms` must be ≤ baseline + 20%. Trim `K0` if needed.
* **Lane drift:** Verify per-lane Hit@5; adjust `λ` or expanded domain matrix to protect specialist lanes.
* **Dim/Norm mismatch:** Assert `||vec||≈1` and `D` equality at all stages; reject requests if violated.

---

## 10) Acceptance Criteria (v0.1)

* Endpoint `/search/hybrid` returns correct JSON and passes schema checks.
* Offline grid finds at least one config with **Hit@5 uplift ≥ +2%** and **P95 latency +≤20%**.
* Lane-wise regression checker shows **no lane worse than −3%** Hit@5.
* All metrics reproducible with fixed seeds and frozen bank.

---

## 11) ASCII Block Diagram (deployment mental model)

```
                ┌────────────┐
Query Vector →  │  FAISS     │  (K0=500..1000, nprobe=16)
                └─────┬──────┘
                      │ candidates_K0
                      ▼
                ┌────────────┐
                │  LVM Rank  │  (Phase-3 1k-ctx; score each)
                └─────┬──────┘
                      │ top-K1 (50)
                      ▼
                ┌────────────┐
                │  TMD Rank  │  (λ=0.7; align Task/Domain)
                └─────┬──────┘
                      │ top-K (5/10)
                      ▼
                   Results
```

---

## 12) Hand-off Checklist

* [ ] Bank NPZ includes `bank_ids`, `tmd_bits`, `lane_indices`
* [ ] Eval NPZ includes `val_curr_ids`, `val_next_ids`
* [ ] API compiled; `/search/hybrid` returns within SLA on dev box
* [ ] `eval-hybrid` produces grid table with a clear winner (store JSON & MD)
* [ ] Summary report “**Hybrid Retrieval v0.1 Results.md**” with chosen K0/λ/nprobe

---

If you want, I can also stub `tools/eval_hybrid.py` (reads NPZs, runs grid, prints a markdown table) so the Architect can run the full experiment in one command.
