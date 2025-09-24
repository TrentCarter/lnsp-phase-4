got it—here’s a tight, role-aware plan you can paste into `chats/conversation_09232025.md` as the **\[P2] Conversation Plan**.

---

# \[P2] Conversation Plan — P4·Day 3 wrap / Day 4 kickoff (25–30 min)

**Attendees:** \[Architect] \[Programmer] \[Consultant]
**Goal:** Ratify Day-3 decisions, clear API blocker, lock eval + ops steps for Day-4 10k ramp.

## 0) Pre-reads (2 min)

* `/docs/enums.md`, `/src/enums.py` (frozen)
* `/src/adapters/lightrag_adapter.py` (vendor glue)
* `eval/day3_report.md` + `eval/day3_results_consultant.jsonl`
* `artifacts/fw1k_vectors.npz`, FAISS `IVF_FLAT (nlist=256)` build notes

## 1) Decisions to ratify (5 min)

* **Dense-only default for L1\_FACTOID** (remove lexical fallback): +61% latency for no quality gain → keep fallback only behind a feature flag `ENABLE_LEX_FALLBACK=0` (env).
* **LightRAG pin/spec**: lock `1.4.9rc1` and the current adapter surface; record in `requirements.txt` and `/docs/architecture.md`.
* **Vector retention policy (Lean)**: store `fused(784)` + optional `question_vec(768)`; recompute `concept_vec` on demand.
* **TMD bits + lane index**: pack to `uint16` (bit layout D:4 | T:5 | M:6 | spare:1), store `lane_index` as `int2` with CHECK `0..32767`.

## 2) API unblock (Pydantic v2) (6 min)

* **Action:** switch settings import and add dep.

  * `requirements.txt`: `pydantic>=2.4`, `pydantic-settings>=2.2`, `fastapi>=0.110`, `uvicorn>=0.30`
  * Code diff (apply wherever settings are used):

    ```diff
    - from pydantic import BaseSettings
    + from pydantic_settings import BaseSettings
    ```
* **Smoke:**

  ```bash
  export NO_DOCKER=1
  source .venv/bin/activate
  uvicorn src.api.retrieve:app --reload
  # curl -s "http://127.0.0.1:8000/search?q=Who%20is%20Cl%C3%A1udia%20Pascoal?&lane=L1_FACTOID&top_k=5"
  ```

## 3) Build & index sanity (4 min)

* Confirm artifacts exist: `artifacts/fw1k_vectors.npz` (from `ingest_1k.sh` + `python -m src.vectorizer`).
* Index check:

  ```bash
  python -m src.faiss_index --index-type IVF_FLAT --nlist 256 --load artifacts/fw1k_vectors.npz
  ```
* Record FAISS params in `conversation_09232025.md` (nlist, trained vectors, build time).

## 4) Evaluation lane (5 min)

* **Offline already logged** (echo 5% sample, vectors from `fw1k_vectors.npz`).
* **Next when network is available:** re-run 20-item eval against live `/search` to re-confirm 100% hybrid baseline; log side-by-side in `eval/day3_report.md`.

  ```bash
  python -m src.eval_runner \
    --queries eval/day3_eval.jsonl \
    --top-k 5 --timeout 15 \
    --out eval/day3_results_live.jsonl
  ```
* **Acceptance snapshot to capture:** echo pass rate, lane distribution, three `/search` examples with inputs/outputs.

## 5) Ops + repo hygiene (3 min)

* `scripts/bootstrap_all.sh` (NO\_DOCKER path) → keep as canonical entrypoint.
* Add `pydantic-settings` to `requirements.txt`; `pip-compile` (if in use) and commit.
* Update `README` quickstart (NO\_DOCKER steps + eval runner examples).

---

## Role-scoped action items

### \[Architect] (owner: TRC)

* ✅ Enums frozen → post final bit-packing note in `/docs/enums.md`.
* Publish LightRAG spec note (pin, interface) in `/docs/architecture.md`.
* Open issue: “Lexical fallback behind flag” with rationale (61% latency, 0 gain).

### \[Programmer]

* Fix Pydantic v2 import; add `pydantic-settings`; bump `fastapi/uvicorn`.
* Re-run API smoke; attach curl output snippet to `conversation_09232025.md`.
* Add unit test `tests/test_retrieve_api.py` (lane routing + top-k shape + score monotonicity).
* Persist FAISS build metadata (nlist, trained, vectors count) to `artifacts/faiss_meta.json`.

### \[Consultant]

* Extend `eval/day3_eval.jsonl` to **20 items** (balanced lanes; keep 5 L1\_FACTOID).
* Update `eval/day3_report.md` with: (a) dense-only decision, (b) latency deltas, (c) three exemplar `/search` traces.
* Log Day-3 acceptance checklist pending items (PG/Neo row counts once dumps land).

---

## Copy-paste commands (for the call)

```bash
# 1) Env (NO_DOCKER path)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) LightRAG pin (already in bootstrap; verify)
python -c "import lightrag, pkgutil; print(lightrag.__version__)"

# 3) API up
uvicorn src.api.retrieve:app --reload

# 4) Offline eval (already done; repeatable)
python -m src.eval_runner --queries eval/day3_eval.jsonl \
  --offline-npz artifacts/fw1k_vectors.npz --top-k 5 --timeout 15 \
  --out eval/day3_results_consultant.jsonl
```

---

## Minutes template to fill after the call

* **Decisions:** dense-only L1\_FACTOID; LightRAG 1.4.9rc1 pinned; Lean vector policy; TMD pack spec.
* **Blockers cleared:** Pydantic v2 import fixed; API boot OK.
* **Artifacts:** `faiss_meta.json`, `eval/day3_results_live.jsonl` (pending), curl samples.
* **Next:** Day-4 10k ingest dry-run → IVF training size, nlist target, perf budget check.

---


