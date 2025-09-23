[P4] Conversation Plan — Day-4 execution & 10k canary (25–30 min)
Attendees: [Architect] [Programmer] [Consultant]
Goal: land 10k ingest canary with dense-only L1, hybrid L2/L3, stable Py 3.11 CI, truthful FAISS metadata, and balanced eval reporting.
0) Pre-reads (2 min)
docs/architecture.md (pin 1.4.9rc1, FAISS SoT, lane behavior)
docs/enums.md (TMD bit-packing, lane extraction)
pytest.ini, src/search_backends/stub.py (light tests)
artifacts/faiss_meta.json (confirm vectors, nlist, index type)
1) Ratified decisions carried into P4 (no debate) (3 min)
Python: hard-pin 3.11.x for runtime, CI, and venvs.
L1_FACTOID: dense-only by default; lexical behind LNSP_LEXICAL_FALLBACK=1.
Indexing:
1k: IVF_FLAT, nlist=32, nprobe=8
10k canary: IVF_FLAT, nlist=128, nprobe=16 (tune after latency check)
2) Acceptance gates (P4 exit) (3 min)
L1 (dense-only) on 10k: P50 ≤ 85 ms, P95 ≤ 200 ms, Top-1 hit-rate ≥ 0.92 (eval-20).
L2/L3 (hybrid) on 10k: P50 ≤ 180 ms, P95 ≤ 400 ms.
Truthfulness: faiss_meta.json reflects real vectors, dim, index_type, nlist.
Stability: pytest -m "not heavy" green in CI 3.11; heavy tests gated to local.
Role-scoped work (P4)
[Architect]
Lock runtime & eval policy
Add Runtime Matrix to README: Py 3.11 ✅; Py 3.13 🚫 (FAISS/BLAS).
Document evaluation policy: L1 dense-only; L2/L3 hybrid; lexical flag is override.
Trace contract
Finalize /search response schema (trace_id, lane, k, scores, support_ids).
Deliverables
README Runtime Matrix section.
docs/architecture.md → “Evaluation Policy & Flags” subsection.
docs/api_response.md (new) with /search JSON schema.
[Programmer]
10k ingest & index
Run canary ingest to 10k chunks → artifacts/fw10k_vectors.npz.
Build IVF_FLAT index nlist=128 → artifacts/fw10k_ivf.index.
Metadata truthing
Run tools/write_faiss_meta.py to regenerate artifacts/faiss_meta.json (auto-reads NPZ + Index).
Add a quick guard in CI that fails if vectors < nlist.
API & flags
Ensure LNSP_LEXICAL_FALLBACK path is not used for L1 unless opt-in.
Keep LNSP_TEST_MODE=1 wiring to stub searcher for unit tests.
CI hardening
Add .github/workflows/ci.yml → Py 3.11, run pytest -m "not heavy".
Deliverables
artifacts/fw10k_vectors.npz, artifacts/fw10k_ivf.index, refreshed faiss_meta.json.
CI run green on Py 3.11; unit tests green w/ stub mode.
Commands (paste-ready)
# 10k canary (adjust input path if needed)
bash scripts/ingest_10k.sh
python -m src.faiss_index --npz artifacts/fw10k_vectors.npz \
  --index-type IVF_FLAT --nlist 128 --out artifacts/fw10k_ivf.index
python tools/write_faiss_meta.py
cat artifacts/faiss_meta.json

# Unit tests (light)
LNSP_TEST_MODE=1 pytest -m "not heavy" -q

# Heavy (local only)
pytest -m heavy -q
[Consultant]
Balanced eval runs on 10k
Use the balanced eval-20 (5×L1 / 8×L2 / 7×L3).
Run two passes:
a) Dense-only L1, hybrid L2/L3 (LNSP_LEXICAL_FALLBACK=0)
b) Lexical enabled for comparison on L1 only (=1)
Report
Create eval/day4_report.md with side-by-side latency (P50/P95), Top-1 hit-rate, and 3 example /search traces per lane.
Call out any regressions vs Day-3; include faiss_meta.json key lines.
Deliverables
eval/day4_results_dense.jsonl, eval/day4_results_lex.jsonl, eval/day4_report.md.
Commands (paste-ready)
# Dense-only L1
export LNSP_LEXICAL_FALLBACK=0
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_dense.jsonl

# Lexical compare
export LNSP_LEXICAL_FALLBACK=1
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_lex.jsonl
Risks & mitigations (blunt)
Wheel drift (Mac/ARM): pinned Py 3.11 + CI enforces it → no segfault roulette.
Index under-sized (nlist too high vs vectors): meta script warns; CI check fails build.
Lexical cost creep: env flag defaults off; L1 stays dense-only.
Paste-ready log snippet
Add this to chats/conversation_09232025_P4.md (or today’s log):
## 2025-09-23 — [P4] Day-4 Execution & 10k Canary

### Decisions (ratified)
- Python 3.11.x only (runtime + CI).
- L1_FACTOID dense-only by default; lexical via LNSP_LEXICAL_FALLBACK=1.
- 10k IVF params: nlist=128, nprobe=16 (tune post-latency check).

### [Architect] Status / Next
- Added Runtime Matrix to README; documented Evaluation Policy & Flags.
- Finalized /search response schema (trace_id, lane, k, scores, support_ids).
- Next: sign-off after P50/P95 + Top-1 gates pass.

### [Programmer] Status / Next
- Run 10k ingest -> fw10k_vectors.npz; build IVF_FLAT (nlist=128).
- Regenerate artifacts/faiss_meta.json via tools/write_faiss_meta.py.
- CI: ensure Py 3.11, run pytest -m "not heavy" (green).
- Next: provide meta excerpt + curl sample for /search L1/L2/L3.

### [Consultant] Status / Next
- Run eval-20 on 10k (dense-only vs lexical-enabled).
- Produce eval/day4_report.md with side-by-side P50/P95, Top-1, and 3 traces/lane.
- Next: flag any regressions vs Day-3; recommend nprobe adjustments if needed.

### Acceptance Gates (P4 exit)
- L1 P50 ≤ 85ms, P95 ≤ 200ms, Top-1 ≥ 0.92.
- L2/L3 P50 ≤ 180ms, P95 ≤ 400ms.
- faiss_meta.json truthful (vectors, dim, index_type, nlist).
- CI green on Py 3.11; unit tests (not heavy) green.


[Consultant] Status — 2025-09-23T15:56:33Z
- Reviewed teammate deliverables: README runtime matrix present (`README.md:25`), `docs/architecture.md` expanded with evaluation flags, and new `docs/api_response.md` schema published.
- Programmer tranche still pending: no `artifacts/fw10k_vectors.npz` or `artifacts/fw10k_ivf.index`; `artifacts/faiss_meta.json` still reports 4 vectors (`artifacts/faiss_meta.json:1`).
- Blocked on Day-4 eval runs (`eval/day4_results_dense.jsonl`, `eval/day4_results_lex.jsonl`) and `eval/day4_report.md` because API remains on 1k index.
- Next: rerun eval commands once 10k ingest/index land and API advertises updated metadata.



UPDATE:  Part 2

P4 Status Snapshot (pretty + high-level)
Track	Where we are	What just landed	What’s next (today)	Gate to pass
Runtime	✅ Python 3.11 only	README Runtime Matrix; guards in bootstrap + API	Keep 3.11 in local + CI	CI green on 3.11
Retrieval policy	✅ Locked	L1 dense-only; L2/L3 hybrid; flags doc’d	Keep lexical off for L1 (compare later)	L1 P50 ≤ 85 ms; P95 ≤ 200 ms
LightRAG	✅ Pinned	1.4.9rc1; FAISS = SoT	No action	—
Enums / TMD	✅ Frozen	TMD pack + lane extraction	No action	—
Data scale	🟨 10k pending	1k done; balanced eval set ready	Run 10k canary ingest	fw10k_vectors.npz present
Indexing	🟨 10k pending	CLI + meta writer shipped	Build IVF_FLAT (nlist 128)	faiss_meta.json truthful
API	✅ Wired	Prefers 10k NPZ if present; lexical flag off by default	Point API at new 10k index	3 lane traces saved
Tests / CI	🟨 CI landed	Stubbed light tests; heavy marked	CI runs -m "not heavy"	CI green
Eval	⏳ Blocked on 10k	Day-4 eval plan staged	Run dense vs lexical passes	Day-4 report committed
Programmer — 10k Runbook (approved, with two tiny tweaks)
Why two tweaks? To keep IP search correct with IVF we should L2-normalize embeddings; and we want a predictable default nprobe for 10k.
0) (Optional 1-line patch you can copy)
Add normalize + metric to the FAISS builder (if not already):
# src/faiss_index.py (build step)
# L2-normalize for inner product search
vecs = vecs.astype("float32")
norms = (vecs**2).sum(axis=1, keepdims=True) ** 0.5
vecs = vecs / (norms + 1e-12)
metric = faiss.METRIC_INNER_PRODUCT
index = faiss.index_factory(dim, "IVF128,Flat", metric)
…and in retrieval code set a sane default:
# during search on 10k
index.nprobe = int(os.getenv("FAISS_NPROBE", "16"))
1) Ingest 10k (offline-ok)
TRANSFORMERS_OFFLINE=1 NO_DOCKER=1 bash scripts/ingest_10k.sh
2) Build IVF index (nlist=128)
python -m src.faiss_index \
  --npz artifacts/fw10k_vectors.npz \
  --index-type IVF_FLAT \
  --nlist 128 \
  --out artifacts/fw10k_ivf.index
3) Truthful metadata
python tools/write_faiss_meta.py
cat artifacts/faiss_meta.json
# Expect: {"vectors": ~10000, "dim": 7xx/768, "index_type": "IndexIVFFlat", "nlist": 128, ...}
4) Light tests (CI parity)
LNSP_TEST_MODE=1 pytest -m "not heavy" -q
5) API points at 10k automatically
You already wired “prefer 10k if present.” Do a quick smoke and capture three traces (L1/L2/L3):
uvicorn src.api.retrieve:app --reload
curl -s 'http://127.0.0.1:8000/healthz'
curl -s 'http://127.0.0.1:8000/search?q=What%20is%20FAISS%3F&lane=L1_FACTOID&top_k=5'
curl -s 'http://127.0.0.1:8000/search?q=Explain%20IVF%20vs%20IVF_PQ&lane=L2_PASSAGE&top_k=5'
curl -s 'http://127.0.0.1:8000/search?q=How%20does%20KG%20help%20RAG%3F&lane=L3_SYNTHESIS&top_k=5'
Consultant — Day-4 Eval (runs after the index flips to 10k)
# Dense-only L1
export LNSP_LEXICAL_FALLBACK=0
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_dense.jsonl

# Lexical compare (L1 only)
export LNSP_LEXICAL_FALLBACK=1
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_lex.jsonl
Report: eval/day4_report.md with side-by-side P50/P95 + Top-1 and 3 traces per lane.
Regression callouts: note any deltas vs Day-3; if L1 misses Top-1 ≥ 0.92, bump nprobe to 24 and re-test.
Architect — “hard items” you asked to push their way
These are the non-trivial bits that benefit from a strong hand:
Stable cpe_id spec & hash
Rule: cpe_id = sha1("{dataset}:{doc_id}:{chunk_start}:{chunk_len}:{version}")[:16]
Publish in docs/api_response.md and enforce in the retrieval layer.
FAISS consistency contract
Enforce L2-normalized vectors for IP search at build time.
Add a startup assertion: if vectors < nlist*4 → warn and set nprobe = max(8, nlist//8).
Healthz deep check
/healthz returns: { "py": "3.11.x", "index": "IVF_FLAT", "vectors": 10000, "nlist": 128, "nprobe": 16, "lexical_L1": false }.
LightRAG ingest policy
Doc the “IDs + relations only” pattern; do not duplicate vectors into LRAG.
Add a nightly validator that compares KG edges count to document count and flags skew.
CI gate for metadata truth
A tiny step that fails PRs if faiss_meta.json is missing or inconsistent (e.g., no IndexIVFFlat or vectors < 9000 for the 10k target).
Paste-ready log block (drop in chats/conversation_09232025_P4.md)
## 2025-09-23 — P4 Updates & Go-Forward

### Programmer (approved plan)
- Run 10k ingest → `fw10k_vectors.npz`
- Build IVF_FLAT (nlist=128) → `fw10k_ivf.index`
- Regenerate `faiss_meta.json` (truthful fields)
- CI: `pytest -m "not heavy"` on Py 3.11 (green)
- API now prefers 10k; capture 3 traces (L1/L2/L3) in report

### Consultant
- Execute Day-4 eval (dense-only vs lexical enabled)
- Publish `eval/day4_report.md` (P50/P95, Top-1, 3 traces/lane)

### Architect (hard items assigned)
- Stable `cpe_id` hashing spec & enforcement
- L2-normalization guarantee for IP; startup guards (vectors vs nlist, default nprobe)
- `/healthz` deep payload; LRAG ingest policy; CI meta truth gate

### Gates
- L1: P50 ≤ 85ms, P95 ≤ 200ms, Top-1 ≥ 0.92
- L2/L3: P50 ≤ 180ms, P95 ≤ 400ms
- `faiss_meta.json` truthful; CI green on 3.11