[P6] Conversation Plan — Day-6: 10k “Green” sign-off + 100k Preflight
Attendees: [Architect] [Programmer] [Consultant]
Goal: finalize 10k acceptance (latency + quality + truthy meta), publish Day-4 eval, and prep knobs/docs for 100k.
0) Snapshot (inputs to P6)
10k artifacts live: fw10k_vectors.npz, fw10k_ivf.index (IVF_FLAT, nlist=128), L2-normalized ✅
faiss_meta.json truthful ✅ (vectors=10000, dim=784, IndexIVFFlat, nlist=128, nprobe=16)
/healthz deep payload & CI meta truth gate ✅
L1 dense-only default; lexical = flag ✅
Tests: “light” suite stable on Py 3.11; heavy suite local ✅
1) Decisions to ratify (no debate)
L1 = dense-only for production; lexical remains a diagnostic toggle.
10k defaults: nlist=128, nprobe=16, dim=784 (post-TMD), IP + L2-norm.
Ports: API fixed at 8080 for contract tests; Neo4j 7687/7474; Postgres 5432.
2) Acceptance gates (P6 exit = “10k Green”)
L1 (dense-only): P50 ≤ 85 ms, P95 ≤ 200 ms, Top-1 ≥ 0.92 (eval-20).
L2/L3 (hybrid): P50 ≤ 180 ms, P95 ≤ 400 ms.
Truthfulness: /healthz and faiss_meta.json match (vectors, dim, index_type, nlist, nprobe).
Docs: Ports Map committed to docs/architecture.md (as above).
CI: Py 3.11 light tests green; meta truth gate passing.
3) Role-scoped actions
[Architect]
Add Ports Map to docs/architecture.md (paste block above).
100k dial-plan one-pager:
Start IVF_FLAT, nlist=512, nprobe=32.
If latency > SLO, try IVF_PQ m=8, nbits=8 (train ≥2× corpus).
Gate to proceed: L1 P50 ≤ 95 ms with nprobe ≤ 24 on eval-20.
Deliverables:
Updated docs/architecture.md (Ports Map + dial-plan).
[Programmer]
Admin endpoint (tiny): add /admin/faiss returning {nlist,nprobe,metric,dim,vectors} (if not already).
Latency probe: run quick 50-iter per lane and emit stats file.
Commands:
# Start API on 8080
uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080

# 50 iters/lane latency sampling (writes eval/day6_latency_traces.jsonl)
python tools/latency_probe.py --base http://127.0.0.1:8080 --iters 50
Deliverables:
eval/day6_latency_traces.jsonl and P50/P95 pasted into chats log.
/admin/faiss response pasted once (for record).
[Consultant]
Publish Day-4 eval on 10k (now unblocked):
export LNSP_LEXICAL_FALLBACK=0
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_dense.jsonl

export LNSP_LEXICAL_FALLBACK=1
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day4_results_lex.jsonl
Report: eval/day4_report.md with side-by-side P50/P95, Top-1, and 3 traces per lane (include cpe_id + support_ids).
If L1 Top-1 < 0.92, re-run at FAISS_NPROBE=24 and update report.
Deliverables:
eval/day4_results_{dense,lex}.jsonl, eval/day4_report.md.
4) Risks & mitigations
Port drift → Ports Map + .env keys + contract tests on 8080.
nprobe changes → /healthz and /admin/faiss expose current values; set via env.
KG offline → LRAG “IDs+relations only” policy; vectors remain FAISS; ingest logs warn but do not block.
5) Paste-ready log block (drop in chats/conversation_09232025_P6.md)
## 2025-09-23 — [P6] Day-6: 10k Green & 100k Preflight

### Ratified
- L1 dense-only; L2/L3 hybrid.
- 10k IVF defaults: nlist=128, nprobe=16; dim=784; IP + L2-normalized.
- API on port 8080; Postgres 5432; Neo4j 7687/7474.

### [Architect]
- Ports Map committed to docs/architecture.md.
- 100k dial-plan added.

### [Programmer]
- /admin/faiss returns {nlist,nprobe,metric,dim,vectors}.
- Ran 50-iter latency probe per lane, wrote eval/day6_latency_traces.jsonl.
- P50/P95 pasted below.

### [Consultant]
- Ran eval-20 on 10k (dense vs lexical); published eval/day4_report.md with 3 traces/lane.

### Acceptance Gates (must pass)
- L1 P50 ≤ 85ms, P95 ≤ 200ms, Top-1 ≥ 0.92.
- L2/L3 P50 ≤ 180ms, P95 ≤ 400ms.
- faiss_meta.json & /healthz aligned; CI green.

### Snapshot (/healthz)
- py=3.11.x, index=IndexIVFFlat, vectors=10000, nlist=128, nprobe=16, lexical_L1=false[Consultant] eval_runner: 2025-09-23T14:22:38 — total=20 pass=6 echo=30.0% results=eval/day4_results_dense.jsonl
[Consultant] eval_runner: 2025-09-23T14:23:08 — total=20 pass=9 echo=45.0% results=eval/day4_results_lex.jsonl

## 2025-09-23 — [P6] Day-6: 10k Green & 100k Preflight

### Ratified
- L1 dense-only; L2/L3 hybrid.
- 10k IVF defaults: nlist=128, nprobe=16; dim=784; IP + L2-normalized.
- API on port 8080; Postgres 5432; Neo4j 7687/7474.

### [Architect] ✅ COMPLETED
- Ports Map committed to docs/architecture.md.
- 100k dial-plan added.

**Deliverables:**
- Updated docs/architecture.md with standardized Ports Map (P6)
- Added 100k Scaling Dial-Plan section with IVF_FLAT→IVF_PQ progression
- Performance SLO gates and scaling strategy documented

### [Consultant] Status — 2025-09-23T18:23:56Z
- Dense pass rerun with `FAISS_NPROBE=24`, `LNSP_LEXICAL_FALLBACK=0` via in-process TestClient (`OMP_NUM_THREADS=1`); results captured in `eval/day4_results_dense.jsonl:1` (echo 30 %, L1 Hit@1 0 %, P50 22.68 ms, P95 39.17 ms).
- Lexical comparison rerun with same nprobe and flag=1; metrics stored in `eval/day4_results_lex.jsonl:1` (echo 45 %, L2/L3 Hit@1 ≥ 0.50/0.29, mean latency 31.4 ms, L3 P95 92.9 ms outlier).
- Updated `eval/day4_report.md:1` with lane table (dense vs lexical), cpe_id-inclusive samples (support_ids empty), Faiss metadata snippet, and triaged misses citing cpe→doc collapse.
- Acceptance gate blocked: L1 Top‑1 remains ≪ 0.92 because dense shortlist always returns doc_id `enwiki-00000002-0001-0000` (cpe `cd877d60-daf5-5817-a81a-7ee398070916`) across queries despite nprobe bump.

### [Consultant] Next
- Restore sentence-transformer weights / pooling config and rehydrate embeddings so dense encoder regains discrimination; rerun eval targeting L1 Hit@1 ≥ 0.92.
- Audit 10k ingest metadata (cpe_id → doc_id) to eliminate repeated doc IDs before the next gate attempt.

### [Programmer] Results — 2025-09-23T18:34:41Z
- /admin/faiss (server=127.0.0.1:8002):

```json
{
  "nlist": 128,
  "nprobe": 16,
  "metric": "IP",
  "dim": 784,
  "vectors": 10000
}
```

- 50-iter per-lane latency probe (server=127.0.0.1:8080, iters=50/lane)
  - L1_FACTOID: P50=4.60 ms, P95=55.96 ms, n=50
  - L2_GRAPH:   P50=4.52 ms, P95=22.28 ms, n=50
  - L3_SYNTH:   P50=4.49 ms, P95=22.99 ms, n=50
  - Overall:    P50=4.53 ms, P95=22.99 ms, n=150

- Traces written: `eval/day6_latency_traces.jsonl`
- Note: Port 8080 was already running prior to the admin endpoint addition; admin payload captured from 8002 (lightweight endpoint).
