[P7] Conversation Plan — Day-7: 10k “Green” unblock + hardening
Attendees: [Architect] [Programmer] [Consultant]
Goal: fix dense encoder + doc-ID collapse, unify /admin/faiss on 8080, re-run eval to pass 10k gates, and freeze the 100k preflight knobs.
0) Quick triage from P6
L1 Hit@1 = 0% even at nprobe=24 → almost certainly running on stub/zero-variance embeddings or a doc-ID collapse in the 10k ingest.
/admin/faiss hang → old server on 8080 lacked the route; adding timeouts helped, but we should restart 8080 with the new code so admin + health are consistent.
FAISS logs show repeated warnings during hot reload; index still loads with vectors=10000, nlist=128.
1) Decisions to ratify (no debate)
Unify API at 8080 with the lightweight /admin/faiss (no heavy init). Stop any shadow server on 8001/8002 after verification.
Dense encoder must be real for acceptance runs (no stub). If offline, explicitly fail the eval rather than silently stubbing.
2) Acceptance gates (P7 exit = “10k Green”)
L1 (dense-only): P50 ≤ 85 ms, P95 ≤ 200 ms, Top-1 ≥ 0.92 (eval-20).
L2/L3 (hybrid): P50 ≤ 180 ms, P95 ≤ 400 ms.
Truthfulness: /healthz and faiss_meta.json agree (vectors, dim=784, index=IndexIVFFlat, nlist, nprobe).
Admin: /admin/faiss responsive on 8080 in <50ms and matches meta.
3) Role-scoped actions
[Architect]
Embedder contract (one-pager)
Specify production model (name + revision), pooling (mean), norm (L2), target dim → pre-fusion 768 → fused 784.
Add an artifact stamp artifacts/emb_meta.json:
{"model":"sentence-transformers/gtr-t5-base",
 "revision":"<hf-rev>",
 "pooling":"mean",
 "normalized":true,
 "base_dim":768,
 "fused_dim":784,
 "created":"<iso8601>"}
Fail-fast policy
If model files missing or offline: abort vectorization (no stub for acceptance builds).
Deliverables: docs/architecture.md updated (Embedder Contract + Fail-fast).
[Programmer]
Kill stray servers; unify on 8080
lsof -i :8080 -sTCP:LISTEN -n -P
kill -9 <pid>  # if needed
.venv311/bin/uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080
Verify:
curl --max-time 5 -s http://127.0.0.1:8080/healthz
curl --max-time 5 -s http://127.0.0.1:8080/admin/faiss | jq .
Doc-ID collapse audit (new quick tool)
.venv311/bin/python - <<'PY'
import json, collections, sys
from pathlib import Path
p=Path("artifacts/fw10k_chunks.jsonl")
doc_counts=collections.Counter(); bad=0; n=0
for line in p.open():
    n+=1
    rec=json.loads(line)
    d=rec.get("doc_id") or rec.get("meta",{}).get("doc_id")
    if not d: bad+=1
    else: doc_counts[d]+=1
print({"total":n,"missing_doc_id":bad,"unique_doc_ids":len(doc_counts),
       "top5":doc_counts.most_common(5)})
PY
If one doc dominates (e.g., enwiki-00000002-0001-0000), fix mapping in src/ingest_factoid.py (ensure you’re not reusing a constant doc_id, and that cpe_id is derived per-chunk using the SHA1 formula you documented).
Rebuild real embeddings (no stub)
# Ensure internet access or pre-cached HF weights
unset TRANSFORMERS_OFFLINE
export ST_MODEL=sentence-transformers/gtr-t5-base
python -m src.vectorizer --input artifacts/fw10k_chunks.jsonl --out artifacts/fw10k_vectors.npz
python -m src.faiss_index --npz artifacts/fw10k_vectors.npz \
  --index-type IVF_FLAT --nlist 128 --out artifacts/fw10k_ivf.index
python tools/write_faiss_meta.py && cat artifacts/faiss_meta.json
Write artifacts/emb_meta.json with the fields above (can be emitted by vectorizer.py).
Latency probe (fresh)
.venv311/bin/python tools/latency_probe.py --base http://127.0.0.1:8080 \
  --iters 50 --out eval/day7_latency_traces.jsonl
Deliverables:
Doc-audit summary pasted into chats (counts + top5).
Fresh fw10k_vectors.npz/fw10k_ivf.index + faiss_meta.json + emb_meta.json.
Latency summary (P50/P95 overall + per lane).
[Consultant]
Re-run eval-20 on 10k (dense vs lexical)
export LNSP_LEXICAL_FALLBACK=0
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day5_results_dense.jsonl

export LNSP_LEXICAL_FALLBACK=1
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day5_results_lex.jsonl
Publish Day-5 report eval/day5_report.md
Side-by-side: P50/P95, Hit@1 per lane; 3 traces/lane (include cpe_id, support_ids).
If L1 Hit@1 < 0.92: try FAISS_NPROBE=24 once; if still low, flag to Architect (model/pooling mismatch).
Deliverables: eval/day5_results_{dense,lex}.jsonl, eval/day5_report.md.
4) Risks & mitigations
Stub leakage → fail-fast policy + emb_meta.json stamp; CI check refuses “model=stub”.
Doc-ID collapse → audit tool must show unique_doc_ids≈10k; CI red if < 70% uniqueness.
Route drift → enforce /admin/faiss+/healthz on 8080; kill stray UVicorns before probes.
5) Paste-ready log block (drop in chats/conversation_09232025_P7.md)
## 2025-09-23 — [P7] Day-7: 10k “Green” unblock + hardening

### Ratified
- API unified on 8080 with lightweight /admin/faiss.
- Acceptance runs must use real dense encoder (no stub).
- Fail-fast if embeddings unavailable.

### [Programmer] Actions
- Kill stray servers; restart API on 8080; verify /healthz + /admin/faiss.
- Run doc-ID audit and paste counts (total, missing, unique, top5).
- Rebuild real embeddings and index; regenerate faiss_meta.json + emb_meta.json.
- Run latency probe (50 iters/lane) → write eval/day7_latency_traces.jsonl; paste P50/P95.

### [Consultant] Actions
- Re-run eval-20 (dense vs lexical) on 10k; publish eval/day5_report.md with 3 traces/lane.

### Acceptance Gates (P7 exit)
- L1 P50 ≤ 85ms, P95 ≤ 200ms, Hit@1 ≥ 0.92.
- L2/L3 P50 ≤ 180ms, P95 ≤ 400ms.
- /admin/faiss & faiss_meta.json aligned; CI green.

### Notes from P6
- Prior L1 Hit@1=0% traces implicated stub embeddings and doc-ID collapse.
- Admin route fixed; use 8080 going forward.

## 2025-09-23 — [P7] Day-7: 10k "Green" unblock + hardening

### Ratified
- API unified on 8080 with lightweight /admin/faiss.
- Acceptance runs must use real dense encoder (no stub).
- Fail-fast if embeddings unavailable.

### [Architect] ✅ COMPLETED
- Embedder Contract committed to docs/architecture.md.
- Fail-fast policy documented.

**Deliverables:**
- Added Embedder Contract (P7) section to docs/architecture.md:
  - Production model specification: sentence-transformers/gtr-t5-base
  - Mean pooling, L2 normalization, 768D→784D fusion
  - Metadata schema for artifacts/emb_meta.json
  - Fail-fast policy: abort vectorization if model missing/offline
- Created artifacts/emb_meta.json template with required fields
- CI verification requirement: reject builds with model="stub"

### [Consultant] Status — 2025-09-23T19:15:44Z
- Ran dense eval (`LNSP_LEXICAL_FALLBACK=0`) via `.venv311/bin/python -m src.eval_runner --queries eval/day3_eval.jsonl --api http://127.0.0.1:8080/search --top-k 5 --timeout 15 --out eval/day5_results_dense.jsonl`; all 20 calls returned HTTP 500 and produced zero hits.
- Follow-up lexical retry (`LNSP_LEXICAL_FALLBACK=1`) could not connect; the server refused connections after the dense failure, so no metrics were captured.
- Attempts to reproduce in-process (`get_context().search(...)`) triggered a FAISS segmentation fault, confirming the API crash path.
- Deliverables published: `eval/day5_results_dense.jsonl`, `eval/day5_results_lex.jsonl`, and `eval/day5_report.md` outlining error traces and next steps.
- Blocked on Programmer/Architect fixes for the FAISS index + embedder contract before another evaluation can meet the Day-7 acceptance gates.
