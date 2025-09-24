P10 Plan — “Real Data On”
[Architect] (hard issues)
A1 — Dimensional contract switch: add a single env flag LNSP_FUSED=0|1. For P10 set LNSP_FUSED=0 → system runs pure 768D. Keep doc noting how to flip back to 784D later.
A2 — Metric & gates: confirm FAISS metric=IP, ntotal=10k, dim=768, nlist=128, nprobe=16. Health gate: /admin/faiss.dim==768, smoke /search returns ≥1 hit.
A3 — Schema/IO: retrieval payload must include stable doc_id/cpe_id, and hydrated Concept/TMD/CPE (no nulls).
[Programmer]
P1 — Real GTR encoder path (768D): implement src/encoders/gtr768.py using sentence-transformers; write scripts/encode_real_gtr.sh to populate artifacts/fw10k_vectors.npz with real values (no zeros).
P2 — Rebuild FAISS (768D): scripts/build_faiss_10k_768.sh → artifacts/fw10k_ivf.index (IP, nlist=128). Update /admin/faiss to reflect 768D.
P3 — API dim toggle: src/api/retrieve.py uses LNSP_FUSED to decide dim=768 vs 784. For LNSP_FUSED=0, bypass TMD fuse and normalize 768D only.
P4 — 3D cloud regeneration: update tools/generate_semantic_cloud.py to refuse to run if vectors are all-zeros; regenerate HTML from real vectors only.
[Consultant]
C1 — 20-item eval on real data: run dense (LNSP_LEXICAL_FALLBACK=0), then optional lexical retry; collect Hit@1/3 and P50/P95.
C2 — Vec2Text demo (1-step): run JXE and IELab decoders with --steps 1 on 3 real IDs (round-trip string ↔ 768D); log cosine.
C3 — Day10 report: /eval/day10_report.md with metrics, three retrieval examples, three vec2text examples, and the real 3D cloud screenshot.
Source for the vec2text usage you attached is here .
Real-data workflow (copy/paste)
Assumes your 10k FactoidWiki chunks are in artifacts/fw10k_chunks.jsonl and contain doc_id, text/concept, and TMD/CPE fields.
Set P10 mode (pure 768D, no fuse)
export LNSP_FUSED=0        # 0 → 768D, 1 → 784D fuse
export LNSP_EMBED_MODEL=sentence-transformers/gtr-t5-base
Encode real GTR embeddings (no stubs)
# scripts/encode_real_gtr.sh
. .venv/bin/activate
python - <<'PY'
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
root = Path(".")
chunks = root/"artifacts/fw10k_chunks.jsonl"
out = root/"artifacts/fw10k_vectors.npz"

model = SentenceTransformer("sentence-transformers/gtr-t5-base")
ids, vecs = [], []
with chunks.open() as f:
    for line in f:
        rec = json.loads(line)
        did = rec.get("doc_id") or rec.get("id")
        txt = rec.get("concept") or rec.get("text") or ""
        if not did or not txt.strip(): 
            continue
        v = model.encode([txt], normalize_embeddings=True)[0]  # (768,), L2-normalized
        ids.append(int(did)); vecs.append(v.astype("float32"))
emb = np.stack(vecs, axis=0)
ids = np.array(ids, dtype=np.int64)
np.savez(out, emb=emb, ids=ids)
print("Wrote:", out, emb.shape, ids.shape, "zero_count:", int((emb==0).all(axis=1).sum()))
PY
Build FAISS index (IP, 768D)
# scripts/build_faiss_10k_768.sh
. .venv/bin/activate
python - <<'PY'
import numpy as np, faiss, json
from pathlib import Path
npz = np.load("artifacts/fw10k_vectors.npz")
emb = npz["emb"].astype("float32")   # (N,768) normalized
N, D = emb.shape
assert D == 768 and N > 0
nlist = 128
quant = faiss.IndexFlatIP(D)
index = faiss.IndexIVFFlat(quant, D, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(emb)
index.add(emb)
index.nprobe = 16
faiss.write_index(index, "artifacts/fw10k_ivf.index")
meta = {"dim": int(D), "ntotal": int(index.ntotal), "metric":"IP", "nlist": nlist, "nprobe": int(index.nprobe)}
Path("artifacts/faiss_meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
PY
Switch API to 768D at query time
In src/api/retrieve.py, set the embed path to GTR-only when LNSP_FUSED=0; make /admin/faiss report dim=768.
Smoke:
. .venv/bin/activate
uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080

curl -s http://127.0.0.1:8080/admin/faiss | jq .
# expect: {"dim":768,"nlist":128,"nprobe":16,"metric":"IP","vectors":10000}

curl -s -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"q":"Which ocean is largest?","top_k":5,"lane":"L1_FACTOID"}' | jq .
Regenerate the 3D cloud (real vectors only)
python3 tools/generate_semantic_cloud.py
open artifacts/semantic_gps_cloud_visualization.html
Tip: have the tool hard-fail if it detects all-zero rows; do not silently randomize.
Vec2Text (1 step) round-trip on 3 real IDs (JXE + IELab)
Use your attached guide to run the decoders with --steps 1 and --vec2text-backend isolated (to guarantee separate wrappers).
Minimal example:
VEC2TEXT_FORCE_PROJECT_VENV=1 ./.venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "Which ocean is largest?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1
Also test vector→text by feeding one of your stored 768D vectors (tool supports reading a single vector from file or inline—follow the doc). Log cosine similarity in /eval/day10_report.md.
Why you saw no clusters
Your log shows the tool detected all-zero vectors and substituted random noise. That yields an isotropic blob (no clusters). After step (2) above, your PCA plot will reflect real semantic structure.
Appending notes to /chats/conversation_09232025_P9.md
Since I can’t push, here’s what to paste at the end of that file to record P10 kickoff and the “real-data only” rule:
### [Owner] P10 Kickoff — REAL DATA ONLY — 2025-09-23T22:40Z
- Rule: Never use fake/simulated/sample data in outputs or visualizations.
- Mode: LNSP_FUSED=0 → 768D GTR-T5 only (vec2text with --steps 1).
- Actions:
  1) Run scripts/encode_real_gtr.sh to populate artifacts/fw10k_vectors.npz with real 768D.
  2) Build FAISS via scripts/build_faiss_10k_768.sh (IP, nlist=128, nprobe=16).
  3) Switch API to dim=768; verify /admin/faiss.dim==768 and /search returns hits.
  4) Regenerate 3D cloud from real vectors; attach screenshot to eval/day10_report.md.
  5) Run JXE & IELab vec2text with --steps 1 on 3 real IDs; log cosine metrics.
### [Consultant] Status — 2025-09-23T22:08Z
- C1 (20-query eval) blocked: faiss_meta still advertises dim=784 and vectors remain zero; dense search would be non-informative. Confirmed via `.venv311/bin/python - <<'PY' ...` (all 10k norms = 0) and `faiss.read_index` (`d=784`).
- C2 (Vec2Text 1-step) blocked: no real 768D inputs to decode; existing vectors are 0 and violate "real data only" rule.
- C3 (Day10 report & cloud) blocked: `tools/generate_semantic_cloud.py` aborts immediately—missing Plotly deps and real embeddings guard.

Evidence
- `artifacts/fw10k_vectors.npz`: keys=`['vectors','cpe_ids','lane_indices','concept_texts','doc_ids']`; `vectors.shape=(10000,784)` with all-zero rows.
- `artifacts/faiss_meta.json`: still `dimension=784`, contradicts LNSP_FUSED=0 contract.
- `.venv311/bin/python tools/generate_semantic_cloud.py` → `ModuleNotFoundError: plotly` before hitting the zero-vector guard.

Next Steps Needed
1. Programmer: run real `scripts/encode_real_gtr.sh` (768D, normalized) and rebuild FAISS via `build_faiss_10k_768.sh`, updating `faiss_meta.json` (dim=768, metric=IP).
2. Architect: flip runtime defaults to `LNSP_FUSED=0`, supply dependencies (plotly, sentence-transformers) in the py311 env, and refresh `/admin/faiss` health gate.
3. Consultant can rerun C1–C3 once the above artifacts exist; expect dense Hit@1/P95 metrics and screenshot-ready 3D cloud.
