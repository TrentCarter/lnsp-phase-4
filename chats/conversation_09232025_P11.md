P10 Plan (hard issues → [Architect])
[Architect] (hard)
Contract & gates
Default LNSP_FUSED=0 (pure 768D).
/admin/faiss.dim == 768 health gate; reject startup if mismatch.
Ensure /search returns hydrated concept_text, tmd_code, lane_index, and stable cpe_id/doc_id.
Vec2Text policy
Pin “GTR-T5-base (768D) → JXE & IELab vec2text” with --steps 1 by default, --vec2text-backend isolated. Inputs must be 768D, L2-normalized.
Zero-vector kill switch
If any batch has mean L2-norm ≈ 0, abort ingest and fail CI (prevents silent “flat” indices).

Encode 10k (real GTR-T5 768D)
# from repo root, py311 venv active
export LNSP_FUSED=0
./scripts/encode_real_gtr.sh  # writes artifacts/fw10k_vectors_768.npz (non-zero, L2-normed)
Build FAISS (768D, IP)
./scripts/build_faiss_10k_768.sh  # creates artifacts/fw10k_ivf_768.index + faiss_meta.json (dim=768, IP, nlist=128, nprobe=16)
Run API in 768D mode
export LNSP_FUSED=0
.venv311/bin/uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080
# verify
curl -s localhost:8080/admin/faiss | jq .
Vec2Text round-trip (1 step)
The attached guide shows exact flags and env vars (uses 768D, isolated backend, steps=1).
Example:
VEC2TEXT_FORCE_PROJECT_VENV=1 ./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --subscribers jxe,ielab --vec2text-backend isolated --steps 1 \
  --input-list '["light dependent reactions split water","what process splits water?"]' --output-format json
(Guide insists on 768D GTR embeddings and isolated backend; don’t swap models or unify backends.)
3D cloud from real vectors
pip install plotly scikit-learn
python3 tools/generate_semantic_cloud.py  # now uses real 768D/fused data, no zero fallback
open artifacts/semantic_gps_cloud_visualization.html
[Consultant]
Eval @ 768D
export LNSP_FUSED=0
tools/latency_probe.py --base http://127.0.0.1:8080 --iters 200 --out eval/day10_latency.jsonl
tests/test_search_smoke.py  # should pass with non-zero hits
Vec2Text sanity (steps=1) & report
JXE vs IELab outputs should be semantically similar, phrasing different; cosine typically ~0.65–0.85.
“Three enriched examples + vector previews” (REAL data)
After you rebuild with 768D, run this to append three examples to chats/conversation_09232025_P10.md:
python3 - <<'PY'
import json, numpy as np, random, datetime, pathlib
from pathlib import Path

chunks = Path("artifacts/fw10k_chunks.jsonl")
vecs   = np.load("artifacts/fw10k_vectors_768.npz")  # created by encode_real_gtr.sh
doc_ids      = vecs["doc_ids"]
concept_texts= vecs["concept_texts"]
tmd_dense    = vecs["tmd_dense"]      if "tmd_dense" in vecs else None
concept_vecs = vecs["vectors"]         # shape: (N,768)
lane_indices = vecs["lane_indices"]    if "lane_indices" in vecs else None

sample_ix = random.sample(range(concept_vecs.shape[0]), 3)
now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
out = ["\n### [Programmer] Enriched Examples — "+now]

for i in sample_ix:
    cid = str(doc_ids[i])
    ctext = concept_texts[i]
    cv = concept_vecs[i]
    tmd = (tmd_dense[i].tolist() if tmd_dense is not None else None)
    # previews (truncate)
    cv_prev  = np.array2string(cv[:32], precision=4, separator=", ")
    tmd_prev = (str([round(x,3) for x in tmd[:8]]) if tmd is not None else "None")
    out.append(f"""
- ID: {cid}
  Concept: {ctext}
  TMD: {tmd_prev}
  CPE lane_index: {(int(lane_indices[i]) if lane_indices is not None else 'NA')}
  768D preview (first 32): {cv_prev}""")

Path("chats/conversation_09232025_P10.md").write_text(
    Path("chats/conversation_09232025_P10.md").read_text() + "\n".join(out)
)
print("Appended 3 examples.")
PY
That writes: (1) ID; (2) human-readable Concept; (3) TMD (first 8 values), lane index; (4) the real 768D preview (first 32).
If you also store CPE fused 784D, you can add a second preview block for fused_vec[:32].
“Query by ID and show 768D/TMD/CPE (truncated)”
Once IDs are known, this prints the real vectors for a given CPE_ID:
python3 - <<'PY'
import numpy as np, sys, uuid
npz = np.load("artifacts/fw10k_vectors_768.npz")
cid = sys.argv[1] if len(sys.argv)>1 else str(npz["doc_ids"][0])
i = int(np.where(npz["doc_ids"]==cid)[0][0])
v768 = npz["vectors"][i]
print("ID:", cid)
print("768D (first 32):", np.array2string(v768[:32], precision=4, separator=", "))
if "tmd_dense" in npz:
    print("TMD16 (all):", np.array2string(npz["tmd_dense"][i], precision=3, separator=", "))
if "fused" in npz:
    print("Fused784 (first 32):", np.array2string(npz["fused"][i][:32], precision=4, separator=", "))
PY  enwiki-00000000-0000-0000
