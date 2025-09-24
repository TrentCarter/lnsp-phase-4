# Start of Day Summary

P13 Roll-Up — What works, what’s complex, what’s tested
Area / Component	Works	Complexity	What we tested / Evidence (real data only)	Notes & Links
Embeddings (GTR-T5 768D, offline)	✅	Med	10k vectors encoded from data/teacher_models/gtr-t5-base; L2-normed; NPZ schema validated (vectors/ids/doc_ids/concept_texts/tmd_dense/lane_indices/cpe_ids).	artifacts/fw10k_vectors_768.npz; offline flags set; doc updated.
NPZ Contract & Boot Gates	✅	Low	Startup checks: required keys present, dim=768, shapes consistent, zero-vector kill switch.	docs/architecture.md (NPZ schema); enforced in src/api/retrieve.py.
FAISS Index (IP, 768D, 10k, ID-mapped)	✅	Med	Built with IndexIDMap2, nlist=128, nprobe=16; /admin/faiss reports dim=768, vectors=10k; returns stable doc_id.	artifacts/fw10k_ivf_768.index, artifacts/faiss_meta.json.
Retrieval API (FastAPI, lane-aware)	✅	Med	Server up on localhost:8092; /search 200 + hydrated fields (doc_id, cpe_id, concept_text, tmd_code, lane_index). Smoke tests pass.	tests/test_search_smoke.py; src/api/retrieve.py.
LightRAG Embedder Adapter (offline GTR)	✅	Low	New adapter exposes embed_batch() + dim=768; smoke shows (N,768) float32, ~1.0 norms.	src/adapters/lightrag/embedder_gtr.py, configs/lightrag.yml.
Knowledge Graph Build (LightRAG)	✅	Med	Build runs with real chunk text; exports nodes/edges/stats; Neo4j load path wired.	src/adapters/lightrag/build_graph.py; artifacts/kg/{nodes,edges,stats}.json(l).
GraphRAG Query Pipeline (LightRAG)	✅	High	End-to-end runner: vector top-k → graph slice (PPR/BFS) → prompt pack → LLM call → Postgres + JSONL.	src/adapters/lightrag/graphrag_runner.py; eval/graphrag_runs.jsonl.
Local LLM (Llama, no cloud)	✅	Med	Provider local_llama enforced; metrics captured (latency, bytes in/out); empty responses treated as failures.	src/llm/local_llama_client.py; docs/howto/how_to_access_local_AI.md.
Vec2Text (JXE / IELab, steps=1)	🟨 Ready	Med	Offline embedder path verified; round-trip hooks ready to run with --steps 1.	how_to_use_jxe_and_ielab.md; next: run on 3 real IDs & log cosine.
Eval & Reporting	✅	Med	20-query harness; latency probe; report scaffold populated.	scripts/run_graphrag_eval.sh; eval/day13_graphrag_report.md.
3D Semantic Cloud (real vectors)	✅	Low	PCA 768→3D HTML (Plotly) generated from real embeddings; zero-vector fallback removed.	tools/generate_semantic_cloud.py; artifacts/semantic_gps_cloud_visualization.html.
CI/Hardening	✅	Med	Fail-fast boot, dim checks, zero-vector guards, API smoke; FAISS/NPZ parity checks.	Documented in docs/architecture.md; tests in tests/*.
Docs & Run Logs	✅	Low	Architecture, local model usage, LLM policy; run log updated.	eval/day13_graphrag_report.md; docs/run_log.md.
quick legend
✅ = implemented & exercised on real data
🟨 Ready = path implemented; final evaluation pending
Complexity: an estimate of ongoing maintenance/operational effort (Low/Med/High)

# Added:

CPE → CPESH (new optional fields; no breaking changes):
{
  "concept": "Light-dependent reactions split water",
  "probe": "What process in photosynthesis splits water molecules?",
  "expected": "Light-dependent reactions",
  "soft_negative": "Calvin cycle",
  "hard_negative": "Mitochondrial respiration"
}

[Architect] 
1) Tighten the CPESH prompt (slot-type & span discipline)
Update your CPESH prompt instructions with these two lines:
Answer type constraint: “Make expected the concrete entity that answers the probe (e.g., a person, track title, year). Do not return section labels like ‘Track listing’, ‘Biography’, or categories.”
Span preference: “Prefer an exact span from the provided text. If you cannot justify the exact answer from the text/context, set insufficient_evidence=true and set the field to null.”
(If you want, paste those into your CPESH_EXTRACT section under the bullet rules.)
2) Add a tiny post-filter/regenerator in your generator script
Drop this helper into tools/make_cpesh_quadruplets.py and re-call the LLM up to 2 extra times when the expected is degenerate:
BAD_HEADINGS = {"track listing","discography","overview","biography","reception","plot","references","external links"}

def cpesh_ok(probe, expected, soft, hard):
    if not expected or expected.strip().lower() in BAD_HEADINGS:
        return False
    # basic type hints from probe word
    p = (probe or "").strip().lower()
    if p.startswith("who") and not any(ch.isalpha() for ch in expected):
        return False
    if "track" in p and expected.strip().lower() in BAD_HEADINGS:
        return False
    # negatives must differ from expected
    if soft and soft.strip().lower() == expected.strip().lower():
        return False
    if hard and hard.strip().lower() == expected.strip().lower():
        return False
    return True

# after you get `cpesh = llm.complete_json(...)`:
attempts = 1
while not cpesh_ok(cpesh.get("probe"), cpesh.get("expected"),
                   cpesh.get("soft_negative"), cpesh.get("hard_negative")) and attempts < 3:
    cpesh = llm.complete_json(prompt, timeout_s=15)
    attempts += 1
3) Add a quick quality score (drop weak items)
Right after you embed A/P/S/H:
# simple quality gates
exp_sim = float(vp @ va)                  # expected vs anchor
soft_sim = float(vs @ va) if s else 0.0   # soft vs anchor
hard_sim = float(vh @ va) if h else 0.0   # hard vs anchor

if not (exp_sim >= 0.55 and hard_sim <= 0.35 and (soft_sim < exp_sim)):
    # skip this record; optionally mark why
    continue
That keeps only quadruplets with a sane margin (tweak thresholds as you like).
4) Re-run 10× and spot-check
PYTHONPATH=src \
LNSP_LLM_ENDPOINT=http://localhost:11434 \
LNSP_LLM_MODEL=llama3.1:8b \
python3 tools/make_cpesh_quadruplets.py --n 10 --embed
Quick audit:
python - <<'PY'
import numpy as np, json, itertools as it
from pathlib import Path
print("jsonl sample:")
p = Path("artifacts/train/cpesh_10x.jsonl")
for i, line in zip(range(3), p.open()):
    print(line.strip()[:240]+"…")
Z = np.load("artifacts/train/cpesh_10x_vectors.npz", allow_pickle=True)
for k in ["anchor","positive","soft","hard"]: print(k, Z[k].shape, Z[k].dtype)
print("doc_ids len:", len(Z["doc_ids"]))
PY
5) (Optional) Score the batch and print margins
python - <<'PY'
import numpy as np, json
Z=np.load("artifacts/train/cpesh_10x_vectors.npz", allow_pickle=True)
A,P,S,H = Z["anchor"],Z["positive"],Z["soft"],Z["hard"]
def cos(a,b): return float((a*b).sum())
ok=0
for i in range(len(A)):
    es, ss, hs = cos(A[i],P[i]), cos(A[i],S[i]), cos(A[i],H[i])
    print(i, dict(expected_sim=round(es,3), soft_sim=round(ss,3), hard_sim=round(hs,3)))
    ok += int(es>=0.55 and hs<=0.35 and ss<es)
print("pass:", ok, "/", len(A))
PY