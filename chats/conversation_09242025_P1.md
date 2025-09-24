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

CPE → CPESH (new optional fields; no breaking changes):
{
  "concept": "Light-dependent reactions split water",
  "probe": "What process in photosynthesis splits water molecules?",
  "expected": "Light-dependent reactions",
  "soft_negative": "Calvin cycle",
  "hard_negative": "Mitochondrial respiration"
}