🚀 Sprint Plan — 092525_S2.md
[Architect]
Finalize Runtime Environment Options
Document FAISS deployment strategies in /docs/runtime_env.md:
Option A: Rebuild FAISS wheel for Python 3.11 arm64.
Option B: Run API under Python 3.9 (x86_64 via Rosetta if macOS, or containerized).
Option C: Use FAISS nightly with Conda-forge if available.
Specify the project’s baseline runtime stack (Python 3.11 + arm64, with fallback guidance).
Update /docs/architecture.md to add “Evaluation Stack Requirements” (FAISS support required for consultant harness).
Pruning Policy Extension
Draft /docs/pruning_policy.md: formal rules for manual pruning:
Low echo_score (<0.82).
Low access_count over N days.
Explicit curator overrides.
Define a JSON schema for pruning manifests (list of CPE_IDs to delete).
[Programmer]
FAISS Compatibility Fix
Attempt to rebuild FAISS wheel for local stack (arm64 + Python 3.11).
If blocked, add a fallback build script:
scripts/setup_faiss_env.sh → provisions a working FAISS environment (e.g., via Conda or Docker).
API Enhancements
Add /health/faiss endpoint to src/api/retrieve.py to report FAISS index status (loaded, trained, size).
Add cache introspection endpoint /cache/stats showing:
total_entries, oldest created_at, most_recent last_accessed.
Pruning Hooks
Add scripts/prune_cache.py:
Reads a pruning manifest JSON.
Removes selected CPESH entries from artifacts/cpesh_cache.jsonl.
Optionally syncs removals into Postgres/Neo4j if enabled.
[Consultant]
Live Evaluation Unblock
Once FAISS stack is working, re-run make consultant-eval end-to-end:
Collect cold/warm latency (P50/P95).
Collect Hit@1/3 with quality weighting vs. baseline.
Log CPESH deltas into eval/day2_eval_report.md.
Cache/Pruning Validation
Add tests in tests/test_pruning.py:
Verify that prune_cache removes targeted entries.
Verify cache size drops, and timestamps remain valid.
Reporting
Generate /eval/day2_eval_report.md:
Live vs. sandbox latency/quality results.
Pruning simulation (before/after cache snapshot).
Recommendations for tightening mission prompts or cache strategies.
Outputs to Repo
/docs/runtime_env.md — FAISS environment fixes & runtime baselines.
/docs/pruning_policy.md — pruning rules & schema.
/scripts/setup_faiss_env.sh — FAISS environment bootstrap.
/scripts/prune_cache.py — pruning utility.
/tests/test_pruning.py — pruning test coverage.
/eval/day2_eval_report.md — consultant report.
/chats/conversation_092525_S2.md — full sprint log.