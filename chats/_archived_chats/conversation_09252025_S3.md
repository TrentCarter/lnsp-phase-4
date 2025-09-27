🚀 Sprint Plan — 092525_S3.md (FactoidWiki → LNSP @ 10k curated chunks)
Objectives (end of S3 = “great” bar)
10k ingest complete with stable CPESH caching (timestamps + access_count already in place).
FAISS dial-plan validated (IVF params + nprobe) with live Hit@1/3 and latency P50/P95 on the FastAPI stack.
Pruning pipeline operational (policy → manifest → prune → verify) and measurable cache quality lift.
Ops Observability: health endpoints + cache stats + index telemetry exposed and tested.
Prompt template finalized with a quick iterative eval loop (echo pass & lane dist snapshots).
[Architect] — Specs, SLOs, and Dial-Plan (done = merged docs + testable targets)
1) 10k Ingest & Index Dial-Plan (docs/architecture.md + docs/runtime_env.md)
Lock initial FAISS plan for 10k vectors:
Phase A (baseline): IVF_FLAT, nlist=512, nprobe ∈ {8, 16, 24}.
Phase B (scale): switchable to IVF_PQ (m=8, nbits=8) at ≥100k.
Define acceptance SLO gates:
Retrieval: Hit@1 ≥ 45%, Hit@3 ≥ 55% on 100-query set (FactoidWiki curated).
Latency with warm cache: P50 ≤ 80 ms, P95 ≤ 450 ms at nprobe ≤ 16.
Document training set rule of thumb (≥40×nlist items; warn when violated).
2) Pruning Policy Final
/docs/pruning_policy.md:
Default rules: echo_score < 0.82 OR access_count == 0 over 14 days OR curator list.
Allow lane-aware overrides (L1_FACTOID stricter than Lx).
Manifest JSON schema (id, reason, timestamp, author).
3) Prompt Template Freeze (docs/prompt_template.md)
Finalize CPESH extraction prompt (low hallucination, high structure).
Include insufficient-evidence guardrail phrasing (now supported by client).
Deliverables
docs/architecture.md (S3 updates), docs/runtime_env.md (confirmed stack),
docs/pruning_policy.md, docs/prompt_template.md.
[Programmer] — Code & Ops (done = code + tests pass + make targets green)
1) Ingest & Index @10k
Update scripts/ingest_10k.sh (new): batch over curated set (10k), idempotent, resumable.
Enhance src/faiss_index.py:
Flags: --metric ip|l2, --nlist 512, --nprobe 8, --type ivf_flat|ivf_pq, --pq-m 8 --pq-nbits 8.
Emit telemetry: vectors, trained, nlist, code size, build time → artifacts/index_meta.json.
2) Observability Endpoints
src/api/retrieve.py:
GET /health/faiss → {loaded, type, metric, nlist, nprobe, ntotal}.
GET /cache/stats → {entries, oldest_created_at, newest_last_accessed, p50_access_age, top_docs_by_access}.
GET /metrics/slo → last run Hit@1/3 + latency summaries (filled by Consultant post-run).
3) Pruning Utilities
scripts/prune_cache.py:
Input: --manifest eval/prune_manifest.json.
Output: artifacts/cpesh_cache_pruned.jsonl + diff stats; atomic swap on success.
Optional: --dry-run with report to eval/prune_report.md.
4) Makefile Targets
make ingest-10k → runs scripts/ingest_10k.sh.
make build-faiss → builds index with current dial-plan flags.
make api → uvicorn with the right PYTHONPATH.
make prune → applies manifest; writes report & backs up old cache.
make slo-snapshot → queries endpoints and writes eval/snapshots/s3_$(date).json.
5) Guardrails / Footguns
Add a shell-safe banner to scripts: never echo comments into terminals.
Exit non-zero if n_train < 40*nlist during index train; print a corrective hint.
Tests
tests/test_index_meta.py (validate index_meta.json keys & plausibility).
tests/test_health_endpoints.py (smoke checks for /health/faiss, /cache/stats).
[Consultant] — Evaluation & Tuning (done = reproducible eval + deltas + recommendations)
1) Live Evaluation @10k
Extend tools/run_consultant_eval.py:
Accept --api http://127.0.0.1:8092, --nprobe {8,16,24}, --metric {ip,l2}.
Log per-query: lane, latency (cold/warm), hits, and whether CPESH used.
Produce eval/day_s3_report.md with:
Table: nprobe vs Hit@1/3 vs P50/P95 latency.
Recommendation: final nprobe for SLO adherence.
Cache impact: % CPESH hits; new entries count; example good/bad CPESH.
2) Pruning Experiment
Generate eval/prune_manifest.json (rule-based from S2/S3 data).
Run make prune, re-index if needed, and re-run eval.
Add before/after section to eval/day_s3_report.md with quality deltas.
3) SLO Snapshot & API Sanity
Capture /health/faiss, /cache/stats, /metrics/slo post-eval via make slo-snapshot.
Confirm timestamps are advancing on cache hits (spot-check).
Concrete Command Plan (end-to-end happy path)
# Build 10k vectors + index (baseline IVF_FLAT)
make ingest-10k
make build-faiss ARGS="--type ivf_flat --metric ip --nlist 512 --nprobe 8"

# Start API
make api

# Live eval sweep
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py \
  --api http://127.0.0.1:8092 --nprobe 8 --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py \
  --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py \
  --api http://127.0.0.1:8092 --nprobe 24 --metric ip --queries eval/100q.jsonl

# Snapshot SLOs after the best setting
make slo-snapshot

# Pruning pass (if needed)
python scripts/prune_cache.py --manifest eval/prune_manifest.json --dry-run
make prune
make build-faiss
PYTHONPATH=src ./.venv/bin/python tools/run_consultant_eval.py \
  --api http://127.0.0.1:8092 --nprobe 16 --metric ip --queries eval/100q.jsonl
Acceptance Checklist (merge gate for S3)
 10k entries visible in /health/faiss (ntotal ≥ 10,000).
 Hit@1 ≥ 45%, Hit@3 ≥ 55% on curated 100-query set.
 P50 ≤ 80 ms, P95 ≤ 450 ms at recommended nprobe.
 eval/day_s3_report.md contains sweep table + final recommendation.
 Pruning pipeline executed at least once with before/after metrics.
 /cache/stats shows plausible timestamp drift and top hits; SLO snapshot saved.
Risks & Mitigations
Under-trained IVF (warning you’ve seen: too few training points): enforce the 40×nlist rule and auto-reduce nlist when violated.
Quality variance post-prune: begin with dry-run and keep automatic backups of JSONL cache.
Prompt brittleness: lock prompt_template.md, add 3-item ablation in day_s3_report.md if Hit@1 stalls.

---

🔍 Consultant Execution Log — 2025-09-25T15:42Z
1) Live Evaluation @10k — Blocked. API + 10k FAISS index not running; prior attempts still hit FAISS segfault until index build completes. No fresh eval data written.
2) Pruning Experiment — Blocked pending successful live eval and presence of eval/prune_manifest.json; command not run.
3) SLO Snapshot & API Sanity — Blocked because live API unavailable; cannot query /health or /metrics endpoints.

Notes:
- Ran `make faiss-setup ARGS="--strategy conda --py 3.11"`; strategy logged (conda) in artifacts/runtime.txt.
- Further progress requires completing ingest/build steps under the Programmer track and standing up the API before Consultant sweeps can proceed.
