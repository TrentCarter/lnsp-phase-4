✅ Sprint Plan — 092525_S1.md
P1 TIMESTAMP Upgrade (All Roles)
Architect
Freeze timestamp schema policy across layers (Text DB, Vector DB, Graph DB).
Update /docs/architecture.md:
created_at (when extracted), last_accessed (on retrieval).
Manual pruning rules (quality/usage/curation).
Remove any mention of 7-day TTL.
Add /docs/timestamps.md with bit-packing rules + ISO8601 usage.
Programmer
Modify:
src/schemas.py: add created_at, last_accessed fields to CPESH.
src/api/retrieve.py:
On cache miss → set created_at.
On cache hit → update last_accessed.
src/utils/timestamps.py: implement get_iso_timestamp(), parse_iso_timestamp().
Extend artifacts/cpesh_cache.jsonl schema with timestamp + access_count.
Backwards compat: auto-migrate entries missing timestamps.
Optional: patch scripts/init_pg.sql to add columns if cpesh_cache table exists.
Consultant
Add tests/test_cpesh_cache.py:
Verify timestamps written on cache insert.
Verify last_accessed updated on read.
Round-trip parse of ISO8601.
Validate auditability: run ingest + retrieval on 5 chunks, confirm timestamps populate.
Produce /eval/day1_timestamp_report.md with findings (include sample JSON lines).
Carry-Forward Tasks from S5
Consultant: retry evaluation harness after fixing LocalLlamaClient:
Initialize requests.Session() inside complete_json.
Honor LNSP_CPESH_TIMEOUT_S.
Collect latency + Hit@1/3 stats once stable.
Outputs to Repo
/docs/timestamps.md — new doc.
/docs/architecture.md — updated CPESH cache policy.
/src/utils/timestamps.py — new util.
/tests/test_cpesh_cache.py — new test file.
/eval/day1_timestamp_report.md — consultant eval.
/chats/conversation_092525_S1.md — full status log.

🔄 Revised Sprint Plan — 092525_S1.md
P1 TIMESTAMP Upgrade (unchanged)
Architect: freeze schema, add /docs/timestamps.md, remove TTL, document pruning.
Programmer: patch schemas.py, retrieve.py, add utils/timestamps.py, extend JSONL format, ensure backwards compat.
Consultant: write tests/test_cpesh_cache.py, validate timestamp persistence, produce /eval/day1_timestamp_report.md.
➡️ This work proceeds as planned.
Carry-Forward from S5 (UPDATED)
✅ The LocalLlamaClient fixes are already in (session handling, timeout, insufficient-evidence fallback).
✅ Consultant’s tools/run_consultant_eval.py works in-process (no FAISS).
✅ Initial report exists (eval/day_s5_report.md).
So:
Consultant (S1) now has a new task:
Re-run tools/run_consultant_eval.py against the real FastAPI + FAISS stack (not the sandbox).
Compare warm-path latency & Hit@1/3 vs. the in-process baseline.
Annotate /eval/day1_timestamp_report.md with a section: “Live vs. Sandbox Delta”.
Programmer: add a make consultant-eval target in the build (calls the harness with API URL + FAISS enabled) so re-running this test doesn’t require manual steps.
Repo Outputs (Revised)
/docs/timestamps.md — new doc.
/src/utils/timestamps.py — new util.
/tests/test_cpesh_cache.py — new test.
/eval/day1_timestamp_report.md — consultant eval (timestamp focus + “Live vs Sandbox Delta”).
/eval/day_s5_report.md — archived baseline (keep as-is).
/chats/conversation_092525_S1.md — full log.
Makefile — new target for consultant-eval.
