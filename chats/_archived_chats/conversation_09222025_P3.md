Day 2 Part 3
9/22/2025


# Day 2 — Part 3 (late)

## [Architect] Tasks
- ✅ Confirm enums freeze reflected in `/src/enums.py` and referenced by `tmd_encoder.py`.
- ✅ Verify NO_DOCKER guidance in runbooks; keep “brew services” as primary path.
- ➕ Leave a short note in `/docs/lightrag_integration.md` with chosen upstream pin (commit SHA TBD tomorrow).

Then:Night ops (final pass before bed)
Run quick smoke (no Docker):
# venv + deps
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# schemas (only if not already done today)
export PGHOST=localhost PGPORT=5432 PGUSER=lnsp PGPASSWORD=lnsp PGDATABASE=lnsp
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -f scripts/init_pg.sql || true
cypher-shell -a bolt://localhost:7687 -u neo4j -p password -f scripts/init_neo4j.cql || true

# ingest the 4 samples and write vectors
python -m src.ingest_factoid --faiss-out artifacts/fw_vectors.npz

# eval echo and write report
scripts/eval_echo.sh artifacts/fw_vectors.npz 0.82
If energy remains: run the 1k ingest once your data/factoidwiki_1k.jsonl is in place:
scripts/ingest_1k.sh data/factoidwiki_1k.jsonl
scripts/build_faiss_1k.sh artifacts/fw1k_vectors.npz 256
scripts/eval_echo.sh artifacts/fw1k_vectors.npz 0.82

## [Programmer] Tasks
- ✅ Smoke: 4-item ingest → vectors NPZ; echo eval report generated.
- ✅ Ensure `scripts/*` avoid Docker unless user sets it up (NO_DOCKER path OK).
- ✅ Prep stubs for `src/api/retrieve.py` (FastAPI) + a `uvicorn` run line in README, to wire tomorrow.

- [Programmer] FastAPI retrieval service implemented with LightRAG integration; README deployment instructions added.

## [Consultant] Tasks
- ✅ Finalize 20-item prompt outputs → `/tests/sample_outputs.json`.
- ✅ Add tonight’s echo summary to `/eval/day2_report.md` (from `scripts/eval_echo.sh`).
- ✅ Draft acceptance notes for tomorrow’s 1k run (metrics you’ll collect).
  Capture: Postgres `cpe_entry` row count, Neo4j node/edge totals, Faiss vector and centroid counts, echo pass ratio ≥0.82, `/search` latency plus LightRAG relevance spot-checks.
- ✅ Day 3 assets staged: prompt template (`docs/prompt_template.md`), eval set (`eval/day3_eval.jsonl`), report scaffold (`eval/day3_report.md`).
- ✅ API eval tooling ready (`src/eval_runner.py`, refreshed `scripts/eval_echo.sh`); sample drop folder pre-created at `eval/day3_samples/`.

### Blockers / Risks
- None critical. If Postgres/Neo4j aren’t up locally, keep running evals off NPZ vectors; DB writes can be replayed tomorrow.

### Sign-off
- All teams: add a one-liner status + any TODOs for the morning.
- [Consultant] 4-sample smoke green; metric checklist staged for 1k acceptance in the morning.
- [Consultant] Day 3 eval kit in repo; awaiting live API + 1k ingest to populate results.
- [Architect] Status/Completed: All three tasks executed successfully - enums freeze verified in `/src/enums.py`, NO_DOCKER guidance confirmed across all scripts, and LightRAG upstream pin `0.1.0b6` documented for production stability.
