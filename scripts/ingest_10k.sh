#!/usr/bin/env bash
# Ingest ~10k FactoidWiki items end-to-end (PG + Neo4j + NPZ vectors)
# Usage: scripts/ingest_10k.sh [path/to/factoid_wiki.jsonl]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Load env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Use sample data if no argument provided
INPUT_JSONL=${1:-data/datasets/factoid-wiki-large/factoid_wiki.jsonl}
ART_DIR=${ART_DIR:-artifacts}
NPZ_PATH=${NPZ_PATH:-$ART_DIR/fw10k_vectors.npz}

mkdir -p "$ART_DIR"

if [ ! -f "$INPUT_JSONL" ]; then
  echo "❌ Missing input JSONL: $INPUT_JSONL"
  echo "   Provide a file with lines: {id, contents, meta}"
  exit 1
fi

# Quick service checks (non-fatal warnings)
if [ "${NO_DOCKER:-0}" = "1" ]; then
  echo "📝 NO_DOCKER mode - ensure PostgreSQL and Neo4j are running manually"
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "⚠️  psql not found; Postgres write will fail unless PG_DSN uses socket/driver auth via psycopg2."
fi
if ! command -v cypher-shell >/dev/null 2>&1; then
  echo "⚠️  cypher-shell not found; ensure Neo4j is running and env vars are set."
fi

# Ingest
echo "▶️  Ingesting ~10k items from $INPUT_JSONL"
python3 -m src.ingest_factoid \
  --jsonl-path "$INPUT_JSONL" \
  --num-samples 10000 \
  --write-pg \
  --write-neo4j \
  --faiss-out "$NPZ_PATH"

# Summaries (best-effort)
if command -v psql >/dev/null 2>&1; then
  echo "
🔎 Postgres counts:"
  psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -c "SELECT COUNT(*) AS cpe_rows FROM cpe_entry; SELECT COUNT(*) AS vec_rows FROM cpe_vectors;"
fi

echo "
✅ Ingest complete. Vectors saved → $NPZ_PATH"
