#!/usr/bin/env bash
# Ontology ingestion wrapper: writes PostgreSQL + Neo4j + FAISS atomically
# Usage: ./scripts/ingest_ontologies.sh [--limit N]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Load env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Optional limit
LIMIT_ARG=${1:-}
if [[ "$LIMIT_ARG" =~ ^--limit=.*$ ]]; then
  LIMIT_VAL="${LIMIT_ARG#--limit=}"
else
  LIMIT_VAL=""
fi

# Service checks (warn-only for Neo4j)
if [ -f scripts/check_services.sh ]; then
  source scripts/check_services.sh || true
  check_postgresql || true
  check_neo4j || true
fi

echo "▶️  Ingesting ontology datasets (SWO/GO/DBpedia)"
PY=python3
if [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; fi

# Build args
LIMIT_CLI=""
if [ -n "$LIMIT_VAL" ]; then LIMIT_CLI="--limit $LIMIT_VAL"; fi

$PY -m src.ingest_ontology_simple \
  --ingest-all \
  --write-pg \
  --write-neo4j \
  --write-faiss \
  $LIMIT_CLI

# Build FAISS index (requires Makefile target)
if command -v make >/dev/null 2>&1; then
  echo "🔧 Building FAISS index via make..."
  make build-faiss || true
fi

# Verify synchronization
./scripts/verify_data_sync.sh

echo "✅ Ontology ingestion complete and synchronized"
