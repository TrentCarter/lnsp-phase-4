#!/usr/bin/env bash
# One-shot environment bootstrap for FactoidWiki → LNSP dev stack
# Brings up Postgres + Neo4j and applies schema migrations.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Default env (can be overridden by the caller or .env)
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-lnsp}"
export PGPASSWORD="${PGPASSWORD:-lnsp}"
export PGDATABASE="${PGDATABASE:-lnsp}"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASS="${NEO4J_PASS:-password}"

# Check for NO_DOCKER flag
if [ "${NO_DOCKER:-0}" = "1" ]; then
    echo "⚠️  NO_DOCKER=1 detected. Skipping Docker operations."
    echo "📝 Please ensure PostgreSQL and Neo4j are running manually:"
    echo "   - PostgreSQL on $PGHOST:$PGPORT"
    echo "   - Neo4j on $NEO4J_URI"

    # NO_DOCKER fast path: venv + deps
    : "${PY:=python3}"
    : "${VENV:=.venv}"

    echo "🐍 Creating virtual environment..."
    $PY -m venv "$VENV"
    source "$VENV/bin/activate"
    echo "📦 Upgrading pip and installing requirements..."
    pip install -U pip wheel
    pip install -r requirements.txt
    # LightRAG pin
    pip install "lightrag-hku==1.4.9rc1"
    echo "✅ NO_DOCKER environment initialized."
else
    # Compose up both services (idempotent)
    echo "▶️  Starting docker services (postgres, neo4j)"
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker or set NO_DOCKER=1 to skip."
        exit 1
    fi
    docker compose up -d postgres neo4j
fi

# Init Postgres schema
bash "$ROOT_DIR/scripts/init_pg.sh"

# Init Neo4j schema
bash "$ROOT_DIR/scripts/init_neo4j.sh"

# Optional quick health checks
echo "🔎 Postgres extensions:"
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT extname FROM pg_extension;" || true

echo "🔎 Neo4j constraints:"
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASS" -q "SHOW CONSTRAINTS" || true

echo "✅ All bootstraps complete."
