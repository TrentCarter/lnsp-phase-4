#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory for locating CQL file reliably
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CQL_FILE="${SCRIPT_DIR}/init_neo4j.cql"

# Prefer modern env names used by Python code; fall back to legacy
NEO4J_URI=${NEO4J_URI:-${NEO4J_BOLT_URL:-bolt://localhost:7687}}
NEO4J_USER=${NEO4J_USER:-neo4j}
# Default password matches docker-compose (NEO4J_AUTH=neo4j/password)
NEO4J_PASS=${NEO4J_PASS:-password}

if ! command -v cypher-shell >/dev/null 2>&1; then
  echo "[init_neo4j] cypher-shell not found on PATH. Install Neo4j client or run via docker exec."
  exit 1
fi

echo "[init_neo4j] Using URI=${NEO4J_URI} USER=${NEO4J_USER} CQL=${CQL_FILE}"
if [[ ! -f "${CQL_FILE}" ]]; then
  echo "[init_neo4j] CQL file not found: ${CQL_FILE}"
  exit 1
fi

cypher-shell -a "${NEO4J_URI}" -u "${NEO4J_USER}" -p "${NEO4J_PASS}" -f "${CQL_FILE}"
echo "[init_neo4j] Initialization complete"
