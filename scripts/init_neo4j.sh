#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${NEO4J_BOLT_URL:-}" ]]; then
  echo "[init_neo4j] Skipping (NEO4J_BOLT_URL not set). LightRAG in-proc KG will be used."
  exit 0
fi
cypher-shell -a "$NEO4J_BOLT_URL" -u "${NEO4J_USER:-neo4j}" -p "${NEO4J_PASS:-neo4j}" -f init_neo4j.cql
