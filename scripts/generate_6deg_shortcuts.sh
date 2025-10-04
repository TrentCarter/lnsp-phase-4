#!/usr/bin/env bash
# Wrapper to add 6-degree shortcut edges to Neo4j graph
# Usage: ./scripts/generate_6deg_shortcuts.sh [--rate 0.01] [--no-tmd-match] [--min-hops 4] [--max-hops 10] [--min-sim 0.5]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PY=python3
if [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; fi

RATE=0.01
NO_TMD_MATCH=0
MIN_HOPS=4
MAX_HOPS=10
MIN_SIM=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate) RATE="$2"; shift 2;;
    --no-tmd-match) NO_TMD_MATCH=1; shift;;
    --min-hops) MIN_HOPS="$2"; shift 2;;
    --max-hops) MAX_HOPS="$2"; shift 2;;
    --min-sim) MIN_SIM="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

ARGS=("-m" "src.graph.add_6deg_shortcuts" "--shortcut-rate" "$RATE" "--min-hops" "$MIN_HOPS" "--max-hops" "$MAX_HOPS" "--min-similarity" "$MIN_SIM")
if [ "$NO_TMD_MATCH" = "1" ]; then ARGS+=("--no-tmd-match"); fi

${PY} "${ARGS[@]}"
