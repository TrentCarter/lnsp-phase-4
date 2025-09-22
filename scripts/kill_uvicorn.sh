#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8080}"
if command -v lsof >/dev/null 2>&1; then
  PID=$(lsof -ti tcp:"$PORT" || true)
  [[ -n "${PID:-}" ]] && kill -9 $PID && echo "[kill] killed $PID" || echo "[kill] nothing on $PORT"
else
  pkill -f "uvicorn .*:app" || true
fi