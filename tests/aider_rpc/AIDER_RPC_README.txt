# Aider RPC Scaffold (PAS)

Files created:
- tools/aider_rpc/server.py
- schemas/*.json

Run locally (example):
    export PAS_PORT=6150
    # optionally: export PAS_REGISTRY_URL=http://127.0.0.1:6121
    python tools/aider_rpc/server.py

Test invoke:
    curl -s http://127.0.0.1:${PAS_PORT:-6150}/invoke -H 'Content-Type: application/json' -d '{
      "target": {"name": "Aider-LCO", "type":"agent", "role":"execution"},
      "payload": {"command": "doc_update", "paths": ["docs/**/*.md"]},
      "policy": {"timeout_s": 60, "require_caps": ["git-edit"]},
      "run_id": "demo-run-001"
    }' | jq

Result includes a routing_receipt persisted to:
    artifacts/costs/<run_id>.json
