
# P1 Scaffolds: Claude-LCO, Gateway, Registry, HMI

## Run
# Terminal 1 - Registry
python tools/pas_registry/server.py

# Terminal 2 - HMI
python tools/hmi/server.py

# Terminal 3 - Claude-LCO
export PAS_REGISTRY_URL=http://127.0.0.1:6121
python tools/claude_rpc/server.py

# (Optional) Terminal 4 - Gateway
export PAS_REGISTRY_URL=http://127.0.0.1:6121
python tools/pas_gateway/server.py

## CLI
export PATH="$PWD/bin:$PATH"
claudia health
claudia describe
claudia invoke --message "Refactor logging" --files src/a.py --run-id cc-001 --open-hmi
