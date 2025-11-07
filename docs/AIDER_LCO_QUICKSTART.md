# Aider-LCO Quick Start (30 seconds)

**Status**: P0 Complete (2025-11-07)
**Aider Version**: v0.86.1 (August 2025)

## Install & Run

```bash
# 1. Install Aider CLI (one-time, latest version)
pipx install aider-chat
aider --version  # Should show v0.86.1+

# 2. Start Ollama (if using local models)
ollama serve &
ollama pull qwen2.5-coder:7b-instruct

# 3. Start Aider-LCO RPC server
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
export PAS_PORT=6150
export AIDER_MODEL="ollama/qwen2.5-coder:7b-instruct"
./.venv/bin/python tools/aider_rpc/server_enhanced.py
```

## Test It

```bash
# Health check
curl http://localhost:6150/health | jq

# Execute a task
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Add docstrings to all functions in src/example.py",
    "files": ["src/example.py"]
  },
  "run_id": "test-001"
}' | jq '.routing_receipt.status'
```

## Files Created (P0)

```
tools/aider_rpc/
├── server_enhanced.py    # ⭐ Full implementation
├── allowlist.py          # Command sandboxing
├── redact.py             # Secrets scrubbing
├── receipts.py           # Cost/KPI tracking
└── heartbeat.py          # Registry client

configs/pas/aider.yaml    # Configuration
docs/AIDER_LCO_SETUP.md   # Complete guide
```

## Next: P1 Integration (Tomorrow)

1. Register with PAS Gateway (port 6120)
2. Route job cards from Claude/Gemini through Gateway
3. Wire HMI visualization (Tree + Sequencer)

---

**Full docs**: `docs/AIDER_LCO_SETUP.md`
