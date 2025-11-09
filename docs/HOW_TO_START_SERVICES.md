# How to Start Services

## Quick Start (HMI Server Only)

The HMI (Human-Machine Interface) server is now running in the background. You can access it at:

**Main Dashboard**: http://localhost:6101/
**Tree View**: http://localhost:6101/tree
**Sequencer**: http://localhost:6101/sequencer
**Actions**: http://localhost:6101/actions

## Starting Services from Scratch

### Option 1: Simple HMI Server (Recommended for Testing)

```bash
# Start in foreground (see logs directly)
./scripts/start_hmi_server.sh

# OR start manually
./.venv/bin/python services/webui/hmi_app.py
```

The server will:
- ✅ Start Flask on port 6101
- ✅ Initialize SSE polling thread
- ✅ Serve Tree View, Sequencer, and Dashboard

### Option 2: All FastAPI Services (For Full Pipeline)

If you need the complete ingestion pipeline (encoders, chunkers, etc.):

```bash
# Start all services
./scripts/start_all_fastapi_services.sh

# This starts:
# - Episode Chunker (8900)
# - Semantic Chunker (8001)
# - GTR-T5 Embeddings (8767)
# - Ingest API (8004)
# - HMI Server (6101)
```

**Stop all services**:
```bash
./scripts/stop_all_fastapi_services.sh
```

## Checking Service Status

### Quick Health Check

```bash
# HMI Server
curl http://localhost:6101/health

# Expected output:
# {
#   "port": 6101,
#   "service": "hmi_app",
#   "status": "ok",
#   "timestamp": "2025-11-08T12:00:00.000000"
# }
```

### Full Service Check

```bash
# Check all running Python processes
ps aux | grep -E "(hmi_app|uvicorn|flask)" | grep -v grep

# Check specific ports
lsof -i :6101  # HMI Server
lsof -i :8900  # Episode Chunker
lsof -i :8001  # Semantic Chunker
lsof -i :8767  # GTR-T5 Embeddings
```

## Current Running Services (Your System)

From the process list, you already have these services running:

| Port | Service | Status |
|------|---------|--------|
| 6101 | **HMI Server** | ✅ Running |
| 6103 | Audio Service | ✅ Running |
| 6104 | Resource Manager | ✅ Running |
| 6105 | Token Governor | ✅ Running |
| 6109 | Heartbeat Monitor | ✅ Running |
| 6121 | Registry Service | ✅ Running |
| 6150 | Aider RPC Server | ✅ Running |
| 7001 | Orchestrator Encoder | ✅ Running |
| 7002 | Orchestrator Decoder | ✅ Running |
| 8050 | Llama 3.1 8B Service | ✅ Running |
| 8051 | TinyLlama Service | ✅ Running |
| 8052 | TLC Classifier | ✅ Running |
| 8766 | Vec2Text Decoder | ✅ Running |
| 8767 | GTR-T5 Encoder | ✅ Running |
| 8900 | Episode Chunker | ✅ Running |
| 8999 | LVM Eval Routes | ✅ Running |
| 9000 | Master Chat API | ✅ Running |

**Most services are already running!** You just needed the HMI server.

## Testing Real-time Updates

Now that the HMI server is running, you can test real-time updates:

```bash
# Run the test script
./scripts/test_realtime_updates.sh

# This will:
# 1. Clean old test data
# 2. Verify HMI server is running ✅ (already done!)
# 3. Prompt you to open browser tabs
# 4. Insert action_logs incrementally (every 3 seconds)
```

## Troubleshooting

### HMI Server Won't Start

**Problem**: Port 6101 already in use

```bash
# Find the process using port 6101
lsof -i :6101

# Kill it (replace PID with actual process ID)
kill -9 <PID>

# Start HMI server again
./scripts/start_hmi_server.sh
```

**Problem**: Database not found

```bash
# Check if registry database exists
ls -lh artifacts/registry/registry.db

# If missing, create it (Registry service should auto-create)
mkdir -p artifacts/registry
```

### SSE Not Working

**Problem**: Real-time updates not appearing

1. **Restart HMI server** (resets SSE tracker)
   ```bash
   # Stop: Press Ctrl+C in HMI terminal
   # Start: ./scripts/start_hmi_server.sh
   ```

2. **Check browser console** (F12 or Cmd+Opt+I)
   - Look for SSE connection messages
   - Check Network tab for `/api/stream/tree/...` connection

3. **Verify test data is NEW** (inserted AFTER server started)
   ```bash
   # Clean old data
   sqlite3 artifacts/registry/registry.db "DELETE FROM action_logs WHERE task_id LIKE 'test-realtime%';"

   # Restart HMI server
   # Then run test script
   ./scripts/test_realtime_updates.sh
   ```

### Other Services Not Running

If you need a service that's not running:

```bash
# Start specific service manually
./.venv/bin/uvicorn services.registry.registry_service:app --host 127.0.0.1 --port 6121 &

# Or use the startup script for all services
./scripts/start_all_fastapi_services.sh
```

## Production Notes

**Warning**: The HMI server uses Flask's built-in development server (not production-ready).

For production use:
```bash
# Use gunicorn instead
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:6101 services.webui.hmi_app:app
```

**Security**: The server currently has no authentication. In production:
- Add JWT/session authentication
- Use HTTPS (TLS/SSL)
- Add rate limiting
- Enable CORS restrictions

## Next Steps

1. ✅ HMI server is running
2. ✅ Open browser: http://localhost:6101/
3. ✅ Run test script: `./scripts/test_realtime_updates.sh`
4. ✅ Watch real-time updates in Tree View and Sequencer!

See `docs/REALTIME_UPDATES_TESTING_GUIDE.md` for complete testing instructions.
