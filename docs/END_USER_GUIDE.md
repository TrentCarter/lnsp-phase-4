# LNSP End User Guide

**Last Updated**: November 7, 2025
**Version**: Phase 4 (PAS + Aider-LCO P0)

Welcome! This guide shows you how to access and use the three main components of the LNSP system:

1. **Aider-LCO RPC** - AI code assistant service
2. **HMI Web Dashboard** - Real-time monitoring and visualization
3. **Test Suite** - Automated testing and validation

---

## 🚀 Quick Start (3 Steps)

### 1. Activate Environment

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
source .venv/bin/activate
```

### 2. Start Services

```bash
# Start PAS infrastructure (if needed)
./scripts/start_all_pas_services.sh

# Start Aider-LCO RPC server (port 6150)
export PAS_PORT=6150
export AIDER_MODEL="ollama/qwen2.5-coder:7b-instruct"
./.venv/bin/python tools/aider_rpc/server_enhanced.py &
```

### 3. Access the System

```bash
# Open HMI Dashboard in browser
open http://localhost:6101

# Check Aider-LCO health
curl http://localhost:6150/health | jq
```

---

## 1️⃣ Aider-LCO RPC (AI Code Assistant)

### What It Does
Aider-LCO is an AI-powered code editing service that can modify files, add features, fix bugs, and refactor code via a JSON API.

### Access Points

**Base URL**: `http://localhost:6150`

**Endpoints**:
- `GET /health` - Service health check
- `POST /invoke` - Execute a coding task
- `GET /status` - Runtime statistics

### Example Usage

#### Basic Health Check
```bash
curl http://localhost:6150/health | jq
```

**Expected Response**:
```json
{
  "status": "ok",
  "service": "Aider-LCO",
  "uptime_seconds": 123.45
}
```

#### Execute a Coding Task
```bash
curl -s http://localhost:6150/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "payload": {
      "message": "Add docstrings to all functions in this file",
      "files": ["src/example.py"],
      "auto_commit": false
    },
    "run_id": "task-001",
    "policy": {
      "timeout_s": 120
    }
  }' | jq
```

**Response Structure**:
```json
{
  "upstream": {
    "outputs": {
      "status": "success",
      "files_edited": ["src/example.py"],
      "exit_code": 0,
      "log_path": "/tmp/aider_xxx.log"
    }
  },
  "routing_receipt": {
    "run_id": "task-001",
    "status": "success",
    "timings_ms": {
      "total": 1234.56
    },
    "ts": "2025-11-07T12:34:56Z"
  }
}
```

### Common Use Cases

#### 1. Add Documentation
```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Add comprehensive docstrings to all classes and functions",
    "files": ["mymodule.py"]
  },
  "run_id": "doc-update-001"
}' | jq '.routing_receipt.status'
```

#### 2. Fix Type Hints
```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Add type hints to all function signatures",
    "files": ["utils/helpers.py"]
  },
  "run_id": "types-001"
}' | jq '.upstream.outputs'
```

#### 3. Refactor Code
```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Extract the authentication logic into a separate AuthHandler class",
    "files": ["api/server.py"]
  },
  "run_id": "refactor-001"
}' | jq
```

### Configuration

**Environment Variables**:
```bash
# Required
export PAS_PORT=6150                                    # Server port
export AIDER_MODEL="ollama/qwen2.5-coder:7b-instruct"   # AI model

# Optional
export PAS_COST_DIR=./artifacts/costs                   # Cost receipts directory
export AIDER_AUTO_COMMIT=false                          # Auto-commit changes
```

**Configuration File**: `configs/pas/aider.yaml`

### Security Features

✅ **Command Allowlist** - Blocks dangerous commands (sudo, rm -rf, fork bombs)
✅ **Path Sandboxing** - Restricts file access to allowed directories
✅ **Secret Redaction** - Scrubs API keys and passwords from logs
✅ **Git Safety** - Never force-push, never skip hooks
✅ **Environment Filtering** - Whitelists safe environment variables

### Troubleshooting

**Problem**: `Connection refused` when calling `/invoke`

**Solution**:
```bash
# Check if server is running
curl http://localhost:6150/health

# If not, start it
./.venv/bin/python tools/aider_rpc/server_enhanced.py
```

**Problem**: `Missing 'message' in payload`

**Solution**: Ensure your request includes both `message` and `files`:
```json
{
  "payload": {
    "message": "Your instruction here",
    "files": ["path/to/file.py"]
  }
}
```

---

## 2️⃣ HMI Web Dashboard (Monitoring & Visualization)

### What It Does
The HMI (Human-Machine Interface) provides real-time monitoring of all PAS services with live metrics, agent visualizations, and system health.

### Access

**URL**: `http://localhost:6101`

Open in your browser:
```bash
open http://localhost:6101
```

### Features

#### Service Health Cards
- **Status**: Green (healthy), Yellow (degraded), Red (down)
- **Metrics**: Request count, uptime, latency
- **Auto-refresh**: Every 5 seconds

#### Agent Tree Visualization (D3.js)
- **Hierarchical view** of all 42 Claude sub-agents
- **Interactive nodes** - Click to expand/collapse
- **Color coding** by agent role (Execution, Design, DevOps, etc.)
- **Auto-refresh**: Every 10 seconds

#### Live System Metrics
- Total active services
- Overall health percentage
- Active WebSocket clients
- Rolling cost metrics (minute/hour/day)

#### Event Stream (WebSocket)
Real-time updates for:
- Service registrations
- Health status changes
- Cost updates
- Agent task completions

### Dashboard Sections

```
┌─────────────────────────────────────────────────┐
│  LNSP Dashboard                    🟢 8/8 OK   │
├─────────────────────────────────────────────────┤
│                                                 │
│  📊 System Metrics                              │
│  ├─ Services: 8 active                          │
│  ├─ Health: 100%                                │
│  └─ Cost (hour): $0.42                          │
│                                                 │
│  🌲 Agent Tree                                  │
│  └─ [Interactive D3 visualization]              │
│                                                 │
│  📡 Service Cards                               │
│  ├─ Registry (6121)        🟢 healthy           │
│  ├─ Gateway (6120)         🟢 healthy           │
│  ├─ Aider-LCO (6150)       🟢 healthy           │
│  └─ ...                                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

### API Endpoints (for custom integrations)

```bash
# Get all metrics
curl http://localhost:6101/api/metrics | jq

# Get service registry
curl http://localhost:6101/api/registry | jq

# Get cost receipts
curl http://localhost:6101/api/costs | jq

# Broadcast custom event (WebSocket)
curl -X POST http://localhost:6102/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"event_type": "custom", "data": {"msg": "Test"}}'
```

### Troubleshooting

**Problem**: Dashboard shows "All services down"

**Solution**:
```bash
# Restart PAS services
./scripts/stop_all_pas_services.sh
./scripts/start_all_pas_services.sh

# Wait 10 seconds for services to register
sleep 10

# Refresh browser
open http://localhost:6101
```

**Problem**: Agent tree not rendering

**Solution**:
1. Check browser console for JavaScript errors
2. Ensure D3.js is loaded (check network tab)
3. Clear browser cache and hard refresh (Cmd+Shift+R)

---

## 3️⃣ Test Suite (Quality Assurance)

### What It Does
Automated test suite covering Aider-LCO RPC, schemas, and PAS infrastructure.

### Running Tests

#### Quick Test (All Aider-LCO Tests)
```bash
pytest tests/aider_rpc/ tests/schemas/ -q
```

**Expected Output**:
```
.......                                                  [100%]
7 passed, 11 warnings in 0.19s
```

#### Verbose Mode (Detailed Output)
```bash
pytest tests/aider_rpc/ tests/schemas/ -v
```

#### Run Specific Test
```bash
# Test allowlist security
pytest tests/aider_rpc/test_allowlist.py -v

# Test secret redaction
pytest tests/aider_rpc/test_redact.py -v

# Test server endpoints
pytest tests/aider_rpc/test_server_health_invoke.py -v
```

### Test Coverage

| Test Module | What It Tests | Status |
|-------------|---------------|--------|
| `test_server_health_invoke.py` | Server health + invoke endpoint | ✅ PASS |
| `test_allowlist.py` | Command security (blocks rm -rf, sudo, etc.) | ✅ PASS |
| `test_redact.py` | Secret scrubbing (API keys, passwords) | ✅ PASS |
| `test_receipts.py` | Receipt persistence helpers | ✅ PASS |
| `test_heartbeat.py` | Heartbeat module presence | ✅ PASS |
| `test_schemas_shapes.py` | Schema validation (10 files) | ✅ PASS |

**Total**: 7 tests, 100% passing

### Example: Running Tests with Coverage

```bash
# Install coverage tool (one-time)
pip install pytest-cov

# Run tests with coverage report
pytest tests/aider_rpc/ tests/schemas/ --cov=tools.aider_rpc --cov-report=term-missing
```

### Test Outputs

Tests create artifacts in:
- **Logs**: `/tmp/aider_*.log` (Aider execution logs)
- **Receipts**: `artifacts/costs/pytest-run-*.json` (cost tracking)
- **Pytest cache**: `.pytest_cache/` (test results)

### CI/CD Integration

```bash
# Add to GitHub Actions / GitLab CI
.venv/bin/pytest tests/aider_rpc/ tests/schemas/ -v --tb=short
```

---

## 📚 Additional Resources

### Quick Reference

| Component | Port | URL | Purpose |
|-----------|------|-----|---------|
| HMI Dashboard | 6101 | http://localhost:6101 | Web UI |
| Event Stream | 6102 | http://localhost:6102 | WebSocket events |
| Gateway | 6120 | http://localhost:6120 | Request routing |
| Registry | 6121 | http://localhost:6121 | Service discovery |
| Aider-LCO | 6150 | http://localhost:6150 | Code assistant |

### Documentation

- **Setup Guide**: `docs/AIDER_LCO_SETUP.md`
- **Quick Start**: `docs/AIDER_LCO_QUICKSTART.md`
- **Security Review**: `docs/SECURITY_REVIEW_AIDER_LCO.md`
- **PAS Architecture**: `docs/PRDs/PRD_Polyglot_Agent_Swarm.md`
- **HMI Guide**: `docs/PHASE2_HMI_STATUS.md`

### Support Scripts

```bash
# Start all services
./scripts/start_all_pas_services.sh

# Stop all services
./scripts/stop_all_pas_services.sh

# Check service health
./scripts/health_check.sh

# View logs
tail -f /tmp/pas_*.log
```

### Common Workflows

#### Workflow 1: Daily Development

```bash
# 1. Start environment
source .venv/bin/activate
./scripts/start_all_pas_services.sh

# 2. Open dashboard
open http://localhost:6101

# 3. Make code changes via Aider
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Refactor database connection logic",
    "files": ["src/db.py"]
  },
  "run_id": "dev-001"
}' | jq

# 4. Run tests
pytest tests/ -v

# 5. Shutdown
./scripts/stop_all_pas_services.sh
```

#### Workflow 2: Debugging Issues

```bash
# 1. Check service health
curl http://localhost:6121/health | jq

# 2. View metrics
curl http://localhost:6101/api/metrics | jq

# 3. Check logs
ls -lt /tmp/pas_*.log | head -5
tail -f /tmp/pas_gateway.log

# 4. Run diagnostic tests
pytest tests/aider_rpc/test_server_health_invoke.py -v

# 5. Check receipts for errors
cat artifacts/costs/*.jsonl | jq -r 'select(.status=="error")'
```

#### Workflow 3: Cost Tracking

```bash
# 1. View recent costs
cat artifacts/costs/*.jsonl | jq -r '[.run_id, .cost_estimate.usd] | @tsv'

# 2. Get rolling window costs
curl "http://localhost:6120/metrics?window=hour" | jq '.cost_total_usd'

# 3. Check budget alerts
curl http://localhost:6101/api/metrics | jq '.budget'
```

---

## 🛠️ Troubleshooting

### General Issues

**Problem**: Port already in use

**Solution**:
```bash
# Find process using port 6150
lsof -i :6150

# Kill process
kill -9 <PID>

# Or use script
./scripts/stop_all_pas_services.sh
```

**Problem**: Tests failing after code changes

**Solution**:
```bash
# 1. Restart services
./scripts/stop_all_pas_services.sh
./scripts/start_all_pas_services.sh

# 2. Clear pytest cache
rm -rf .pytest_cache/

# 3. Re-run tests
pytest tests/aider_rpc/ tests/schemas/ -v
```

**Problem**: HMI showing stale data

**Solution**:
```bash
# 1. Clear browser cache
# 2. Hard refresh (Cmd+Shift+R on macOS)
# 3. Check WebSocket connection in browser DevTools > Network
```

### Getting Help

1. **Check logs**: `tail -f /tmp/pas_*.log`
2. **Read docs**: See "Additional Resources" section above
3. **Run diagnostics**: `pytest tests/ -v`
4. **View receipts**: `cat artifacts/costs/*.jsonl | jq`

---

## 📝 Summary

You now have access to three powerful components:

1. **Aider-LCO** (`http://localhost:6150`) - AI code assistant with JSON API
2. **HMI Dashboard** (`http://localhost:6101`) - Real-time monitoring and visualization
3. **Test Suite** (`pytest tests/`) - Automated quality assurance

**Next Steps**:
- Explore the HMI dashboard features
- Try the example Aider-LCO tasks above
- Run the test suite to verify everything works
- Read the detailed setup guide: `docs/AIDER_LCO_SETUP.md`

**Happy coding!** 🚀
