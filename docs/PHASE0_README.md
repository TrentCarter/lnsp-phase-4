# Phase 0: Core Infrastructure — Registry + Heartbeat Monitor

**Status:** ✅ Ready for Testing
**Date:** 2025-11-06
**Services:** Registry (6121), Heartbeat Monitor (6109)

---

## Overview

Phase 0 establishes the **core infrastructure** that all other PAS services depend on:

1. **Service Registry (6121)** - Service registration, discovery, heartbeats, TTL
2. **Heartbeat Monitor (6109)** - Monitors heartbeats, detects misses, emits alerts

---

## Quick Start

```bash
# 1. Start services
./scripts/start_phase0_services.sh

# 2. Run integration tests (takes ~2 minutes)
./scripts/test_phase0.sh

# 3. Stop services when done
./scripts/stop_phase0_services.sh
```

---

## What Phase 0 Provides

### Service Registry (Port 6121)

**Capabilities:**
- ✅ Service registration (models, tools, agents)
- ✅ Service discovery with filters (type, role, capability)
- ✅ Heartbeat tracking
- ✅ TTL-based eviction (mark 'down' after 2 misses, deregister after 3)
- ✅ Role promotion/demotion (staging → production)
- ✅ SQLite persistence (survives restarts)

**API Endpoints:**
```bash
POST   /register        # Register a service
PUT    /heartbeat       # Update service heartbeat
GET    /discover        # Find services by filters
POST   /promote         # Change service role
POST   /deregister      # Remove a service
GET    /services        # List all services
GET    /services/{id}   # Get service details
GET    /health          # Health check
```

**Database:** `artifacts/registry/registry.db` (SQLite)

### Heartbeat Monitor (Port 6109)

**Capabilities:**
- ✅ Background task checks heartbeats every 30s
- ✅ Detects missed heartbeats (based on TTL)
- ✅ Marks services 'down' after 2 misses
- ✅ Deregisters services after 3 misses
- ✅ Emits alerts to HMI event log
- ✅ Supports service recovery (auto-restore when heartbeats resume)

**API Endpoints:**
```bash
GET /health     # Health check
GET /stats      # Monitor statistics
GET /alerts     # Recent heartbeat alerts
```

**Event Log:** `artifacts/hmi/events/heartbeat_alerts_YYYYMMDD.jsonl` (LDJSON format)

---

## Manual Testing

### 1. Start Services

```bash
./scripts/start_phase0_services.sh
```

**Expected output:**
```
✅ Phase 0 services started successfully!

Registry:          http://localhost:6121
Heartbeat Monitor: http://localhost:6109
```

### 2. Register a Test Service

```bash
curl -X POST http://localhost:6121/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-service",
    "type": "model",
    "role": "experimental",
    "url": "http://127.0.0.1:8888",
    "caps": ["infer"],
    "ctx_limit": 32768,
    "heartbeat_interval_s": 60,
    "ttl_s": 90
  }'
```

**Expected response:**
```json
{
  "service_id": "uuid-here",
  "registered_at": "2025-11-06T10:30:00.000Z"
}
```

### 3. Discover the Service

```bash
# By name
curl "http://localhost:6121/discover?name=test-service"

# By type and role
curl "http://localhost:6121/discover?type=model&role=experimental"

# By capability
curl "http://localhost:6121/discover?cap=infer"
```

### 4. Send Heartbeat

```bash
curl -X PUT http://localhost:6121/heartbeat \
  -H "Content-Type: application/json" \
  -d '{
    "service_id": "uuid-from-step-2",
    "status": "ok",
    "p95_ms": 123.4,
    "queue_depth": 2,
    "load": 0.35
  }'
```

### 5. Check Heartbeat Monitor Stats

```bash
curl http://localhost:6109/stats
```

**Expected response:**
```json
{
  "total_services": 1,
  "healthy_services": 1,
  "degraded_services": 0,
  "down_services": 0,
  "last_check": "2025-11-06T10:32:00.000Z"
}
```

### 6. Wait for Missed Heartbeat (90s TTL)

Wait 2 minutes without sending heartbeats, then check:

```bash
# Should show service marked 'down'
curl http://localhost:6109/stats

# Check alerts
curl http://localhost:6109/alerts
```

**Expected alert:**
```json
{
  "alert_type": "service_down",
  "service_id": "uuid",
  "service_name": "test-service",
  "missed_beats": 2,
  "last_seen": "2025-11-06T10:30:00.000Z",
  "action": "marked_down",
  "ts": "2025-11-06T10:32:00.000Z"
}
```

### 7. View Event Log

```bash
cat artifacts/hmi/events/heartbeat_alerts_$(date +%Y%m%d).jsonl
```

---

## Database Schema

### Services Table

```sql
CREATE TABLE services (
    service_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'model' | 'tool' | 'agent'
    role TEXT NOT NULL,  -- 'production' | 'staging' | 'canary' | 'experimental'
    url TEXT NOT NULL,
    caps TEXT NOT NULL,  -- JSON array
    labels TEXT,         -- JSON object
    ctx_limit INTEGER,
    cost_hint_usd_per_1k REAL,
    heartbeat_interval_s INTEGER DEFAULT 60,
    ttl_s INTEGER DEFAULT 90,
    status TEXT DEFAULT 'ok',
    last_heartbeat_ts TEXT,
    registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    p95_ms REAL,
    queue_depth INTEGER,
    load REAL
);
```

**Indexes:**
- `idx_services_name_role` on `(name, role)`
- `idx_services_type_role` on `(type, role)`
- `idx_services_status` on `(status)`
- `idx_services_last_heartbeat` on `(last_heartbeat_ts)`

---

## Logs

**Location:** `/tmp/pas_logs/`

```bash
# Watch Registry logs
tail -f /tmp/pas_logs/registry.log

# Watch Heartbeat Monitor logs
tail -f /tmp/pas_logs/heartbeat_monitor.log
```

---

## Troubleshooting

### Services won't start

**Check ports are free:**
```bash
lsof -i :6121  # Should be empty
lsof -i :6109  # Should be empty
```

**Kill existing processes:**
```bash
./scripts/stop_phase0_services.sh
```

### Registry database locked

```bash
# Remove database and restart
rm artifacts/registry/registry.db
./scripts/start_phase0_services.sh
```

### Heartbeat Monitor not detecting misses

**Check interval settings:**
- Heartbeat Monitor checks every 30s
- Default TTL is 90s
- 2 misses = 120s = service marked 'down'
- 3 misses = 180s = service deregistered

**Wait at least 2 minutes** after last heartbeat to see alerts.

---

## Acceptance Criteria

- [x] Registry service starts on port 6121
- [x] Can register a mock service via `POST /register`
- [x] Can discover services via `GET /discover`
- [x] Heartbeat Monitor detects missed beats (30s intervals)
- [x] Services marked `down` after 2 misses
- [x] Services deregistered after 3 misses
- [x] SQLite database persists across restarts
- [x] Event log written to `artifacts/hmi/events/`
- [x] Health endpoints return 200 OK

---

## Next Steps

Once Phase 0 is tested and approved:

1. **Phase 1:** Resource Manager (6104) + Token Governor (6105)
2. **Phase 2:** Flask HMI Dashboard (6101) + WebSocket (6102)
3. **Phase 3:** Gateway (6120) + Router (6103)

---

## Files Created

```
services/
  registry/
    registry_service.py           # FastAPI service (6121)
  heartbeat_monitor/
    heartbeat_monitor.py          # FastAPI service (6109)

contracts/
  service_registration.schema.json
  heartbeat.schema.json
  status_update.schema.json
  heartbeat_alert.schema.json
  resource_request.schema.json

scripts/
  start_phase0_services.sh        # Start services
  stop_phase0_services.sh         # Stop services
  test_phase0.sh                  # Integration tests

artifacts/
  registry/
    registry.db                   # SQLite database
  hmi/
    events/
      heartbeat_alerts_*.jsonl    # Event logs
```

---

**END OF PHASE 0 README**
