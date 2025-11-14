# TRON System Status Dashboard Integration

**Date:** 2025-11-14
**Status:** ✅ Complete
**Issue:** TRON Heartbeat Monitor (port 6109) was not visible in HMI System Status dashboard

---

## Problem

The user reported that TRON (port 6109) was not appearing in Settings → System Status → Core Services section of the HMI dashboard, despite the service being running.

### Root Causes Identified

1. **Services not registered with Registry Service** - TRON and other core infrastructure services were not calling the Registry Service (port 6121) `/register` endpoint on startup

2. **No heartbeat transmission** - Even when manually registered, services were not sending periodic heartbeats to `/heartbeat` endpoint, causing TTL expiration (90s) and automatic de-registration

3. **HMI API format mismatch** - HMI was expecting Registry to return `{"services": {...}}` but Registry actually returns `{"items": [...]}`

4. **Dashboard grouping incomplete** - Dashboard JavaScript only grouped services containing "gateway", "pas root", or "registry" in their names as Core Services, missing TRON/heartbeat services

---

## Solution

### 1. Created Registry Heartbeat Utility

**File:** `services/common/registry_heartbeat.py`

Provides automatic service registration and heartbeat transmission for all services:

```python
from services.common.registry_heartbeat import start_registry_heartbeat

@app.on_event("startup")
async def startup_event():
    await start_registry_heartbeat(
        service_id="tron-heartbeat-monitor",
        name="TRON Heartbeat Monitor",
        type="agent",
        role="production",
        url="http://localhost:6109",
        caps=["health_monitoring", "timeout_detection", "service_alerts"],
        labels={"tier": "core", "category": "infrastructure", "port": 6109},
        heartbeat_interval_s=30,
        ttl_s=90
    )
```

**Features:**
- Automatic registration on startup
- Background heartbeat loop (every 30s by default)
- Graceful shutdown handling
- Error logging and retry logic
- Uses HTTP PUT (not POST) for `/heartbeat` endpoint

**Key Fix:** Changed from POST to PUT method (line 122):
```python
response = await client.put(  # Was: client.post
    f"{REGISTRY_URL}/heartbeat",
    json=payload,
    timeout=5.0
)
```

### 2. Integrated Heartbeat into TRON

**File:** `services/heartbeat_monitor/heartbeat_monitor.py`

**Changes:**
- Added automatic registration in `startup_event` (lines 274-293)
- Added heartbeat cleanup in `shutdown_event` (lines 308-315)

**Startup Logic (lines 266-297):**
```python
@app.on_event("startup")
async def startup_event():
    # Register with Registry Service and start sending heartbeats
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from common.registry_heartbeat import start_registry_heartbeat

        await start_registry_heartbeat(
            service_id="tron-heartbeat-monitor",
            name="TRON Heartbeat Monitor",
            ...
        )
    except Exception as e:
        print(f"⚠️  Failed to register: {e}")
```

### 3. Fixed HMI API Format Mismatch

**File:** `services/webui/hmi_app.py:216-219`

**Before:**
```python
# Registry returns {"services": {...}} - convert dict to list
services_dict = data.get('services', {})
services_list = list(services_dict.values()) if isinstance(services_dict, dict) else []
```

**After:**
```python
# Registry returns {"items": [...]} - extract list
services_list = data.get('items', [])
if not isinstance(services_list, list):
    services_list = []
```

### 4. Expanded Dashboard Core Services Grouping

**File:** `services/webui/templates/dashboard.html:1244-1246`

**Before:**
```javascript
} else if (name.includes('gateway') || name.includes('pas root') || name.includes('registry')) {
    grouped.core.push(agent);
```

**After:**
```javascript
} else if (name.includes('gateway') || name.includes('pas root') || name.includes('registry') ||
           name.includes('heartbeat') || name.includes('tron') || name.includes('webui') ||
           name.includes('events') || name.includes('router') || name.includes('aider')) {
    grouped.core.push(agent);
```

This ensures TRON, Event Stream, Router, Aider-LCO, and HMI itself are grouped under Core Services.

---

## Verification

### ✅ Registration Confirmed
```bash
$ curl -s http://localhost:6121/services | python3 -m json.tool | grep -A 10 "TRON"
{
    "service_id": "tron-heartbeat-monitor",
    "name": "TRON Heartbeat Monitor",
    "type": "agent",
    "role": "production",
    "url": "http://localhost:6109",
    "status": "ok",
    "last_heartbeat_ts": "2025-11-14T18:50:19.035040"
}
```

### ✅ Heartbeat Updates Every 30s
```
Initial:  2025-11-14T18:49:49.022434 | Status: ok
After 35s: 2025-11-14T18:50:19.035040 | Status: ok
```
Timestamp advanced by 30 seconds, confirming heartbeat loop is working.

### ✅ Visible in HMI Dashboard
```bash
$ curl -s http://localhost:6101/api/services | python3 -m json.tool
{
    "services": [
        {
            "name": "TRON Heartbeat Monitor",
            "status": "ok",
            "url": "http://localhost:6109",
            "last_heartbeat_ts": "2025-11-14T18:50:19.035040"
        }
    ]
}
```

### ✅ Grouped as Core Service
JavaScript grouping logic confirms TRON will appear under "Core Services" section due to "heartbeat" in name.

---

## Additional Tools Created

### Service Registration Utility

**File:** `tools/register_core_services.py`

Manual registration script for core infrastructure services that don't have startup integration yet:

```bash
# Register all core services
python tools/register_core_services.py

# Register specific service
python tools/register_core_services.py --service tron
```

**Services Supported:**
- TRON Heartbeat Monitor (6109)
- Resource Manager (6104)
- Token Governor (6105)
- Event Stream (6102)
- Router (6103)
- Aider-LCO (6130)

**Note:** This is a fallback tool. Services with integrated `registry_heartbeat.py` startup don't need this.

---

## Next Steps

### Immediate (Required for Full Functionality)

1. **Integrate heartbeat into other core services:**
   - Resource Manager (port 6104)
   - Token Governor (port 6105)
   - Event Stream (port 6102)
   - Router (port 6103)

2. **Integrate heartbeat into PAS agents:**
   - Gateway (6120)
   - PAS Root (6100)
   - Registry (6121) - should self-register
   - Architect (6110)
   - Directors (6111-6115)
   - Managers (6141-6147)
   - Programmers (6151-6160)

### Implementation Pattern

For any FastAPI service, add to startup event:

```python
from services.common.registry_heartbeat import start_registry_heartbeat

@app.on_event("startup")
async def startup_event():
    await start_registry_heartbeat(
        service_id="unique-service-id",
        name="Human Readable Name",
        type="agent",  # or "model" or "tool"
        role="production",  # or "staging", "canary", "experimental"
        url="http://localhost:PORT",
        caps=["capability1", "capability2"],
        labels={"tier": "core", "agent_role": "director"}  # etc
    )
```

### Testing

With heartbeat system fully deployed, Test 13 (TRON Timeout Detection) can be completed:
1. ✅ Phase 1: TRON connectivity check (already passing)
2. **Phase 2:** Simulate agent timeout (stop sending heartbeats)
3. **Phase 3:** TRON detects timeout and marks service as "down"
4. **Phase 4:** Parent agent receives alert via Event Stream

---

## Technical Details

### Registry Service Endpoints

- **POST /register** - Register new service (returns service_id)
- **PUT /heartbeat** - Update heartbeat timestamp (⚠️ Must use PUT, not POST)
- **GET /services** - List all registered services (returns `{"items": [...]}`)
- **POST /deregister** - Remove service from registry

### Heartbeat Parameters

- **heartbeat_interval_s:** 30s (recommended) - How often to send heartbeats
- **ttl_s:** 90s (3x interval) - Time before service is considered down if no heartbeat received
- **CHECK_INTERVAL_S:** 30s (TRON) - How often TRON checks for missed heartbeats
- **MAX_MISSES_BEFORE_DOWN:** 2 (TRON) - Mark service "down" after 2 missed beats
- **MAX_MISSES_BEFORE_DEREGISTER:** 3 (TRON) - Remove from registry after 3 missed beats

### Service Registration Schema

Required fields:
- `service_id` (string, UUID) - Unique identifier
- `name` (string) - Display name
- `type` ("model" | "tool" | "agent")
- `role` ("production" | "staging" | "canary" | "experimental")
- `url` (string) - Service endpoint
- `caps` (array) - Capabilities list
- `labels` (object, optional) - Metadata (tier, category, port, etc.)

---

## Related Documentation

- **PRD:** `docs/PRDs/PRD_Hierarchical_Health_Monitoring_Retry_System.md`
- **Service Ports:** `docs/SERVICE_PORTS.md`
- **Agent Family Tests:** `docs/AGENT_FAMILY_TESTS_RESOURCE_MANAGEMENT.md`
- **Last Session Summary:** `docs/last_summary.md`

---

## Summary

TRON Heartbeat Monitor (port 6109) is now:
- ✅ Automatically registering with Registry Service on startup
- ✅ Sending heartbeats every 30 seconds via PUT /heartbeat
- ✅ Visible in HMI System Status dashboard under "Core Services"
- ✅ Maintaining "ok" status (not expiring due to TTL)
- ✅ Ready for Test 13 timeout detection scenarios

The `registry_heartbeat.py` utility provides a reusable pattern for integrating all other PAS services with the Registry and TRON monitoring system.
