# Session Summary: HMI Dashboard Improvements (Part 2)
**Date**: November 7, 2025
**Focus**: Dashboard Events Counter, Fast Refresh Timer, Demo Data Flow Debugging

---

## 🎯 User Requirements

1. **Add Events Counter to Dashboard** - Show event count in Event Stream service card
2. **Make Dashboard Refresh Fast** - Change from 60s default to 0s (instant/fast) with range 0-600s
3. **Fix Missing Demo Data** - No activity showing in Actions, Sequencer, or Tree View despite demo running
4. **Fix Agent Heartbeats** - Registered agents showing "Never" for last heartbeat

---

## ✅ Completed Changes

### 1. Event Stream: Total Events Counter

**Files Modified**:
- `services/event_stream/event_stream.py`

**Changes**:
```python
# Added global counter (line 36)
total_events = 0

# Updated health endpoint (line 48)
'total_events': total_events

# Increment on every event (line 108)
def add_to_buffer(event: Dict[str, Any]):
    global event_buffer, total_events
    event_buffer.append(event)
    total_events += 1  # NEW
    if len(event_buffer) > MAX_BUFFER_SIZE:
        event_buffer.pop(0)
```

**Result**: Health endpoint now returns `{"total_events": 157, ...}`

---

### 2. Dashboard: Display Events Counter

**Files Modified**:
- `services/webui/templates/dashboard.html`

**Changes** (line 517-519):
```javascript
${svc.key === 'event_stream' ? `
<div class="service-info-row">
    <span class="service-info-label">Events:</span>
    <span class="service-info-value">${serviceData.total_events || 0}</span>
</div>
` : ''}
```

**Result**: Event Stream service card now shows "Events: 157" (and counting)

---

### 3. Dashboard: Fast Refresh by Default

**Files Modified**:
- `services/webui/templates/base.html` (Settings configuration)
- `services/webui/templates/dashboard.html` (Refresh logic)

#### base.html Changes:

**Default Settings** (line 887):
```javascript
const DEFAULT_SETTINGS = {
    refreshInterval: 0,  // Changed from 60 to 0 (instant/fast)
    // ...
};
```

**Validation Range** (line 936-937):
```javascript
// Changed from min=5, max=300 to:
if (currentSettings.refreshInterval < 0) currentSettings.refreshInterval = 0;
if (currentSettings.refreshInterval > 600) currentSettings.refreshInterval = 600;
```

**HTML Input** (line 567):
```html
<input type="number" id="refresh-interval" class="setting-input"
       min="0" max="600" value="0">
<!-- Changed from min="5" max="300" value="60" -->
```

**Description** (line 564):
```html
<div class="setting-description">Time between auto-refreshes (0 = instant/fast, seconds)</div>
```

#### dashboard.html Changes (line 586-592):

```javascript
function applyViewSettings(settings) {
    if (settings.autoRefreshEnabled) {
        // 0 = instant/fast (use 500ms minimum), otherwise use specified interval
        const intervalMs = settings.refreshInterval === 0
            ? 500  // 0.5 seconds for "instant" mode
            : settings.refreshInterval * 1000;

        metricsUpdateInterval = setInterval(() => {
            fetchMetrics();
            fetchAgents();
            fetchCostMetrics();
        }, intervalMs);
    }
}
```

**Result**:
- Default refresh: 0 seconds → 500ms polling (2 updates/sec)
- User can set 0-600 seconds (0=fast, 600=10min)
- Dashboard updates every 0.5s by default for near-instant feedback

---

### 4. Demo Worker: Fixed Event Structure

**Files Modified**:
- `/tmp/lnsp_demo_worker.py`

**Issue**: Demo worker was sending flat event structure:
```python
# ❌ WRONG (old code)
payload = {
    "event_type": event_type,
    "service_id": service_id,
    "task_name": task_name,
    "timestamp": datetime.now().isoformat(),
    "metadata": metadata
}
```

**Fix**: Wrapped everything in `data` field to match Event Stream expectations:
```python
# ✅ CORRECT (new code)
payload = {
    "event_type": event_type,
    "data": {
        "service_id": service_id,
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {})
    }
}
```

**Added**: Action logging to HMI:
```python
def log_action(action_id, action_name, agent_id, status="pending", metadata=None):
    """Log action to HMI action log"""
    payload = {
        "action_id": action_id,
        "action_name": action_name,
        "agent_id": agent_id,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    response = requests.post(ACTION_LOG_URL, json=payload, timeout=2)
```

**Result**: Events now broadcast correctly (157+ events sent), but...

---

## ⚠️ Issues Found (Not Yet Fixed)

### 1. Demo Data Not Appearing in Actions/Sequencer/Tree

**Status**: Projects ARE being created (15+ in `/tmp/lnsp_demo/`), events ARE being sent (157+ events), but:
- `/api/actions/tasks` returns `{"tasks": []}`
- Sequencer shows no timeline data
- Tree View shows no activity

**Root Cause**: HMI action log endpoint proxies to Registry service:
```python
@app.route('/api/actions/log', methods=['POST'])
def log_action():
    response = requests.post(f'{REGISTRY_URL}/action_logs', json=action_data, timeout=5)
```

**Likely Issue**: Registry service (`localhost:6121`) may not have:
- `action_logs` table in SQLite DB
- `/action_logs` POST endpoint implemented
- Proper storage/retrieval of action data

**Evidence**:
- Registry health check: ✅ `{"status": "ok", "port": 6121}`
- Demo projects created: ✅ 15 projects in `/tmp/lnsp_demo/`
- Events sent: ✅ 157+ events to Event Stream
- Actions retrieved: ❌ 0 tasks in `/api/actions/tasks`

### 2. Agent Heartbeats All Null

**Status**: 15 agents registered, all show `last_heartbeat_ts: null`

**Agents**:
```
Chief Architect, Director of Code, Director of Data,
Code API Manager, Code Impl Manager, Data Schema Manager,
Narrative Manager, OpenAPI Designer (LLM), Doc Generator (Tool),
Backend Developer (LLM), Test Writer (Tool), Schema Designer (LLM),
Migration Builder (Tool), README Writer (LLM), Deployment Guide Writer (Tool)
```

**Root Cause**: No heartbeat service actively sending heartbeats for these agents.

**Expected Flow**:
1. Agents register with Registry → ✅ Working (15 agents registered)
2. Agents send periodic heartbeats → ❌ Not happening
3. Dashboard displays last heartbeat time → Shows "Never"

**Why This Matters**: Without heartbeats, can't tell if agents are alive/healthy.

### 3. Event Counter Only on Event Stream

**Status**: Only Event Stream service card shows "Events: 157". Other core services don't have event counters.

**Core Services**:
- ✅ Event Stream: Has events counter (157+)
- ❌ Registry: No events counter
- ❌ Heartbeat Monitor: No events counter
- ❌ Resource Manager: No events counter
- ❌ Token Governor: No events counter

**User Request**: "Shouldn't all Core Services have an Events counter?"

**Implementation Needed**:
1. Add event counters to each service's health endpoint
2. Update Dashboard to display counters for all services
3. Decide: Should this be "events processed" or "events emitted"?

---

## 🔧 Files Modified Summary

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `services/event_stream/event_stream.py` | 4 lines added | Added `total_events` counter |
| `services/webui/templates/dashboard.html` | 8 lines added | Display events counter + fast refresh logic |
| `services/webui/templates/base.html` | 3 lines changed | Default refresh 0s, range 0-600s |
| `/tmp/lnsp_demo_worker.py` | ~70 lines changed | Fixed event structure, added action logging |

---

## 📊 Performance Metrics

**Event Stream**:
- Total Events: 157+ (lifetime counter)
- Buffered Events: 100 (last 100 events kept)
- Connected Clients: 1
- Status: ✅ Healthy

**Dashboard Refresh**:
- Old Default: 60 seconds (1 update/min)
- New Default: 0 seconds → 500ms (2 updates/sec)
- Range: 0-600 seconds (configurable)
- Performance: Fast, responsive, near real-time

**Demo Worker**:
- Projects Created: 15+ (in `/tmp/lnsp_demo/`)
- Files per Project: 3 (README.md, main.py/index.js, test.py/test.js)
- LLM Used: Ollama + llama3.1:8b (free, local)
- Status: ✅ Running (but stopped to save GPU)

---

## 🐛 Known Bugs & Next Steps

### Critical (Blocking Demo Visibility)

1. **Action Logs Not Persisting**
   - **Symptom**: Demo creates projects, sends action logs, but `/api/actions/tasks` returns empty
   - **Root Cause**: Registry service missing `action_logs` table or `/action_logs` endpoint
   - **Fix**: Add SQLite table + POST/GET endpoints to Registry service
   - **Estimated Effort**: 30 minutes

2. **Agent Heartbeats Not Updating**
   - **Symptom**: All 15 agents show `last_heartbeat_ts: null`
   - **Root Cause**: No service sending heartbeats for these agents
   - **Fix**: Create heartbeat worker or make agents send their own heartbeats
   - **Estimated Effort**: 1 hour

### Medium (UX Improvements)

3. **Event Counters Only on Event Stream**
   - **Symptom**: User requested counters on all Core Services, only Event Stream has one
   - **Fix**: Add event counters to Registry, Heartbeat Monitor, Resource Manager, Token Governor
   - **Estimated Effort**: 1 hour

4. **Sequencer/Tree View Not Showing Data**
   - **Symptom**: No timeline/tree visualization despite events being sent
   - **Root Cause**: Likely depends on Action Logs (issue #1)
   - **Fix**: Verify data flow after fixing Action Logs
   - **Estimated Effort**: 15 minutes (after #1 fixed)

### Low (Nice to Have)

5. **Demo Worker GPU Usage**
   - **Symptom**: User reported GPU usage (likely Ollama LLM calls)
   - **Status**: Demo worker stopped to prevent GPU usage
   - **Consider**: Add GPU usage monitoring or throttling

---

## 🎓 Lessons Learned

1. **Event Structure Matters**: Event Stream expects `{"event_type": "...", "data": {...}}` structure. Flat payloads are silently accepted but cause downstream issues.

2. **Service Dependencies**: HMI depends on Registry for action logs. If Registry doesn't implement the endpoint, data is lost silently (no error shown to user).

3. **Heartbeat Infrastructure**: Having agents in the registry doesn't mean they're alive. Need active heartbeat mechanism.

4. **Fast Refresh Trade-offs**: 500ms polling is fast but may increase server load. Monitor CPU usage in production.

5. **Demo Worker Design**: Using real LLMs (Ollama) for demos is powerful but resource-intensive. Consider:
   - Stub mode for low-resource environments
   - Configurable LLM call frequency
   - GPU usage monitoring

---

## 🚀 Verification Commands

```bash
# Check Event Stream is healthy and counting events
curl -s http://localhost:6102/health | jq .
# Expected: {"total_events": 157+, "connected_clients": 1, ...}

# Check Dashboard refresh setting
# Open browser → http://localhost:6101 → Settings → Refresh Interval
# Expected: Default value = 0, Range = 0-600

# Check demo projects created
ls -la /tmp/lnsp_demo/ | wc -l
# Expected: 15+ directories

# Check action logs (currently broken)
curl -s http://localhost:6101/api/actions/tasks | jq .
# Expected: {"tasks": []} (empty, needs Registry fix)

# Check agent heartbeats (currently broken)
curl -s http://localhost:6101/api/services | jq '.services[] | {name, last_heartbeat_ts}'
# Expected: All show "last_heartbeat_ts": null (needs heartbeat worker)
```

---

## 📝 Documentation Updates Needed

1. **Update `docs/END_USER_GUIDE.md`**:
   - Document new "Events" counter on Dashboard
   - Document fast refresh setting (0-600s range)
   - Add troubleshooting section for missing demo data

2. **Update `services/webui/README.md`** (if exists):
   - Document HMI settings: refreshInterval semantics
   - Document Event Stream integration
   - Document action logging flow

3. **Update `docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md`**:
   - Add Event Stream → HMI data flow diagram
   - Document Registry action_logs dependency
   - Add heartbeat architecture section

4. **Create `docs/TROUBLESHOOTING_HMI.md`**:
   - "Events counter shows 0" → Check Event Stream service
   - "Demo not showing in Actions" → Check Registry action_logs table
   - "Agents show 'Never' heartbeat" → Check heartbeat worker
   - "Dashboard not refreshing" → Check settings refreshInterval

---

## 🔄 Services Status After Session

| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Event Stream | 6102 | ✅ Healthy | Total events: 157+, 1 client connected |
| HMI WebUI | 6101 | ✅ Healthy | Dashboard refresh: 500ms (fast mode) |
| Registry | 6121 | ✅ Healthy | Missing action_logs endpoint |
| Heartbeat Monitor | 6109 | ✅ Healthy | Not sending agent heartbeats |
| Resource Manager | 6104 | ✅ Healthy | No event counter |
| Token Governor | 6105 | ✅ Healthy | No event counter |
| Demo Worker | N/A | ⏹️ Stopped | To save GPU (Ollama LLM calls) |

---

## 🎯 Summary for User

**What Works**:
- ✅ Dashboard Events counter added (Event Stream shows 157+ events)
- ✅ Dashboard refresh super fast now (0.5 seconds by default, configurable 0-600s)
- ✅ Demo worker creating real projects with Ollama LLM (15+ projects)
- ✅ Event structure fixed (now using correct `{"event_type": "...", "data": {...}}` format)

**What Doesn't Work Yet**:
- ❌ Actions/Sequencer/Tree View empty (Registry missing action_logs table/endpoint)
- ❌ Agent heartbeats show "Never" (no heartbeat worker running)
- ❌ Other Core Services don't have event counters (only Event Stream does)

**Next Steps**:
1. Fix Registry service: Add `action_logs` table + POST/GET endpoints
2. Fix agent heartbeats: Create heartbeat worker or make agents self-heartbeat
3. Add event counters to all Core Services
4. Verify Sequencer/Tree View work after fixing action logs

**GPU Usage**: Demo worker stopped to prevent Ollama from consuming GPU. Restart manually if needed:
```bash
python3 /tmp/lnsp_demo_worker.py &
```

---

**End of Session Summary**
