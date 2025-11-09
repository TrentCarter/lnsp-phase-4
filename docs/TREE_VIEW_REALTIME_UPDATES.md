# Tree View Real-time Updates

## Overview

The Tree View now supports **real-time updates** via Server-Sent Events (SSE). When viewing a specific task, the tree will automatically update as new actions arrive, showing new agents, delegation edges, and status changes **without requiring page reloads**.

## Features

### 1. Auto-updating Tree
- Automatically fetches and displays new agent nodes as they appear in `action_logs`
- Updates delegation edges (parent → child relationships) in real-time
- Updates node status colors when agents transition between states

### 2. Visual Animations
- **New Node Animation**: Green pulsing glow effect (3 iterations, 1.5s each)
- **New Edge Animation**: Green highlighted edge with pulse effect (2 iterations)
- **Node Status Update**: Smooth color transition with brief highlight pulse

### 3. State Preservation
- Tree expansion state is preserved across updates (expanded nodes stay expanded)
- Zoom/pan position is maintained during live updates
- URL updates automatically with `task_id` parameter

## How It Works

### Backend (SSE Endpoint)

**File**: `services/webui/hmi_app.py`

1. **Polling Thread** (`poll_action_logs()`):
   - Runs in background, polling `action_logs` table every 1 second
   - Detects new entries by tracking `last_known_log_id`
   - Notifies all SSE subscribers when new actions arrive

2. **SSE Endpoints**:
   - `/api/stream/tree/<task_id>` - Filtered stream for a specific task
   - `/api/stream/action_logs?task_id=...` - General action_log stream

3. **Event Types**:
   - `connected` - Initial handshake when client connects
   - `new_node` - New agent appeared in the task
   - `new_edge` - New delegation relationship created
   - `update_node` - Existing agent status changed
   - `ping` - Keep-alive heartbeat (every 15s)

### Frontend (JavaScript)

**File**: `services/webui/templates/tree.html`

1. **SSE Connection** (`connectSSE()`):
   - Automatically connects when `task_id` is present in URL
   - Registers event listeners for `new_node`, `new_edge`, `update_node`
   - Auto-reconnects on connection failure (5s retry)

2. **Event Handlers**:
   - `handleNewNode()` - Fetches fresh tree data, applies pulse animation
   - `handleNewEdge()` - Refreshes tree, highlights new delegation edge
   - `handleNodeUpdate()` - Updates node color/status without full refresh

3. **Animations**:
   - CSS classes: `.new-node`, `.new-edge`
   - D3.js transitions for smooth color changes
   - Automatic cleanup after animation completes

## Usage

### 1. Viewing Real-time Updates

```bash
# Step 1: Start the HMI server (if not already running)
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./.venv/bin/python services/webui/hmi_app.py

# Step 2: Open Tree View in browser with a task_id
http://localhost:6101/tree?task_id=test-realtime-001
```

### 2. Testing with Simulated Data

```bash
# Run the test script to simulate a live project
./.venv/bin/python tests/test_tree_realtime_updates.py

# This will:
# - Insert action_log entries every 3 seconds
# - Simulate a delegation flow: User → VP → Director → Manager → Programmer
# - Show 9 steps total (delegation, work, review, approval)
```

**Expected Behavior**:
1. Tree starts empty (or with previous historical data)
2. Every 3 seconds, a new action appears:
   - **New nodes** pulse with green glow
   - **New edges** light up in green
   - **Status changes** smoothly transition colors
3. After 27 seconds (9 steps × 3s), the full delegation hierarchy is visible

### 3. Manual Testing

You can manually insert action_log entries to test specific scenarios:

```python
import sqlite3
import json
from datetime import datetime

db_path = 'artifacts/registry/registry.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO action_logs
    (task_id, action_type, action_name, from_agent, to_agent, tier_from, tier_to, status, timestamp, action_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    'test-realtime-001',          # task_id
    'delegate',                    # action_type
    'Test delegation',             # action_name
    'manager_001',                 # from_agent
    'programmer_002',              # to_agent
    3,                             # tier_from (Manager)
    4,                             # tier_to (Programmer)
    'running',                     # status
    datetime.now().isoformat(),    # timestamp
    json.dumps({'test': True})     # action_data
))

conn.commit()
conn.close()
```

## API Reference

### SSE Endpoint: `/api/stream/tree/<task_id>`

**Request**:
```
GET /api/stream/tree/test-realtime-001
```

**Response** (text/event-stream):
```
event: connected
data: {"status": "ok", "task_id": "test-realtime-001"}

event: new_node
data: {"agent_id": "vp_001", "name": "Vp 001", "tier": 1, "status": "running"}

event: new_edge
data: {"from": "user", "to": "vp_001", "action_type": "delegate"}

event: update_node
data: {"agent_id": "vp_001", "status": "completed", "action_name": "Task completed"}

event: ping
data: {"timestamp": "2025-11-08T10:30:00.000Z"}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Browser (Tree View)                                             │
│                                                                 │
│  ┌──────────────┐    EventSource    ┌─────────────────────┐    │
│  │ JavaScript   │ ←──────────────── │ SSE Connection      │    │
│  │ Event        │                   │ /api/stream/tree/   │    │
│  │ Handlers     │                   │ {task_id}           │    │
│  └──────────────┘                   └─────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ D3.js Tree Visualization                             │      │
│  │ - Nodes (agents)                                     │      │
│  │ - Edges (delegations)                                │      │
│  │ - Animations (pulse, color transitions)              │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │ SSE Events
                             │
┌────────────────────────────┴────────────────────────────────────┐
│ Flask HMI App (services/webui/hmi_app.py)                       │
│                                                                 │
│  ┌──────────────────┐       ┌──────────────────────────────┐   │
│  │ SSE Generator    │       │ Background Polling Thread    │   │
│  │ stream_tree_     │ ←───  │ poll_action_logs()          │   │
│  │ updates()        │       │ - Polls DB every 1s         │   │
│  └──────────────────┘       │ - Notifies subscribers      │   │
│                             └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │ SQL Queries
                             │
┌────────────────────────────┴────────────────────────────────────┐
│ SQLite (artifacts/registry/registry.db)                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ action_logs Table                                         │ │
│  │ - log_id (auto-increment)                                 │ │
│  │ - task_id                                                 │ │
│  │ - from_agent, to_agent                                    │ │
│  │ - tier_from, tier_to                                      │ │
│  │ - status, timestamp                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Considerations

1. **Polling Interval**: 1 second (adjustable in `poll_action_logs()`)
2. **SSE Keep-alive**: 15 seconds (prevents connection timeout)
3. **Reconnect Delay**: 5 seconds on error
4. **Animation Duration**:
   - New node: 4.5 seconds total (3 iterations × 1.5s)
   - New edge: 3 seconds total (2 iterations × 1.5s)
5. **Queue Size**: 100 events per subscriber (prevents memory overflow)

## Troubleshooting

### SSE Not Connecting

**Check browser console**:
```javascript
// Should see:
SSE connected: {status: "ok", task_id: "..."}

// If not:
// 1. Verify task_id is in URL: ?task_id=...
// 2. Check server logs for SSE errors
// 3. Verify Registry DB exists at artifacts/registry/registry.db
```

### Tree Not Updating

**Check action_logs table**:
```bash
sqlite3 artifacts/registry/registry.db
SELECT * FROM action_logs WHERE task_id='test-realtime-001' ORDER BY log_id DESC LIMIT 5;
```

**Check server logs**:
```
# Should see:
Initialized last_known_log_id to 123
Started background action_logs polling thread
```

### Animations Not Playing

**Check CSS class application**:
```javascript
// In browser console:
document.querySelectorAll('.new-node')  // Should find nodes with animation
document.querySelectorAll('.new-edge')  // Should find edges with animation
```

## Limitations

1. **Task-Specific**: SSE only works when `task_id` is specified in URL
2. **Polling-Based**: Uses database polling (not native triggers)
3. **Full Tree Refresh**: New nodes/edges trigger full tree rebuild (preserves state but re-renders)
4. **Single Database**: Only monitors local SQLite `action_logs` table

## Future Enhancements

1. **Incremental Updates**: Add nodes/edges without full tree rebuild
2. **WebSocket Support**: Bi-directional communication for richer interactions
3. **Multi-Task View**: Subscribe to multiple tasks simultaneously
4. **Filtered Subscriptions**: Filter by agent tier, status, or action type
5. **Replay Mode**: Scrub timeline to view historical states

## References

- **SSE Spec**: https://html.spec.whatwg.org/multipage/server-sent-events.html
- **D3.js Transitions**: https://github.com/d3/d3-transition
- **Flask Streaming**: https://flask.palletsprojects.com/en/2.3.x/patterns/streaming/
