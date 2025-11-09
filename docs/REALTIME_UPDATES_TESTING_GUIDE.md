# Real-time Updates Testing Guide

## Overview

The HMI (Human-Machine Interface) supports real-time updates for both the **Tree View** and **Sequencer** using Server-Sent Events (SSE). This guide explains how to test and verify that real-time updates are working correctly.

## Quick Start

### Step 1: Start the HMI Server

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./.venv/bin/python services/webui/hmi_app.py
```

**Important**: The server must be running BEFORE you run the test script. This is because the server initializes its SSE polling from the current max log_id in the database. Any action_logs inserted AFTER the server starts will trigger real-time updates.

### Step 2: Run the Test Script

```bash
./scripts/test_realtime_updates.sh
```

This script will:
1. Clean up old test data
2. Verify the HMI server is running
3. Prompt you to open the browser
4. Insert action_logs incrementally (every 3 seconds)

### Step 3: Open the Browser

Open two browser tabs:

**Tree View**:
```
http://localhost:6101/tree?task_id=test-realtime-003
```

**Sequencer**:
```
http://localhost:6101/sequencer?task_id=test-realtime-003
```

**Important**: Open the browser tabs BEFORE the test script starts inserting data. This ensures the SSE connection is established before events start flowing.

### Step 4: Watch the Updates

Press ENTER in the terminal when you're ready. You should see:

#### Tree View:
- ✅ New nodes appearing with **green pulsing glow**
- ✅ New delegation edges lighting up in **green**
- ✅ Status changes with smooth color transitions
- ✅ Tree expanding automatically as hierarchy grows

#### Sequencer:
- ✅ Agents appearing in the left sidebar
- ✅ **Tasks appearing on the timeline** with text labels
- ✅ Tasks **color-coded** by status (running, completed, blocked, etc.)
- ✅ Timeline **auto-playing at 1x speed**
- ✅ Playhead moving through the timeline

## What Was Fixed

### Sequencer Fixes

1. **Task Text Labels** (sequencer.html:703-725)
   - Added text rendering on task blocks (if width > 60px)
   - Text truncates with ellipsis if too long
   - White bold text for visibility

2. **Canvas Width Calculation** (sequencer.html:520-523)
   - Fixed canvas width to accommodate full timeline
   - Canvas now expands to `AGENT_LABEL_WIDTH + (timeRangeSeconds * pixelsPerSecond)`
   - Default 1-hour window = 7350px canvas width

3. **Auto-play at 1x Speed** (sequencer.html:579-583)
   - Fixed to set playback speed to 1.0x when auto-playing
   - Previously would auto-play at whatever speed was last set
   - Now always starts at normal speed for active projects

4. **Fixed Function Name Bug** (sequencer.html:583)
   - Changed `togglePlay()` to `togglePlayPause()` (function didn't exist)
   - This was causing auto-play to fail silently

5. **Timeline Offset Fix (2025-11-08)** (sequencer.html:618-668, 694-714)
   - **Problem**: Timeline showed past hour (NOW - 1hr to NOW), current time at RIGHT edge
   - **Solution**: Timeline now shows from TASK START (earliest action) to PRESENT
   - **Result**: Current time appears at left edge, relative time shown as 0:00, 5:00, 10:00...
   - Time labels show elapsed time from task start (MM:SS format)
   - Timeline origin (t=0) is the earliest task's start_time
   - This makes it easy to see task progression from beginning

### Tree View Fixes

1. **SSE Event Handlers** (tree.html:857-887)
   - Already implemented correctly
   - Listens for: `connected`, `new_node`, `new_edge`, `update_node`, `ping`

2. **Animation Classes** (tree.html:889-998)
   - New nodes: green pulse (3 iterations, 1.5s each)
   - New edges: green highlight (2 iterations)
   - Node updates: smooth color transitions

### Backend Fixes

1. **SSE Polling Thread** (hmi_app.py:1454-1517)
   - Already implemented correctly
   - Polls action_logs table every 1 second
   - Notifies all SSE subscribers when new logs appear

2. **SSE Initialization Issue (NOT A BUG!)** (hmi_app.py:1695-1709)
   - **Behavior**: Server initializes `last_known_log_id` from `MAX(log_id)` on startup
   - **Why**: This is CORRECT design - prevents replaying old events on every server restart
   - **Implication**: Test data inserted BEFORE server starts will NOT trigger SSE events
   - **Solution**: Test script now cleans old data, waits for user to restart server, then inserts new data

3. **SSE Endpoints** (hmi_app.py:1519-1689)
   - `/api/stream/action_logs` - General action log stream
   - `/api/stream/tree/<task_id>` - Task-specific tree updates

## Testing Checklist

Run through this checklist to verify everything works:

- [ ] HMI server starts without errors
- [ ] Test script cleans up old data
- [ ] Test script confirms server is running
- [ ] Browser shows Tree View with PAS Agent Swarm root
- [ ] Browser shows Sequencer with empty timeline
- [ ] Browser console shows "SSE connected" message
- [ ] Test script inserts action_logs every 3 seconds
- [ ] Tree view shows new nodes appearing with green pulse
- [ ] Tree view shows new edges in green
- [ ] Sequencer shows agents appearing in left sidebar
- [ ] Sequencer shows tasks appearing on timeline
- [ ] Tasks have text labels (action names)
- [ ] Tasks are color-coded by status
- [ ] Timeline auto-plays at 1x speed
- [ ] Playhead moves smoothly through timeline
- [ ] No errors in browser console
- [ ] No errors in server logs

## Common Issues

### Issue: Tasks don't appear on timeline

**Cause**: Tasks might be outside the visible time window, or canvas width is too small.

**Solution**:
- Check that canvas width >= `AGENT_LABEL_WIDTH + (timeRangeSeconds * PIXELS_PER_SECOND * zoomLevel)`
- Default 1-hour window requires ~7350px canvas
- Use browser dev tools to check canvas dimensions

### Issue: SSE events not received

**Cause**: Server was started AFTER test data was inserted, so `last_known_log_id` is already at max.

**Solution**:
1. Stop the HMI server
2. Run the test script (it cleans up old data)
3. Start the HMI server
4. Open browser tabs
5. Press ENTER in test script to start simulation

**Correct Order**:
1. Clean old data
2. Start HMI server
3. Open browser
4. Insert new data

### Issue: Tree view only shows first node

**Cause**: SSE connection dropped or events are not being sent for subsequent nodes.

**Solution**:
- Check browser console for SSE error messages
- Check server logs for polling thread errors
- Verify `seen_agents` and `seen_edges` tracking in SSE handler

### Issue: Timeline doesn't auto-play

**Cause**: Function name bug or auto-play logic not triggered.

**Solution**:
- Verify `isCurrentTask` is true (task is active)
- Check that `togglePlayPause()` function exists
- Check that `updatePlaybackSpeed(1.0, 'auto')` is called before auto-play

## Architecture

### Data Flow

```
action_logs table (SQLite)
        ↓
poll_action_logs() thread (every 1s)
        ↓
subscriber_queue (in-memory)
        ↓
SSE endpoints (/api/stream/*)
        ↓
Browser EventSource
        ↓
Event handlers (new_node, new_edge, update_node)
        ↓
Tree/Sequencer UI updates
```

### SSE Event Types

| Event | Data | Triggered By |
|-------|------|--------------|
| `connected` | `{status: 'ok', task_id: '...'}` | SSE connection established |
| `new_node` | `{agent_id, name, tier, status}` | New agent appears in action_logs |
| `new_edge` | `{from, to, action_type}` | New delegation between agents |
| `update_node` | `{agent_id, status, action_name}` | Agent status changes |
| `ping` | `{timestamp}` | Keep-alive (every 15s) |

### Timeline Calculations

**Canvas Width**:
```javascript
const pixelsPerSecond = PIXELS_PER_SECOND * zoomLevel; // 2 * 1.0 = 2
const timelineWidth = AGENT_LABEL_WIDTH + (timeRangeSeconds * pixelsPerSecond);
// For 1-hour window: 150 + (3600 * 2) = 7350px
```

**Task Position**:
```javascript
const now = Date.now() / 1000; // Current time (seconds)
const startTime = now - timeRangeSeconds; // Left edge of timeline
const x = AGENT_LABEL_WIDTH + ((taskStartTime - startTime) * pixelsPerSecond);
```

**Task Width**:
```javascript
const duration = taskEndTime - taskStartTime;
const width = Math.max(duration * pixelsPerSecond, 2); // Min 2px
```

## Files Modified

### Frontend
- `services/webui/templates/sequencer.html`
  - Lines 520-523: Canvas width calculation
  - Lines 579-583: Auto-play at 1x speed
  - Lines 703-725: Task text labels

- `services/webui/templates/tree.html`
  - Lines 857-887: SSE event listeners
  - Lines 889-998: Event handlers with animations

### Backend
- `services/webui/hmi_app.py`
  - Lines 1436-1517: SSE polling infrastructure
  - Lines 1519-1689: SSE endpoints

### Test Scripts
- `scripts/test_realtime_updates.sh` - Comprehensive end-to-end test
- `tests/test_tree_realtime_updates.py` - Original Python-only test (still works)

## Performance Considerations

- **Polling Interval**: 1 second (hmi_app.py:1512)
  - Tradeoff: Lower = more responsive, higher CPU
  - 1s is good balance for development

- **SSE Keep-alive**: 15 seconds (hmi_app.py:1553, 1622)
  - Prevents connection timeout
  - Browser auto-reconnects on disconnect

- **Canvas Size**: Dynamic based on time window
  - 1-hour window = 7350px
  - 2-hour window = 14550px
  - May cause memory issues for very long time windows

- **Queue Size**: 100 events (hmi_app.py:1538, 1604)
  - Prevents memory overflow if clients disconnect
  - Old events are dropped when queue fills

## Future Improvements

1. **WebSocket instead of SSE**
   - Bidirectional communication
   - More efficient for high-frequency updates

2. **Incremental Tree Updates**
   - Currently does full tree rebuild on each event
   - Could update individual nodes without refresh

3. **Smarter Canvas Sizing**
   - Virtual scrolling for very long timelines
   - Dynamic time window adjustment

4. **Event Batching**
   - Group multiple events into single SSE message
   - Reduces network overhead

## References

- [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [D3.js Tree Layout](https://d3js.org/d3-hierarchy/tree)
- [Canvas API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
