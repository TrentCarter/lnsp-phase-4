# Real-time Updates Fixes (2025-11-08)

## Summary

Fixed critical issues with Tree View and Sequencer real-time updates. Both views now properly display live data with correct timing and visual feedback.

## Issues Fixed

### 1. Tree View SSE Not Working ✅

**Problem**: Tree View was not receiving SSE events for new action_logs.

**Root Cause**:
- Server initializes `last_known_log_id = MAX(log_id)` on startup (line hmi_app.py:1695-1709)
- If test data exists BEFORE server starts, it won't trigger SSE events
- This is **correct behavior** (prevents replaying old events on every restart)

**Solution**:
- Updated test script to guide users through correct testing procedure:
  1. Clean old test data
  2. Restart HMI server (to reset SSE tracker)
  3. Open browser tabs
  4. Insert new data (triggers SSE events)

**Files Changed**:
- `scripts/test_realtime_updates.sh` - Added server restart prompt

### 2. Sequencer Timeline Offset Wrong ✅

**Problem**: Timeline showed past hour (NOW - 1hr to NOW), with current time at RIGHT edge. User wanted current time at LEFT edge (relative time 0).

**Root Cause**:
```javascript
// BEFORE (wrong):
const startTime = now - timeRangeSeconds;  // Shows last hour ending NOW
```

**Solution**:
```javascript
// AFTER (correct):
const startTime = earliestTaskTime;  // Shows from task start to present
```

Timeline now shows:
- **Left edge**: First action's timestamp (t=0)
- **Right edge**: Current time (or latest action)
- **Time labels**: Relative elapsed time (0:00, 5:00, 10:00... MM:SS format)

**Files Changed**:
- `services/webui/templates/sequencer.html:618-668` - drawTimelineGrid()
- `services/webui/templates/sequencer.html:694-714` - drawTaskBlocks()

**Visual Result**:
```
Before:                    After:
┌────────────────────┐    ┌────────────────────┐
│ -60m  -30m   NOW → │    │ 0:00  5:00  10:00  │
│  [=========●]      │    │ ●[=============]   │
└────────────────────┘    └────────────────────┘
  Past hour view           Task progression view
  (current at right)       (start at left)
```

## Testing Procedure

### ✅ Correct Order (Events Fire)

1. **Clean old data** (test script does this automatically)
2. **Restart HMI server** (important!)
   ```bash
   ./.venv/bin/python services/webui/hmi_app.py
   ```
3. **Run test script** (prompts you to open browser)
   ```bash
   ./scripts/test_realtime_updates.sh
   ```
4. **Open browser tabs** when prompted
5. **Watch real-time updates** appear

### ❌ Wrong Order (No Events)

1. Insert test data
2. Start HMI server (initializes last_known_log_id to max)
3. Open browser
4. **Result**: No SSE events (data already exists)

## Expected Results

### Tree View
- ✅ 9 nodes appear with **green pulsing glow**
- ✅ Edges connect with **green highlights**
- ✅ Hierarchical layout (auto-expanding tree)
- ✅ Smooth status color transitions

### Sequencer
- ✅ 9 tasks visible on timeline
- ✅ **Timeline starts at 0:00** (left edge = task start)
- ✅ Time labels: **0:00, 5:00, 10:00...** (MM:SS from start)
- ✅ Tasks have **text labels** (action names)
- ✅ **Color-coded** by status:
  - Blue/Yellow/Green: Running (0-25%, 25-75%, 75-100%)
  - Orange: Blocked/Waiting
  - Purple: Awaiting Approval
  - Red: Error/Stuck
  - Gray: Done/Idle
- ✅ **Auto-plays at 1x speed** (for active tasks)

## Technical Details

### SSE Architecture

```
┌─────────────────────────────────────────────┐
│ HMI Server (Flask)                          │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │ poll_action_logs() (background)      │  │
│  │ - Polls DB every 1s                  │  │
│  │ - Tracks last_known_log_id           │  │
│  │ - Notifies all SSE subscribers       │  │
│  └──────────────────────────────────────┘  │
│               │                             │
│               ↓                             │
│  ┌──────────────────────────────────────┐  │
│  │ /api/stream/tree/<task_id>           │  │
│  │ - SSE endpoint                       │  │
│  │ - Filters by task_id                 │  │
│  │ - Sends: new_node, new_edge, update  │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                │
                ↓ (EventSource)
┌─────────────────────────────────────────────┐
│ Browser (tree.html)                         │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │ SSE Event Handlers                   │  │
│  │ - connected → log message            │  │
│  │ - new_node → add node + green glow   │  │
│  │ - new_edge → add edge + highlight    │  │
│  │ - update_node → change status/color  │  │
│  │ - ping → keep-alive (every 15s)      │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Timeline Calculation (Sequencer)

**Before (wrong)**:
```javascript
const now = Date.now() / 1000;
const startTime = now - timeRangeSeconds;  // 1 hour ago
const x = AGENT_LABEL_WIDTH + ((taskStartTime - startTime) * pixelsPerSecond);
// Result: Tasks appear based on absolute time
//         Current time at right edge
```

**After (correct)**:
```javascript
// Find earliest task to set timeline origin
let earliestTime = Infinity;
tasks.forEach(task => {
    if (task.start_time < earliestTime) earliestTime = task.start_time;
});

const startTime = earliestTime;  // Timeline origin (t=0)
const x = AGENT_LABEL_WIDTH + ((taskStartTime - startTime) * pixelsPerSecond);
// Result: Tasks appear at relative time from task start
//         First task at left edge (t=0)
```

**Time Label Rendering**:
```javascript
// Show relative time from task start (MM:SS format)
const minutesFromStart = Math.floor(t / 60);
const secondsRemainder = Math.floor(t % 60);
const timeStr = t === 0 ? '0:00' :
               `${minutesFromStart}:${secondsRemainder.toString().padStart(2, '0')}`;
```

## Files Modified

1. **services/webui/templates/sequencer.html**
   - Lines 618-668: `drawTimelineGrid()` - Timeline calculation and labels
   - Lines 694-714: `drawTaskBlocks()` - Task positioning

2. **scripts/test_realtime_updates.sh**
   - Lines 32-55: Added server restart prompt and instructions
   - Lines 238-259: Updated expected results output

3. **docs/REALTIME_UPDATES_TESTING_GUIDE.md**
   - Lines 86-92: Added timeline offset fix documentation
   - Lines 112-116: Added SSE initialization explanation

## Troubleshooting

### Tree View not updating?
1. ✅ Restart HMI server
2. ✅ Check browser console for SSE connection
3. ✅ Verify Network tab shows `/api/stream/tree/...` connection
4. ✅ Check server logs for polling activity

### Sequencer timeline wrong?
1. ✅ Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+F5)
2. ✅ Check that tasks have `start_time` fields
3. ✅ Verify timeline shows "0:00" at left edge
4. ✅ Check browser console for JavaScript errors

### No SSE events firing?
1. ✅ Restart HMI server (resets `last_known_log_id`)
2. ✅ Clean old test data (script does this automatically)
3. ✅ Insert NEW data AFTER server starts
4. ✅ Check that action_logs have unique log_id values

## Performance Notes

- SSE polling: 1 second intervals (configurable at hmi_app.py:1512)
- SSE keep-alive: 15 seconds (prevents connection timeout)
- Timeline redraw: On data fetch, every playhead update
- Tree animations: CSS-based (GPU accelerated)

## References

- SSE Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
- Tree View Implementation: `services/webui/templates/tree.html`
- Sequencer Implementation: `services/webui/templates/sequencer.html`
- Backend SSE: `services/webui/hmi_app.py:1454-1689`
- Test Script: `scripts/test_realtime_updates.sh`
- Testing Guide: `docs/REALTIME_UPDATES_TESTING_GUIDE.md`
