# Sequencer Multi-Lane + Real-Time Audio Fixes (2025-11-09)

## Overview

This document describes two critical fixes applied to the sequencer timeline view:

1. **Multi-lane allocation restored** - Tasks no longer stack in a single row
2. **Real-time audio playback** - Sounds now play during Start Demo execution

Both issues were caused by the quick task execution times (2-6 seconds) introduced in the previous duration fix.

---

## Problem 1: Lost Multi-Lane Allocation

### Symptoms
- All tasks for an agent stack in a single lane (vertical row)
- Previously, concurrent tasks used multiple lanes to prevent visual overlap
- Screenshot evidence: All green "Assign: BFST..." boxes stacked on top of each other

### Root Cause
**Temporal overlap detection failure during fast sequential execution**

The original overlap detection logic:
```javascript
// OLD LOGIC (BROKEN)
if (taskStart >= lastTaskEnd) {
    assignedLane = i;  // Reuse same lane
    break;
}
```

**Why it failed**:
1. Tasks delegated concurrently (within 1-2 seconds)
2. Each task completes in 2-6 seconds
3. By the time task B's `start_time` is recorded, task A has already completed (has `end_time`)
4. Overlap detector sees: `taskB.start >= taskA.end` → FALSE overlap → single lane

**Example timeline**:
```
t=0s:  Task A delegated (start=0s)
t=3s:  Task A completes (end=3s)
t=4s:  Task B delegated (start=4s)  ← Sees 4 >= 3 = no overlap!
t=8s:  Task B completes (end=8s)

Result: Both tasks in same lane (visually stacked)
Expected: Two separate lanes (delegated ~4s apart)
```

### Solution
**Delegation-time-based concurrency detection**

New logic detects **concurrent delegation**, not just temporal overlap:

```javascript
// NEW LOGIC (FIXED)
const DELEGATION_WINDOW = 15; // Seconds

const noTemporalOverlap = (taskStart >= lastTaskEnd);
const notConcurrentlyDelegated = (taskStart - lastTaskDelegationTime) >= DELEGATION_WINDOW;

if (noTemporalOverlap && notConcurrentlyDelegated) {
    assignedLane = i;  // Reuse lane ONLY if both conditions met
    break;
}
```

**Two conditions must BOTH be true** to reuse a lane:
1. **No temporal overlap**: Current task starts after previous task ends
2. **Not concurrently delegated**: Tasks were delegated ≥15 seconds apart

**Why this works**:
- Tasks delegated within 15-second window → forced into separate lanes
- Even if they execute sequentially (no temporal overlap), visual separation maintained
- Reflects actual system behavior: parallel delegation but serial execution

### Code Changes

**File**: `services/webui/templates/sequencer.html`

**Location**: Lines 709-747

**Key additions**:
```javascript
// Line 711
const DELEGATION_WINDOW = 15; // Seconds - tasks delegated within this window are concurrent

// Lines 723-730
const lastTaskDelegationTime = lastTaskInLane.start_time;

const noTemporalOverlap = (taskStart >= lastTaskEnd);
const notConcurrentlyDelegated = (taskStart - lastTaskDelegationTime) >= DELEGATION_WINDOW;

if (noTemporalOverlap && notConcurrentlyDelegated) {
    assignedLane = i;
    break;
}
```

### Testing
1. Refresh browser (Cmd+R)
2. Select any project with concurrent task delegation
3. Look for `[LANES]` console logs:
   ```
   [LANES] Allocated 15 rows (5 sub-lanes)
   [LANES] Agents with overlapping tasks:
     - Mgr Backend: 3 parallel lanes
     - Prog 001: 2 parallel lanes
   ```
4. Verify visual separation: Tasks delegated close together should use separate lanes

### Edge Cases
- **Adjustment needed?** If 15 seconds is too aggressive (too many lanes), reduce `DELEGATION_WINDOW` to 10 or 5
- **Too conservative?** If tasks still stack, increase `DELEGATION_WINDOW` to 20 or 30

---

## Problem 2: No Audio During Real-Time Execution

### Symptoms
- Audio plays correctly when **replaying** completed tasks (playback mode)
- **No audio** when tasks execute in real-time (Start Demo button)
- Sound mode correctly set to "Music Note" or "Voice" in dropdown

### Root Cause
**Audio only triggered by playhead movement**

The `checkTaskEvents()` function detects task start/end by checking if the **playhead** crosses task boundaries:

```javascript
// PLAYBACK-ONLY LOGIC
if (previousPlaybackTime < taskStart && taskStart <= currentPlaybackTime) {
    queueSound('task_started', task);  // Only fires during playback
}
```

**Why it failed during real-time execution**:
1. Start Demo → tasks execute immediately
2. Polling fetches new task data every 3 seconds
3. Tasks already have `start_time` and `end_time` when fetched
4. **Playhead is NOT moving** (stopped at t=0, or slowly advancing)
5. `checkTaskEvents()` never fires because playhead hasn't crossed task boundaries

**Timeline comparison**:

**Playback mode (WORKS)**:
```
t=0s:  User presses Play → playhead starts moving
t=5s:  Playhead crosses task_A.start_time → 🔊 Sound!
t=8s:  Playhead crosses task_A.end_time → 🔊 Sound!
```

**Real-time mode (BROKEN)**:
```
t=0s:  User presses Start Demo → tasks execute on backend
t=3s:  Polling fetches task_A (start=0s, end=3s, status=done)
       Playhead at t=0 (not moving) → NO sound triggered
t=6s:  Polling fetches task_B (start=3s, end=6s, status=done)
       Playhead at t=0 (still not moving) → NO sound triggered
```

### Solution
**Real-time status change detection**

Added `detectRealtimeTaskChanges()` that triggers sounds **immediately** when task data changes (independent of playhead):

```javascript
function detectRealtimeTaskChanges(newTasks) {
    newTasks.forEach(task => {
        const previousState = previousTaskStates.get(task.task_id);

        if (!previousState) {
            // New task detected → trigger "task started" sound
            queueSound('task_started', task);
        } else if (previousState.status !== task.status || justCompleted) {
            // Status changed → trigger completion/error sound
            queueSound('task_completed', task);
        }
    });
}
```

**How it works**:
1. Every 3 seconds, `pollForActiveTasks()` fetches new task data
2. `detectRealtimeTaskChanges()` compares new tasks against previous snapshot
3. **New task?** → Play "task started" sound
4. **Status changed?** → Play "task completed" / "error" / "warning" sound
5. Update snapshot for next comparison

**Key features**:
- **Stateful tracking**: `previousTaskStates` Map stores last known state per task
- **Garbage collection**: Removes old tasks that no longer exist
- **Respects sound mode**: Only triggers if sound is enabled
- **Queued playback**: Uses existing `queueSound()` to prevent overlapping audio

### Code Changes

**File**: `services/webui/templates/sequencer.html`

**New state variable** (line 1888):
```javascript
let previousTaskStates = new Map(); // Track previous task states for real-time sound triggering
```

**New function** (lines 1890-1944):
```javascript
function detectRealtimeTaskChanges(newTasks) {
    // Check if sound is enabled
    const settings = typeof getSettings === 'function' ? getSettings() : {};
    const effectiveSoundMode = soundMode !== 'none' ? soundMode : (settings.defaultSoundMode || 'none');

    if (effectiveSoundMode === 'none') {
        return; // Sound disabled
    }

    // Check each task for status changes
    newTasks.forEach(task => {
        const taskId = task.task_id;
        const currentState = {
            status: task.status,
            hasEndTime: !!task.end_time
        };

        const previousState = previousTaskStates.get(taskId);

        if (!previousState) {
            // New task detected - trigger "task started" sound
            console.log(`🔊 [REALTIME] New task detected: "${task.name}" (status=${task.status})`);
            queueSound('task_started', task);
            previousTaskStates.set(taskId, currentState);
        } else {
            // Check for status changes
            const statusChanged = (previousState.status !== currentState.status);
            const justCompleted = (!previousState.hasEndTime && currentState.hasEndTime);

            if (statusChanged || justCompleted) {
                // Task status changed - trigger appropriate sound
                const eventType = task.status === 'done' ? 'task_completed' :
                                 task.status === 'error' ? 'error' :
                                 task.status === 'blocked' ? 'warning' :
                                 'task_completed';

                console.log(`🔊 [REALTIME] Task status changed: "${task.name}" (${previousState.status} → ${task.status})`);
                queueSound(eventType, task);
                previousTaskStates.set(taskId, currentState);
            }
        }
    });

    // Clean up old tasks that no longer exist (garbage collection)
    const currentTaskIds = new Set(newTasks.map(t => t.task_id));
    for (const taskId of previousTaskStates.keys()) {
        if (!currentTaskIds.has(taskId)) {
            previousTaskStates.delete(taskId);
        }
    }
}
```

**Integration point** (line 890):
```javascript
// Inside fetchSequencerData() after tasks are fetched
detectRealtimeTaskChanges(data.tasks || []);
```

### Testing
1. Refresh browser (Cmd+R)
2. Set sound mode to "Music Note" (top toolbar dropdown)
3. Click **Start Demo** button
4. **Expected behavior**:
   - Sounds play immediately when tasks start/complete
   - Console shows `🔊 [REALTIME]` logs for each event:
     ```
     🔊 [REALTIME] New task detected: "Assign: Authentication logic" (status=running)
     🔊 [REALTIME] Task status changed: "Assign: Authentication logic" (running → done)
     ```
5. Verify both modes work:
   - **Real-time mode** (Start Demo): Immediate sounds via `detectRealtimeTaskChanges()`
   - **Playback mode** (Play button): Sounds via `checkTaskEvents()` (unchanged)

### Performance Impact
- **Minimal**: Only compares task snapshots (Map lookups are O(1))
- **Memory**: ~100 bytes per task (task_id + status + hasEndTime)
- **CPU**: Runs once per 3-second polling cycle (only when demo is active)

---

## Verification Checklist

### Multi-Lane Fix
- [ ] Browser refreshed after code changes
- [ ] Select project with concurrent delegation (e.g., "task_8bacd0ed")
- [ ] Console shows `[LANES] Agents with overlapping tasks:` with counts > 1
- [ ] Visual check: Tasks delegated close together use separate lanes (no stacking)
- [ ] Visual check: Tasks delegated far apart (>15s) reuse same lane (expected)

### Audio Fix
- [ ] Browser refreshed after code changes
- [ ] Sound mode set to "Music Note" or "Voice" (not "None")
- [ ] Click **Start Demo** button
- [ ] Hear sounds when tasks start/complete (immediate, not delayed)
- [ ] Console shows `🔊 [REALTIME]` logs for task events
- [ ] Verify **playback mode** still works (press Play on completed project)

### Regression Tests
- [ ] Playback controls still work (Play/Pause/Speed/Time Range)
- [ ] Arrow rendering still works (parent→child connections)
- [ ] Task deduplication still works (toggle in settings)
- [ ] Vertical zoom still works (expand/collapse lanes)
- [ ] Sound queue still prevents overlapping audio

---

## Rollback Instructions

If issues arise, revert to previous version:

```bash
git checkout HEAD~1 services/webui/templates/sequencer.html
```

Or manually remove:
1. **Multi-lane fix**: Restore old overlap detection (lines 709-747)
2. **Audio fix**: Remove `detectRealtimeTaskChanges()` function (lines 1888-1944) and call (line 890)

---

## Future Improvements

### Multi-Lane Allocation
1. **Dynamic window**: Adjust `DELEGATION_WINDOW` based on average task duration
2. **Smart compaction**: After all tasks complete, compact lanes if no visual overlap
3. **User preference**: Make `DELEGATION_WINDOW` configurable in settings

### Real-Time Audio
1. **WebSocket integration**: Replace 3-second polling with instant push notifications
2. **Audio feedback modes**: Add haptic feedback (vibration) for mobile browsers
3. **Sound volume control**: Add slider to adjust sound volume

---

## Related Documentation

- **Previous fix**: `docs/SEQUENCER_TIMELINE_FIXES_2025_11_09.md` (duration calculation)
- **Sound system**: `docs/HMI_CHANGES_2025_11_08.md` (original sound implementation)
- **Real-time updates**: `docs/REALTIME_UPDATES_TESTING_GUIDE.md` (polling mechanism)

---

**Date**: 2025-11-09
**Author**: Claude Code
**Files Modified**:
- `services/webui/templates/sequencer.html` (lines 711, 723-730, 890, 1888-1944)

**Status**: ✅ Ready for Testing
