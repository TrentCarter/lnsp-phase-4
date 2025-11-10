# Debug Guide: Flash Only Works for First 5 Tasks

## ✅ FIXED (2025-11-09)

**Root Cause**: Off-screen viewport culling was preventing flashes for tasks not visible in the current scroll position.

**Fix Location**: `services/webui/templates/sequencer.html` lines 1202-1206

The issue was that line 1203 had:
```javascript
// Skip if task block is not visible on screen
if (y + blockHeight < HEADER_HEIGHT || y > canvas.height) return;
```

This caused tasks scrolled out of view to skip ALL rendering, including flash overlays.

**Solution**: Check if task should flash BEFORE the visibility check, then only skip if not visible AND not flashing:
```javascript
// Check if task is currently playing sound (needs flash even if off-screen)
const shouldFlash = currentlyPlayingTasks.has(task.task_id);

// Skip if task block is not visible on screen AND not currently flashing
if (!shouldFlash && (y + blockHeight < HEADER_HEIGHT || y > canvas.height)) return;
```

Now flashing tasks are always rendered, even if off-screen!

---

## Problem (Original)
Visual flashes only appear for the first 5 tasks when sounds play, despite all sounds triggering correctly.

## Diagnostic Steps

### Step 1: Check Console Logs
Open browser console (F12) and look for these patterns:

```
🔊 [SOUND] Task STARTED: "Task Name" (id=XXX)
✨ [FLASH] Added task "Task Name" to flash set (size=N)
🎨 [DRAW] Drawing N tasks, currentlyPlayingTasks.size=N
✨ [FLASH RENDER] Drawing flash for "Task Name" (id=XXX)
✨ [FLASH] Removed task "Task Name" from flash set (size=N)
```

**Questions to answer:**
1. Do you see `[FLASH] Added` logs for MORE than 5 tasks?
2. Do you see `[FLASH RENDER]` logs for MORE than 5 tasks?
3. What is the `currentlyPlayingTasks.size` when sounds 6+ play?

### Step 2: Check Task IDs
In console, run:
```javascript
// Check if all tasks have unique IDs
console.table(tasks.map(t => ({name: t.name, id: t.task_id})));

// Check current flash set
console.log('Currently flashing:', Array.from(currentlyPlayingTasks));
```

### Step 3: Manual Test
In console, manually add task IDs to the flash set:
```javascript
// Add task 6's ID manually
currentlyPlayingTasks.add(tasks[5].task_id);
drawSequencer();
// Does task 6 flash now?
```

## Hypotheses

### Hypothesis 1: Timing Issue
**Theory**: Tasks 6+ are added to `currentlyPlayingTasks`, but removed before next draw cycle.

**Test**: Increase note duration to 500ms in settings. Do more tasks flash?

**Fix**: Change line 2043 from `setTimeout(processNextSound, 10)` to `setTimeout(processNextSound, 0)` to process sounds faster.

### Hypothesis 2: Set Size Limit
**Theory**: JavaScript Set has some unexpected limit (unlikely, but possible).

**Test**:
```javascript
// In console
const testSet = new Set();
for (let i = 0; i < 100; i++) { testSet.add(i); }
console.log('Set size:', testSet.size); // Should be 100
```

### Hypothesis 3: Task ID Collision
**Theory**: Tasks 6+ have duplicate task_ids, causing Set overwrites.

**Test**:
```javascript
// Check for duplicate IDs
const ids = tasks.map(t => t.task_id);
const uniqueIds = new Set(ids);
console.log(`Total tasks: ${ids.length}, Unique IDs: ${uniqueIds.size}`);
```

### Hypothesis 4: Draw Loop Issue
**Theory**: `drawSequencer()` is called, but canvas rendering stops after 5 tasks.

**Test**: Add debug log inside the forEach at line 1178:
```javascript
tasks.forEach((task, index) => {
    console.log(`Drawing task ${index}: "${task.name}", shouldFlash=${currentlyPlayingTasks.has(task.task_id)}`);
    // ... rest of code
});
```

### Hypothesis 5: Canvas Clipping
**Theory**: Tasks 6+ are off-screen or outside canvas viewport.

**Test**: Check X positions in `[FLASH RENDER]` logs. Are tasks 6+ outside the visible area?

## Most Likely Cause

Based on the code review, I believe the issue is **Hypothesis 4: Draw Loop stops checking after 5 tasks**.

Look at line 1178 - the `tasks.forEach()` loop should process ALL tasks, but there might be:
1. A `break` or `return` statement hidden somewhere
2. An exception thrown after 5 iterations
3. The `tasks` array itself only has 5 elements when flashing occurs

## Quick Fix to Try

Add more verbose logging to see exactly what's happening:

```javascript
// Around line 1178, BEFORE tasks.forEach:
console.log(`🎨 [DRAW START] About to draw ${tasks.length} tasks`);
console.log(`   currentlyPlayingTasks contents:`, Array.from(currentlyPlayingTasks));

tasks.forEach((task, index) => {
    const shouldFlash = currentlyPlayingTasks.has(task.task_id);
    console.log(`  Task ${index}: "${task.name}" (id=${task.task_id}), shouldFlash=${shouldFlash}`);

    // ... rest of existing code
});
```

This will show:
- Which task IDs are in the flash set
- Which tasks are being checked
- Which tasks should flash but don't

## Expected Behavior

With 10 tasks playing sounds with 150ms duration and 10ms spacing:
- All 10 tasks should be added to `currentlyPlayingTasks`
- All 10 should show `[FLASH RENDER]` logs
- At t=200ms, task 1 should be removed (200ms after it was added)
- Rolling window of ~15 concurrent flashes should be visible

## Contact

If none of these diagnostics reveal the issue, please share:
1. Console log output (especially `[FLASH]` and `[DRAW]` messages)
2. `tasks.length` value
3. Result of `console.log(Array.from(currentlyPlayingTasks))` during playback
