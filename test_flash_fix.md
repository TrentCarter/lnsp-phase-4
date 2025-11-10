# Test Plan: Verify Flash Fix Works

## Prerequisites
- WebUI running on http://localhost:6101
- Browser with console access (F12)

## Test Case 1: All Tasks Flash Regardless of Scroll Position

### Setup
1. Navigate to http://localhost:6101/sequencer
2. Wait for data to load (should see task blocks in timeline)
3. Open browser console (F12)

### Test Steps
1. **Count visible tasks**: Note how many task rows are visible without scrolling (should be ~5-8)
2. **Enable sound**: Settings → Sound Mode → "Music Note"
3. **Reset playhead**: Double-click Stop button (rewind to t=0)
4. **Start playback**: Click Play ▶️
5. **Watch console**: Look for `✨ [FLASH RENDER]` logs

### Expected Results
✅ Should see `✨ [FLASH RENDER]` logs for ALL tasks that trigger sounds, not just first 5
✅ Tasks should flash bright white when their sound plays
✅ Console should show logs like:
```
✨ [FLASH RENDER] Drawing flash for "Task 1" (id=abc) at x=123
✨ [FLASH RENDER] Drawing flash for "Task 2" (id=def) at x=456
...
✨ [FLASH RENDER] Drawing flash for "Task 10" (id=xyz) at x=789  ← Should now work!
```

## Test Case 2: Off-Screen Tasks Flash

### Setup
1. Same as Test Case 1
2. Scroll timeline DOWN so first 5 tasks are OFF-SCREEN
3. Only tasks 6-10 should be visible

### Test Steps
1. Double-click Stop (rewind)
2. Click Play ▶️
3. Watch for flashes on VISIBLE tasks (6-10)
4. Check console for `✨ [FLASH RENDER]` logs

### Expected Results
✅ Visible tasks (6-10) should flash
✅ Console should show flash logs for ALL tasks (1-10), even off-screen ones
✅ No errors about undefined positions

## Test Case 3: Rapid Scrolling During Playback

### Setup
1. Same as Test Case 1
2. Start playback

### Test Steps
1. Click Play ▶️
2. While sounds are playing, rapidly scroll timeline up and down
3. Watch for flashes

### Expected Results
✅ Tasks should continue flashing even as you scroll
✅ No visual artifacts or rendering glitches
✅ Console shows continuous `✨ [FLASH RENDER]` logs

## Regression Tests

### No Performance Degradation
**Test**: Load sequencer with 50+ tasks, scroll rapidly
**Expected**: Smooth scrolling, no frame drops

### Off-Screen Culling Still Works
**Test**: Scroll to bottom of long task list
**Expected**: Tasks at top should NOT render (except when flashing)

### Flash Duration Setting
**Test**: Change note duration slider (50ms → 500ms)
**Expected**: Flash duration changes to match setting

## Debug Commands

If issues occur, run these in browser console:

```javascript
// Check flash set size
console.log('Currently flashing:', currentlyPlayingTasks.size);

// Check flash set contents
console.log('Flash IDs:', Array.from(currentlyPlayingTasks));

// Check task array
console.log('Total tasks:', tasks.length);

// Manual flash test
currentlyPlayingTasks.add(tasks[9].task_id); // Force task 10 to flash
drawSequencer(); // Redraw
// Should see task 10 flash even if off-screen!
```

## Success Criteria
- [ ] All tasks flash when sounds play (verified by console logs)
- [ ] Off-screen tasks still flash (verified by adding task ID manually)
- [ ] No performance degradation
- [ ] No visual artifacts during scrolling
- [ ] Flash duration setting works correctly

## Failure Modes

### If Still Only 5 Tasks Flash
1. Check console for `✨ [FLASH]` Added logs - are ALL tasks being added to set?
2. Check `currentlyPlayingTasks.size` - does it grow beyond 5?
3. Check for JavaScript errors in console
4. Verify WebUI restarted after code changes

### If Performance Degrades
1. Check if too many tasks are being force-rendered
2. Verify `shouldFlash` check happens BEFORE position calculations
3. Check canvas draw calls - should not increase significantly

### If Flashes Appear at Wrong Position
1. Check Y-position calculation for off-screen tasks
2. Verify `agentIndex` is correct even when off-screen
3. Check console for position warnings

## Notes
- Fix location: `services/webui/templates/sequencer.html` lines 1202-1206
- Key change: `shouldFlash` computed BEFORE visibility culling
- Performance impact: Minimal (only forces render for actively flashing tasks)
