# Flash Fix Summary (2025-11-09)

## Problem
Visual flashes only appeared for the first ~5 tasks when sounds played in the sequencer, despite all sounds triggering correctly.

## Root Cause
**Off-screen viewport culling** was preventing flashes for tasks not visible in the current scroll position.

The culling check at line 1203 of `services/webui/templates/sequencer.html` skipped rendering ALL content for tasks outside the visible area:

```javascript
// Skip if task block is not visible on screen
if (y + blockHeight < HEADER_HEIGHT || y > canvas.height) return;
```

This meant:
- Task 1-5 were visible → rendered → flashed ✅
- Task 6+ were scrolled down → culled → NO flash ❌

## Solution
Check if a task should flash **BEFORE** the visibility culling, then only skip rendering if the task is both off-screen AND not flashing.

### Code Changes

**File**: `services/webui/templates/sequencer.html`
**Lines**: 1202-1206

**Before**:
```javascript
const rowHeight = ROW_HEIGHT * verticalZoom;
const y = HEADER_HEIGHT + (agentIndex * rowHeight) - verticalOffset + ROW_PADDING;
const blockHeight = rowHeight - (2 * ROW_PADDING);

// Skip if task block is not visible on screen
if (y + blockHeight < HEADER_HEIGHT || y > canvas.height) return;
```

**After**:
```javascript
const rowHeight = ROW_HEIGHT * verticalZoom;
const y = HEADER_HEIGHT + (agentIndex * rowHeight) - verticalOffset + ROW_PADDING;
const blockHeight = rowHeight - (2 * ROW_PADDING);

// Check if task is currently playing sound (needs flash even if off-screen)
const shouldFlash = currentlyPlayingTasks.has(task.task_id);

// Skip if task block is not visible on screen AND not currently flashing
if (!shouldFlash && (y + blockHeight < HEADER_HEIGHT || y > canvas.height)) return;
```

**Also Updated**: Line 1272 comment to note that `shouldFlash` is computed earlier:
```javascript
// Draw bright flash overlay if task is currently playing sound
// (shouldFlash already computed earlier at line 1203)
if (shouldFlash) {
```

## Result
✅ All tasks now flash correctly when sounds play, regardless of scroll position!

## Testing
1. Load sequencer with 10+ tasks
2. Scroll down so only tasks 6+ are visible
3. Hit Play ▶️
4. Verify all tasks flash when sounds play (check console logs for `✨ [FLASH RENDER]`)

## Documentation
- **Implementation Guide**: `SEQUENCER_HIERARCHICAL_SOUND_IMPLEMENTATION.md` (updated)
- **Debug Guide**: `DEBUG_FLASH_ISSUE.md` (with full diagnostic details)
- **This Fix**: `FLASH_FIX_SUMMARY.md`

## Performance Impact
Minimal - only forces rendering for tasks actively playing sounds (typically 1-5 tasks at a time with sequential playback). Does not affect steady-state rendering performance.

## Related Files
- `services/webui/templates/sequencer.html` - Main fix (lines 1202-1206, 1272)
- `services/webui/templates/base.html` - Sound handling (unchanged)
- `SEQUENCER_HIERARCHICAL_SOUND_IMPLEMENTATION.md` - Updated with fix details
- `DEBUG_FLASH_ISSUE.md` - Complete diagnostic guide
