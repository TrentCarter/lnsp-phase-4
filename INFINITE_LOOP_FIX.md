# Infinite Loop Fix - Sound Queue (2025-11-09)

## Critical Bug Found & Fixed

### The Problem
**`processNextSound()` was creating an infinite loop** that scheduled thousands of setTimeout callbacks in seconds.

**Root Cause (Line 2122):**
```javascript
// ❌ WRONG - This ALWAYS schedules another call, even when queue is empty!
setTimeout(processNextSound, 10); // Small delay to avoid blocking
```

This line executed **unconditionally** after processing each sound, creating a runaway setTimeout loop:
- Process sound → schedule next call in 10ms
- 10ms later → check queue (empty) → return
- **BUT timeout was already scheduled!** → schedule another call
- Infinite recursion → thousands of pending timeouts

### The Fix

**Changed from unconditional recursion to conditional:**

```javascript
// ✅ CORRECT - Only schedule next call if queue has items
if (soundQueue.length > 0) {
    setTimeout(() => {
        processNextSound(); // Process next item in queue
    }, 10); // Small delay to avoid blocking
} else {
    // Queue is empty - mark as not playing
    currentlyPlaying = false;
}
```

**Now the function:**
1. Checks if queue is empty at the **start** → exits immediately if true
2. Processes one sound from queue
3. Checks if queue **still has items** before scheduling next call
4. Sets `currentlyPlaying = false` when queue drains completely

---

## Code Changes

### `processNextSound()` function (lines 2087-2133)

**Before:**
- Checked queue at start (returned if empty)
- **Never set `currentlyPlaying = true`** (so guard in `queueSound()` didn't work)
- **Always scheduled next call** with `setTimeout(processNextSound, 10)`
- Result: Infinite loop of setTimeout callbacks

**After:**
- Checks queue at start → exits if empty
- **Sets `currentlyPlaying = true`** when processing
- **Only schedules next call if queue.length > 0**
- **Sets `currentlyPlaying = false`** when queue drains
- Result: Processes queue cleanly, stops when done

---

## Testing

**Hard refresh browser:** `Cmd+Shift+R` or `Ctrl+Shift+R`

**Expected behavior:**
- ✅ No setTimeout accumulation
- ✅ Console doesn't flood with thousands of logs
- ✅ Sound queue processes items then stops
- ✅ Playback works normally

**Before fix:**
- 1000+ pending timeouts in seconds
- Browser slowdown/crash
- DevTools unusable

**After fix:**
- Only active timeouts are for queued sounds
- Clean queue processing
- Normal browser performance

---

## Files Modified

- `services/webui/templates/sequencer.html` (lines 2087-2133)

---

## Why This Happened

The original code tried to process sounds **concurrently** by always scheduling the next call, but forgot to actually **check if there were more items to process**.

The guard `if (!currentlyPlaying)` in `queueSound()` should have prevented duplicate processing, but `currentlyPlaying` was never set to `true`, so it didn't work.

**The fix:** Properly manage the `currentlyPlaying` flag AND only recurse when needed.
