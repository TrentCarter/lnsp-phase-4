# Quick Testing Guide: Multi-Lane + Audio Fixes (2025-11-09)

## 🎯 What to Test

Two critical fixes:
1. **Multi-lane allocation** - Tasks no longer stack in single row
2. **Real-time audio** - Sounds play during Start Demo

---

## 📋 Test Procedure

### Step 1: Refresh Browser
```bash
# Press Cmd+R (Mac) or Ctrl+R (Windows/Linux)
# Or hard refresh: Cmd+Shift+R / Ctrl+Shift+R
```

### Step 2: Check Multi-Lane Allocation

1. **Open browser console** (F12 or Cmd+Option+I)
2. **Select a project** with concurrent tasks from dropdown (e.g., "task_8bacd0ed")
3. **Look for console output**:
   ```
   [LANES] Allocated 15 rows (5 sub-lanes)
   [LANES] Agents with overlapping tasks:
     - Mgr Backend: 3 parallel lanes
     - Prog 001: 2 parallel lanes
   ```

4. **Visual check**:
   - Tasks delegated close together (within 15s) → **separate lanes** ✅
   - Tasks delegated far apart (>15s) → **same lane** ✅
   - **WRONG**: All tasks stacked in one row ❌

**Example (CORRECT)**:
```
Mgr Backend    ┌────────┐           ┌────────┐
  └─ Lane 2                ┌────────┐
  └─ Lane 3                          ┌────────┐
```

**Example (BROKEN)**:
```
Mgr Backend    ┌────────┬────────┬────────┬────────┐  ← All stacked!
```

### Step 3: Check Real-Time Audio

1. **Enable sound**:
   - Top toolbar → "Sound" dropdown → Select **"Music Note"**
   - (Default is "None" which disables sound)

2. **Start demo**:
   - Click **"Start Demo"** button (top toolbar)
   - OR navigate to Dashboard → click "Start Demo"

3. **Listen for sounds**:
   - ✅ Hear beeps/notes when tasks start
   - ✅ Hear completion sounds when tasks finish
   - ❌ **WRONG**: No sounds (but playback works)

4. **Check console**:
   ```
   🔊 [REALTIME] New task detected: "Assign: Authentication logic" (status=running)
   🔊 [REALTIME] Task status changed: "Assign: Authentication logic" (running → done)
   ```

### Step 4: Verify Playback Still Works

1. **Wait for demo to complete** (or select completed project)
2. **Press Play button** (▶️) at normal speed (1x)
3. **Verify**:
   - Playhead moves across timeline
   - Sounds play when playhead crosses task boundaries
   - Console shows `🔊 [SOUND]` logs (not `🔊 [REALTIME]`)

---

## ✅ Success Criteria

### Multi-Lane Fix
- [ ] Console shows lane allocation (sub-lanes > 0)
- [ ] Visual: Tasks with close delegation use separate lanes
- [ ] Visual: Tasks with distant delegation reuse same lane

### Audio Fix
- [ ] Sounds play **immediately** when tasks start/complete during demo
- [ ] Console shows `🔊 [REALTIME]` logs during demo execution
- [ ] Playback mode still works (sounds via `🔊 [SOUND]` logs)

### No Regressions
- [ ] Playback controls work (Play/Pause/Speed)
- [ ] Arrows render correctly (parent→child)
- [ ] Task deduplication works (settings toggle)
- [ ] Vertical zoom works (expand/collapse)

---

## 🐛 Known Issues / Edge Cases

### Multi-Lane Tuning
- **Too many lanes?** → Reduce `DELEGATION_WINDOW` from 15s to 10s or 5s
- **Still stacking?** → Increase `DELEGATION_WINDOW` to 20s or 30s
- **Location**: `services/webui/templates/sequencer.html` line 711

### Audio Timing
- **First sound delayed?** → Browser requires user interaction before playing audio (click Play first)
- **Sounds overlap?** → Sound queue should prevent this, but check console for errors
- **No sounds at all?** → Check sound mode dropdown (must be "Music Note" or "Voice", not "None")

---

## 🔧 Debugging

### If multi-lane NOT working:
```javascript
// Check console for this line:
[LANES] No overlapping tasks detected - all agents have single lanes

// If you see this, tasks are executing too far apart (>15s)
// Solution: Reduce DELEGATION_WINDOW constant (line 711)
```

### If audio NOT working:
```javascript
// Check for these logs:
🔇 [SOUND] Sound mode is "none", no sounds will play  ← Wrong! Change dropdown

🔊 [REALTIME] New task detected: ...  ← Correct! Real-time working
🔊 [SOUND] Task STARTED: ...          ← Correct! Playback working
```

### If audio works in playback but NOT real-time:
```javascript
// Check if detectRealtimeTaskChanges() is being called:
// Add temporary debug log in fetchSequencerData() (line 890):
console.log('Calling detectRealtimeTaskChanges with', data.tasks.length, 'tasks');

// If you don't see this log, polling may not be active
// Check: /api/demo/status should return {"running": true}
```

---

## 📞 Report Issues

If tests fail, provide:
1. Browser console output (full logs)
2. Screenshot of timeline view
3. Project name / task_id being tested
4. Sound mode setting

**File location**: `services/webui/templates/sequencer.html`

**Documentation**: `docs/SEQUENCER_MULTI_LANE_AUDIO_FIXES_2025_11_09.md`

---

**Date**: 2025-11-09
**Status**: ✅ Ready for User Testing
