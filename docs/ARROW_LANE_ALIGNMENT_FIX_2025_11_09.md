# Arrow-to-Lane Alignment Fix (2025-11-09)

## Problem

**Symptom**: Arrows pointing to empty space instead of task boxes when agents have multiple lanes.

**Example (from screenshot)**:
- Db 001 has a green dashed arrow going UP
- Arrow terminates at "Mgr Db Lane 2" position
- **No task box at that location** - the actual task is in "Mgr Db" main lane!

**Visual**:
```
Mgr Db (2 lanes)     [Task A]  ← Arrow should point HERE
  └─ Lane 2          (empty)   ← But arrow points HERE (wrong!)
```

## Root Cause

**Arrow Y-position calculation used agent's service_id, not lane-specific position.**

### Before Fix (Broken)

```javascript
// Build simple map: service_id → Y position
const agentRowMap = {};
agents.forEach((agent, idx) => {
    agentRowMap[agent.service_id] = yCenter;  // Only maps FIRST lane
});

// Draw arrow using parent agent's service_id
const fromY = agentRowMap[task.from_agent];  // ❌ Always uses first lane!
```

**Problem**:
1. Multi-lane agents have rows like: "Mgr Db" (Lane 0), "  └─ Lane 2" (Lane 1), "  └─ Lane 3" (Lane 2)
2. All share same `service_id` ("Mgr Db")
3. `agentRowMap[service_id]` only stores Y position of **first** row (Lane 0)
4. When parent task is in Lane 1 or Lane 2, arrow uses Lane 0 Y position → **points to empty space**

### After Fix (Correct)

```javascript
// Build TWO-LEVEL map: both service_id AND (service_id, lane_index)
const agentRowMap = {};
agents.forEach((agent, idx) => {
    const yCenter = ...;

    // Primary key: service_id (fallback for single-lane agents)
    if (!agentRowMap[agent.service_id]) {
        agentRowMap[agent.service_id] = yCenter;
    }

    // Secondary key: service_id + laneIndex (for multi-lane agents)
    const laneKey = `${agent.service_id}_lane${agent._laneIndex || 0}`;
    agentRowMap[laneKey] = yCenter;  // ✅ Stores Y position per lane!
});

// Draw arrow using parent task's LANE-SPECIFIC key
const parentLaneKey = `${task.from_agent}_lane${parentTask._lane || 0}`;
const parentY = agentRowMap[parentLaneKey] || agentRowMap[task.from_agent];
```

**Why this works**:
1. Each lane has unique key: `"Mgr Db_lane0"`, `"Mgr Db_lane1"`, `"Mgr Db_lane2"`
2. Parent task knows its lane: `parentTask._lane = 1`
3. Arrow looks up correct Y position: `agentRowMap["Mgr Db_lane1"]`
4. Arrow points to **actual task location** ✅

## Implementation Details

### Key Data Structures

**Lane Assignment** (from lane allocation):
```javascript
// Each task gets assigned a lane during allocation
task._lane = 0;  // Lane index (0 = main lane, 1 = Lane 2, 2 = Lane 3, etc.)
```

**Agent Rows** (after expansion):
```javascript
// For multi-lane agent "Mgr Db" with 3 lanes:
[
    { service_id: "Mgr Db", name: "Mgr Db", _laneIndex: 0, _isParent: true },
    { service_id: "Mgr Db", name: "  └─ Lane 2", _laneIndex: 1, _isParent: false },
    { service_id: "Mgr Db", name: "  └─ Lane 3", _laneIndex: 2, _isParent: false }
]
```

**AgentRowMap Structure**:
```javascript
{
    "Mgr Db": 250,              // Fallback (first lane)
    "Mgr Db_lane0": 250,        // Main lane Y position
    "Mgr Db_lane1": 290,        // Lane 2 Y position
    "Mgr Db_lane2": 330,        // Lane 3 Y position
    "Prog 001": 370,            // Single-lane agent (no sub-lanes)
    ...
}
```

### Code Changes

**File**: `services/webui/templates/sequencer.html`

**Locations**:
1. **AgentRowMap construction** (lines 1296-1313)
2. **Child task Y position** (lines 1401-1410)
3. **Delegation arrows** (lines 1442-1470)
4. **Report arrows** (lines 1481-1509)
5. **Code generation delegation** (lines 1522-1546)
6. **Code generation completion** (lines 1560-1589)
7. **Generic delegation** (lines 1600-1625)
8. **Generic completion** (lines 1633-1659)

**Changes Made**:
1. ✅ Build lane-specific keys in `agentRowMap`
2. ✅ Use parent task's `_lane` to lookup correct Y position
3. ✅ Reduce tolerance from 50px to 10px (more strict now that we have correct positions)
4. ✅ Add lane index to console logs for debugging

### Before/After Comparison

**Before** (broken):
```javascript
const fromY = agentRowMap[task.from_agent];  // "Mgr Db" → 250px (Lane 0)
const actualParentY = parentTask._bounds.y;  // 290px (Lane 1)
const yDiff = Math.abs(290 - 250);           // 40px difference
if (yDiff < 50) {  // ✅ Passes tolerance check
    drawArrow(..., fromY, ...);  // ❌ Arrow drawn to Lane 0 (wrong!)
}
```

**After** (fixed):
```javascript
const parentLaneKey = `${task.from_agent}_lane${parentTask._lane}`;  // "Mgr Db_lane1"
const parentY = agentRowMap[parentLaneKey];  // 290px (Lane 1 - correct!)
const actualParentY = parentTask._bounds.y;  // 290px (Lane 1)
const yDiff = Math.abs(290 - 290);           // 0px difference ✅
if (yDiff < 10) {  // ✅ Passes tolerance check
    drawArrow(..., actualParentY, ...);  // ✅ Arrow drawn to Lane 1 (correct!)
}
```

## Testing

### Manual Verification

1. **Refresh browser** (Cmd+R)
2. **Select multi-lane project** (e.g., "task_8bacd0ed")
3. **Check console logs**:
   ```
   [Arrows] Built agentRowMap for 42 agent rows (including lanes)
   ✅ [Arrow] Delegation arrow drawn for "Design database schema" (parent lane=1)
   ```
4. **Visual check**:
   - ✅ Every arrow should start/end at a task box
   - ❌ No arrows pointing to empty space

### Console Debug Logs

**Success**:
```
✅ [Arrow] Delegation arrow drawn for "Implement feature X" (parent lane=2)
✅ [Arrow] Report arrow drawn for "Completed: task Y" (worker lane=1)
```

**Skipped** (expected for mismatched lanes):
```
[Arrow Skip] Delegation: "Task Z" - Parent Y mismatch: expected=250, actual=290, diff=40px (parent lane=1)
```
**Note**: With correct implementation, you should see very few "Y mismatch" skips!

### Edge Cases

**Single-lane agents**:
- Use `agentRowMap[service_id]` directly (no lane suffix)
- Still works because we set both keys

**Missing lane info**:
- Falls back to `task._lane || 0` (defaults to first lane)
- Falls back to `agentRowMap[service_id]` if lane key not found

**Tolerance**:
- Reduced from 50px to 10px (safe because we now have correct Y positions)
- Allows for minor rounding errors but catches real misalignment

## Performance Impact

**Minimal**:
- Map construction: +3 lines per agent (insignificant)
- Map lookup: O(1) hash lookup (same as before)
- Memory: +1 key per lane (~50 bytes per lane, negligible)

## Rollback Instructions

If arrows break, revert to single-key mapping:

```bash
git diff HEAD~1 services/webui/templates/sequencer.html | grep -A5 -B5 "agentRowMap"
git checkout HEAD~1 services/webui/templates/sequencer.html
```

Or manually restore:
1. Remove lane-specific keys (lines 1308-1310)
2. Restore `const fromY = agentRowMap[task.from_agent]` (line 1408)
3. Restore 50px tolerance (all arrow sections)

## Related Fixes

This fix builds on two previous improvements:
1. **Multi-lane allocation** (earlier today) - Tasks delegated within 15s use separate lanes
2. **Duration calculation** (yesterday) - Tasks show real durations (2-6s)

All three fixes work together to create accurate timeline visualization:
- Multi-lane allocation → Creates lanes for concurrent delegation
- Duration calculation → Shows when tasks actually run
- **Arrow-to-lane mapping** → Arrows point to correct lane where task exists

---

**Date**: 2025-11-09
**Author**: Claude Code
**Files Modified**:
- `services/webui/templates/sequencer.html` (lines 1296-1659, 8 arrow sections)

**Status**: ✅ Ready for Testing

**Verification Checklist**:
- [ ] Browser refreshed
- [ ] Multi-lane project selected
- [ ] Console shows lane-specific arrow logs
- [ ] No arrows pointing to empty space
- [ ] All arrows start/end at task boxes
