# Vertical Arrows & Receive Report Tasks - 2025-11-09

## User Requirement

**"Arrows should always be completely VERTICAL, not at an angle. When a child completes and the parent task finished, a NEW task is started in the parent lane showing 'Receiving report from child'. This task exists until the parent reports to ITS parent. Green UP arrows point VERTICALLY to these receive_report task bubbles."**

## Solution Architecture

### Before (Broken)
- Child task completes
- Arrow tries to point to parent's historical delegation task (which ended long ago)
- Result: L-shaped arrows pointing to empty space

### After (Fixed)
- Child task completes
- **Parent creates a NEW "receive_report" task** at the same time
- Arrow points VERTICALLY from child → parent's receive_report task
- Result: Vertical green arrows pointing to actual task bubbles

## Complete Reporting Flow

```
WORKER (Tier 3)
  ↓ Executes work
  | Task: "Create api.py" (status=done)
  ↓
  ↑ VERTICAL green arrow
  |
MANAGER (Tier 2)
  | NEW Task: "Processing completion from prog_001" (receive_report)
  | (This task bubble appears when worker completes)
  ↓ Manager finishes processing, reports up
  |
  ↑ VERTICAL green arrow
  |
DIRECTOR (Tier 1)
  | NEW Task: "Processing completion from mgr_backend" (receive_report)
  | (This task bubble appears when manager reports)
  ↓ Director finishes processing, reports up
  |
  ↑ VERTICAL green arrow
  |
PAS_ROOT (Tier 0)
  | NEW Task: "Processing completion from dir_code" (receive_report)
  | (This task bubble appears when director reports)
  ↓ PAS Root finishes, notifies user
```

## Data Changes

### New Action Type: `receive_report`

**Purpose**: Create visible task bubbles showing "parent is processing child's completion report"

**Created When**: Immediately after child reports completion

**Example**:
```python
# Worker completes
log_action(task_id, "prog_001", "mgr_backend", "report",
           "Completed api.py", status="completed")

# Manager receives report (NEW TASK BUBBLE)
log_action(task_id, "mgr_backend", "mgr_backend", "receive_report",
           "Processing completion from prog_001", status="running")
           #         ^^^^^^^^^^^^  ^^^^^^^^^^^^
           #         from_agent    to_agent (same = task appears in this agent's lane)
```

**Key Fields**:
- `from_agent` = Parent agent (e.g., "mgr_backend")
- `to_agent` = Parent agent (same as from_agent - task appears in parent's lane)
- `action_type` = "receive_report"
- `action_name` = "Processing completion from {child_id}"
- `status` = "running" (until parent reports up)

### Timeline of Tasks

1. **Manager delegates**: `mgr_backend → prog_001` (action_type=delegate)
2. **Worker executes**: `prog_001` task (action_type=code_generation, status=running)
3. **Worker completes**: Same task (status=done)
4. **Worker reports**: `prog_001 → mgr_backend` (action_type=report) ← NOT shown as task bubble
5. **Manager receives**: `mgr_backend → mgr_backend` (action_type=receive_report) ← NEW TASK BUBBLE ✨
6. **Manager reports**: `mgr_backend → dir_code` (action_type=report) ← NOT shown as task bubble
7. **Director receives**: `dir_code → dir_code` (action_type=receive_report) ← NEW TASK BUBBLE ✨
8. And so on...

## Visualization Changes

### Arrow Logic (VERTICAL only)

**Old Approach** (removed):
```javascript
// L-shaped arrow from child's end → parent's historical position
drawArrow(
    childEndX, taskY,      // Start: child's right edge
    parentEndX, parentY,   // End: parent's historical position
    ...
);
// Result: Angled arrows pointing to empty space
```

**New Approach** (implemented):
```javascript
// Find parent's receive_report task at same time
const receiveReportTask = allTasks.find(t =>
    t.to_agent === task.from_agent &&  // Parent agent
    t.action_type === 'receive_report' &&
    Math.abs(t.start_time - task.end_time) < 2.0  // Within 2 seconds
);

// VERTICAL arrow at same X position
drawArrow(
    arrowX, taskY,         // Start: child's position
    arrowX, receiveY,      // End: SAME X (vertical!), parent's receive_report task
    ...
);
// Result: Perfect vertical arrows to actual task bubbles
```

### Arrow Colors & Types

- **Green dashed arrows (UP)**: Completion reports (child → parent's receive_report task)
- **Purple dashed arrows (DOWN)**: Delegation (parent → child)
- All arrows are now **perfectly VERTICAL** (no angles)

## Files Modified

### 1. Data Generation: `/tmp/lnsp_llm_driven_demo.py`

**Lines 343-352**: Create receive_report tasks when workers complete
```python
# Worker reports + Manager receives
report_log_id = log_action(..., "report", ...)
log_action(task_id, manager_id, manager_id, "receive_report",
           f"Processing completion from {worker_id}", status="running")
```

**Lines 365-380**: Manager → Director reporting with receive_report tasks
```python
# Manager reports
mgr_report_log = log_action(task_id, manager_id, director_id, "report", ...)

# Director receives (NEW TASK BUBBLE)
log_action(task_id, director_id, director_id, "receive_report",
           f"Processing completion from {manager_id}", status="running")
```

**Lines 385-397**: Director → PAS Root reporting with receive_report tasks
```python
# Director reports
dir_report_log = log_action(task_id, director_id, "pas_root", "report", ...)

# PAS Root receives (NEW TASK BUBBLE)
log_action(task_id, "pas_root", "pas_root", "receive_report",
           f"Processing completion from {director_id}", status="running")
```

### 2. Visualization: `services/webui/templates/sequencer.html`

**Lines 1575-1611**: Code generation completion arrows (VERTICAL)
- Find receive_report task at same time
- Draw vertical arrow (same X coordinate)
- Green dashed, upward

**Lines 1650-1679**: Generic completion arrows (VERTICAL)
- Find receive_report task at same time
- Draw vertical arrow
- Purple dashed, upward

**Lines 1480-1483**: Removed old report arrow logic (replaced with receive_report)

### 3. Data Filtering: `services/webui/hmi_app.py`

**No changes needed!** The existing filter already works:
```python
# This filters OUT "report" actions (which don't create task bubbles)
if action_type == 'report':
    continue

# But "receive_report" is NOT filtered, so it creates task bubbles ✅
```

## Testing Instructions

1. **Stop current demo**:
   ```bash
   ps aux | grep lnsp_llm_driven_demo | awk '{print $2}' | xargs kill
   ```

2. **Clear old data** (optional, recommended):
   ```bash
   rm artifacts/registry/registry.db
   # Restart registry service
   ```

3. **Refresh browser**:
   ```
   Cmd+Shift+R (hard refresh)
   ```

4. **Start new demo**:
   - Click "Play" button in sequencer
   - OR run: `python3 /tmp/lnsp_llm_driven_demo.py`

5. **Observe arrows**:
   - ✅ All arrows should be **perfectly VERTICAL**
   - ✅ Green UP arrows point to **visible task bubbles** (receive_report tasks)
   - ✅ Task bubbles appear in parent lanes showing "Processing completion from {child}"
   - ✅ Complete chain: Worker → Manager → Director → PAS_ROOT

6. **Verify data**:
   ```bash
   sqlite3 artifacts/registry/registry.db "
   SELECT action_type, COUNT(*) as count
   FROM action_logs
   GROUP BY action_type
   ORDER BY action_type;"
   ```

   **Expected output**:
   ```
   code_generation|X
   delegate|Y
   finalize|1
   notify|1
   receive_report|Z   ← NEW action type!
   report|W
   ```

## Expected Visual Result

Looking at the timeline, you should see:

```
Prog 001 (Lane 1):
  ████████████ [Create api.py - Done]
              ↑ (green vertical arrow)

Mgr Backend (Lane 1):
              ████ [Processing completion from prog_001]
                  ↑ (green vertical arrow)

Dir Code (Lane 1):
                  ██ [Processing completion from mgr_backend]
                    ↑ (green vertical arrow)

PAS Root:
                    █ [Processing completion from dir_code]
```

All arrows are **straight vertical lines** connecting task bubbles.

## Benefits

1. ✅ **Visual clarity**: Arrows always point to visible task bubbles
2. ✅ **Vertical alignment**: No confusing angled arrows
3. ✅ **Living audit trail**: See exactly when each tier processed reports
4. ✅ **Complete chain**: Every completion flows up through hierarchy to PAS_ROOT
5. ✅ **Time accuracy**: receive_report tasks appear at the actual time processing occurred

---

**Status**: ✅ Ready for testing
**Date**: 2025-11-09
**Impact**: Complete redesign of reporting chain visualization
