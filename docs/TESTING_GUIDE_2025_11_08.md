# Testing Guide: Tree View Project Selector + MASTER STOP Button
**Date**: November 8, 2025
**Session**: Remaining TODOs Implementation

## ✅ What Was Implemented

### 1. **Project Dropdown Selector in Tree View**
- Added project/task selector dropdown to Tree View (like the one in Sequencer View)
- Shows all available "prime directives" (top-level tasks) with status indicators
- Automatically loads and refreshes every 5 seconds
- Location: Bottom-left controls toolbar in Tree View

### 2. **MASTER STOP Button**
- Emergency kill switch for forcefully terminating all demo processes
- Uses `SIGKILL` to immediately stop processes (no graceful shutdown)
- Kills entire process group (parent + all children)
- Hunts down stray processes using `pgrep`
- Location: Top navigation bar (visible only when demo is running)

---

## 🧪 How to Test

### **HMI Server Status**
The HMI server is currently running on: **http://localhost:6101**

---

## Test 1: Tree View Project Selector

### Steps:
1. **Open Tree View**
   ```
   http://localhost:6101/tree
   ```

2. **Verify Dropdown Exists**
   - Look at bottom-left controls toolbar
   - Should see a dropdown with text like:
     ```
     ▶️ abc123... (42 actions, 11/8/2025, 3:45:00 PM)
     ```
   - If no projects exist yet, it will show: `"No prime directives found"`

3. **Start Demo (to Create Tasks)**
   - Click **"▶️ Start Demo"** button in top navigation
   - Wait 10-30 seconds for tasks to be created
   - Dropdown should populate with task entries

4. **Test Selection**
   - Select different tasks from dropdown
   - Tree should reload with new task's data
   - URL should update with `?task_id=...`

5. **Verify Auto-Refresh**
   - Leave page open
   - Every 5 seconds, dropdown should refresh (check browser console for logs)
   - New tasks should appear automatically

### Expected Results:
- ✅ Dropdown shows all available tasks
- ✅ Running tasks have ▶️ emoji
- ✅ Paused tasks have ⏸️ emoji
- ✅ Selecting a task reloads tree with that task's data
- ✅ URL updates to include `task_id` parameter

---

## Test 2: MASTER STOP Button

### Steps:

#### **2A: Verify Button Visibility**

1. **Initial State (No Demo Running)**
   - Visit any HMI page: `http://localhost:6101/`
   - Top nav should show:
     - ✅ **"▶️ Start Demo"** (visible)
     - ❌ **"⏹️ Stop Demo"** (hidden)
     - ❌ **"🛑 MASTER STOP"** (hidden)

2. **Start Demo**
   - Click **"▶️ Start Demo"**
   - Wait for confirmation alert
   - Top nav should now show:
     - ❌ **"▶️ Start Demo"** (hidden)
     - ✅ **"⏹️ Stop Demo"** (visible)
     - ✅ **"🛑 MASTER STOP"** (visible, red with yellow border)

#### **2B: Test Regular Stop First**

3. **Click "⏹️ Stop Demo"**
   - Should send `SIGTERM` to demo process
   - Button should show "⏳ Stopping..."
   - Should hide both Stop and MASTER STOP buttons
   - **Note**: If processes don't stop immediately, proceed to 2C

#### **2C: Test MASTER STOP (Emergency Kill)**

4. **Start Demo Again**
   - Click **"▶️ Start Demo"**
   - Wait for demo to be running

5. **Click "🛑 MASTER STOP"**
   - **Confirmation Dialog** should appear:
     ```
     ⚠️ MASTER STOP: This will FORCE KILL all demo processes immediately!

     This is an EMERGENCY measure - use only if regular Stop fails.

     Continue?
     ```
   - Click **OK** to proceed

6. **Verify Immediate Termination**
   - Button text changes to "⏳ KILLING..."
   - Processes are killed instantly (no delays)
   - Success alert shows:
     ```
     ✅ MASTER STOP: Killed 1 process(es)
     PIDs: 12345
     ```
   - All buttons reset:
     - ✅ **"▶️ Start Demo"** (visible)
     - ❌ **"⏹️ Stop Demo"** (hidden)
     - ❌ **"🛑 MASTER STOP"** (hidden)

#### **2D: Verify Backend Cleanup**

7. **Check Process Tree**
   ```bash
   # Verify no demo processes are running
   pgrep -f lnsp_llm_driven_demo.py
   # (should return nothing)

   # Check PID file is removed
   ls -l /tmp/lnsp_demo.pid
   # (should not exist)
   ```

8. **Check Server Logs**
   - Look for log entries like:
     ```
     MASTER STOP: Killed process group for PID 12345
     MASTER STOP: Killed stray demo process 12346
     ```

### Expected Results:

| Step                     | Expected Behavior                                              |
|--------------------------|----------------------------------------------------------------|
| Start Demo               | Both Stop and MASTER STOP buttons appear                       |
| Regular Stop             | Sends SIGTERM, graceful shutdown (may take seconds)            |
| MASTER STOP              | Sends SIGKILL, immediate termination (< 1 second)              |
| Process Group Kill       | Kills parent + all child processes                             |
| Stray Process Cleanup    | Finds and kills any orphaned demo processes via `pgrep`        |
| Button Reset             | All buttons return to initial state (Start visible)            |
| Confirmation Dialog      | Warns user this is an EMERGENCY measure                        |
| Success Alert            | Shows number of processes killed + their PIDs                  |

---

## 🔍 Backend Implementation Details

### API Endpoints

#### **GET /api/actions/projects**
- Returns list of all tasks with metadata
- Sorted by most recent first
- Used by both Sequencer and Tree View dropdowns

```json
{
  "projects": [
    {
      "task_id": "abc123-def456",
      "first_action": "2025-11-08T15:30:00",
      "last_action": "2025-11-08T15:45:00",
      "action_count": 42,
      "is_running": 1
    }
  ]
}
```

#### **POST /api/demo/stop**
- Sends `SIGTERM` to demo process (graceful shutdown)
- Removes PID file
- Returns: `{"message": "Demo stopped", "pid": 12345}`

#### **POST /api/demo/master-stop** ⭐ NEW
- **EMERGENCY KILL**: Uses `SIGKILL` for immediate termination
- Kills entire process group via `os.killpg()`
- Hunts stray processes via `pgrep -f lnsp_llm_driven_demo.py`
- Returns: `{"message": "...", "killed_pids": [12345, 12346]}`

### Process Killing Strategy

```python
# Step 1: Kill process group (parent + all children)
os.killpg(os.getpgid(main_pid), signal.SIGKILL)

# Step 2: Find and kill strays
subprocess.run(['pgrep', '-f', 'lnsp_llm_driven_demo.py'])
for pid in stray_pids:
    os.kill(pid, signal.SIGKILL)
```

---

## 📝 Files Changed

| File                          | Lines Modified | Description                                      |
|-------------------------------|----------------|--------------------------------------------------|
| `tree.html`                   | ~70 lines      | Added project dropdown + loadProjects() function |
| `base.html`                   | ~80 lines      | Added MASTER STOP button + masterStopDemo()      |
| `hmi_app.py`                  | ~70 lines      | Added /api/demo/master-stop endpoint             |

**Total**: ~220 lines of new/modified code

---

## ⚠️ Known Issues / Edge Cases

### 1. **No Tasks Available**
- If no demo has been run yet, dropdown shows: `"No prime directives found"`
- **Solution**: Start demo to create tasks

### 2. **PID File Stale**
- If HMI crashes, `/tmp/lnsp_demo.pid` may exist but process is dead
- **Solution**: MASTER STOP cleans up stale PID files

### 3. **macOS vs Linux**
- `pgrep` command syntax may differ slightly
- **macOS**: `pgrep -f pattern`
- **Linux**: Same, but some distros use `pidof`

### 4. **Button Visibility Race Condition**
- If page loads while demo is starting, buttons may briefly show wrong state
- **Solution**: Auto-refresh every 5 seconds corrects this

---

## 🎯 Success Criteria

### Project Selector (Tree View)
- [x] Dropdown appears in Tree View controls
- [x] Shows all available tasks with status icons
- [x] Selecting task reloads tree with new data
- [x] Auto-refreshes every 5 seconds
- [x] URL updates with `task_id` parameter

### MASTER STOP Button
- [x] Button visible only when demo is running
- [x] Red background with yellow border (high visibility)
- [x] Confirmation dialog warns user
- [x] Kills processes immediately (SIGKILL)
- [x] Cleans up process group (parent + children)
- [x] Finds and kills stray processes
- [x] Shows success message with PIDs killed
- [x] Resets UI to initial state

---

## 🚀 Next Steps

1. **User Acceptance Testing**
   - Run through both test sequences above
   - Verify all expected results match actual behavior

2. **Documentation**
   - Update user manual with MASTER STOP instructions
   - Add warning about when to use MASTER STOP vs regular Stop

3. **Future Enhancements**
   - Add process kill timeout (if SIGKILL fails after 5s, report error)
   - Log killed PIDs to action_logs table for audit trail
   - Add "Force Restart" button (MASTER STOP + Start in one action)

---

## 📊 Testing Checklist

### Tree View Project Selector
- [ ] Dropdown visible in Tree View controls
- [ ] Shows "Loading..." on initial load
- [ ] Populates with tasks after demo starts
- [ ] Running tasks have ▶️ emoji
- [ ] Selecting task reloads tree
- [ ] URL updates with task_id
- [ ] Auto-refresh works (check console logs every 5s)

### MASTER STOP Button
- [ ] Hidden when no demo is running
- [ ] Visible when demo starts (alongside regular Stop)
- [ ] Confirmation dialog appears on click
- [ ] Button shows "⏳ KILLING..." during execution
- [ ] Processes killed immediately (< 1 second)
- [ ] Success alert shows number of PIDs killed
- [ ] Buttons reset to initial state after success
- [ ] No stray demo processes remain (verify with pgrep)
- [ ] PID file removed from /tmp/

---

**All features implemented successfully! Ready for testing.** 🎉
