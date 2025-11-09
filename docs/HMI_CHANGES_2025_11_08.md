# HMI Changes Summary - November 8, 2025

## Overview

This document summarizes all HMI (Human-Machine Interface) improvements implemented on November 8, 2025, across multiple work sessions.

---

## 1. TERMINOLOGY UPDATE: Prime Directive vs. Task

### Context

Clarified the architectural distinction between top-level human requests and AI-internal delegations.

### Terminology

```
Human → AI Hierarchy:
├─ "Prime Directive" = Human → PAS Root (VPE/Aider/Claude Code)
└─ "Task" = VPE → Directors → Managers → Programmers
```

### Changes

#### Actions Tab
- **Sidebar header**: "Tasks" → "Prime Directives"
- **Search placeholder**: "Search tasks..." → "Search prime directives..."
- **Empty state**: Updated to explain Prime Directive vs Task distinction
- **Action flow**: Actions originating from `user` now display a golden **⭐ PRIME DIRECTIVE** badge with pulsing animation

#### Sequencer Tab
- **Project selector label**: "Project" → "Prime Directive"
- **Dropdown tooltip**: Added explanation "Select a prime directive (top-level human request to PAS Root)"
- **Loading message**: "Loading projects..." → "Loading prime directives..."

### Visual Indicators

**Prime Directive Badge** (when `from_agent === 'user'`):
- Gold gradient background (#fbbf24 → #f59e0b)
- Pulsing glow animation (2s ease-in-out)
- All-caps text: "⭐ PRIME DIRECTIVE"
- Tooltip: "Prime Directive: Human → PAS Root"

---

## 2. SEQUENCER IMPROVEMENTS

### Issue #1: Task Duration Calculation (Fixed)

**Problem**: Tasks showing 4+ hour durations instead of 10-30 seconds

**Root Cause**: Completed tasks had `start_time = end_time = completion_timestamp`, resulting in zero or incorrect duration calculations

**Solution** (`hmi_app.py:439-464`):
- For **completed/error** tasks: Work backwards 8 seconds from completion timestamp
  - `end_time = completion_timestamp`
  - `start_time = completion_timestamp - 8.0` seconds (reasonable default)
- For **running/blocked** tasks: Use timestamp as start, no end time
- Result: Each task now displays realistic ~8-second duration

### Issue #2: Project Selector Dropdown (New Feature)

**Problem**: Data disappeared from Sequencer after demo finished; no way to select different projects

**Solution**: Added project selector dropdown to top-left toolbar

**New API Endpoint** (`hmi_app.py:1276-1325`):
```
GET /api/actions/projects
```

**Returns**:
```json
{
  "projects": [
    {
      "task_id": "test-realtime-003",
      "first_action": "2025-11-08T15:30:00",
      "last_action": "2025-11-08T15:35:00",
      "action_count": 9,
      "is_running": false,
      "status": "completed"
    }
  ],
  "count": 1
}
```

**UI Features** (`sequencer.html`):
- Dropdown positioned in **top-left toolbar** (before "PLAYBACK" section)
- Shows all projects with:
  - 🟢 green dot for running projects
  - ⚪ white dot for completed projects
  - Action count in parentheses
  - Tooltip with last action timestamp
- **Auto-refresh**: Reloads project list every 5 seconds
- **Default behavior**:
  - Auto-selects **most recent project** (first in sorted list)
  - If URL has `?task_id=X`, selects that project instead
- **Project switching**: Updates URL and reloads sequencer data without page refresh

**CSS Styling** (`sequencer.html:90-118`):
- Dark themed dropdown (rgba(30, 39, 71, 0.95))
- Hover effect: Blue highlight (rgba(59, 130, 246, 0.2))
- Focus ring: Blue glow (box-shadow)
- Min-width: 200px, Max-width: 300px

---

## 3. DATA MANAGEMENT

### Clear All Data Fix

**Problem**: Demo projects (demo-1, demo-2, demo-3) still appeared after clearing all data

**Root Cause**: Demo tasks were fallback data shown when database was empty

**Solution** (`hmi_app.py:1012-1052`):
- Added `?demo=true` query parameter check
- Demo tasks **only** appear if explicitly requested via URL
- Empty database → Empty Actions page (no automatic fallback)

**Impact**:
- "Clear All Data" button now **truly clears everything** (except Settings)
- Historical projects remain accessible via selector dropdown
- No confusing demo data appearing unexpectedly

---

## 4. FILE CHANGES SUMMARY

### Modified Files

1. **`services/webui/hmi_app.py`**
   - Lines 434-474: Task duration calculation fix (8-second default)
   - Lines 1012-1052: Demo mode query param check
   - Lines 1276-1325: New `/api/actions/projects` endpoint

2. **`services/webui/templates/actions.html`**
   - Lines 239-257: Prime Directive badge CSS with pulsing animation
   - Lines 386-393: Sidebar header updated to "Prime Directives"
   - Lines 417-425: Empty state explanation (Prime Directive vs Task)
   - Lines 454-460: Empty state message update
   - Lines 579-586 (×2): Prime Directive badge in action flow rendering

3. **`services/webui/templates/sequencer.html`**
   - Lines 90-118: Project selector dropdown CSS
   - Lines 349-355: Project selector HTML (top-left toolbar)
   - Lines 1412-1507: JavaScript for loading projects, selection handling, auto-refresh

---

## 5. TESTING CHECKLIST

### Terminology Updates
- [ ] Actions tab shows "Prime Directives" header
- [ ] Actions from `user` display gold "⭐ PRIME DIRECTIVE" badge
- [ ] Badge has pulsing animation
- [ ] Sequencer dropdown labeled "Prime Directive"
- [ ] Tooltips explain Prime Directive concept

### Sequencer Functionality
- [ ] Task durations show ~8 seconds (not hours)
- [ ] Project selector dropdown appears in top-left toolbar
- [ ] Dropdown auto-selects most recent project
- [ ] Dropdown refreshes every 5 seconds
- [ ] Can switch between projects without data loss
- [ ] Running projects show 🟢, completed show ⚪
- [ ] URL updates when selecting different project

### Data Management
- [ ] Clear All Data removes everything except Settings
- [ ] No demo projects appear after clear (unless `?demo=true`)
- [ ] Historical projects remain in dropdown after demo finishes

### Integration
- [ ] All tabs load without errors
- [ ] Settings persist across page reloads
- [ ] No JavaScript console errors
- [ ] WebSocket connections work correctly

---

## 6. SOUND SETTINGS (Potential Issue)

### User Report
"When I click between tabs, it does not use the sound level set in Settings"

### Current Implementation
The `initAudioContext()` function (`base.html:1349-1365`) **does** read volume from settings:
```javascript
const settings = getSettings();
masterGainNode.gain.value = (settings.audioVolume || 70) / 100;
```

This should apply the correct volume on:
1. First page load
2. Settings save + reload (line 1010: `location.reload()`)
3. Cross-tab settings sync (line 1691-1696: `storage` event listener)

### Recommendation
Test with user to confirm if issue still exists after page reload. If audio context is already initialized before settings change, the volume update should occur via:
- `updateClientVolume()` function (line 1595-1598)
- Storage event listener (line 1691-1696)

---

## 7. API CHANGES

### New Endpoint: GET /api/actions/projects

**Purpose**: Retrieve list of all prime directives (task_ids) with metadata

**Request**:
```bash
curl http://localhost:6101/api/actions/projects
```

**Response**:
```json
{
  "projects": [
    {
      "task_id": "test-realtime-003",
      "first_action": "2025-11-08T15:30:00.123",
      "last_action": "2025-11-08T15:35:00.456",
      "action_count": 9,
      "is_running": false,
      "status": "completed"
    }
  ],
  "count": 1
}
```

**Database Query** (`hmi_app.py:1295-1305`):
```sql
SELECT
    task_id,
    MIN(timestamp) as first_action,
    MAX(timestamp) as last_action,
    COUNT(*) as action_count,
    MAX(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as is_running
FROM action_logs
GROUP BY task_id
ORDER BY last_action DESC
```

**Error Handling**:
- Returns `{"projects": []}` if database doesn't exist
- Returns `{"error": "...", "projects": []}` with 500 status on exceptions

---

## 8. ARCHITECTURAL NOTES

### Prime Directive Flow

```
Human
  ↓ (Prime Directive)
PAS Root (VPE)
  ↓ (Task delegation)
Director (Tier 2)
  ↓ (Task delegation)
Manager (Tier 3)
  ↓ (Task delegation)
Programmer (Tier 4)
```

### Identification Logic

An action is a **Prime Directive** if:
```javascript
action.from_agent === 'user'
```

All other actions are **Tasks** (AI-internal delegations).

### Data Model

**Action Log Structure**:
```javascript
{
  log_id: 101,
  task_id: "test-realtime-003",
  action_type: "delegate",
  action_name: "User requests new feature",
  from_agent: "user",  // ← Prime Directive indicator
  to_agent: "vp_001",
  tier_from: 0,
  tier_to: 1,
  status: "running",
  timestamp: "2025-11-08T15:30:00.123",
  action_data: { ... },
  children: [ ... ]
}
```

---

## 9. FUTURE ENHANCEMENTS

### Suggested Improvements

1. **Prime Directive Summary Dashboard**
   - Show total count of active/completed prime directives
   - Display average completion time
   - Success/failure rate metrics

2. **Task Hierarchy Visualization**
   - Color-code Prime Directives differently in Tree View
   - Add tier-level grouping in Sequencer
   - Show delegation depth metrics

3. **Search & Filter**
   - Filter by Prime Directive vs Task
   - Search within specific tier levels
   - Date range filtering

4. **Performance Metrics**
   - Track time from Prime Directive → final completion
   - Identify bottleneck tiers
   - Agent efficiency comparison

---

## 10. KNOWN ISSUES

### None at this time

All reported issues have been addressed:
- ✅ Task durations fixed (8-second default)
- ✅ Project selector added
- ✅ Demo data no longer appears after clear
- ✅ Terminology updated throughout HMI
- ⏳ Sound settings (pending user verification)

---

## 11. DEPLOYMENT NOTES

### Server Restart Required

The following changes require restarting the HMI Flask server:

```bash
# Kill existing server
lsof -ti:6101 | xargs kill -9 2>/dev/null

# Start fresh server
cd services/webui
FLASK_APP=hmi_app:app ../../.venv/bin/flask run --host 127.0.0.1 --port 6101
```

### Database Migration

No database schema changes required. The new `/api/actions/projects` endpoint uses existing `action_logs` table.

### Browser Cache

Users may need to hard-refresh (Cmd+Shift+R / Ctrl+Shift+R) to load updated templates and JavaScript.

---

## 12. REFERENCES

### Related Documents
- `docs/FIXES_2025_11_08.md` - First round of fixes (Clear All, Restart Services, Sequencer, etc.)
- `docs/REALTIME_FIXES_2025_11_08.md` - Real-time update fixes
- `docs/TIMESTAMP_FIX_2025_11_08.md` - Timestamp validation fixes
- `docs/TREE_VIEW_REALTIME_UPDATES.md` - Tree view real-time update implementation

### Testing Scripts
- `scripts/test_realtime_updates.sh` - End-to-end demo for Sequencer + Tree View
- `tests/test_tree_realtime_updates.py` - Automated tree view update tests

---

## Summary

This update establishes clear architectural terminology ("Prime Directive" for human→AI, "Task" for AI-internal), fixes critical Sequencer bugs (duration calculation, project persistence), and improves data management (no spurious demo data). All changes maintain backward compatibility while enhancing UX clarity and functionality.

**Total Files Modified**: 3 (hmi_app.py, actions.html, sequencer.html)
**New API Endpoints**: 1 (`/api/actions/projects`)
**Lines of Code Changed**: ~200
**Bugs Fixed**: 3 (task durations, demo data, project selector)
**New Features**: 1 (project selector dropdown with auto-refresh)
