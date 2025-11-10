# Prime Directive Completion Protocol

## Overview
When a Prime Directive (top-level project) finishes execution, the PAS ROOT agent must signal completion to the HMI, which will:
1. Stop timeline auto-scroll
2. Display "END OF PROJECT" banner
3. Show final project report
4. Prevent further execution

## Current Architecture

### Communication Flow
```
PAS Stub (Port 6200)
    ↓ Updates run status in memory (RUNS dict)
    ↓ Background worker completes all tasks
    ↓
Registry DB (action_logs table)
    ↓ HMI polls via background thread (poll_action_logs)
    ↓ SSE stream pushes updates to browser
    ↓
HMI (Port 6101)
    ↓ Sequencer polls /api/actions/tasks
    ↓ Renders timeline + detects completion
    ↓
Browser (Sequencer Timeline)
```

### Key Components

**PAS Stub** (`services/pas/stub/app.py`):
- `_execute_run()` - Background worker that executes tasks sequentially
- Sets `RUNS[run_id]["status"]` to `"completed"` when done
- No direct notification to HMI (isolated service)

**Registry DB** (`artifacts/registry/registry.db`):
- `action_logs` table stores all task actions
- `status` column: running, done, error, blocked, waiting
- **Missing**: No run-level completion flag or project end marker

**HMI** (`services/webui/hmi_app.py`):
- `poll_action_logs()` - Background thread polls DB every 1 second
- `stream_action_logs()` - SSE endpoint pushes updates to browser
- `get_action_tasks()` - REST API for sequencer data fetch

**Sequencer** (`services/webui/templates/sequencer.html`):
- `fetchSequencerData()` - Polls `/api/actions/tasks` every 5 seconds
- Renders timeline with playhead scrolling
- **Missing**: No detection of project completion

## Completion Signal Design

### Option 1: Special Action Log Entry (RECOMMENDED)

**Pros**: Uses existing infrastructure, SSE pushes immediately, no schema changes
**Cons**: Slight semantic stretch (action_logs for run-level events)

**Implementation**:
```python
# PAS Stub: After _execute_run() completes all tasks
action_log = {
    "task_id": run_id,  # Use run_id as pseudo-task
    "parent_log_id": None,
    "timestamp": datetime.utcnow().isoformat(),
    "from_agent": "PAS_ROOT",
    "to_agent": "HMI",
    "action_type": "directive_complete",
    "action_name": "Prime Directive Complete",
    "action_data": json.dumps({
        "run_id": run_id,
        "project_id": project_id,
        "tasks_total": len(tasks),
        "tasks_succeeded": tasks_succeeded,
        "tasks_failed": tasks_failed,
        "duration_seconds": duration,
        "validation_pass": all_kpis_passed
    }),
    "status": "done",
    "tier_from": 0,  # PAS ROOT is tier 0
    "tier_to": None
}

# POST to Registry
requests.post(
    "http://localhost:6121/action_logs",
    json=action_log
)
```

**HMI Detection** (sequencer.html):
```javascript
// Inside fetchSequencerData() or SSE handler
if (task.action_type === 'directive_complete' && task.from_agent === 'PAS_ROOT') {
    handleDirectiveComplete(task);
}

function handleDirectiveComplete(completionData) {
    // 1. Stop timeline auto-scroll
    isPlaying = false;

    // 2. Show END OF PROJECT banner
    showEndOfProjectBanner(completionData);

    // 3. Stop polling
    clearInterval(refreshIntervalId);

    // 4. Scroll to end of timeline
    scrollToTimelineEnd();
}
```

### Option 2: New Registry Endpoint

**Pros**: Cleaner separation, explicit run-level API
**Cons**: Requires new endpoint, polling instead of push, higher latency

**Implementation**:
```python
# Registry: Add new endpoint
@app.get("/runs/{run_id}/status")
def get_run_status(run_id: str):
    # Query action_logs to compute run status
    cursor.execute("""
        SELECT
            COUNT(*) as total_tasks,
            SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done_tasks,
            MAX(timestamp) as last_update
        FROM action_logs
        WHERE task_id LIKE ?
    """, (f"{run_id}%",))

    row = cursor.fetchone()
    is_complete = (row[0] == row[1])  # All tasks done

    return {
        "run_id": run_id,
        "complete": is_complete,
        "tasks_total": row[0],
        "tasks_done": row[1],
        "last_update": row[2]
    }
```

**HMI Polling**:
```javascript
// Poll every 5 seconds
async function checkRunStatus() {
    const response = await fetch(`/api/runs/${currentRunId}/status`);
    const data = await response.json();

    if (data.complete && !projectComplete) {
        handleDirectiveComplete(data);
    }
}
```

### Option 3: New Table (project_runs)

**Pros**: Most robust, explicit schema, supports multiple runs
**Cons**: Requires migration, most complex

**Schema**:
```sql
CREATE TABLE project_runs (
    run_id TEXT PRIMARY KEY,
    project_id INTEGER,
    status TEXT NOT NULL,  -- executing, completed, terminated, needs_review
    started_at TEXT NOT NULL,
    completed_at TEXT,
    tasks_total INTEGER,
    tasks_succeeded INTEGER,
    tasks_failed INTEGER,
    validation_pass BOOLEAN
);
```

## Recommended Approach: **Option 1 (Special Action Log)**

### Why?
1. **Immediate push notification** via existing SSE stream
2. **Zero schema changes** - works with current DB
3. **Minimal code changes** - reuses existing infrastructure
4. **Fast time-to-market** - can ship in 2-4 hours

### Implementation Plan

#### 1. PAS Stub Changes (30 min)
**File**: `services/pas/stub/app.py`

```python
def _execute_run(run_id: str):
    """Background worker to execute a run (topological order)."""
    start_time = time.time()
    tasks = DAG[run_id]

    # ... existing task execution loop ...

    # Update run status (existing code)
    failed_tasks = [t for t in tasks if TASKS[t]["status"] == "failed"]
    if failed_tasks:
        RUNS[run_id]["status"] = "needs_review"
        RUNS[run_id]["validation_pass"] = False
    else:
        RUNS[run_id]["status"] = "completed"
        RUNS[run_id]["validation_pass"] = True

    # 🆕 NEW: Notify HMI of completion via action_log
    duration = time.time() - start_time
    _notify_directive_complete(run_id, duration, tasks, failed_tasks)


def _notify_directive_complete(run_id: str, duration: float, tasks: List[str], failed_tasks: List[str]):
    """Send Prime Directive completion signal to HMI."""
    try:
        run = RUNS[run_id]

        completion_log = {
            "task_id": run_id,  # Use run_id as pseudo-task
            "parent_log_id": None,
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": "PAS_ROOT",
            "to_agent": "HMI",
            "action_type": "directive_complete",
            "action_name": "Prime Directive Complete",
            "action_data": json.dumps({
                "run_id": run_id,
                "project_id": run["project_id"],
                "tasks_total": len(tasks),
                "tasks_succeeded": len(tasks) - len(failed_tasks),
                "tasks_failed": len(failed_tasks),
                "duration_seconds": round(duration, 2),
                "validation_pass": run["validation_pass"],
                "status": run["status"]
            }),
            "status": "done",
            "tier_from": 0,  # PAS ROOT is tier 0
            "tier_to": None
        }

        # POST to Registry
        response = requests.post(
            "http://localhost:6121/action_logs",
            json=completion_log,
            timeout=2
        )
        response.raise_for_status()
        print(f"✅ Notified HMI of Prime Directive completion: {run_id}")

    except Exception as e:
        print(f"⚠️ Failed to notify HMI of completion: {e}")
        # Don't fail the run if notification fails (non-critical)
```

#### 2. HMI Detection Logic (45 min)
**File**: `services/webui/templates/sequencer.html`

```javascript
// Add flag to track completion state
let projectComplete = false;

// Modify fetchSequencerData() to detect completion
async function fetchSequencerData() {
    try {
        const response = await fetch('/api/actions/tasks');
        const data = await response.json();

        tasks = data.items || [];

        // 🆕 Check for directive completion
        const completionTask = tasks.find(t =>
            t.action_type === 'directive_complete' &&
            t.from_agent === 'PAS_ROOT'
        );

        if (completionTask && !projectComplete) {
            handleDirectiveComplete(completionTask);
        }

        // ... existing code ...
    } catch (error) {
        console.error('Failed to fetch sequencer data:', error);
    }
}

function handleDirectiveComplete(completionTask) {
    console.log('🎯 PRIME DIRECTIVE COMPLETE', completionTask);

    projectComplete = true;

    // 1. Stop playback and auto-scroll
    isPlaying = false;
    document.getElementById('play-icon').textContent = '▶️';
    document.getElementById('play-text').textContent = 'Play';
    document.getElementById('play-pause-btn').classList.remove('active');

    // 2. Stop polling for updates
    if (refreshIntervalId) {
        clearInterval(refreshIntervalId);
        refreshIntervalId = null;
    }

    // 3. Parse completion data
    let completionData = {};
    try {
        completionData = JSON.parse(completionTask.action_data || '{}');
    } catch (e) {
        console.warn('Could not parse completion data:', e);
    }

    // 4. Show END OF PROJECT banner
    showEndOfProjectBanner(completionData);

    // 5. Scroll to end of timeline (show final state)
    setTimeout(() => {
        scrollToTimelineEnd();
    }, 500);
}

function showEndOfProjectBanner(data) {
    // Create banner overlay
    const banner = document.createElement('div');
    banner.id = 'end-of-project-banner';
    banner.style.cssText = `
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 60px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        z-index: 10000;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        animation: slideDown 0.5s ease-out;
    `;

    // Add animation keyframes
    if (!document.getElementById('banner-animation-styles')) {
        const style = document.createElement('style');
        style.id = 'banner-animation-styles';
        style.textContent = `
            @keyframes slideDown {
                from { transform: translate(-50%, -150%); opacity: 0; }
                to { transform: translate(-50%, -50%); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    const passIcon = data.validation_pass ? '✅' : '⚠️';
    const statusText = data.validation_pass ? 'Completed Successfully' : 'Completed with Issues';

    banner.innerHTML = `
        <div style="font-size: 72px; margin-bottom: 20px;">🏁</div>
        <h1 style="font-size: 48px; margin: 0 0 10px 0; font-weight: bold;">
            END OF PROJECT
        </h1>
        <p style="font-size: 24px; margin: 10px 0; opacity: 0.9;">
            ${passIcon} ${statusText}
        </p>
        <div style="margin-top: 30px; font-size: 18px; opacity: 0.8;">
            <div>Tasks: ${data.tasks_succeeded || 0} succeeded, ${data.tasks_failed || 0} failed</div>
            <div>Duration: ${data.duration_seconds || 0}s</div>
        </div>
        <button onclick="closeEndOfProjectBanner()" style="
            margin-top: 30px;
            padding: 15px 40px;
            font-size: 18px;
            background: white;
            color: #667eea;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        ">
            View Final Report
        </button>
    `;

    document.body.appendChild(banner);
}

function closeEndOfProjectBanner() {
    const banner = document.getElementById('end-of-project-banner');
    if (banner) {
        banner.style.animation = 'slideUp 0.3s ease-in';
        setTimeout(() => banner.remove(), 300);
    }

    // Open final report (could navigate to /report page)
    // For now, just scroll to show all tasks
    scrollToTimelineEnd();
}

function scrollToTimelineEnd() {
    // Find last task time
    if (tasks.length === 0) return;

    const lastTask = tasks.reduce((latest, task) => {
        const taskEnd = task.end_time || task.start_time;
        const latestEnd = latest.end_time || latest.start_time;
        return taskEnd > latestEnd ? task : latest;
    });

    const lastTime = lastTask.end_time || lastTask.start_time;
    const startTime = window._sequencerStartTime || Date.now() / 1000;

    // Scroll to show last 30 seconds
    timelineOffset = Math.max(0, lastTime - startTime - 30);
    updatePlayhead();
    drawSequencer();
}
```

#### 3. CSS for Banner (15 min)
**File**: `services/webui/templates/sequencer.html` (add to `<style>` block)

```css
@keyframes slideDown {
    from { transform: translate(-50%, -150%); opacity: 0; }
    to { transform: translate(-50%, -50%); opacity: 1; }
}

@keyframes slideUp {
    from { transform: translate(-50%, -50%); opacity: 1; }
    to { transform: translate(-50%, -150%); opacity: 0; }
}

#end-of-project-banner button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
```

## Testing Plan

### Unit Tests
1. **PAS Stub**: Run completes → notification sent
2. **HMI**: Completion action log → banner shown
3. **Sequencer**: Timeline stops scrolling

### Integration Test
```bash
# 1. Start services
make run-pas-stub &
make run-hmi &

# 2. Submit demo project
curl -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "run_id": "test-completion-123",
    "run_kind": "baseline"
  }'

# 3. Submit task cards
curl -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 1,
    "run_id": "test-completion-123",
    "lane": "Code-Impl",
    "payload": {}
  }'

# 4. Open browser: http://localhost:6101/sequencer
# 5. Wait for tasks to complete (watch synthetic execution)
# 6. Verify:
#    - Timeline stops scrolling ✅
#    - "END OF PROJECT" banner appears ✅
#    - Polling stops (check network tab) ✅
#    - Final report button works ✅
```

## Future Enhancements

1. **Final Report Page** (`/report/{run_id}`):
   - Task completion timeline
   - KPI violations summary
   - Cost breakdown
   - Energy consumption
   - Downloadable PDF

2. **Project Archive**:
   - Mark completed runs as archived
   - Prevent accidental replay
   - Historical runs browser

3. **Celebration Effects**:
   - Confetti animation on 100% success
   - Sound effect (optional, settings toggle)
   - Agent hierarchy collapse animation

4. **Notification Integration**:
   - Browser notification API
   - Email notification (if configured)
   - Slack webhook (if configured)

## Rollout Plan

### Phase 1: Core Completion Signal (Day 1)
- [x] Design protocol (this document)
- [ ] Implement PAS notification
- [ ] Implement HMI detection
- [ ] Add END OF PROJECT banner
- [ ] Test with demo project

### Phase 2: Polish (Day 2)
- [ ] Final report page
- [ ] Animation refinements
- [ ] Error handling (what if notification fails?)
- [ ] Logging/telemetry

### Phase 3: Advanced Features (Week 2)
- [ ] Project archive
- [ ] Browser notifications
- [ ] Downloadable reports
- [ ] Replay protection

## Open Questions

1. **Multiple concurrent runs**: How to handle if user starts another project while one is completing?
   - **Answer**: Use `run_id` in banner, allow concurrent banners (stack vertically)

2. **Partial completion**: What if some tasks fail?
   - **Answer**: Show "COMPLETED WITH ISSUES" banner (yellow), link to failure report

3. **Cancelled runs**: Should we show END banner for terminated/cancelled runs?
   - **Answer**: Show different banner: "PROJECT TERMINATED" (red), reason from termination_reason

4. **Resume after completion**: Can user "replay" timeline after completion?
   - **Answer**: Yes, allow replay but mark as "Historical Replay" (read-only)

## Summary

**Recommended**: Option 1 (Special Action Log Entry)
- **Fastest**: Ships in 2-4 hours
- **Zero migration**: Uses existing schema
- **Push-based**: Immediate SSE notification
- **Minimal risk**: Reuses proven infrastructure

**Key Changes**:
1. PAS Stub: Add `_notify_directive_complete()` after `_execute_run()`
2. Sequencer: Add `handleDirectiveComplete()` to detect and show banner
3. CSS: Add banner animations

**Timeline**: 2-4 hours implementation, 1 hour testing, ships same day!
