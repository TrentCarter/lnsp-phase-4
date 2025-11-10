# Prime Directive Completion Signal Flow

## Visual Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    PRIME DIRECTIVE EXECUTION TIMELINE                           │
└────────────────────────────────────────────────────────────────────────────────┘

TIME:    T=0s           T=10s          T=20s          T=30s          T=30.1s
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼

PAS:     Start Run  →  Task 1     →  Task 2     →  Task 3     →  ALL DONE
         ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐       ┌──────┐
         │START│       │ Run │       │ Run │       │ Run │       │NOTIFY│
         └─────┘       └─────┘       └─────┘       └─────┘       └──────┘
                                                                       │
                                                                       │ POST
                                                                       ▼
Registry:                                                         Insert Log
                                                                  ┌─────────┐
                                                                  │ action_ │
                                                                  │  logs   │
                                                                  └─────────┘
                                                                       │
                                                                       │ Poll
                                                                       ▼
HMI:     Polling...    Polling...    Polling...    Polling...    DETECT!
         ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
         │Fetch │      │Fetch │      │Fetch │      │Fetch │      │HANDLE│
         └──────┘      └──────┘      └──────┘      └──────┘      └──────┘
                                                                       │
                                                                       ▼
Browser:  Scrolling     Scrolling     Scrolling     Scrolling     🏁 BANNER
         ├──────────────────────────────────────────────────────┤
         │                  Timeline Playback                    │
         └───────────────────────────────────────────────────────┘
                                                                  ↓
                                                              STOP!
```

## Detailed Component Flow

### 1. PAS Stub (services/pas/stub/app.py)

```python
def _execute_run(run_id: str):
    start_time = time.time()
    tasks = DAG[run_id]

    # Execute all tasks sequentially
    for task_id in tasks:
        # ... task execution ...
        pass

    # Update run status
    if failed_tasks:
        status = "needs_review"
    else:
        status = "completed"

    # 🆕 NOTIFY HMI
    duration = time.time() - start_time
    _notify_directive_complete(run_id, duration, tasks, failed_tasks)
```

**Output**:
```json
POST http://localhost:6121/action_logs
{
  "task_id": "run-123",
  "from_agent": "PAS_ROOT",
  "to_agent": "HMI",
  "action_type": "directive_complete",
  "action_data": "{\"tasks_total\": 3, \"tasks_succeeded\": 3, ...}",
  "status": "done",
  "tier_from": 0
}
```

### 2. Registry DB (SQLite)

**Before**:
```
action_logs table:
log_id | task_id  | action_type    | from_agent | to_agent | status
-------|----------|----------------|------------|----------|--------
1001   | run-123  | assign_task    | VP_001     | Dir_001  | done
1002   | run-123  | complete_task  | Dir_001    | VP_001   | done
1003   | run-123  | assign_task    | Dir_001    | Mgr_001  | done
...
```

**After PAS Notification**:
```
action_logs table:
log_id | task_id  | action_type         | from_agent | to_agent | action_data
-------|----------|---------------------|------------|----------|------------------
1001   | run-123  | assign_task         | VP_001     | Dir_001  | {...}
1002   | run-123  | complete_task       | Dir_001    | VP_001   | {...}
1003   | run-123  | assign_task         | Dir_001    | Mgr_001  | {...}
...
1099   | run-123  | directive_complete  | PAS_ROOT   | HMI      | {run summary} ← NEW!
```

### 3. HMI Polling (services/webui/hmi_app.py)

```python
def poll_action_logs():
    """Background thread polling every 1 second"""
    while True:
        cursor.execute("""
            SELECT * FROM action_logs
            WHERE log_id > ?
            ORDER BY log_id ASC
            LIMIT 100
        """, (last_known_log_id,))

        new_rows = cursor.fetchall()

        for row in new_rows:
            if row['action_type'] == 'directive_complete':
                # Push to SSE subscribers
                notify_subscribers(row)

        time.sleep(1)
```

### 4. Browser (services/webui/templates/sequencer.html)

```javascript
// Polling loop (every 5 seconds)
async function fetchSequencerData() {
    const response = await fetch('/api/actions/tasks');
    const data = await response.json();
    tasks = data.items;

    // 🆕 CHECK FOR COMPLETION
    const completionTask = tasks.find(t =>
        t.action_type === 'directive_complete' &&
        t.from_agent === 'PAS_ROOT'
    );

    if (completionTask && !projectComplete) {
        handleDirectiveComplete(completionTask);
        //   ├─ Stop playback
        //   ├─ Clear auto-refresh
        //   ├─ Show banner
        //   └─ Scroll to end
    }
}
```

## State Transition Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   HMI Sequencer State Machine                    │
└─────────────────────────────────────────────────────────────────┘

    INITIAL
       │
       │ User opens sequencer
       ▼
  ┌────────┐
  │ LOADING│────fetch data────▶┌─────────┐
  └────────┘                    │ STOPPED │
                                └─────────┘
                                     │
                         User clicks Play
                                     │
                                     ▼
                                ┌─────────┐
                      ┌────────▶│ PLAYING │◀────────┐
                      │         └─────────┘         │
                      │              │              │
            User clicks Stop         │         Timeline
            or reaches end           │           scrolls
                                     │
                         Completion detected
                                     │
                                     ▼
                               ┌──────────┐
                               │ COMPLETE │
                               └──────────┘
                                     │
                                     ├─ isPlaying = false
                                     ├─ Stop auto-refresh
                                     ├─ Show banner
                                     └─ projectComplete = true

                               (Terminal state)
```

## Banner Display Sequence

```
┌──────────────────────────────────────────────────────────────┐
│              "END OF PROJECT" Banner Animation                │
└──────────────────────────────────────────────────────────────┘

T=0ms:    Banner created (off-screen, above viewport)
          Position: translate(-50%, -150%)
          Opacity: 0

          ┌────────────────┐
          │   🏁 BANNER    │ ← Above screen
          └────────────────┘
                            ╲
                             ╲ slideDown animation
                              ╲ (0.5s ease-out)
                               ╲
T=500ms:  Banner visible (centered)
          Position: translate(-50%, -50%)
          Opacity: 1

          ─────────────────────

          ┌────────────────────┐
          │    🏁 BANNER       │ ← Centered
          │                    │
          │  END OF PROJECT    │
          │  ✅ Success        │
          │                    │
          │  [View Final]      │ ← Button
          └────────────────────┘

T=X:      User clicks button

          ┌────────────────────┐
          │  🏁 BANNER (hover) │
          └────────────────────┘
                             ╱
                            ╱ slideUp animation
                           ╱  (0.3s ease-in)
                          ╱
T=X+300ms: Banner removed (above screen)
           Position: translate(-50%, -150%)
           Opacity: 0

           ┌────────────────┐
           │   🏁 BANNER    │ ← Above screen (removed from DOM)
           └────────────────┘
```

## Data Structure: Completion Action Log

```javascript
{
  // Standard action_log fields
  "log_id": 1099,
  "task_id": "run-123",               // Use run_id as pseudo-task
  "parent_log_id": null,
  "timestamp": "2025-11-09T17:30:45.123Z",
  "from_agent": "PAS_ROOT",           // Identifies completion signal
  "to_agent": "HMI",
  "action_type": "directive_complete", // Special type for completion
  "action_name": "Prime Directive Complete",
  "status": "done",
  "tier_from": 0,                     // PAS ROOT is tier 0
  "tier_to": null,

  // Completion-specific data (JSON string)
  "action_data": {
    "run_id": "run-123",
    "project_id": 1,
    "tasks_total": 10,
    "tasks_succeeded": 9,
    "tasks_failed": 1,
    "duration_seconds": 45.2,
    "validation_pass": false,          // Any KPI failures?
    "status": "needs_review"           // completed | needs_review | terminated
  }
}
```

## Timeline Scroll Behavior

```
┌─────────────────────────────────────────────────────────────────┐
│                     Timeline Viewport                            │
└─────────────────────────────────────────────────────────────────┘

DURING PLAYBACK:
┌─────────────────────────────────────────────────────┐
│ ▶ Playing                                           │
│ ┌────────────────────────────────────────────────┐  │
│ │ [T1] [T2] [T3]      [NOW]       [T4] [T5]     │◀─── Auto-scrolls
│ └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
           └─ Playhead moves right, timeline scrolls

AFTER COMPLETION:
┌─────────────────────────────────────────────────────┐
│ 🏁 BANNER OVERLAY                                   │
│ ┌────────────────────────────────┐                  │
│ │   END OF PROJECT               │                  │
│ │   ✅ Success                   │                  │
│ └────────────────────────────────┘                  │
│                                                      │
│ ┌────────────────────────────────────────────────┐  │
│ │ [T1] [T2] [T3] [T4] [T5] [T6] [T7] [T8] [T9] │◀─── Scrolled to end
│ └────────────────────────────────────────────────┘  │
│                                            ↑         │
│                                    Shows last 30s   │
└─────────────────────────────────────────────────────┘
           └─ No more auto-scroll, shows final state
```

## Error Handling Flow

```
┌──────────────────────────────────────────────────────────┐
│                Error Handling Decision Tree               │
└──────────────────────────────────────────────────────────┘

PAS Sends Notification
         │
         ▼
    ┌─────────┐
    │Registry │
    │Running? │
    └─────────┘
      │      │
     Yes     No
      │      │
      │      └──▶ Log Warning
      │           "Registry unavailable"
      │           Continue (non-critical)
      │
      ▼
  Insert Row
      │
      ├──▶ Success ─────▶ HMI Detects
      │                       │
      └──▶ Fail             ▼
            │          Show Banner
            │               │
            └──▶ Log Error  ▼
                         Done!
```

## Performance Metrics

```
┌────────────────────────────────────────────────────────┐
│                    Latency Budget                       │
└────────────────────────────────────────────────────────┘

Component            Action                Time     Notes
─────────────────────────────────────────────────────────
PAS Stub             Execute tasks         15-45s   Synthetic delays
PAS Stub             Send notification     ~10ms    HTTP POST
Registry             Insert row            ~5ms     SQLite write
HMI Polling Thread   Detect new row        0-1s     Poll interval
HMI                  Push SSE              ~1ms     WebSocket
Browser              Detect completion     ~1ms     Array.find()
Browser              Show banner           500ms    Animation
─────────────────────────────────────────────────────────
TOTAL (after tasks)                        0.5-2.5s End-to-end latency
```

## Security Considerations

```
┌────────────────────────────────────────────────────────┐
│                  Security Analysis                      │
└────────────────────────────────────────────────────────┘

Threat Model:

1. ✅ MITIGATED: Malicious completion signal
   - Only PAS_ROOT can send (from_agent check)
   - action_type must be exact string match
   - No privilege escalation possible

2. ✅ MITIGATED: SQL injection
   - Using parameterized queries
   - JSON data properly escaped

3. ✅ MITIGATED: XSS in banner
   - Data sanitized before display
   - No eval() or innerHTML with user data

4. ⚠️ RESIDUAL: Replay attack
   - Completion can be triggered multiple times
   - Mitigated by projectComplete flag (idempotency)

5. ✅ MITIGATED: Denial of service
   - Single completion per run
   - Non-blocking notification (async)
   - Graceful degradation if Registry down
```

## Summary

This diagram shows the complete flow from task execution to banner display, including:
- Timeline visualization
- State transitions
- Data structures
- Performance metrics
- Error handling paths
- Security analysis

**Key Insight**: The entire flow leverages existing infrastructure (action_logs + polling), making it robust, fast, and easy to maintain!
