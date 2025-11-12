# TRON Task Resend Testing Guide

## Overview

This document outlines the testing procedure for the TRON task resend functionality implemented on 2025-11-12.

## What Was Implemented

### Part A: TRON Banner UI ✅
1. **Moved TRON banner below header** - Now positioned using `--header-height` CSS variable
2. **Added dismiss button (✕)** - User can manually clear the banner
3. **Dynamic header height calculation** - Calculates actual header height on page load

### Part B: Task Resend Logic ✅
1. **Task tracking** - `CHILD_ACTIVE_TASKS` dictionary tracks what each Director is working on
2. **Task capture on timeout** - When TRON detects timeout, task context is preserved
3. **Automatic task resend** - After successful restart, parent (Architect) automatically resends the task
4. **Task cleanup** - When task completes, tracking is cleaned up

## Test Scenarios

### Test 1: TRON Banner UI (Visual)

**Steps:**
1. Open http://localhost:6101 in browser
2. Send test event:
   ```bash
   curl -X POST http://localhost:6102/broadcast \
     -H "Content-Type: application/json" \
     -d '{"event_type":"hhmrs_timeout","data":{"agent_id":"Dir-Code","restart_count":1}}'
   ```
3. **Verify:**
   - TRON banner appears **below the header** (not at very top)
   - Banner shows: "⚡ TRON" + event badges + "HHMRS Phase 1"
   - Dismiss button (✕) is visible on right side
   - Clicking ✕ hides the banner and clears events

**Expected Result:** ✅ Banner positioned correctly with working dismiss button

---

### Test 2: Task Resend After Timeout (End-to-End)

**Scenario:** Director-Code times out while working on a task. TRON detects timeout, restarts Dir-Code, and Architect automatically resends the task.

**Prerequisites:**
- All services running (HMI, EventStream, Architect, Dir-Code)
- Logs available for monitoring

**Steps:**

1. **Monitor logs in separate terminals:**
   ```bash
   # Terminal 1: Architect logs
   tail -f logs/pas/architect.log | grep -E "resend|restart|timeout"

   # Terminal 2: Communication logs
   tail -f artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt | grep -E "TRON|resend|Dir-Code"
   ```

2. **Submit a Prime Directive to Architect:**
   ```bash
   curl -X POST http://localhost:6110/submit \
     -H "Content-Type: application/json" \
     -d '{
       "run_id": "test-resend-001",
       "title": "Test Task Resend",
       "prd": "This is a test Prime Directive to verify TRON task resend functionality. Create a simple code change.",
       "budget": {"max_llm_calls": 5, "max_tokens": 10000}
     }'
   ```

3. **Wait for task delegation:**
   ```bash
   # Check that Dir-Code received the task
   tail -n 20 artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt | grep "Job card submitted"
   ```

4. **Simulate Dir-Code timeout:**
   ```bash
   # Kill Dir-Code process
   lsof -ti:6111 | xargs kill -9
   echo "Dir-Code killed at $(date)"
   ```

5. **Wait ~60 seconds for TRON to detect timeout**
   - TRON checks heartbeats every 5s
   - Timeout threshold is 60s (configurable in HMI Settings → HHMRS)

6. **Observe the flow:**
   - **TRON detects timeout** → sends alert to Architect
   - **Architect receives alert** → logs "Child timeout alert: Dir-Code"
   - **Architect restarts Dir-Code** → logs "Attempting process restart"
   - **Dir-Code restarts** → new process spawns, health check passes
   - **Architect looks up task** → finds task in CHILD_ACTIVE_TASKS
   - **Architect resends task** → logs "Resending task to restarted Dir-Code"
   - **Dir-Code receives task** → continues working

7. **Verify in logs:**
   ```bash
   # Check Architect logs for key messages
   grep -E "Successfully restarted.*Dir-Code" logs/pas/architect.log
   grep -E "Resending task to restarted Dir-Code" logs/pas/architect.log
   grep -E "Task resent successfully" logs/pas/architect.log
   ```

8. **Check TRON bar in HMI:**
   - Should show: ⏱️ Dir-Code (timeout event)
   - Then: 🔄 Dir-Code (restart event)
   - Chimes should play for both events

**Expected Results:**
- ✅ TRON detects timeout within ~60s
- ✅ Architect restarts Dir-Code process
- ✅ Health check passes after restart
- ✅ Task is automatically resent to restarted Dir-Code
- ✅ Dir-Code receives task and continues working
- ✅ TRON bar shows both timeout and restart events
- ✅ All steps logged in communication logs

**Success Criteria:**
- Response status: `"restarted_and_resent"`
- Logs contain: "Task resent successfully"
- Dir-Code continues processing the task

---

### Test 3: Task Cleanup After Completion

**Scenario:** Verify that CHILD_ACTIVE_TASKS is cleaned up when a Director completes its task normally.

**Steps:**

1. **Submit a simple Prime Directive:**
   ```bash
   curl -X POST http://localhost:6110/submit \
     -H "Content-Type: application/json" \
     -d '{
       "run_id": "test-cleanup-001",
       "title": "Test Task Cleanup",
       "prd": "Simple task: Create a README.md file.",
       "budget": {"max_llm_calls": 3, "max_tokens": 5000}
     }'
   ```

2. **Monitor communication logs:**
   ```bash
   tail -f artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt | grep -E "Job card submitted|Lane report received|Cleared active task"
   ```

3. **Wait for task completion:**
   - Director processes task
   - Director sends lane report back to Architect

4. **Verify cleanup:**
   ```bash
   # Look for "Cleared active task tracking" message
   grep "Cleared active task tracking for Dir-Code" artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt
   ```

**Expected Results:**
- ✅ Task submitted to Director
- ✅ Director completes task
- ✅ Lane report received by Architect
- ✅ CHILD_ACTIVE_TASKS cleaned up (logged)

---

### Test 4: Multiple Restarts (Escalation)

**Scenario:** Kill Dir-Code multiple times to test escalation to grandparent after max_restarts exceeded.

**Steps:**

1. **Check current max_restarts setting:**
   ```bash
   curl -s http://localhost:6101/api/settings/hhmrs | jq '.hhmrs.max_restarts'
   ```
   Default: 3

2. **Submit Prime Directive**

3. **Kill Dir-Code repeatedly:**
   ```bash
   # Kill 1
   lsof -ti:6111 | xargs kill -9
   # Wait 60s for TRON to restart (restart_count=1)

   # Kill 2
   lsof -ti:6111 | xargs kill -9
   # Wait 60s for TRON to restart (restart_count=2)

   # Kill 3
   lsof -ti:6111 | xargs kill -9
   # Wait 60s for TRON to restart (restart_count=3)

   # Kill 4 (should escalate to PAS Root)
   lsof -ti:6111 | xargs kill -9
   # Wait 60s - should see escalation, not restart
   ```

4. **Verify escalation:**
   ```bash
   # Should see escalation event in logs
   grep "Escalated.*Dir-Code.*failure to PAS Root" logs/pas/architect.log
   ```

**Expected Results:**
- ✅ First 3 timeouts: Task resent each time
- ✅ 4th timeout: Escalation to PAS Root
- ✅ TRON bar shows escalation event (⬆️)

---

## Log Inspection Commands

```bash
# View all TRON-related logs
tail -100 artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt | grep TRON

# View Architect restart logs
grep -E "restart|resend" logs/pas/architect.log | tail -20

# View Director-Code heartbeats
tail -50 artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt | grep "Dir-Code.*heartbeat"

# Check TRON retry counts
curl -s http://localhost:6110/health | jq '.tron_retry_counts'
```

---

## Troubleshooting

### Issue: TRON doesn't detect timeout

**Check:**
- Is TRON running? `curl -s http://localhost:6110/health | jq '.tron_enabled'`
- What's the timeout threshold? Default 60s
- Are heartbeats being sent? Check comms logs

**Fix:**
- Reduce timeout threshold in HMI Settings → HHMRS → timeout_threshold_s (try 30s)

### Issue: Task not resent after restart

**Check:**
- Was task tracked in CHILD_ACTIVE_TASKS?
- Did restart succeed?
- Check Architect logs for "Resending task" message

**Fix:**
- Verify delegate_to_directors() recorded task (check logs for "Track active task")

### Issue: Process won't restart

**Check:**
- Port already in use? `lsof -ti:6111`
- Logs show error? `tail logs/pas/director_code.log`

**Fix:**
- Manually kill lingering processes
- Check AGENT_RESTART_CONFIG in architect/app.py

---

## Success Checklist

After running all tests, verify:

- [ ] TRON banner appears below header
- [ ] Dismiss button (✕) works
- [ ] TRON detects timeout within configured threshold
- [ ] Process restarts successfully
- [ ] Health check passes after restart
- [ ] Task automatically resent to restarted process
- [ ] Logs contain all expected messages
- [ ] TRON bar shows events in HMI
- [ ] Chimes play for HHMRS events
- [ ] Task cleanup happens when task completes
- [ ] Escalation works after max_restarts exceeded

---

## Implementation Summary

**Files Modified:**
1. `services/webui/templates/base.html` - TRON banner positioning + dismiss button
2. `services/pas/architect/app.py` - Task tracking + resend logic

**Key Changes:**
- Added `CHILD_ACTIVE_TASKS` dictionary (tracks active tasks per Director)
- Updated `delegate_to_directors()` to record tasks
- Updated `handle_child_timeout()` to resend tasks after restart
- Updated `receive_lane_report()` to clean up task tracking
- Added `dismissTronBar()` and `setHeaderHeight()` JavaScript functions

**Next Steps:**
1. Run tests above
2. Fix any issues found
3. Document in session summary
4. Commit changes

---

**Test Date:** 2025-11-12
**Tested By:** [Your Name]
**Result:** [PASS/FAIL]
**Notes:** [Any observations or issues]
