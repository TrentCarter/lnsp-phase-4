# Agent Family Tests - Resource Management & Health Monitoring

**Date**: 2025-11-14
**Status**: ✅ Implemented
**Location**: HMI → Agent Family Test tab

---

## Overview

Added 3 new system tests to the Agent Family Test suite to verify resource management and health monitoring systems. These tests complement the existing 10 tests by focusing on **resource contention**, **busy state handling**, and **TRON health monitoring**.

---

## New Tests

### Test 11: 🏊 Resource Pool Exhaustion

**Purpose**: Test Manager's ability to handle programmer pool exhaustion and escalate to Director

**Scenario**:
1. Fill all 5 programmers (Prog-001 to Prog-005) with long-running tasks
2. Manager receives 6th task when pool is full
3. Manager tries all programmers (all busy)
4. Manager should either:
   - Queue the task
   - Escalate to Director
   - Return "retry later" response

**What This Tests**:
- ✅ Manager fallback logic (try Prog-01 → Prog-02 → Prog-03...)
- ✅ Manager escalation to Director when pool exhausted
- ✅ Graceful degradation under load
- ✅ Real production scenario (resource contention)

**Expected Behavior**:
- **Success**: Manager queues or escalates (doesn't fail)
- **Failure**: Manager crashes or loses task

**Code Location**: `services/webui/templates/model_pool_enhanced.html:2017-2075`

---

### Test 12: ⏸️ Programmer Busy State

**Purpose**: Test programmer busy state rejection and Manager fallback routing

**Scenario**:
1. Manager A (Mgr-Code-01) sends long task to Prog-001
2. Manager B (Mgr-Models-01) immediately tries to send to same Prog-001
3. Prog-001 should reject with HTTP 503 "Service Unavailable" (busy)
4. Manager B falls back to Prog-002 (succeeds)

**What This Tests**:
- ✅ Programmer busy state detection
- ✅ HTTP 503 rejection when busy
- ✅ Manager automatic fallback to next available programmer
- ✅ Concurrent resource access handling

**Expected Behavior**:
- **Success**: Prog-001 rejects 2nd task, Manager B uses Prog-002
- **Failure**: Prog-001 accepts concurrent tasks (no busy state tracking)

**Implementation Status**:
- ⚠️ **Test reveals**: Programmers may not currently track busy state
- 🔧 **TODO**: Add busy state tracking to Programmer services

**Code Location**: `services/webui/templates/model_pool_enhanced.html:2077-2124`

---

### Test 13: ⏱️ TRON Timeout Detection

**Purpose**: Verify TRON (HeartbeatMonitor) is running and configured for timeout detection

**Scenario**:
1. Check TRON service health (port 6109)
2. Verify TRON configuration (30s heartbeat, 60s timeout)
3. Document full test requirements (agent crash simulation)

**What This Tests**:
- ✅ TRON service is running
- ✅ TRON health endpoint responding
- ✅ System infrastructure for timeout detection
- 📝 **Mock test only** - Full test requires agent crash simulation

**Full Test Requirements** (TODO):
1. Agent stops sending heartbeats
2. TRON detects timeout after 60s (2 missed heartbeats)
3. TRON alerts parent via RPC to `/handle_child_timeout`
4. Parent restarts agent or routes work elsewhere

**Expected Behavior**:
- **Success**: TRON service running and responding
- **Failure**: TRON not running (port 6109 not listening)

**Related PRD**: `docs/PRDs/PRD_Hierarchical_Health_Monitoring_Retry_System.md`

**Code Location**: `services/webui/templates/model_pool_enhanced.html:2126-2179`

---

## Integration with Existing Architecture

### TRON (HeartbeatMonitor) - Port 6109
- **Service**: `services/common/heartbeat.py`
- **Purpose**: Centralized health monitoring (pure Python, no LLM)
- **Features**:
  - Tracks all agent heartbeats (30s intervals)
  - Detects timeouts (60s = 2 missed heartbeats)
  - Alerts parent when child fails
  - Pure heuristic (no LLM calls)

### Resource Manager - Port 6104
- **Service**: `services/resource_manager/resource_manager.py`
- **Purpose**: CPU/memory/GPU quota tracking
- **Features**:
  - Resource reservations
  - Quota enforcement
  - Port allocation tracking

### Manager Fallback Logic
- **Current State**: Managers have agent chat endpoints
- **Need**: Fallback routing logic when programmers busy
- **TODO**: Implement try-next-programmer logic in Manager services

---

## Test Results Summary

| Test | Status | Pass Criteria | Notes |
|------|--------|---------------|-------|
| **Resource Pool Exhaustion** | 🟡 Partial | Manager queues or escalates | Reveals Manager routing logic needed |
| **Programmer Busy State** | 🔴 Fail | Prog rejects when busy | Reveals busy state tracking needed |
| **TRON Timeout Detection** | 🟢 Pass | TRON running | Infrastructure verified, full test TODO |

**Legend**:
- 🟢 Pass: Test validates expected behavior
- 🟡 Partial: Test runs but reveals missing features
- 🔴 Fail: Test exposes bugs or missing functionality

---

## Next Steps

### Immediate (High Priority)
1. **Add busy state tracking to Programmer services**
   - Track current task execution
   - Return HTTP 503 when busy
   - Clear busy state on task completion

2. **Implement Manager fallback routing**
   - Try programmers in sequence (Prog-01 → Prog-02 → Prog-03)
   - Escalate to Director if all busy
   - Track programmer availability

3. **Complete TRON full test**
   - Add endpoint to simulate agent crash
   - Test TRON timeout detection (60s)
   - Verify parent receives alert
   - Test parent restart logic

### Future (Medium Priority)
4. **Add Resource Manager integration**
   - Managers query Resource Manager before routing
   - Resource Manager tracks real-time programmer state
   - Provide optimal routing recommendations

5. **Add retry logic to Managers**
   - Exponential backoff when pool full
   - Circuit breaker for failing programmers
   - Cost-aware routing (prefer cheaper LLMs)

---

## Code Changes

### Files Modified
1. `services/webui/templates/model_pool_enhanced.html`
   - Added 3 test buttons (lines 567-580)
   - Added 3 test implementations (lines 2016-2179)

### Files to Modify (TODO)
1. `services/pas/programmer_*/app.py`
   - Add busy state tracking
   - Return HTTP 503 when busy

2. `services/pas/manager_*/app.py`
   - Add fallback routing logic
   - Add escalation to Director

3. `services/common/heartbeat.py` (TRON)
   - Add parent alert RPC
   - Complete timeout handling

---

## Related Documentation

- **TRON PRD**: `docs/PRDs/PRD_Hierarchical_Health_Monitoring_Retry_System.md` (1420 lines)
- **Service Ports**: `docs/SERVICE_PORTS.md`
- **Programmer Pool**: `docs/PROGRAMMER_POOL_ARCHITECTURE.md`
- **Multi-Tier PAS**: `docs/MULTITIER_PAS_ARCHITECTURE.md`

---

## Usage

1. Open HMI: http://localhost:6101
2. Navigate to "Agent Family Test" tab
3. Click any of the new test buttons:
   - 🏊 Resource Pool Exhaustion
   - ⏸️ Programmer Busy State
   - ⏱️ TRON Timeout Detection
4. View test results in visualization panel
5. Check comms logs: `artifacts/logs/pas_comms_2025-11-14.txt`

---

**Status**: Ready for testing
**Blockers**: None (tests run in current state, reveal missing features)
**Impact**: Critical for production-ready resource management
