# ⚡ QUICK START - Test Everything NOW (15 Minutes)

**Date**: November 7, 2025
**Status**: ✅ Ready to Execute

---

## 🎯 What You'll Prove in 15 Minutes

1. ✅ PAS stub works (execution spine operational)
2. ✅ VP CLI works (terminal client operational)
3. ✅ End-to-end integration works (LCO → PAS)

---

## 📋 Step-by-Step Instructions

### **Step 1: Start PAS Stub** (2 minutes)

```bash
# Terminal 1: Start PAS stub
make run-pas-stub
```

**Expected Output**:
```
[PAS Stub] Starting on port 6200...
INFO:     Uvicorn running on http://127.0.0.1:6200 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Keep this terminal open!**

---

### **Step 2: Health Check** (1 minute)

```bash
# Terminal 2: Health check
curl http://localhost:6200/health | jq
```

**Expected Output**:
```json
{
  "status": "ok",
  "active_runs": 0,
  "total_tasks": 0,
  "total_receipts": 0
}
```

✅ If you see this, PAS stub is operational!

---

### **Step 3: OpenAPI Docs** (1 minute)

```bash
# Open Swagger UI in browser
open http://localhost:6200/docs
# Or manually navigate to: http://localhost:6200/docs
```

**Expected**: Swagger UI showing all 12 PAS API endpoints

✅ If you see this, API contracts are stable!

---

### **Step 4: Run PAS End-to-End Demo** (5 minutes)

```bash
# Terminal 2: Run PAS stub demo
bash tests/demos/demo_pas_stub_e2e.sh
```

**Expected Output** (abbreviated):
```
=== PAS Stub End-to-End Demo ===

✓ PAS stub is running

1️⃣  Health Check
{ "status": "ok", ... }

2️⃣  Start Run (baseline)
{ "status": "executing", "run_id": "r-2025-11-07-143052" }

3️⃣  Submit Job Cards (Code-Impl + Vector-Ops lanes)
   Task 1 (Code-Impl): task-abc12345
   Task 2 (Vector-Ops): task-def67890

...

✅ End-to-End Demo Complete!

📊 Summary:
   - Run ID: r-2025-11-07-143052
   - Tasks submitted: 2
   - Lanes tested: Code-Impl, Vector-Ops
   - Idempotency: ✓ Verified
   - Synthetic execution: ✓ Completed
   - KPIs emitted: ✓ (check run status for violations)
```

✅ If you see this, PAS execution works!

---

### **Step 5: Run VP CLI Integration Demo** (6 minutes)

```bash
# Terminal 2: Run VP + PAS integration demo
bash tests/demos/demo_vp_pas_integration.sh
```

**Expected Output** (abbreviated):
```
=== VP + PAS Integration Demo ===

✓ PAS stub is running
✓ Cleaned VP state

📋 Test Workflow:

1️⃣  vp new --name demo-project
Creating project: demo-project
✓ Project created
  Project ID: 1234
  Run ID: r-2025-11-07-143102

2️⃣  vp plan
Planning project 1234...
✓ Plan generated (stub)

3️⃣  vp estimate
📊 Estimates (90% confidence intervals):
  Tokens:   15,000 (13,200 - 16,800)
  Duration: 25 min (20 - 30 min)
  Cost:     $2.50 ($2.20 - $2.80)
  Energy:   0.35 kWh (0.30 - 0.40 kWh)

4️⃣  vp simulate --rehearsal 0.01
✓ Rehearsal simulation complete
  Strata coverage:   100%
  Rehearsal tokens:  150
  Projected tokens:  15,000
  Confidence band:   13,200 - 16,800
  Risk score:        0.14

5️⃣  vp start
✓ Run started: executing
Submitting sample tasks...
  ✓ Code-Impl: task-xyz
  ✓ Vector-Ops: task-abc

⏳ Waiting 20 seconds for synthetic execution...

6️⃣  vp status
Run: r-2025-11-07-143102
Status: completed

📊 Progress:
  Tasks:       2/2 completed
  Failed:      0

💰 Spend:
  Cost:        $0.29
  Energy:      0.042 kWh

⏰ Runway:     0 minutes

Tasks:
  ✓ Code-Impl: succeeded
  ✓ Vector-Ops: succeeded

✅ Integration Demo Complete!

📊 Summary:
   - VP CLI: ✓ Operational
   - PAS Stub: ✓ Executing tasks
   - End-to-end flow: ✓ Working
```

✅ If you see this, **FULL INTEGRATION WORKS!**

---

## 🎉 Success Criteria

After completing all 5 steps, you should have:

- [x] PAS stub running on port 6200
- [x] Health check returns `{"status": "ok"}`
- [x] OpenAPI docs accessible
- [x] PAS end-to-end demo passes
- [x] VP CLI integration demo passes
- [x] Tasks execute synthetically (Code-Impl + Vector-Ops)
- [x] KPIs emitted (test_pass_rate, index_freshness)
- [x] Status shows 2/2 tasks completed

---

## 🚧 Troubleshooting

### Issue: "PAS stub not running"

**Solution**:
```bash
# Terminal 1: Check if port 6200 is in use
lsof -ti:6200 | xargs kill -9

# Restart PAS stub
make run-pas-stub
```

---

### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Solution**:
```bash
# Install dependencies
./.venv/bin/pip install fastapi uvicorn click requests
```

---

### Issue: "curl: command not found"

**Solution**:
```bash
# macOS: Install curl
brew install curl

# Or use httpie
brew install httpie
http http://localhost:6200/health
```

---

## 📚 Next Steps (After 15-Minute Test)

### If All Tests Pass ✅

**Immediate**:
1. **Review**: `docs/SHIP_IT_SUMMARY.md` (comprehensive overview)
2. **Review**: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md` (10-week plan)
3. **Sign-off**: Approve phased rollout

**Week 1 (Nov 8-14)**:
1. **Start Phase 1**: LightRAG Code Index
2. **Daily standup**: 09:30 ET
3. **Allocate resources**: PAS design review

---

### If Tests Fail ❌

**Debug Steps**:
1. Check PAS stub logs (Terminal 1)
2. Verify Python dependencies:
   ```bash
   ./.venv/bin/pip install -r requirements.txt
   ```
3. Check port availability:
   ```bash
   lsof -i:6200
   ```
4. Review error messages in demo scripts

**Escalation**: #plms-rollout (Slack)

---

## 📊 What You Just Proved

| Component | Test | Result |
|-----------|------|--------|
| PAS Stub API | Health check | ✅ Operational |
| PAS Stub Execution | End-to-end demo | ✅ Tasks execute |
| VP CLI | Commands work | ✅ All 7 commands |
| Integration | VP → PAS → Execution | ✅ Full flow |
| Idempotency | Re-submit same key | ✅ Returns same task_id |
| Synthetic KPIs | Lane-specific | ✅ test_pass_rate, BLEU, etc. |
| OpenAPI | Swagger UI | ✅ All 12 endpoints documented |

---

## 🎯 You're Ready!

**Time Spent**: 15 minutes
**Components Tested**: 7
**Tests Passed**: 100%

**Next**: Review `docs/SHIP_IT_SUMMARY.md` for full rollout plan

---

**🚀 LET'S SHIP IT!**

_Quick Start - Nov 7, 2025_
