# Prime Directive Completion Signal - Deployment Checklist

## Pre-Deployment Verification

### Code Review
- [x] PAS Stub changes reviewed (`services/pas/stub/app.py`)
- [x] HMI Sequencer changes reviewed (`services/webui/templates/sequencer.html`)
- [x] No hardcoded values or debug code left in
- [x] Error handling implemented for all edge cases
- [x] Code follows existing patterns and conventions

### Documentation
- [x] Protocol specification written
- [x] Implementation details documented
- [x] Test script created
- [x] Visual diagrams created
- [x] This deployment checklist created

### Testing
- [ ] Local test passed (run `scripts/test_completion_signal.sh`)
- [ ] Manual browser verification completed
- [ ] Banner displays correctly
- [ ] Timeline stops scrolling
- [ ] Auto-refresh stops
- [ ] Browser console shows correct logs
- [ ] Registry database contains completion entry

## Deployment Steps

### 1. Backup Current State
```bash
# Backup current files
cp services/pas/stub/app.py services/pas/stub/app.py.backup
cp services/webui/templates/sequencer.html services/webui/templates/sequencer.html.backup

# Backup Registry DB (optional)
cp artifacts/registry/registry.db artifacts/registry/registry.db.backup
```

### 2. Deploy PAS Stub Changes
```bash
# Stop PAS Stub
lsof -ti:6200 | xargs kill -9 2>/dev/null

# Deploy updated file (already in place from implementation)
# services/pas/stub/app.py

# Verify syntax
./.venv/bin/python -m py_compile services/pas/stub/app.py

# Start PAS Stub
./.venv/bin/python services/pas/stub/app.py &

# Wait for startup
sleep 2

# Health check
curl -s http://localhost:6200/health | jq .
```

### 3. Deploy HMI Changes
```bash
# Stop HMI
lsof -ti:6101 | xargs kill -9 2>/dev/null

# Deploy updated file (already in place from implementation)
# services/webui/templates/sequencer.html

# Start HMI
./.venv/bin/python services/webui/hmi_app.py &

# Wait for startup
sleep 2

# Health check
curl -s http://localhost:6101/health | jq .
```

### 4. Verify Deployment
```bash
# Run automated test
bash scripts/test_completion_signal.sh

# Expected output:
# ✓ All services running
# ✓ Run completes
# ✓ Completion signal found in Registry
# ✓ All checks passed
```

### 5. Manual Browser Test
1. Open: http://localhost:6101/sequencer?task_id={run_id}
2. Verify banner appears
3. Check Network tab: auto-refresh stopped
4. Check Console: `🎯 [PRIME DIRECTIVE COMPLETE]`
5. Click "View Final State" button
6. Verify banner closes and timeline scrolls

## Post-Deployment Monitoring

### Immediate (First 30 Minutes)
```bash
# Watch PAS logs
tail -f /tmp/pas_stub.log

# Expected: ✅ [PAS] Notified HMI of Prime Directive completion

# Watch HMI logs
tail -f /tmp/hmi_app.log

# Expected: No errors related to completion handling
```

### Short Term (First 24 Hours)
- [ ] Monitor Registry DB size (should grow by ~1 row per completed run)
- [ ] Check for error logs in PAS/HMI
- [ ] Verify banner appears for all completed runs
- [ ] Collect user feedback on banner UX

### Long Term (First Week)
- [ ] Analyze completion notification latency (target: <2s)
- [ ] Verify no memory leaks (banner cleanup)
- [ ] Check for edge cases (concurrent runs, failed runs, etc.)
- [ ] Plan Phase 2 enhancements (final report page)

## Rollback Procedure

If critical issues occur:

### 1. Stop Services
```bash
lsof -ti:6200 | xargs kill -9 2>/dev/null  # PAS Stub
lsof -ti:6101 | xargs kill -9 2>/dev/null  # HMI
```

### 2. Restore Backups
```bash
# Restore PAS Stub
cp services/pas/stub/app.py.backup services/pas/stub/app.py

# Restore HMI Sequencer
cp services/webui/templates/sequencer.html.backup services/webui/templates/sequencer.html

# (Optional) Restore Registry DB
cp artifacts/registry/registry.db.backup artifacts/registry/registry.db
```

### 3. Restart Services
```bash
./.venv/bin/python services/pas/stub/app.py &
sleep 2
./.venv/bin/python services/webui/hmi_app.py &
sleep 2
```

### 4. Verify Rollback
```bash
# Health checks
curl -s http://localhost:6200/health
curl -s http://localhost:6101/health

# Test basic functionality (without completion signal)
```

### 5. Clean Up Completion Entries (if needed)
```bash
# Remove completion action logs from Registry
sqlite3 artifacts/registry/registry.db \
  "DELETE FROM action_logs WHERE action_type='directive_complete'"
```

## Troubleshooting Guide

### Issue: Banner Doesn't Appear

**Check 1**: Verify Registry has completion entry
```bash
sqlite3 artifacts/registry/registry.db \
  "SELECT * FROM action_logs WHERE action_type='directive_complete' ORDER BY log_id DESC LIMIT 5"
```

**Check 2**: Verify browser console logs
- Open: http://localhost:6101/sequencer
- F12 → Console
- Look for: `🎯 [PRIME DIRECTIVE COMPLETE]`

**Check 3**: Verify JavaScript loaded correctly
- F12 → Sources
- Check `sequencer.html` for `handleDirectiveComplete` function

**Fix**: If missing, re-deploy HMI file

### Issue: Timeline Keeps Scrolling

**Check 1**: Verify `isPlaying` flag set to false
- F12 → Console
- Type: `isPlaying`
- Expected: `false`

**Check 2**: Verify auto-refresh stopped
- F12 → Network tab
- Expected: No more `/api/actions/tasks` calls

**Fix**: Verify `handleDirectiveComplete()` was called

### Issue: PAS Notification Fails

**Check 1**: Verify Registry is running
```bash
curl -s http://localhost:6121/health
```

**Check 2**: Check PAS console logs
```bash
tail -f /tmp/pas_stub.log
# Look for: ⚠️ Failed to notify HMI of completion
```

**Fix**: Restart Registry, then restart PAS Stub

### Issue: Multiple Banners Appear

**Symptom**: Banner appears multiple times for same run

**Cause**: `projectComplete` flag not persisting across refreshes

**Fix**: This is expected behavior - banner re-appears on page reload to show final state

**To Prevent**: Store `projectComplete` in sessionStorage
```javascript
// Add to handleDirectiveComplete()
sessionStorage.setItem(`project_complete_${run_id}`, 'true');

// Check before showing banner
if (sessionStorage.getItem(`project_complete_${run_id}`)) {
    return; // Already shown
}
```

## Performance Benchmarks

### Expected Latency
- PAS notification: <10ms
- Registry insert: <5ms
- HMI detection: 0-1s (poll interval)
- Banner render: <500ms (animation)
- **Total**: 0.5-2.5s after last task completes

### Load Testing
```bash
# Submit 10 concurrent runs
for i in {1..10}; do
  (
    RUN_ID="stress-test-$i"
    curl -X POST http://localhost:6200/pas/v1/runs/start \
      -H "Content-Type: application/json" \
      -d "{\"project_id\": 1, \"run_id\": \"$RUN_ID\", \"run_kind\": \"baseline\"}"

    # Submit 5 tasks per run
    for j in {1..5}; do
      curl -X POST http://localhost:6200/pas/v1/jobcards \
        -H "Content-Type: application/json" \
        -d "{\"project_id\": 1, \"run_id\": \"$RUN_ID\", \"lane\": \"Code-Impl\"}"
    done
  ) &
done

wait

# Check Registry for 10 completion entries
sqlite3 artifacts/registry/registry.db \
  "SELECT COUNT(*) FROM action_logs WHERE action_type='directive_complete' AND task_id LIKE 'stress-test-%'"
```

Expected: 10 completion entries (one per run)

## Success Criteria

Deployment is successful if:
- [x] All services start without errors
- [ ] Test script passes (`scripts/test_completion_signal.sh`)
- [ ] Banner appears in browser for completed runs
- [ ] Timeline stops scrolling on completion
- [ ] Auto-refresh stops on completion
- [ ] No errors in PAS/HMI logs
- [ ] Registry DB contains completion entries
- [ ] Performance within expected range (<2.5s latency)
- [ ] No memory leaks after 1 hour
- [ ] Rollback procedure tested and documented

## Sign-Off

### Technical Review
- [ ] Code changes approved by: __________________
- [ ] Date: __________________

### QA Testing
- [ ] Functional testing passed by: __________________
- [ ] Date: __________________

### Deployment
- [ ] Deployed by: __________________
- [ ] Date: __________________
- [ ] Production URL: __________________

### Post-Deployment
- [ ] Monitoring configured: __________________
- [ ] Alerts configured: __________________
- [ ] Documentation updated: __________________

## Contact Information

**For Issues**:
- Development: @trentcarter
- Operations: @operations-team
- Emergency: #pas-incidents Slack channel

**Documentation**:
- Protocol: `docs/design_documents/PRIME_DIRECTIVE_COMPLETION_PROTOCOL.md`
- Implementation: `PRIME_DIRECTIVE_COMPLETION_IMPLEMENTATION.md`
- This Checklist: `DEPLOYMENT_CHECKLIST.md`

---

**Deployment Status**: 🟡 Ready for Testing
**Next Step**: Run `bash scripts/test_completion_signal.sh`
