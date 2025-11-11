# Parent-Child Communication Tracking Implementation

**Date**: November 10, 2025
**Status**: ✅ Production Ready
**Version**: 1.0

---

## 🎯 Summary

Implemented complete parent-child tracking for all agent communications in the PAS (Polyglot Agent Swarm). Every message now links back to its parent request, creating a complete audit trail from user input to final output.

---

## 🔧 What Was Implemented

### Core Infrastructure (Already Existed)

The communication logging infrastructure was already production-ready:

- ✅ **CommsLogger** (`services/common/comms_logger.py`)
  - Accepts `parent_log_id` parameter in all log methods
  - Returns `log_id` for use in child logs
  - Writes to both flat `.txt` files and SQLite database

- ✅ **Registry Service** (`services/registry/registry_service.py`)
  - Stores `parent_log_id` in `action_logs` table
  - Provides query endpoints for retrieving log hierarchies

- ✅ **HMI** (`services/webui/hmi_app.py`)
  - Tree and Sequencer views ready to consume parent-child data

### What Was Missing (Fixed Nov 10, 2025)

**Services weren't capturing or passing `log_id` values!**

#### 1. PAS Root Service (`services/pas/root/app.py`)

**Problem**: Wasn't capturing log_id or passing parent_log_id

**Fix Applied:**
```python
# Before (no parent tracking)
logger.log_cmd(
    from_agent="Gateway",
    to_agent="PAS Root",
    message="Submit Prime Directive",
    run_id=run_id
)

# After (full parent tracking)
gateway_log_id = logger.log_cmd(  # ← Capture log_id
    from_agent="Gateway",
    to_agent="PAS Root",
    message="Submit Prime Directive",
    run_id=run_id
)
RUNS[run_id]["gateway_log_id"] = gateway_log_id  # ← Store for background task

# Later: Link child operations
status_log_id = logger.log_status(
    from_agent="PAS Root",
    to_agent="Gateway",
    message="Started execution",
    run_id=run_id,
    parent_log_id=gateway_log_id  # ← Link to parent
)
```

**Changes Made:**
- Lines 246-273: Capture `gateway_log_id` and store in RUNS dict
- Lines 85-127: Retrieve `gateway_log_id` in background task, link all child operations
- Lines 155-212: Pass `parent_log_id` to completion/error responses

#### 2. Aider-LCO RPC Service (`services/tools/aider_rpc/app.py`)

**Problem**: Wasn't accepting or using parent_log_id from PAS Root

**Fix Applied:**

1. **Update Request Model** (line 99):
```python
class EditRequest(BaseModel):
    message: str
    files: List[str]
    run_id: Optional[str] = None
    parent_log_id: Optional[int] = None  # ← Added this field
```

2. **Capture and Use parent_log_id** (lines 125-145):
```python
# Extract parent_log_id from request
parent_log_id = req.parent_log_id

# Log incoming command with parent link
cmd_log_id = logger.log(
    from_agent="PAS Root",
    to_agent="Aider-LCO",
    msg_type=MessageType.CMD,
    message=f"Execute: {req.message[:100]}...",
    run_id=run_id,
    parent_log_id=parent_log_id  # ← Link to parent
)
```

3. **Link All Responses** (lines 150-300):
```python
# Validation errors
logger.log_response(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="File not allowed",
    run_id=run_id,
    parent_log_id=cmd_log_id  # ← Link to parent
)

# Execution status
status_log_id = logger.log_status(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Starting Aider execution",
    run_id=run_id,
    parent_log_id=cmd_log_id  # ← Link to parent
)

# Final response
logger.log_response(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Completed successfully",
    run_id=run_id,
    parent_log_id=status_log_id  # ← Link to parent
)
```

4. **Pass parent_log_id in HTTP Request** (PAS Root → Aider-LCO):
```python
payload = {
    "message": nl,
    "files": files,
    "run_id": run_id,
    "parent_log_id": aider_cmd_log_id  # ← Pass parent_log_id via HTTP
}
```

---

## 🧪 Test Results

### Test Case: Prime Directive Submission

**Input:**
```bash
curl -X POST http://localhost:6100/pas/prime_directives \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Test parent-child tracking",
    "description": "Verify parent-child log tracking",
    "goal": "Add comment to docs/readme.txt",
    "repo_root": "/path/to/project",
    "entry_files": ["docs/readme.txt"]
  }'
```

**Resulting Log Hierarchy:**

```sql
SELECT log_id, parent_log_id, from_agent, to_agent, action_name, status
FROM action_logs
WHERE task_id = '2a0b4805-b23c-4bf2-9a32-ba0dbccc7e38'
ORDER BY log_id ASC;
```

**Output:**
```
log_id | parent_log_id | from_agent  | to_agent   | action_name                                  | status
-------|---------------|-------------|------------|----------------------------------------------|--------
2417   | NULL          | Gateway     | PAS Root   | Submit Prime Directive: Add Test Message...  | unknown
2418   | 2417          | PAS Root    | Gateway    | Queued: Add Test Message...                  | queued
2419   | 2417          | PAS Root    | Gateway    | Started execution: Add Test Message...       | running
2420   | 2419          | PAS Root    | Aider-LCO  | Execute Prime Directive: Add a comment...    | unknown
2421   | 2420          | PAS Root    | Aider-LCO  | Execute: You are refactoring/creating...     | queued
2422   | 2421          | Aider-LCO   | PAS Root   | File not allowed: /path/to/readme.txt        | error
2423   | 2420          | PAS Root    | Gateway    | Error: {"detail":"File not allowed..."}      | error
```

### Visual Hierarchy

```
2417: Gateway → PAS Root (ROOT: Submit Prime Directive)
├─ 2418: PAS Root → Gateway (Queued)
├─ 2419: PAS Root → Gateway (Started execution)
   └─ 2420: PAS Root → Aider-LCO (Execute Prime Directive)
      ├─ 2421: PAS Root → Aider-LCO (Execute: You are refactoring...)
      │  └─ 2422: Aider-LCO → PAS Root (ERROR: File not allowed)
      └─ 2423: PAS Root → Gateway (ERROR: File not allowed)
```

**Validation:** ✅ **PASS**
- Every message (except root) has a non-NULL `parent_log_id`
- Complete audit trail from user input to error response
- Error traced back through Aider-LCO → PAS Root → Gateway

---

## 📊 Benefits

### 1. Complete Audit Trail
Every message links back to its originating request, creating an unbroken chain from user input to final output.

### 2. Error Traceability
When an error occurs, you can trace it back through the entire call stack:
```
Error at log_id 2422 (Aider-LCO)
  ← Caused by command at log_id 2421 (PAS Root → Aider-LCO)
    ← Caused by execution start at log_id 2419 (PAS Root)
      ← Originated from user request at log_id 2417 (Gateway)
```

### 3. Performance Analysis
Measure latency at each delegation level:
- Gateway → PAS Root: Time between log 2417 and 2418
- PAS Root → Aider-LCO: Time between log 2420 and 2422
- Total execution time: Time from log 2417 to log 2423

### 4. Tree Visualization
HMI can build accurate hierarchical trees showing:
- Which agents communicated with whom
- The order of operations
- Where errors occurred in the flow
- Which LLM was used at each step

---

## 🔍 Database Schema

### action_logs Table

```sql
CREATE TABLE action_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    parent_log_id INTEGER,  -- ← Links to parent log entry
    timestamp TEXT NOT NULL,
    from_agent TEXT,
    to_agent TEXT,
    action_type TEXT NOT NULL,
    action_name TEXT,
    action_data TEXT,
    status TEXT,
    tier_from INTEGER,
    tier_to INTEGER,
    FOREIGN KEY (parent_log_id) REFERENCES action_logs(log_id)
);

CREATE INDEX idx_action_logs_parent_log_id ON action_logs(parent_log_id);
```

### Querying Hierarchies

**Get all children of a log entry:**
```sql
SELECT * FROM action_logs
WHERE parent_log_id = 2420
ORDER BY log_id ASC;
```

**Get complete tree for a task:**
```sql
WITH RECURSIVE tree AS (
  -- Root messages (parent_log_id IS NULL)
  SELECT log_id, parent_log_id, from_agent, to_agent, action_name, 0 as depth
  FROM action_logs
  WHERE task_id = '2a0b4805-b23c-4bf2-9a32-ba0dbccc7e38'
    AND parent_log_id IS NULL

  UNION ALL

  -- Children
  SELECT a.log_id, a.parent_log_id, a.from_agent, a.to_agent, a.action_name, t.depth + 1
  FROM action_logs a
  INNER JOIN tree t ON a.parent_log_id = t.log_id
)
SELECT * FROM tree ORDER BY log_id ASC;
```

---

## 🚀 Next Steps

### Phase 1: HMI Tree Visualization (Recommended)

Update `services/webui/hmi_app.py` to use `parent_log_id` for building trees:

**Current approach** (flat list):
```python
def build_tree_from_actions(task_id):
    actions = fetch_actions(task_id)
    # Builds tree by inferring relationships from agent names
    return build_tree_heuristically(actions)
```

**New approach** (use parent_log_id):
```python
def build_tree_from_actions(task_id):
    actions = fetch_actions(task_id)
    # Build tree using parent_log_id foreign key
    tree = {}
    for action in actions:
        tree[action['log_id']] = action
        if action['parent_log_id']:
            parent = tree[action['parent_log_id']]
            parent.setdefault('children', []).append(action)
    return [a for a in tree.values() if not a['parent_log_id']]
```

### Phase 2: Performance Metrics

Add latency tracking:
```python
def calculate_delegation_latency(task_id):
    """Calculate time between parent command and child response"""
    query = """
        SELECT
            p.log_id as parent_id,
            p.timestamp as command_time,
            c.timestamp as response_time,
            JULIANDAY(c.timestamp) - JULIANDAY(p.timestamp) * 86400 as latency_seconds
        FROM action_logs p
        JOIN action_logs c ON c.parent_log_id = p.log_id
        WHERE p.task_id = ?
          AND p.action_type = 'delegate'
          AND c.action_type = 'response'
    """
    return execute_query(query, task_id)
```

### Phase 3: Gateway Integration

Add parent-child tracking to Gateway service (`services/gateway/app.py`) so the complete flow from user HTTP request to final response is tracked.

---

## 📚 Documentation Updates

Updated the following documents:

1. **docs/COMMS_LOGGING_GUIDE.md**
   - Added "Parent-Child Tracking" section with examples
   - Updated "Developer Usage" section with log_id capture examples
   - Added "Current Integration Points" section

2. **docs/PARENT_CHILD_TRACKING_IMPLEMENTATION.md** (this document)
   - Complete implementation details
   - Test results and validation
   - Next steps for HMI integration

---

## ✅ Checklist

- [x] CommsLogger infrastructure supports parent_log_id (already existed)
- [x] PAS Root captures and passes parent_log_id
- [x] Aider-LCO accepts and uses parent_log_id
- [x] Database stores parent_log_id correctly
- [x] End-to-end test validates complete hierarchy
- [x] Documentation updated
- [ ] HMI Tree View updated to use parent_log_id (Phase 1)
- [ ] Performance metrics implemented (Phase 2)
- [ ] Gateway integration (Phase 3)

---

## 🔗 Related Files

**Code:**
- `services/common/comms_logger.py` - Logger implementation
- `services/pas/root/app.py` - PAS Root service (updated)
- `services/tools/aider_rpc/app.py` - Aider-LCO RPC (updated)
- `services/registry/registry_service.py` - Database storage

**Documentation:**
- `docs/COMMS_LOGGING_GUIDE.md` - Usage guide
- `docs/FLAT_LOG_FORMAT.md` - Log format specification
- `docs/COMMS_LOGGING_SUMMARY.md` - Quick reference

**Database:**
- `artifacts/registry/registry.db` - SQLite database (action_logs table)

---

## 📞 Contact

For questions or issues with parent-child tracking:
1. Check `docs/COMMS_LOGGING_GUIDE.md` for usage examples
2. Review test script: `tests/test_comms_logger.py`
3. Query database directly: `sqlite3 artifacts/registry/registry.db`

---

**End of Implementation Summary**
