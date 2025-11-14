# Agent Chat Coverage Matrix

**Date:** 2025-11-13
**Status:** Phase 7 Complete - Programmer Integration (60% Coverage)
**Branch:** feature/aider-lco-p0

---

## Overview

This document tracks which agents in the PAS hierarchy have Agent Chat integration enabled. Agent Chat provides bidirectional parent-child communication with:

- **Thread-based conversations** (context preservation across messages)
- **Bidirectional Q&A** (children can ask questions, parents can answer)
- **Status updates** (real-time progress visibility)
- **LLM-powered ask_parent tool** (intelligent question generation)
- **Real-time SSE streaming** (messages appear in HMI Sequencer instantly)

---

## Coverage Status by Tier

### Architect Tier

| Agent      | Port | Chat Enabled | Can Send | Can Receive | SSE Events | Implementation Location |
|------------|------|--------------|----------|-------------|------------|-------------------------|
| Architect  | 6110 | ✅ YES        | ✅ Answers | ✅ Questions | ✅ YES      | `services/pas/architect/app.py:48-63` |

**Features:**
- Creates threads for delegation to Directors
- Answers questions from Directors using LLM
- Monitors thread events via polling (no SSE consumption yet)

---

### Director Tier

| Agent          | Port | Chat Enabled | Can Send | Can Receive | SSE Events | Implementation Location |
|----------------|------|--------------|----------|-------------|------------|-------------------------|
| Dir-Code       | 6111 | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/director_code/app.py:46-59` |
| Dir-Data       | 6113 | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/director_data/app.py:42-55` |
| Dir-Docs       | 6115 | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/director_docs/app.py:42-55` |
| Dir-DevSecOps  | 6114 | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/director_devsecops/app.py:42-55` |
| Dir-Models     | 6112 | ❌ NO         | ❌ No     | ❌ No        | ❌ NO       | *Not yet implemented* |

**All Directors (Code, Data, Docs, DevSecOps) have:**
- `/agent_chat/receive` endpoint for incoming delegation messages
- `process_agent_chat_message()` with LLM processing
- `execute_job_card_with_chat()` for status updates
- Status messages during decomposition and delegation
- Completion/error messages with thread closure
- ask_parent tool integration (via `call_llm_with_tools`)

**Message Types Supported:**
- **Send:** Questions (via ask_parent), Status Updates, Completion, Errors
- **Receive:** Delegation, Answers (from Architect)

**Dir-Models Status:**
- Not yet implemented (planned for future phases)
- Will follow same pattern as other Directors when added

---

### Manager Tier

| Agent             | Port    | Architecture | Chat Enabled | Can Send | Can Receive | SSE Events | Implementation Location |
|-------------------|---------|--------------|--------------|----------|-------------|------------|-------------------------|
| Mgr-Code-01       | 6141    | FastAPI HTTP | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/manager_code/app.py` |
| Mgr-Code-02       | 6142    | FastAPI HTTP | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/manager_code/app.py` |
| Mgr-Code-03       | 6143    | FastAPI HTTP | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/pas/manager_code/app.py` |
| Mgr-Data-01       | 6145    | Not Created  | ❌ NO         | ❌ No     | ❌ No        | ❌ NO       | *Not yet created* |
| Mgr-Docs-01       | 6147    | Not Created  | ❌ NO         | ❌ No     | ❌ No        | ❌ NO       | *Not yet created* |
| Mgr-DevSecOps-01  | 6146    | Not Created  | ❌ NO         | ❌ No     | ❌ No        | ❌ NO       | *Not yet created* |
| Mgr-Models-01     | 6144    | Not Created  | ❌ NO         | ❌ No     | ❌ No        | ❌ NO       | *Not yet created* |

**Status:** Phase 6 (Partially Complete - 2025-11-13)

**✅ Architecture Change:** Managers now use FastAPI HTTP servers (like Directors) instead of file-based coordination!

**All Manager-Code services (Mgr-Code-01/02/03) have:**
- `/agent_chat/receive` endpoint for incoming delegation messages
- `process_agent_chat_message()` for LLM processing
- `execute_task_with_chat()` for status updates during Aider RPC execution
- Status messages during code execution and acceptance testing
- Completion/error messages with thread closure
- Direct integration with Aider RPC (port 6130)

**Message Types Supported:**
- **Send:** Status Updates, Completion, Errors
- **Receive:** Delegation (from Dir-Code)

**Remaining Manager Services:**
- Mgr-Data-01, Mgr-Docs-01, Mgr-DevSecOps-01, Mgr-Models-01 (not yet created)
- Will follow same FastAPI HTTP pattern when implemented

---

### Programmer Tier

| Agent         | Port | Chat Enabled | Can Send | Can Receive | SSE Events | Implementation Location |
|---------------|------|--------------|----------|-------------|------------|-------------------------|
| Prog-Qwen-001 | 6130 | ✅ YES        | ✅ Yes    | ✅ Yes       | ✅ YES      | `services/tools/aider_rpc/app.py` |

**Status:** Phase 7 (Complete - 2025-11-13)

**Prog-Qwen-001 (Aider-LCO) has:**
- `/agent_chat/receive` endpoint for incoming delegation messages
- `process_agent_chat_message()` for background task processing
- `execute_aider_with_chat()` for status updates during Aider CLI execution
- Heartbeat monitoring during execution
- Status messages at key stages (received, validating, executing, completed)
- Error handling with agent chat notifications
- Completion messages with output preview
- Automatic thread closure on success/failure

**Message Types Supported:**
- **Send:** Status Updates, Completion, Errors
- **Receive:** Delegation (from Manager-Code)

---

## Integration Summary

### ✅ Fully Integrated (9/15 agents = 60.0%)

1. **Architect** (services/pas/architect/app.py)
   - Creates threads for delegation
   - Answers questions from Directors using LLM
   - Monitors thread events

2. **Dir-Code** (services/pas/director_code/app.py)
   - Receives delegation messages
   - Asks questions using ask_parent tool
   - Sends status updates during execution
   - Sends completion/error messages

3. **Dir-Data** (services/pas/director_data/app.py)
   - Same pattern as Dir-Code
   - Data lane specific status messages

4. **Dir-Docs** (services/pas/director_docs/app.py)
   - Same pattern as Dir-Code
   - Docs lane specific status messages

5. **Dir-DevSecOps** (services/pas/director_devsecops/app.py)
   - Same pattern as Dir-Code
   - DevSecOps lane specific status messages

6. **Mgr-Code-01** (services/pas/manager_code/app.py)
   - Receives delegation messages from Dir-Code
   - Executes code changes via Aider RPC
   - Sends status updates during execution
   - Runs acceptance tests (lint, tests, coverage)
   - Sends completion/error messages

7. **Mgr-Code-02** (services/pas/manager_code/app.py)
   - Same pattern as Mgr-Code-01
   - Independent FastAPI HTTP server

8. **Mgr-Code-03** (services/pas/manager_code/app.py)
   - Same pattern as Mgr-Code-01
   - Independent FastAPI HTTP server

9. **Prog-Qwen-001** (services/tools/aider_rpc/app.py) **← NEW!**
   - Receives delegation messages from Managers
   - Executes Aider CLI with guardrails
   - Sends status updates during execution
   - Heartbeat monitoring
   - Sends completion/error messages
   - Automatic thread closure

### ❌ Not Integrated (6/15 agents = 40.0%)

- **1 Director:** Dir-Models (not yet implemented for any functionality)
- **4 Managers:** Mgr-Data-01, Mgr-Docs-01, Mgr-DevSecOps-01, Mgr-Models-01 (not yet created)
- **2 Programmers:** Prog-Qwen-002, Prog-Qwen-003 (not yet created - will use same Aider-LCO codebase)

---

## Phase Progress

### ✅ Phase 3: Backend Integration (Complete)
- Agent Chat service with thread/message persistence
- Event broadcasting to Event Stream (port 6102)
- LLM-powered ask_parent tool
- Integration in Architect and all active Directors

### ✅ Phase 4: Real-Time SSE Streaming (Complete)
- SSE endpoint in HMI (`/api/stream/agent_chat/<run_id>`)
- Frontend EventSource consumer
- Message rendering in Sequencer timeline
- <100ms latency (100x faster than polling)

### ✅ Phase 5: Director Coverage (Complete - 2025-11-13)
- ✅ Dir-Code (already had it)
- ✅ Dir-Data (fixed run_id bug)
- ✅ Dir-Docs (fixed run_id bug)
- ✅ Dir-DevSecOps (fixed run_id bug)

**All active Directors now have full agent chat integration!**

### ✅ Phase 6: Manager Integration (Partially Complete - 2025-11-13)
- ✅ **Mgr-Code-01, Mgr-Code-02, Mgr-Code-03** (NEW FastAPI HTTP architecture!)
- [ ] Mgr-Data-01 (not yet created)
- [ ] Mgr-Docs-01 (not yet created)
- [ ] Mgr-DevSecOps-01 (not yet created)
- [ ] Mgr-Models-01 (not yet created)

**Architecture Change:** Managers migrated from file-based coordination to FastAPI HTTP servers!

**All Manager-Code services operational:**
- Port 6141: Mgr-Code-01
- Port 6142: Mgr-Code-02
- Port 6143: Mgr-Code-03

**Benefits:**
- Consistent architecture across all PAS tiers
- Direct HTTP communication (no file queues)
- Standard `/agent_chat/receive` endpoints
- Standard `/health` endpoints for monitoring
- Real-time status updates via SSE
- Better scalability and observability

### ✅ Phase 7: Programmer Integration (Complete - 2025-11-13)
- ✅ **Prog-Qwen-001 (Aider-LCO)** - Full agent chat integration!

**Implementation:** `services/tools/aider_rpc/app.py`

**Features:**
- `/agent_chat/receive` endpoint for delegation messages
- Background task processing with `process_agent_chat_message()`
- `execute_aider_with_chat()` with status updates at all stages
- Heartbeat monitoring during Aider CLI execution
- Filesystem allowlist validation with error reporting
- Status messages: received → validating → executing → completed/failed
- Automatic thread closure on completion/error
- Output preview in completion messages (last 1000 chars)
- Error messages with stderr preview (last 500 chars)

**Benefits:**
- Real-time visibility into Aider CLI execution
- Managers can track Programmer progress via agent chat
- Consistent pattern across all PAS tiers (Architect → Director → Manager → Programmer)
- Automatic error handling with parent notification
- Complete execution trace via agent chat messages

### 🔲 Phase 8: Advanced Features (Future Work)
- [ ] Thread detail panel (sidebar in Sequencer)
- [ ] TRON Tree View animations for message flow
- [ ] User intervention (approve/reject/modify answers)
- [ ] Sound effects for different message types
- [ ] Batch SSE events for efficiency
- [ ] Message compression for large content

---

## Implementation Pattern (for Future Extensions)

When adding agent chat to Managers or Programmers, follow this pattern:

### 1. Add Imports (top of app.py)

```python
# Agent chat for Parent-Child communication
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage

# LLM with tool support
from services.common.llm_tool_caller import call_llm_with_tools, LLMResponse
from services.common.llm_tools import get_ask_parent_tool, validate_ask_parent_args
```

### 2. Initialize Agent Chat Client

```python
agent_chat = get_agent_chat_client()
```

### 3. Add `/agent_chat/receive` Endpoint

```python
@app.post("/agent_chat/receive")
async def receive_agent_message(
    request: AgentChatMessage,
    background_tasks: BackgroundTasks
):
    """Receive message from parent via Agent Chat thread."""
    thread_id = request.thread_id
    run_id = request.run_id or "unknown"

    logger.log_cmd(
        from_agent=request.from_agent,
        to_agent="YourAgentName",
        message=f"Agent chat message received: {request.message_type}",
        run_id=run_id,
        metadata={"thread_id": thread_id, "message_type": request.message_type}
    )

    background_tasks.add_task(process_agent_chat_message, request)

    return {"status": "ok", "thread_id": thread_id}
```

### 4. Add Message Processing Handler

```python
async def process_agent_chat_message(request: AgentChatMessage):
    """Process agent chat message with LLM (background task)."""
    thread_id = request.thread_id
    run_id = request.run_id or "unknown"

    try:
        # Load thread history
        thread = await agent_chat.get_thread(thread_id)

        # Send initial status
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="YourAgentName",
            to_agent=request.from_agent,
            message_type="status",
            content="Processing task..."
        )

        # Execute task with status updates
        await execute_task_with_chat(task_data, thread_id)

    except Exception as e:
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="YourAgentName",
            to_agent=request.from_agent,
            message_type="error",
            content=f"Error: {str(e)}"
        )

        await agent_chat.close_thread(
            thread_id=thread_id,
            status="failed",
            error=str(e)
        )
```

### 5. Send Status Updates During Execution

```python
# During task execution
await agent_chat.send_message(
    thread_id=thread_id,
    from_agent="YourAgentName",
    to_agent="ParentAgentName",
    message_type="status",
    content="Step 1 of 3 complete",
    metadata={"progress": 33}
)
```

### 6. Ask Questions Using ask_parent Tool

```python
# Let LLM decide if it needs clarification
llm_response = await call_llm_with_tools(
    model="google/gemini-2.5-flash",
    system_prompt=get_system_prompt_with_ask_parent("YourAgentName"),
    messages=[{"role": "user", "content": task_description}],
    tools=[get_ask_parent_tool()],
    thread_id=thread_id,
    from_agent="YourAgentName",
    to_agent="ParentAgentName"
)

# If LLM used ask_parent tool, question was sent and answer received
```

### 7. Send Completion/Error and Close Thread

```python
# On success
await agent_chat.send_message(
    thread_id=thread_id,
    from_agent="YourAgentName",
    to_agent="ParentAgentName",
    message_type="completion",
    content="Task completed successfully"
)

await agent_chat.close_thread(
    thread_id=thread_id,
    status="completed",
    result="Task output here"
)

# On failure
await agent_chat.send_message(
    thread_id=thread_id,
    from_agent="YourAgentName",
    to_agent="ParentAgentName",
    message_type="error",
    content=f"Task failed: {error_message}"
)

await agent_chat.close_thread(
    thread_id=thread_id,
    status="failed",
    error=error_message
)
```

---

## Testing

### Verify Director Integration

```bash
# Check if service is running
lsof -ti:6113  # Dir-Data
lsof -ti:6114  # Dir-DevSecOps
lsof -ti:6115  # Dir-Docs

# Test agent chat endpoint
curl -X POST http://localhost:6113/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-thread-001",
    "run_id": "test-run-001",
    "from_agent": "Architect",
    "to_agent": "Dir-Data",
    "message_type": "delegation",
    "content": "Test delegation message",
    "metadata": {}
  }'

# Check SSE stream
curl -N http://localhost:6101/api/stream/agent_chat/test-run-001
```

### Create Test Data

```bash
# Create sample agent chat messages
./.venv/bin/python tools/test_agent_chat_visualization.py

# View in HMI Sequencer
open http://localhost:6101/sequencer
# Select "Test Agent Chat Visualization" from dropdown
```

### Verify SSE Events

```bash
# Monitor Event Stream
curl -N http://localhost:6102/subscribe

# Should see events like:
# event: agent_chat_message_sent
# data: {"run_id": "...", "thread_id": "...", "message_type": "delegation", ...}
```

---

## Performance Metrics

- **Latency:** <100ms from message send to SSE delivery
- **Throughput:** Event Stream handles 100+ concurrent SSE connections
- **Reliability:** Non-blocking event broadcasting (failures don't affect chat ops)
- **Coverage:** 53.3% of agents (8/15) fully integrated
- **Architecture:** 100% of active agents now use FastAPI HTTP (Architect, Directors, Managers)

---

## Next Steps

1. **Phase 6: Manager Integration**
   - Start with Mgr-Code-01 (most active Manager)
   - Replicate Dir-Code pattern
   - Test with real task delegation from Dir-Code

2. **Phase 7: Programmer Integration**
   - Integrate Aider-LCO with agent chat
   - Enable questions during code editing
   - Status updates for git operations

3. **Phase 8: Advanced Features**
   - Thread detail panel (sidebar)
   - TRON Tree animations
   - User intervention capabilities

---

## References

- **Agent Chat Implementation:** `services/common/agent_chat.py`
- **Event Stream Service:** `services/event_stream/event_stream_app.py` (port 6102)
- **HMI SSE Endpoint:** `services/webui/hmi_app.py:2764-2921`
- **Frontend SSE Consumer:** `services/webui/templates/sequencer.html:2990-3141`
- **Phase 4 Documentation:** `docs/AGENT_CHAT_PHASE4_SSE_IMPLEMENTATION.md`
- **Visualization Documentation:** `docs/AGENT_CHAT_VISUALIZATION_IMPLEMENTATION.md`

---

**End of Coverage Matrix**
