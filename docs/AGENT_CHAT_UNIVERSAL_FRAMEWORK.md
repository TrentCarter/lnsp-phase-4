# Agent Chat Universal Framework

**Status**: ✅ Production Ready (2025-11-14)
**Coverage**: 23/23 agents (100%)
**Single Source of Truth**: `services/common/agent_chat_mixin.py`

## Overview

All 23 PAS agents now use a **centralized Agent Chat framework** that provides universal bidirectional messaging, SSE events, and thread management. This eliminates code duplication and ensures consistency across the entire hierarchy.

## Architecture

### Core Components

1. **`services/common/agent_chat.py`** - Database-backed Agent Chat client
   - Thread management (create, close, get status)
   - Message storage (SQLite: `artifacts/registry/registry.db`)
   - Event broadcasting (SSE to HMI via Event Stream)

2. **`services/common/agent_chat_mixin.py`** - Universal integration mixin
   - Route injection (`add_agent_chat_routes()`)
   - Message polling (`start_message_poller()`)
   - Helper functions (`send_message_to_parent()`, `send_message_to_child()`)

3. **`tools/add_agent_chat_to_all.py`** - Automation script
   - Automatically adds Agent Chat to any agent
   - Handles import injection, handler creation, startup events
   - Works with dynamic IDs (e.g., Programmers using `PROGRAMMER_ID` env var)

### Integration Pattern

Every agent follows this pattern:

```python
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage
from services.common.agent_chat_mixin import (
    add_agent_chat_routes,
    start_message_poller,
    send_message_to_parent,
    send_message_to_child
)

# Initialize
agent_chat = get_agent_chat_client()

# Define message handler
async def handle_incoming_message(message: AgentChatMessage):
    """Handle incoming messages from parent or children"""
    if message.message_type == "delegation":
        # Process delegation
        pass
    elif message.message_type == "question":
        # Answer question
        pass
    # ... etc

# Add routes (POST /agent-chat/send, GET /agent-chat/events, etc.)
add_agent_chat_routes(
    app=app,
    agent_id="Agent-Name",
    agent_chat=agent_chat,
    on_message_received=handle_incoming_message
)

# Start message poller on startup
@app.on_event("startup")
async def startup():
    await start_message_poller(
        agent_id="Agent-Name",
        agent_chat=agent_chat,
        poll_interval=2.0
    )
```

## API Endpoints

Every agent automatically gets these endpoints:

- **POST `/agent-chat/send`** - Send message to another agent
- **POST `/agent-chat/create-thread`** - Create new conversation thread
- **GET `/agent-chat/threads/{run_id}`** - Get all threads for a run
- **GET `/agent-chat/thread/{thread_id}`** - Get thread with message history
- **POST `/agent-chat/close-thread`** - Close a thread
- **GET `/agent-chat/events`** - SSE endpoint for real-time events

## Coverage Status

### Architect (1 agent)
- ✅ Architect (6110)

### Directors (5 agents)
- ✅ Dir-Code (6111)
- ✅ Dir-Models (6112) - **NEW**
- ✅ Dir-Data (6113)
- ✅ Dir-DevSecOps (6114)
- ✅ Dir-Docs (6115)

### Managers (7 agents)
- ✅ Mgr-Code-01 (6141)
- ✅ Mgr-Code-02 (6142)
- ✅ Mgr-Code-03 (6143)
- ✅ Mgr-Models-01 (6144) - **NEW**
- ✅ Mgr-Data-01 (6145) - **NEW**
- ✅ Mgr-DevSecOps-01 (6146) - **NEW**
- ✅ Mgr-Docs-01 (6147) - **NEW**

### Programmers (10 agents)
- ✅ Prog-001 (6151)
- ✅ Prog-002 (6152) - **NEW**
- ✅ Prog-003 (6153) - **NEW**
- ✅ Prog-004 (6154) - **NEW**
- ✅ Prog-005 (6155) - **NEW**
- ✅ Prog-006 (6156) - **NEW**
- ✅ Prog-007 (6157) - **NEW**
- ✅ Prog-008 (6158) - **NEW**
- ✅ Prog-009 (6159) - **NEW**
- ✅ Prog-010 (6160) - **NEW**

**Total**: 23/23 agents (100%)

## Message Types

The framework supports these message types:

- **delegation** - Parent assigns work to child
- **question** - Child asks parent for clarification
- **answer** - Parent answers child's question
- **status** - Progress update
- **completion** - Task completed successfully
- **error** - Task failed with error
- **escalation** - Child escalates issue to parent
- **abort** - Parent cancels child's task

## Database Schema

**Tables** (in `artifacts/registry/registry.db`):

1. **agent_conversation_threads**
   - `thread_id` (PRIMARY KEY)
   - `run_id` (indexed)
   - `parent_agent_id` (indexed)
   - `child_agent_id` (indexed)
   - `status` (active, completed, failed, timeout, abandoned)
   - `created_at`, `updated_at`, `completed_at`
   - `result`, `error`, `metadata`

2. **agent_conversation_messages**
   - `message_id` (PRIMARY KEY)
   - `thread_id` (FOREIGN KEY, indexed)
   - `from_agent`, `to_agent`
   - `message_type`
   - `content`
   - `created_at` (indexed)
   - `metadata`

## How to Add Agent Chat to a New Agent

### Option 1: Automated (Recommended)

1. Add agent to `tools/add_agent_chat_to_all.py` AGENTS_TO_UPDATE list
2. Run: `./.venv/bin/python tools/add_agent_chat_to_all.py --dry-run`
3. Review changes, then run: `./.venv/bin/python tools/add_agent_chat_to_all.py`

### Option 2: Manual

1. Import mixin functions
2. Initialize `agent_chat = get_agent_chat_client()`
3. Define `handle_incoming_message(message: AgentChatMessage)` function
4. Call `add_agent_chat_routes()` after app creation
5. Call `start_message_poller()` in `@app.on_event("startup")`

## Future Changes

**To modify Agent Chat behavior across ALL agents:**

1. Edit `services/common/agent_chat_mixin.py` (single file)
2. Changes automatically affect all 23 agents
3. No need to update individual agent files

**Examples:**
- Change polling interval: Update `start_message_poller()` default
- Add new message type: Update `handle_incoming_message()` template
- Modify SSE format: Update event generator in `add_agent_chat_routes()`
- Change routing logic: Update route handlers

## Testing

### Test Agent Chat on a Single Agent

```bash
# Start agent
./services/pas/manager_models_01/start_manager_models_01.sh

# Check health (should show agent_chat_enabled: true)
curl http://localhost:6144/health

# Test message sending
curl -X POST http://localhost:6144/agent-chat/send \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-thread",
    "to_agent": "Dir-Models",
    "message_type": "status",
    "content": "Test message"
  }'
```

### Test SSE Events

```bash
curl -N http://localhost:6144/agent-chat/events
```

## Benefits

### 1. DRY (Don't Repeat Yourself)
- **Before**: Agent Chat code duplicated across 23 agents
- **After**: Single source of truth (`agent_chat_mixin.py`)

### 2. Consistency
- All agents use identical message routing
- Guaranteed compatibility across hierarchy
- Uniform error handling and logging

### 3. Maintainability
- Future changes: Update 1 file, affects all 23 agents
- Easy to test: Test mixin once, applies everywhere
- Clear documentation: Single reference implementation

### 4. Scalability
- Adding new agents: 1 function call (`add_agent_chat_routes()`)
- New message types: Update mixin, auto-propagates
- New features (retries, auth, etc.): Centralized implementation

## Related Documentation

- **PRD**: `docs/PRDs/PRD_Parent_Child_Chat_Communications.md`
- **Agent Status**: `docs/AGENT_CHAT_COVERAGE_MATRIX.md`
- **Database**: `artifacts/registry/registry.db`
- **HMI Integration**: `services/webui/templates/model_pool_enhanced.html` (Agent Status tab)

## Implementation Timeline

- **2025-11-13**: Initial Agent Chat implementation (9/23 agents)
- **2025-11-14**: Universal framework created
- **2025-11-14**: Automated integration (6 files updated)
- **2025-11-14**: 100% coverage achieved (23/23 agents)

## Maintenance

### Adding a New Message Type

1. Edit `services/common/agent_chat_mixin.py`:
   ```python
   elif message.message_type == "new_type":
       print(f"[{{agent_id}}] New type: {{message.content}}")
   ```

2. No other changes needed - all agents get the new handler

### Changing Polling Behavior

1. Edit `services/common/agent_chat_mixin.py` - `_poll_for_messages()` function
2. All agents automatically use new behavior on next restart

### Modifying API Endpoints

1. Edit `services/common/agent_chat_mixin.py` - `add_agent_chat_routes()` function
2. All agents get updated endpoints automatically
