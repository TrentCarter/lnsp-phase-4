# PRD: Parent-Child Chat Communications (Agent-to-Agent)

**Status:** 🔴 Not Started
**Priority:** P0 (Critical for P0 Stack Communication)
**Created:** 2025-11-13
**Owner:** TBD
**Related:** `PRD_Chat_Aider_LLM.md` (Human ↔ Agent chat)

---

## Problem Statement

### Current State: Stateless Job Cards

The P0 stack currently uses **one-shot job cards** for Parent → Child delegation:

```
PAS Root → Architect → Job Card → Dir-Code → Job Card → Mgr-Code-01 → Aider RPC
```

**Problems:**
1. ❌ **No context preservation** - Each job card is stateless, no history
2. ❌ **No bidirectional Q&A** - Child cannot ask Parent clarifying questions
3. ❌ **No status updates** - Child cannot stream progress back to Parent
4. ❌ **No error escalation** - Child cannot request guidance when stuck
5. ❌ **Fire-and-forget** - Parent delegates and waits, no interactive communication

### Real-World Scenario (Currently Impossible)

**User submits:** "Refactor authentication to use OAuth2"

```
Architect → Dir-Code (Job Card):
    "Refactor authentication to use OAuth2"

Dir-Code needs to ask:
    ❌ "Should I use library 'authlib' or 'python-oauth2'?"
    ❌ "Found 3 auth files - which to refactor first?"
    ❌ "Tests are failing - should I fix them now?"

Currently: Dir-Code must GUESS or FAIL
```

**What's needed:** Parent ↔ Child **conversational thread** that:
- ✅ Preserves full message history until task complete
- ✅ Allows Child to ask questions → Parent responds
- ✅ Streams status updates (planning, executing, 50% done, etc.)
- ✅ Handles completion/error/timeout/abandonment gracefully

---

## Solution: Agent Conversation Threads

Add **stateful conversation threads** between Parent and Child agents with:

1. **Thread Lifecycle:** Created on delegation, closed on completion/error/timeout
2. **Context Preservation:** Full message history stored in DB
3. **Bidirectional Messaging:** Both Parent and Child can send messages
4. **Message Types:** Delegation, question, answer, status, completion, error
5. **Stream Support:** Real-time status updates (SSE)

---

## Architecture

### New Flow (Agent ↔ Agent Chat)

```
┌──────────────────────────────────────────────────────────────────┐
│ Parent Agent (Architect)                                         │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ 1. Create thread_id                                       │   │
│ │ 2. POST /agent_chat (delegation message)                  │   │
│ │ 3. Wait for response or poll /thread/{id}/status          │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼ POST /agent_chat
         ┌───────────────────────────────────────────┐
         │  Child Agent (Dir-Code)                   │
         │  ┌────────────────────────────────────┐   │
         │  │ 1. Load thread history             │   │
         │  │ 2. Process message with LLM        │   │
         │  │ 3. Need clarification?             │   │
         │  │    → POST back to Parent           │   │
         │  │ 4. Stream status updates           │   │
         │  │ 5. Complete → Send final result    │   │
         │  └────────────────────────────────────┘   │
         └───────────────────────────────────────────┘
```

### Message Flow Example

```
[Thread abc-123 created]

Parent → Child: "Refactor auth to OAuth2"
Child → Parent: "I found 3 auth implementations (basic, JWT, session).
                 Should I refactor all or focus on one?"
Parent → Child: "Focus on JWT first, keep others unchanged."
Child → Parent: [STATUS] "Planning refactor... (10%)"
Child → Parent: [STATUS] "Updating auth.py... (50%)"
Child → Parent: [STATUS] "Running tests... (80%)"
Child → Parent: [COMPLETE] "Done! Refactored JWT auth, all tests pass."

[Thread abc-123 closed]
```

---

## Technical Design

### 1. Database Schema

**New Table: `agent_conversation_threads`**

```sql
CREATE TABLE agent_conversation_threads (
    thread_id TEXT PRIMARY KEY,              -- UUID
    run_id TEXT NOT NULL,                    -- Links to Prime Directive run
    parent_agent_id TEXT NOT NULL,           -- e.g., "Architect"
    child_agent_id TEXT NOT NULL,            -- e.g., "Dir-Code"
    created_at TEXT NOT NULL,                -- ISO 8601 timestamp
    updated_at TEXT NOT NULL,                -- Last message timestamp
    completed_at TEXT,                       -- When thread closed
    status TEXT NOT NULL DEFAULT 'active',   -- active, completed, failed, abandoned, timeout
    result TEXT,                             -- Final result (if completed)
    error TEXT,                              -- Error message (if failed)
    metadata TEXT DEFAULT '{}',              -- JSON metadata

    INDEX idx_run_id (run_id),
    INDEX idx_status (status),
    INDEX idx_parent (parent_agent_id),
    INDEX idx_child (child_agent_id)
);
```

**New Table: `agent_conversation_messages`**

```sql
CREATE TABLE agent_conversation_messages (
    message_id TEXT PRIMARY KEY,             -- UUID
    thread_id TEXT NOT NULL,                 -- FK to threads
    from_agent TEXT NOT NULL,                -- Sender agent ID
    to_agent TEXT NOT NULL,                  -- Recipient agent ID
    message_type TEXT NOT NULL,              -- delegation, question, answer, status, completion, error
    content TEXT NOT NULL,                   -- Message content
    created_at TEXT NOT NULL,                -- ISO 8601 timestamp
    metadata TEXT DEFAULT '{}',              -- JSON metadata (tool calls, attachments, etc.)

    FOREIGN KEY (thread_id) REFERENCES agent_conversation_threads(thread_id) ON DELETE CASCADE,
    INDEX idx_thread (thread_id),
    INDEX idx_created (created_at)
);
```

**Location:** `artifacts/registry/registry.db` (existing database)

### 2. Message Types

| Type | Sender | Purpose | Example |
|------|--------|---------|---------|
| `delegation` | Parent | Initial task assignment | "Refactor auth to OAuth2" |
| `question` | Child | Needs clarification | "Which library should I use?" |
| `answer` | Parent | Response to question | "Use authlib, it's more mature" |
| `status` | Child | Progress update | "Planning refactor... (10%)" |
| `completion` | Child | Task finished successfully | "Done! All tests pass." |
| `error` | Child | Task failed, needs help | "Tests failing, need guidance" |
| `escalation` | Child | Cannot proceed, escalate | "Blocked by missing dependency" |
| `abort` | Parent | Cancel task | "Stop work, requirements changed" |

### 3. API Endpoints

#### Create Thread (Parent initiates)

```python
POST /agent_chat/create
{
    "run_id": "run-456",
    "child_agent_id": "Dir-Code",
    "initial_message": "Refactor authentication to OAuth2",
    "metadata": {
        "entry_files": ["src/auth.py"],
        "budget_tokens": 10000
    }
}

Response:
{
    "thread_id": "thread-abc-123",
    "status": "active",
    "created_at": "2025-11-13T10:30:00Z"
}
```

#### Send Message (Bidirectional)

```python
POST /agent_chat/send
{
    "thread_id": "thread-abc-123",
    "from_agent": "Dir-Code",
    "to_agent": "Architect",
    "message_type": "question",
    "content": "Should I use library 'authlib' or 'python-oauth2'?"
}

Response:
{
    "message_id": "msg-789",
    "thread_id": "thread-abc-123",
    "created_at": "2025-11-13T10:31:00Z"
}
```

#### Get Thread History

```python
GET /agent_chat/thread/{thread_id}

Response:
{
    "thread_id": "thread-abc-123",
    "run_id": "run-456",
    "parent_agent_id": "Architect",
    "child_agent_id": "Dir-Code",
    "status": "active",
    "messages": [
        {
            "message_id": "msg-001",
            "from_agent": "Architect",
            "to_agent": "Dir-Code",
            "message_type": "delegation",
            "content": "Refactor authentication to OAuth2",
            "created_at": "2025-11-13T10:30:00Z"
        },
        {
            "message_id": "msg-002",
            "from_agent": "Dir-Code",
            "to_agent": "Architect",
            "message_type": "question",
            "content": "Should I use library 'authlib' or 'python-oauth2'?",
            "created_at": "2025-11-13T10:31:00Z"
        }
    ]
}
```

#### Close Thread

```python
POST /agent_chat/close
{
    "thread_id": "thread-abc-123",
    "status": "completed",  // or "failed", "timeout", "abandoned"
    "result": "Successfully refactored JWT auth to OAuth2. All tests pass.",
    "error": null
}

Response:
{
    "thread_id": "thread-abc-123",
    "status": "completed",
    "completed_at": "2025-11-13T10:45:00Z",
    "duration_s": 900
}
```

#### Stream Status Updates (SSE)

```python
GET /agent_chat/stream/{thread_id}

Response: (Server-Sent Events)
event: status
data: {"message": "Planning refactor...", "progress": 10}

event: status
data: {"message": "Updating auth.py...", "progress": 50}

event: status
data: {"message": "Running tests...", "progress": 80}

event: complete
data: {"result": "Done! All tests pass."}
```

### 4. Implementation in Agents

#### Parent (Architect) - Create Thread and Delegate

```python
# services/pas/architect/app.py

from services.common.agent_chat import AgentChatClient

agent_chat = AgentChatClient()

async def delegate_to_director(lane: str, task: str, run_id: str):
    """Delegate task to Director with conversation thread"""

    # Create thread
    thread = await agent_chat.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id=f"Dir-{lane}",
        initial_message=task
    )

    # Monitor thread for questions/status
    async for event in agent_chat.stream_thread(thread.thread_id):
        if event.type == "question":
            # Child asking question - use LLM to generate answer
            answer = await generate_answer_with_llm(event.content, thread.history)
            await agent_chat.send_message(
                thread_id=thread.thread_id,
                from_agent="Architect",
                to_agent=f"Dir-{lane}",
                message_type="answer",
                content=answer
            )
        elif event.type == "status":
            # Log progress
            logger.log_status(f"Dir-{lane}", "Architect", event.content)
        elif event.type == "complete":
            # Task done
            return event.result
        elif event.type == "error":
            # Task failed - escalate or retry
            return handle_child_error(event.error)
```

#### Child (Dir-Code) - Process with Q&A

```python
# services/pas/director_code/app.py

from services.common.agent_chat import AgentChatClient

agent_chat = AgentChatClient()

@app.post("/agent_chat/receive")
async def receive_agent_message(request: AgentChatMessage):
    """Receive message from parent on existing thread"""

    # Load thread history
    thread = await agent_chat.get_thread(request.thread_id)
    messages = thread.messages

    # Build LLM context with full history
    llm_context = build_llm_context(messages)

    # Add system prompt with "you can ask questions" capability
    system_prompt = """
    You are Dir-Code, responsible for code lane tasks.

    IMPORTANT: If you need clarification:
    - Use the 'ask_parent' tool to ask Architect questions
    - Wait for response before proceeding
    - Don't guess - always clarify ambiguities

    Tools:
    - ask_parent(question: str) -> str
    - send_status(message: str, progress: int)
    - call_aider(message: str, files: list[str]) -> str
    """

    # Process with LLM
    response = await llm_with_tools(
        messages=llm_context,
        system=system_prompt,
        tools=[ask_parent_tool, send_status_tool, call_aider_tool]
    )

    # Handle tool calls
    if response.tool_call == "ask_parent":
        await agent_chat.send_message(
            thread_id=thread.thread_id,
            from_agent="Dir-Code",
            to_agent="Architect",
            message_type="question",
            content=response.tool_args["question"]
        )
        # Wait for parent response (polling or webhook)
        answer = await agent_chat.wait_for_response(thread.thread_id)
        # Continue processing with answer...
```

### 5. Common Library: `services/common/agent_chat.py`

```python
"""
Agent Chat Client - Shared library for Parent ↔ Child communication
"""
import sqlite3
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime
from pydantic import BaseModel

DB_PATH = Path("artifacts/registry/registry.db")

class AgentChatThread(BaseModel):
    thread_id: str
    run_id: str
    parent_agent_id: str
    child_agent_id: str
    status: str
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    messages: List[Dict[str, Any]] = []

class AgentChatClient:
    """Client for agent-to-agent conversation threads"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_conversation_threads (
                thread_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                parent_agent_id TEXT NOT NULL,
                child_agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                result TEXT,
                error TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_conversation_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (thread_id) REFERENCES agent_conversation_threads(thread_id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_run ON agent_conversation_threads(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_thread ON agent_conversation_messages(thread_id)")

        conn.commit()
        conn.close()

    async def create_thread(
        self,
        run_id: str,
        parent_agent_id: str,
        child_agent_id: str,
        initial_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentChatThread:
        """Create new conversation thread with initial delegation message"""
        thread_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + 'Z'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create thread
        cursor.execute("""
            INSERT INTO agent_conversation_threads
            (thread_id, run_id, parent_agent_id, child_agent_id, created_at, updated_at, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
        """, (thread_id, run_id, parent_agent_id, child_agent_id, now, now, json.dumps(metadata or {})))

        # Add initial delegation message
        message_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO agent_conversation_messages
            (message_id, thread_id, from_agent, to_agent, message_type, content, created_at)
            VALUES (?, ?, ?, ?, 'delegation', ?, ?)
        """, (message_id, thread_id, parent_agent_id, child_agent_id, initial_message, now))

        conn.commit()
        conn.close()

        return await self.get_thread(thread_id)

    async def send_message(
        self,
        thread_id: str,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send message on existing thread"""
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + 'Z'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO agent_conversation_messages
            (message_id, thread_id, from_agent, to_agent, message_type, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, thread_id, from_agent, to_agent, message_type, content, now, json.dumps(metadata or {})))

        # Update thread timestamp
        cursor.execute("""
            UPDATE agent_conversation_threads SET updated_at = ? WHERE thread_id = ?
        """, (now, thread_id))

        conn.commit()
        conn.close()

        return message_id

    async def get_thread(self, thread_id: str) -> AgentChatThread:
        """Get thread with full message history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get thread
        cursor.execute("SELECT * FROM agent_conversation_threads WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Thread {thread_id} not found")

        # Get messages
        cursor.execute("""
            SELECT message_id, from_agent, to_agent, message_type, content, created_at, metadata
            FROM agent_conversation_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
        """, (thread_id,))

        messages = [
            {
                "message_id": r[0],
                "from_agent": r[1],
                "to_agent": r[2],
                "message_type": r[3],
                "content": r[4],
                "created_at": r[5],
                "metadata": json.loads(r[6]) if r[6] else {}
            }
            for r in cursor.fetchall()
        ]

        conn.close()

        return AgentChatThread(
            thread_id=row[0],
            run_id=row[1],
            parent_agent_id=row[2],
            child_agent_id=row[3],
            created_at=row[4],
            updated_at=row[5],
            completed_at=row[6],
            status=row[7],
            messages=messages
        )

    async def close_thread(
        self,
        thread_id: str,
        status: str,  # "completed", "failed", "timeout", "abandoned"
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Close conversation thread"""
        now = datetime.utcnow().isoformat() + 'Z'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE agent_conversation_threads
            SET status = ?, completed_at = ?, result = ?, error = ?
            WHERE thread_id = ?
        """, (status, now, result, error, thread_id))

        conn.commit()
        conn.close()
```

---

## Integration with Existing Systems

### 1. Job Cards vs. Agent Chat

**Job Cards** (Existing):
- ✅ Keep for simple, non-interactive tasks
- ✅ Used when no clarification needed
- ✅ Fire-and-forget pattern

**Agent Chat** (New):
- ✅ Use for complex, ambiguous tasks
- ✅ Use when clarification likely needed
- ✅ Interactive, bidirectional pattern

**Decision Logic in Architect:**
```python
if task_requires_clarification(task):
    # Use Agent Chat (new)
    thread = await agent_chat.create_thread(...)
else:
    # Use Job Card (existing)
    job_card = create_job_card(...)
```

### 2. Communication Logging

**Update `services/common/comms_logger.py`:**
- Log all agent chat messages to flat logs (audit trail)
- Tag with `thread_id` for correlation
- Use existing `MessageType` enum (add `AGENT_CHAT` type)

### 3. TRON Visualization

**Update TRON to show:**
- Active conversation threads
- Message count per thread
- Thread status (active, waiting_for_response, completed)
- Click thread → View full conversation history

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
**Goal:** Database + API + Common Library

**Tasks:**
1. ✅ Create DB schema (`agent_conversation_threads`, `agent_conversation_messages`)
2. ✅ Implement `services/common/agent_chat.py` (CRUD operations)
3. ✅ Add unit tests for DB operations
4. ✅ Add API endpoints to Gateway (or new service?)

**Deliverable:** Can create threads, send messages, retrieve history

### Phase 2: Parent Integration (Week 2)
**Goal:** Architect can create threads and handle responses

**Tasks:**
1. ✅ Update Architect to create threads on delegation
2. ✅ Implement LLM-powered response generation for questions
3. ✅ Add thread monitoring/polling logic
4. ✅ Update comms logger integration

**Deliverable:** Architect can delegate via threads and answer questions

### Phase 3: Child Integration (Week 3)
**Goal:** Directors can ask questions and send status

**Tasks:**
1. ✅ Update Directors to process thread messages
2. ✅ Add "ask_parent" tool to LLM system prompts
3. ✅ Implement status update streaming
4. ✅ Add completion/error reporting

**Deliverable:** Directors can ask questions, stream status, complete tasks

### Phase 4: TRON Visualization (Week 4)
**Goal:** Users can see conversation threads in TRON

**Tasks:**
1. ✅ Add thread list view to TRON
2. ✅ Add thread detail view (message history)
3. ✅ Add real-time updates (SSE)
4. ✅ Add manual intervention (user can answer questions)

**Deliverable:** Full visibility into agent conversations

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/test_agent_chat_db.py -v     # DB operations
pytest tests/test_agent_chat_client.py -v # Client library
```

### Integration Tests
```bash
pytest tests/test_agent_chat_flow.py -v   # Full Parent → Child → Parent flow
```

### Manual Testing
1. Submit Prime Directive with ambiguous task
2. Verify Architect creates thread
3. Verify Dir-Code asks question
4. Verify Architect responds
5. Verify Dir-Code completes task
6. Check TRON shows full conversation

---

## Success Metrics

**Phase 1:**
- [ ] Can create thread and send 10 messages without errors
- [ ] Full history retrievable with correct ordering
- [ ] Thread status transitions work (active → completed)

**Phase 2:**
- [ ] Architect successfully delegates via thread (not job card)
- [ ] Architect responds to child questions within 5s
- [ ] Comms logs show thread_id correlation

**Phase 3:**
- [ ] Director asks at least 1 clarifying question per complex task
- [ ] Director streams status updates every 30s
- [ ] Director completion rate increases by 20% (fewer failures due to guessing)

**Phase 4:**
- [ ] TRON shows active threads in real-time
- [ ] Users can view full conversation history
- [ ] Users can manually intervene (answer questions)

---

## Risks & Mitigations

### Risk 1: LLM Quality (Generating Good Questions/Answers)
**Problem:** LLM may ask irrelevant questions or give poor answers
**Mitigation:**
- Use high-quality models (Claude Sonnet 4.5 for Architect)
- Provide clear system prompts with examples
- Add validation (reject trivial questions)

### Risk 2: Infinite Question Loops
**Problem:** Child keeps asking questions, never completes task
**Mitigation:**
- Limit questions per thread (max 5)
- Add timeout (30 min max)
- System prompt: "Ask minimal necessary questions"

### Risk 3: Performance (DB Queries)
**Problem:** Loading full history becomes slow (100+ messages)
**Mitigation:**
- Add pagination for message retrieval
- Cache recent threads in memory
- Archive old threads

### Risk 4: Deadlocks (Both Waiting for Response)
**Problem:** Parent waiting for child, child waiting for parent
**Mitigation:**
- Clear protocol: Child asks, Parent must respond within timeout
- Add watchdog: If no response in 2min, escalate to human
- Fallback: If stuck, child proceeds with best guess

---

## Future Enhancements

### Multi-Party Threads
- Architect ↔ Multiple Directors (group chat)
- Director ↔ Multiple Managers (parallel delegation)

### Rich Media
- Attach code diffs, logs, screenshots to messages
- Inline visualizations (graphs, charts)

### Human-in-the-Loop
- User can subscribe to thread notifications
- User can inject messages ("Use library X, not Y")
- Approval gates for critical decisions

### Analytics
- Track question types (clarification, error, escalation)
- Measure response times
- Identify frequently asked questions

---

## References

- **Related PRD:** `PRD_Chat_Aider_LLM.md` (Human ↔ Agent chat)
- **Existing Schema:** `services/webui/llm_chat_db.py` (inspiration for message storage)
- **Comms Logger:** `services/common/comms_logger.py` (audit trail integration)
- **P0 Integration:** `docs/P0_END_TO_END_INTEGRATION.md`
- **Job Cards:** `services/common/job_queue.py`

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-13 | Separate `AgentConversationThread` table vs. reusing `ConversationSession` | Clean separation, different access patterns |
| 2025-11-13 | SQLite storage in `registry.db` | Centralized DB, already used for registry |
| 2025-11-13 | Limit 5 questions per thread | Prevent infinite loops, force agents to be efficient |
| 2025-11-13 | Keep Job Cards for simple tasks | Don't overcomplicate, use chat only when needed |

---

## Appendix: Example Full Conversation

```
Thread: thread-abc-123
Run: run-456
Parent: Architect
Child: Dir-Code

[2025-11-13 10:30:00] Architect → Dir-Code (delegation)
"Refactor authentication to use OAuth2. Entry file: src/auth.py"

[2025-11-13 10:30:15] Dir-Code → Architect (question)
"I found 3 auth implementations (BasicAuth, JWTAuth, SessionAuth).
Should I refactor all to OAuth2 or focus on one?"

[2025-11-13 10:30:45] Architect → Dir-Code (answer)
"Focus on JWTAuth first. Keep BasicAuth and SessionAuth unchanged for backward compatibility."

[2025-11-13 10:31:00] Dir-Code → Architect (status)
"Planning refactor for JWTAuth... (10%)"

[2025-11-13 10:32:00] Dir-Code → Architect (question)
"Should I use library 'authlib' (MIT license, 5k stars) or 'python-oauth2' (Apache, 2k stars)?"

[2025-11-13 10:32:30] Architect → Dir-Code (answer)
"Use 'authlib' - more active maintenance and better docs."

[2025-11-13 10:33:00] Dir-Code → Architect (status)
"Installing authlib... (20%)"

[2025-11-13 10:35:00] Dir-Code → Architect (status)
"Refactoring JWTAuth class... (50%)"

[2025-11-13 10:40:00] Dir-Code → Architect (status)
"Running tests... (80%)"

[2025-11-13 10:42:00] Dir-Code → Architect (error)
"Test 'test_jwt_refresh' failing. Stack trace:
AssertionError: Token expired
Should I extend token lifetime or fix test expectations?"

[2025-11-13 10:42:30] Architect → Dir-Code (answer)
"Fix test expectations - use proper token refresh flow. Don't extend lifetime."

[2025-11-13 10:44:00] Dir-Code → Architect (status)
"Fixed test. All tests passing... (95%)"

[2025-11-13 10:45:00] Dir-Code → Architect (completion)
"✅ COMPLETE: Refactored JWTAuth to OAuth2 using authlib.
- Modified: src/auth.py
- Added: requirements.txt (authlib==1.2.0)
- Tests: 15/15 passing
- Git commit: a1b2c3d"

[Thread closed: completed]
```
