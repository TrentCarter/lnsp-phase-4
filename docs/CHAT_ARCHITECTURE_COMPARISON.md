# Chat Architecture Comparison: Two Distinct Systems

**Created:** 2025-11-13

This document clarifies the two separate chat systems in the LNSP architecture.

---

## Overview: Two Chat Types

| Aspect | Human ↔ Agent Chat | Agent ↔ Agent Chat |
|--------|-------------------|-------------------|
| **Purpose** | User interacts with PAS agents | Parent delegates to Child with Q&A |
| **Initiated By** | User (via LLM Chat UI) | Parent agent (Architect, Directors) |
| **Endpoint** | `/chat` | `/agent_chat` |
| **Context Storage** | `ConversationSession` (existing) | `AgentConversationThread` (new) |
| **Lifecycle** | User-controlled (create/archive/delete) | Task-bound (create on delegation, close on complete) |
| **PRD** | `PRD_PAS_Agent_Chat_Interface.md` | `PRD_Parent_Child_Chat_Communications.md` |

---

## Architecture Diagrams

### 1. Human ↔ Agent Chat (LLM Chat UI)

```
┌─────────────────────────────────────────────────────────────────┐
│ User (Web Browser)                                              │
│ http://localhost:6101/llm                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ POST /chat/stream
         ┌───────────────────────────────────────┐
         │  Gateway (Port 6120)                  │
         │  - Route to correct agent/model       │
         │  - Cost tracking                      │
         │  - Session management                 │
         └────────────┬──────────────────────────┘
                      │
      ┌───────────────┴──────────────────┐
      │                                   │
      ▼ agent_id="direct"                 ▼ agent_id="architect"
┌─────────────────┐               ┌──────────────────────┐
│ Direct LLM      │               │ PAS Agent (6110)     │
│ - Ollama        │               │ - Architect          │
│ - Anthropic     │               │ - /chat endpoint     │
│ - Google        │               │ - System prompt      │
└─────────────────┘               └──────────┬───────────┘
                                             │
                                             ▼ Tools
                                  ┌──────────────────────┐
                                  │ Aider RPC (6130)     │
                                  │ - read_file()        │
                                  │ - list_directory()   │
                                  │ - aider_edit()       │
                                  └──────────────────────┘

Storage: services/webui/data/llm_chat.db
Tables: conversation_sessions, messages
```

**Flow Example:**
```
User: "What files are in src/?"
→ Gateway routes to Architect /chat
→ Architect LLM calls list_directory("src/")
→ Aider RPC returns file list
→ Architect streams response: "Found 5 files: main.py, auth.py, ..."
→ Stored in ConversationSession
```

---

### 2. Agent ↔ Agent Chat (Parent-Child Communication)

```
┌─────────────────────────────────────────────────────────────────┐
│ Prime Directive Execution (Background Task)                     │
│ User → Verdict CLI → Gateway → PAS Root → Architect             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ Architect needs to delegate
         ┌───────────────────────────────────────┐
         │  Architect (Parent)                   │
         │  1. Create thread_id                  │
         │  2. Send delegation message           │
         │  3. Monitor for questions/status      │
         │  4. Answer questions with LLM         │
         └────────────┬──────────────────────────┘
                      │
                      ▼ POST /agent_chat/send
         ┌───────────────────────────────────────┐
         │  Dir-Code (Child)                     │
         │  1. Load thread history               │
         │  2. Process with LLM + context        │
         │  3. Ask questions if needed           │
         │  4. Stream status updates             │
         │  5. Complete and close thread         │
         └────────────┬──────────────────────────┘
                      │
                      ▼ Questions back to Parent
         ┌───────────────────────────────────────┐
         │  Architect (Parent)                   │
         │  - Receives question                  │
         │  - Uses LLM to generate answer        │
         │  - Sends answer back                  │
         └───────────────────────────────────────┘

Storage: artifacts/registry/registry.db
Tables: agent_conversation_threads, agent_conversation_messages
```

**Flow Example:**
```
Architect → Dir-Code: "Refactor auth to OAuth2"
Dir-Code → Architect: "Should I use authlib or python-oauth2?"
Architect → Dir-Code: "Use authlib - better maintained"
Dir-Code → Architect: [STATUS] "Installing authlib... (20%)"
Dir-Code → Architect: [STATUS] "Refactoring... (50%)"
Dir-Code → Architect: [COMPLETE] "Done! All tests pass."
→ Stored in AgentConversationThread
```

---

## Key Differences

### Context Preservation

**Human ↔ Agent:**
- User-controlled session lifecycle
- Multiple conversations per user
- Can archive/delete conversations
- Session persists across browser refreshes

**Agent ↔ Agent:**
- Task-bound thread lifecycle
- One thread per delegation
- Auto-closes on completion/error/timeout
- Thread tied to `run_id` (Prime Directive)

### Message Types

**Human ↔ Agent:**
- `user` - User message
- `assistant` - Agent response
- `system` - System notifications
- `status` - Task status updates

**Agent ↔ Agent:**
- `delegation` - Initial task from Parent
- `question` - Child asks Parent
- `answer` - Parent responds to question
- `status` - Progress update
- `completion` - Task finished
- `error` - Task failed
- `escalation` - Cannot proceed

### Tools Available

**Human ↔ Agent:**
- `read_file(path)` - Show user file contents
- `list_directory(path)` - Browse filesystem
- `aider_edit(message, files)` - Make code changes at user request

**Agent ↔ Agent:**
- `ask_parent(question)` - Ask Parent for clarification
- `send_status(message, progress)` - Update Parent on progress
- `call_aider(message, files)` - Make code changes (same as human chat)

---

## Database Schemas

### Human ↔ Agent (Existing)

**Table: `conversation_sessions`**
```sql
CREATE TABLE conversation_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,          -- "architect", "dir-code", etc.
    agent_name TEXT NOT NULL,
    parent_role TEXT NOT NULL,
    model_id TEXT,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT DEFAULT 'active',    -- active, archived, deleted
    title TEXT
);
```

**Table: `messages`**
```sql
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL,      -- user, assistant, system, status
    role TEXT,                       -- (legacy)
    content TEXT NOT NULL,
    agent_id TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL
);
```

### Agent ↔ Agent (New)

**Table: `agent_conversation_threads`**
```sql
CREATE TABLE agent_conversation_threads (
    thread_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,            -- Links to Prime Directive
    parent_agent_id TEXT NOT NULL,   -- "Architect"
    child_agent_id TEXT NOT NULL,    -- "Dir-Code"
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'active',    -- active, completed, failed, abandoned, timeout
    result TEXT,                     -- Final result if completed
    error TEXT                       -- Error message if failed
);
```

**Table: `agent_conversation_messages`**
```sql
CREATE TABLE agent_conversation_messages (
    message_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    from_agent TEXT NOT NULL,        -- Sender agent ID
    to_agent TEXT NOT NULL,          -- Recipient agent ID
    message_type TEXT NOT NULL,      -- delegation, question, answer, status, completion, error
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

---

## When to Use Which System?

### Use Human ↔ Agent Chat When:
- ✅ User wants to explore codebase interactively
- ✅ User wants to make small edits ("add docstring")
- ✅ User wants to ask questions ("what does this function do?")
- ✅ User wants conversational assistance

**Example:** User opens LLM Chat UI, selects Dir-Code, asks "Show me src/main.py"

### Use Agent ↔ Agent Chat When:
- ✅ Parent agent delegates complex task to child
- ✅ Task is ambiguous and may need clarification
- ✅ Child needs to ask questions during execution
- ✅ Parent wants real-time status updates

**Example:** Architect delegates "Refactor auth" to Dir-Code, Dir-Code asks "Which library?"

### Use Job Cards (No Chat) When:
- ✅ Task is simple and unambiguous
- ✅ No clarification needed
- ✅ Fire-and-forget pattern acceptable

**Example:** "Run linter on all Python files" - straightforward, no questions needed

---

## Implementation Strategy

### Phase Ordering

**Phase 1:** Human ↔ Agent Chat (Higher Priority)
- **Why:** Directly improves user experience
- **Deliverable:** Users can chat with PAS agents in LLM Chat UI
- **PRD:** `PRD_PAS_Agent_Chat_Interface.md`

**Phase 2:** Agent ↔ Agent Chat (Foundation for Autonomy)
- **Why:** Enables smarter agent delegation
- **Deliverable:** Agents can ask each other questions
- **PRD:** `PRD_Parent_Child_Chat_Communications.md`

**Rationale:** Human chat provides immediate value and tests the infrastructure. Agent chat builds on learned patterns.

---

## Common Infrastructure

### Shared Components
- ✅ Aider RPC (port 6130) - Used by both systems
- ✅ Comms Logger - Logs both human and agent messages
- ✅ LLM Clients - Shared Ollama/Anthropic/Google integrations
- ✅ Gateway - Routes both human and agent requests

### Separate Components
- ❌ Database tables - Different schemas for different use cases
- ❌ API endpoints - `/chat` vs `/agent_chat`
- ❌ Client libraries - `ConversationSession` vs `AgentChatClient`

---

## Future Integration Possibilities

### Unified Chat History View
**TRON Enhancement:** Show both human and agent conversations in one timeline

```
[10:30] User → Dir-Code: "What's in src/?"
[10:31] Dir-Code → User: "Found 5 files..."
[10:32] Architect → Dir-Code (agent): "Refactor auth"
[10:33] Dir-Code → Architect (agent): "Which library?"
[10:34] Architect → Dir-Code (agent): "Use authlib"
```

### Human-in-the-Loop for Agent Chat
**Allow user to monitor and intervene in agent conversations**

```
Architect → Dir-Code: "Refactor auth"
Dir-Code → Architect: "Should I use authlib?"

[User notification: "Dir-Code asking question"]
[User can respond: "Yes, use authlib and add rate limiting"]
```

### Cross-System Context Sharing
**Agent can reference human conversation history**

```
User (in LLM Chat): "I prefer using authlib for OAuth"
...later...
Dir-Code (in agent chat): "User previously mentioned preferring authlib, using that"
```

---

## References

- **PRD (Human ↔ Agent):** `docs/PRDs/PRD_PAS_Agent_Chat_Interface.md`
- **PRD (Agent ↔ Agent):** `docs/PRDs/PRD_Parent_Child_Chat_Communications.md`
- **Existing Chat DB:** `services/webui/llm_chat_db.py`
- **Comms Logger:** `services/common/comms_logger.py`
- **P0 Integration:** `docs/P0_END_TO_END_INTEGRATION.md`
