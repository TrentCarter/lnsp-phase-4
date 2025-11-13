# PRD: LLM Task Interface (Conversational AI Page)

**Version**: 1.0
**Date**: 2025-11-12
**Status**: Draft
**Owner**: LNSP Core Team

---

## Executive Summary

**LLM Task Interface** is a conversational AI page in the HMI that provides a rich, chat-based interface for submitting Prime Directives to the PAS hierarchy. It enables natural language task submission with the full feature set of modern LLM interfaces (streaming responses, conversation history, model selection, agent selection, etc.) while maintaining the PAS hierarchy's safety guardrails and communication patterns.

**What It Is**:
- New HMI page at `/llm` (alongside Tree, Sequencer, Dashboard, Actions)
- Rich chat interface with streaming LLM responses
- **Agent selector dropdown**: "SSH-like" connection to any agent in hierarchy (Architect, Directors, Managers, Programmers)
- **Default agent**: Architect (Port 6110) - LLM-powered top-level coordinator
- Model selection dropdown (shared with Settings page)
- Real-time token tracking (count, context window %, cost)
- Conversation history, export, and session management
- Optional voice features (Whisper STT + open-source TTS)

**What It Is NOT**:
- A replacement for Verdict CLI (complements CLI with visual interface)
- A direct LLM API wrapper (maintains PAS hierarchy communication pattern)
- An independent agent (all communication flows through selected agent in hierarchy)

**Core Innovation**: Brings modern chat-based UX to the PAS hierarchy with **SSH-like agent selection**, allowing users to "connect" to any agent (Architect, Directors, Managers, Programmers) and communicate as if they were that agent's parent. Includes real-time token/cost tracking and optional voice interface.

---

## 1. Problem Statement

### Current State
- Users can submit Prime Directives via:
  - **Verdict CLI** (`./bin/verdict send`) - Text-based, scripting-friendly
  - **`/pas-task` slash command** (Claude Code DirEng) - Conversational but external to HMI
- No visual, chat-based interface within the HMI for task submission
- No way to have back-and-forth conversation with **Architect** or other agents
- No visibility into LLM reasoning, plan generation, or iterative improvements
- No agent selection (can't "SSH into" a specific Director or Manager to debug/instruct)

### Pain Points
1. **No conversational refinement**: Users must fully specify task upfront (no iteration)
2. **Limited visibility**: Can't see Architect's planning, questions, or reasoning process
3. **No agent access**: Can't "SSH into" a specific Director/Manager to debug or give direct instructions
4. **No token tracking**: Can't see real-time token usage, context window %, or costs during conversation
5. **No rich UX features**: Missing streaming responses, syntax highlighting, code blocks, export
6. **Model lock-in**: Can't easily switch LLM models for different task types
7. **Disconnected workflows**: Chat interface (Claude Code) is separate from HMI monitoring (Tree/Sequencer)

### Why Now?
- PAS P0 stack is production-ready (Gateway → PAS Root → Aider-LCO working end-to-end)
- HMI infrastructure mature (Flask @ 6101, WebSocket/SSE streaming @ 6102)
- Users requesting "ChatGPT-like interface" for task submission
- Need bridge between conversational task intake and visual progress monitoring

---

## 2. Goals & Success Criteria

### Goals
1. **Rich Chat UX**: Modern LLM interface with streaming, markdown, code blocks, syntax highlighting
2. **Agent Selection**: "SSH-like" dropdown to connect to any agent (Architect, Directors, Managers, Programmers)
3. **Model Selection**: Dropdown to choose LLM model (same list as Settings page)
4. **Real-time Token Tracking**: Display tokens used, context window %, cost per message and cumulative
5. **Conversational Refinement**: Multi-turn conversations to clarify, plan, and approve tasks
6. **PAS Integration**: All messages route through selected agent in hierarchy
7. **Session Management**: Conversation history, export to markdown/JSON, clear/reset
8. **Voice Interface** (V2): Whisper STT + open-source TTS for hands-free operation
9. **Seamless Monitoring**: One-click navigation from chat to Tree/Sequencer to watch execution

### Success Criteria (V1 MVP)
- [ ] Chat interface live at `/llm` route
- [ ] **Agent selector dropdown** with all agents (Architect, Dir-*, Mgr-*, Programmer-*)
- [ ] Default agent: Architect (Port 6110)
- [ ] Streaming responses with typing indicators
- [ ] Markdown rendering with syntax highlighting (code blocks)
- [ ] **Model selection dropdown** (populated from Settings → LLM Models)
- [ ] **Real-time token tracking**: tokens/message, cumulative, context %, cost
- [ ] **Token meter** visual gauge (progress bar showing context window usage)
- [ ] Conversation history persisted (SQLite or JSON files)
- [ ] Export conversation to markdown/JSON
- [ ] Clear conversation and start new session
- [ ] Submit task → selected agent flow working
- [ ] Display task status inline (submitted → planning → executing → complete)
- [ ] Link to Tree/Sequencer views for active task
- [ ] Mobile-responsive (works on tablet/phone)

### Non-Goals (V1)
- Voice input/output (Whisper STT + TTS) → V2 (architecture planned, defer implementation)
- Multi-modal inputs (images, PDFs, diagrams) → V2
- Branching conversations (tree of dialogue paths) → V2
- Real-time collaboration (multiple users in same chat) → V2
- Agentic loop visualization (show Director/Manager sub-conversations inline) → V2

---

## 3. User Experience

### 3.1 Page Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Header: PAS Agent Swarm (OK) | Dashboard | Tree | Sequencer | ... │
│                                                            [LLM] ← │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Toolbar                                                         │ │
│ │ ┌────────────────────┐ ┌────────────────────┐ ┌──────────────┐│ │
│ │ │Agent: [Architect ▼]│ │Model: [Sonnet 4 ▼] │ │Export│Clear││ │
│ │ └────────────────────┘ └────────────────────┘ └──────────────┘│ │
│ │ Tokens: 2,458/200k (1.2%) | Cost: $0.03 | Window: [▓░░░░] 1% │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Conversation Area (Scrollable)                                  │ │
│ │                                                                  │ │
│ │ ┌─────────────────────────────────────────────────────────────┐ │ │
│ │ │ 👤 User                                        [10:23 AM]    │ │ │
│ │ │ Please implement a new feature to track user login times    │ │ │
│ │ └─────────────────────────────────────────────────────────────┘ │ │
│ │                                                                  │ │
│ │ ┌─────────────────────────────────────────────────────────────┐ │ │
│ │ │ 🏛️ Architect (Claude Sonnet 4.5)              [10:23 AM]    │ │ │
│ │ │ I'll help you implement user login tracking. Let me gather  │ │ │
│ │ │ some requirements:                                           │ │ │
│ │ │                                                              │ │ │
│ │ │ 1. Where should login times be stored? (DB, logs, both?)    │ │ │
│ │ │ 2. Do you need timezone support?                            │ │ │
│ │ │ 3. Should we track login/logout or just login?              │ │ │
│ │ │                                                              │ │ │
│ │ │ [View in Tree] [View in Sequencer]                          │ │ │
│ │ └─────────────────────────────────────────────────────────────┘ │ │
│ │                                                                  │ │
│ │ ┌─────────────────────────────────────────────────────────────┐ │ │
│ │ │ 👤 User                                        [10:24 AM]    │ │ │
│ │ │ Store in PostgreSQL, yes to timezone, login only is fine    │ │ │
│ │ └─────────────────────────────────────────────────────────────┘ │ │
│ │                                                                  │ │
│ │ ┌─────────────────────────────────────────────────────────────┐ │ │
│ │ │ 🏛️ Architect (Claude Sonnet 4.5)              [10:24 AM]    │ │ │
│ │ │ ┌───────────────────────────────────────────────────────┐   │ │ │
│ │ │ │ Tokens: 1,234 | Context: 0.6% | Cost: $0.015         │   │ │ │
│ │ │ └───────────────────────────────────────────────────────┘   │ │ │
│ │ │ Perfect! Here's my implementation plan:                      │ │ │
│ │ │                                                              │ │ │
│ │ │ **Phase 1: Database Schema**                                │ │ │
│ │ │ - Add `user_logins` table with columns:                     │ │ │
│ │ │   - `user_id` (FK to users)                                 │ │ │
│ │ │   - `login_timestamp` (TIMESTAMPTZ)                         │ │ │
│ │ │   - `ip_address`, `user_agent`                              │ │ │
│ │ │                                                              │ │ │
│ │ │ **Phase 2: Backend Logic**                                  │ │ │
│ │ │ - Middleware to capture login events                        │ │ │
│ │ │ - Insert login records on successful auth                   │ │ │
│ │ │                                                              │ │ │
│ │ │ **Phase 3: API Endpoint**                                   │ │ │
│ │ │ - GET /api/users/{id}/login-history                         │ │ │
│ │ │                                                              │ │ │
│ │ │ Estimated tokens: ~15,000 | Time: ~8 minutes                │ │ │
│ │ │                                                              │ │ │
│ │ │ [✓ Approve & Execute] [✎ Refine Plan] [✗ Cancel]           │ │ │
│ │ └─────────────────────────────────────────────────────────────┘ │ │
│ │                                                                  │ │
│ │ [Typing indicator: PAS Root is thinking...]                    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Input Area                                                      │ │
│ │ ┌───────────────────────────────────────────────────────────┐ │ │
│ │ │ Type your message... (Shift+Enter for new line)          │ │ │
│ │ │                                                            │ │ │
│ │ └───────────────────────────────────────────────────────────┘ │ │
│ │ [📎 Attach] [🎤 Voice (V2)] [💾 Templates (V2)]    [Send ➤]  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Features

#### Agent Selection Dropdown ("SSH Mode")
- **Location**: Top toolbar, left side
- **Concept**: "SSH into" any agent in the PAS hierarchy
- **Content**: All active agents from Registry (6121)
  - **Tier 2 - Architect**: `Architect` (Port 6110) - **DEFAULT**
  - **Tier 3 - Directors**: `Dir-Code`, `Dir-Models`, `Dir-Data`, `Dir-DevSecOps`, `Dir-Docs` (6111-6115)
  - **Tier 4 - Managers**: `Mgr-Code-01`, `Mgr-Code-02`, etc. (6141-6150)
  - **Tier 5 - Programmers**: `Programmer-Qwen-001`, `Programmer-Claude-001`, etc. (6151-6199)
- **Behavior**:
  - Shows currently selected agent (e.g., "🏛️ Architect", "💻 Dir-Code", "👨‍💻 Programmer-Claude-001")
  - Dropdown organized by tier with visual separators
  - Role icons displayed next to agent names (🏛️ Architect, 💻 Director, ⚙️ Manager, 👨‍💻 Programmer)
  - Port numbers shown in dropdown (e.g., "Architect (6110)")
  - Agent selection persists per conversation
  - When switched, next message goes to new agent (can switch mid-conversation)
- **Communication Pattern**: User communicates as if they are the agent's **parent**
  - Chat with Architect → You are PAS Root (the Architect's parent)
  - Chat with Dir-Code → You are Architect (the Director's parent)
  - Chat with Mgr-Code-01 → You are Dir-Code (the Manager's parent)
  - Chat with Programmer-Claude-001 → You are Mgr-Code-01 (the Programmer's parent)

#### Model Selection Dropdown
- **Location**: Top toolbar, center
- **Content**: Same model list as Settings → LLM Models
- **Behavior**:
  - Shows currently selected model (e.g., "Claude Sonnet 4.5", "GPT-4o", "Llama 3.1:8b")
  - Dropdown lists all configured models with metadata:
    - Model name
    - Provider (Anthropic, OpenAI, Ollama)
    - Context window size
    - Cost per 1M tokens (if applicable)
  - Model selection persists per conversation (can switch mid-conversation)
  - Displays model name in each AI response bubble

#### Real-Time Token Tracking
- **Location**: Top toolbar, below dropdowns (second row)
- **Display**:
  - **Cumulative tokens**: `2,458 / 200,000` (current conversation total / model context window)
  - **Context percentage**: `(1.2%)` - visual indicator of how full the context window is
  - **Cumulative cost**: `$0.03` - total cost for current conversation
  - **Visual gauge**: Progress bar `[▓░░░░]` showing context window fill percentage
- **Per-Message Tracking**: Each AI message bubble shows:
  - Tokens used in that response: `Tokens: 1,234`
  - Context percentage after that message: `Context: 0.6%`
  - Cost for that message: `Cost: $0.015`
- **Color Coding** (Context Window %):
  - Green: 0-50% (plenty of room)
  - Yellow: 50-75% (getting full)
  - Orange: 75-90% (consider clearing)
  - Red: 90-100% (almost full, clear soon)

#### Streaming Responses
- **Typing Indicator**: "🤖 PAS Root is thinking..." while waiting
- **Token-by-Token Streaming**: Text appears word-by-word (via SSE/WebSocket)
- **Markdown Rendering**: Real-time markdown parsing as text streams
- **Code Block Highlighting**: Syntax highlighting applied after code block completes

#### Message Bubbles
- **User Messages**: Right-aligned, blue background, white text
- **AI Messages**: Left-aligned, dark gray background, white text
- **Metadata**: Timestamp, model name (for AI), role icon (👤 user, 🤖 AI)
- **Status Badges**: For task-related messages:
  - 🟡 Planning
  - 🔵 Executing
  - 🟢 Complete
  - 🔴 Error
  - 🟣 Awaiting Approval

#### Conversation History
- **Persistence**: SQLite table `llm_conversations`
  - Columns: `id`, `session_id`, `role` (user/assistant), `content`, `model`, `timestamp`, `metadata` (JSON)
- **Session Management**:
  - Each conversation has unique `session_id`
  - Sessions listed in sidebar (V2) or accessible via "History" dropdown
  - Can resume previous sessions
- **Search**: Full-text search across conversation history (V2)

#### Export Functionality
- **Formats**:
  - **Markdown**: Human-readable, formatted with headers/code blocks
  - **JSON**: Machine-readable, includes all metadata (timestamps, models, status)
  - **PDF**: Formatted conversation with syntax highlighting (V2)
- **Filename**: `pas_conversation_{session_id}_{timestamp}.{ext}`

#### Clear Conversation
- **Behavior**: Clears current conversation, starts new session
- **Confirmation**: "Are you sure? This will clear the current conversation."
- **Preservation**: Old conversation saved in history, not deleted

---

## 4. Architecture & Technical Design

### 4.1 Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ User Types Message in HMI Chat Interface                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────────┐
         │ HMI Flask Route                 │
         │ POST /api/chat/message          │
         │ (Port 6101)                     │
         └──────────────┬──────────────────┘
                        │
                        ▼ POST /prime_directives or /chat
         ┌─────────────────────────────────┐
         │ Gateway (Port 6120)             │
         │ - Idempotency key               │
         │ - Request routing               │
         └──────────────┬──────────────────┘
                        │
                        ▼ POST /pas/chat
         ┌─────────────────────────────────┐
         │ PAS Root (Port 6100)            │
         │ - LLM orchestration             │
         │ - Plan generation               │
         │ - Director delegation           │
         └──────────────┬──────────────────┘
                        │
                        ▼ Streaming response (SSE)
         ┌─────────────────────────────────┐
         │ Event Stream (Port 6102)        │
         │ - Token-by-token streaming      │
         │ - Status updates                │
         └──────────────┬──────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────┐
         │ HMI Frontend (JavaScript)       │
         │ - Markdown rendering            │
         │ - Syntax highlighting           │
         │ - Status badges                 │
         └─────────────────────────────────┘
```

### 4.2 Communication Pattern: Human ↔ PAS Hierarchy

**Key Insight**: Each agent thinks their **parent** is the human. The actual human can "SSH into" any agent and communicate as that agent's parent.

**Example 1: Chat with Architect (Default)**
```
Human (HMI Chat) - Selects "Architect" from dropdown
  │
  ├─ "Implement login tracking"
  │  [Human acts as PAS Root, Architect's parent]
  │
  ▼
Architect (Tier 2 - LLM Coordinator)
  │
  ├─ [Thinks PAS Root is talking to it]
  ├─ [Generates Plan]
  │  "I'll delegate schema to Dir-Data, API to Dir-Code"
  │
  ├─ → Dir-Data (Director)
  │     │
  │     ├─ [Receives] "Create user_logins table schema"
  │     │
  │     ├─ → Mgr-Schema (Manager)
  │     │     │
  │     │     ├─ [Receives] "Write migration for user_logins table"
  │     │     │
  │     │     ├─ → DB-Programmer (Worker)
  │     │     │     │
  │     │     │     └─ [Thinks Mgr-Schema is human]
  │     │     │         "Here's the migration SQL..."
  │     │     │
  │     │     └─ [Aggregates] "Migration ready"
  │     │
  │     └─ [Reports to PAS Root] "Schema complete"
  │
  ├─ → Dir-Code (Director)
  │     │
  │     └─ [Similar delegation pattern]
  │
  └─ [Aggregates All Results]
     "Login tracking implemented! Here's the summary..."
     │
     ▼
Human (HMI Chat)
  └─ [Sees final result + all intermediate steps (optional)]
```

**Example 2: Chat with Director (Direct Access)**
```
Human (HMI Chat) - Selects "Dir-Code" from dropdown
  │
  ├─ "Review the API implementation for security issues"
  │  [Human acts as Architect, Dir-Code's parent]
  │
  ▼
Dir-Code (Tier 3 - Code Lane Director)
  │
  ├─ [Thinks Architect is talking to it]
  ├─ [Analyzes API code]
  │  "Found 3 potential SQL injection vulnerabilities"
  │
  ├─ → Mgr-Code-01 (Manager)
  │     "Fix SQL injection in user_search endpoint"
```

**Example 3: Chat with Programmer (Deep Dive)**
```
Human (HMI Chat) - Selects "Programmer-Claude-001" from dropdown
  │
  ├─ "Why did you use a list comprehension here instead of a loop?"
  │  [Human acts as Mgr-Code-01, Programmer's parent]
  │
  ▼
Programmer-Claude-001 (Tier 5 - Code Executor)
  │
  ├─ [Thinks Mgr-Code-01 is asking]
  └─ "I used a list comprehension because it's more Pythonic and
      has better performance for this dataset size (N < 10,000)..."
```

**Implementation Notes**:
- **Agent Selection**: User can "SSH into" any agent in the hierarchy
- **Parent Simulation**: HMI impersonates the agent's parent when communicating
- **Direct Access**: Allows debugging, clarification, or direct instruction of any agent
- **No Hierarchy Bypass**: Selected agent still follows normal delegation patterns
- **Visibility**: All messages in current conversation go to/from selected agent only

### 4.3 Data Models

#### Conversation Session
```python
class ConversationSession:
    session_id: str           # UUID
    user_id: str              # User identifier
    model_name: str           # Selected LLM model
    created_at: datetime      # Session start time
    updated_at: datetime      # Last message time
    status: str               # active, archived, deleted
    metadata: dict            # Custom data (tags, project_id, etc.)
```

#### Message
```python
class Message:
    id: str                   # UUID
    session_id: str           # FK to ConversationSession
    role: str                 # user, assistant, system
    content: str              # Message text (markdown)
    model_name: str           # Model used for response (if assistant)
    timestamp: datetime       # Message time
    metadata: dict            # {
                              #   "task_id": "...",          # If message spawned task
                              #   "status": "planning",       # Task status
                              #   "tokens_used": 1234,       # Token count
                              #   "elapsed_seconds": 5.2,    # Response time
                              #   "parent_agent": "PAS Root", # Agent name
                              # }
```

#### Model Configuration (Shared with Settings)
```python
class ModelConfig:
    model_id: str             # Unique identifier
    model_name: str           # Display name
    provider: str             # anthropic, openai, ollama
    endpoint: str             # API endpoint URL
    context_window: int       # Max tokens
    cost_per_1m_input: float  # Cost (USD)
    cost_per_1m_output: float # Cost (USD)
    capabilities: list        # ["streaming", "function_calling", "vision"]
    is_active: bool           # Enabled/disabled
```

### 4.4 API Endpoints

#### HMI Backend (Flask @ 6101)
```python
# Render chat page
GET /llm
GET /tasks
  → Returns: HTML template with chat interface

# Send message
POST /api/chat/message
  Body: {
    "session_id": "uuid",      # Optional, creates new if omitted
    "message": "...",          # User message text
    "model": "claude-4-sonnet" # Optional, uses default if omitted
  }
  → Returns: {
    "session_id": "uuid",
    "message_id": "uuid",
    "status": "streaming"  # Message queued for streaming
  }

# Get conversation history
GET /api/chat/sessions/{session_id}
  → Returns: {
    "session_id": "uuid",
    "messages": [
      {"id": "...", "role": "user", "content": "...", ...},
      {"id": "...", "role": "assistant", "content": "...", ...}
    ],
    "metadata": {...}
  }

# List all sessions
GET /api/chat/sessions
  Query: ?limit=20&offset=0
  → Returns: {
    "sessions": [
      {"session_id": "...", "created_at": "...", "message_count": 5},
      ...
    ],
    "total": 42
  }

# Export conversation
GET /api/chat/sessions/{session_id}/export?format=markdown
GET /api/chat/sessions/{session_id}/export?format=json
  → Returns: File download (Content-Disposition: attachment)

# Clear conversation (archive)
POST /api/chat/sessions/{session_id}/clear
  → Returns: {"status": "archived", "new_session_id": "uuid"}

# Delete conversation
DELETE /api/chat/sessions/{session_id}
  → Returns: {"status": "deleted"}

# Get available models (from Settings)
GET /api/models
  → Returns: {
    "models": [
      {"model_id": "...", "model_name": "...", "provider": "...", ...},
      ...
    ]
  }
```

#### Gateway (FastAPI @ 6120)
```python
# Forward chat message to PAS Root
POST /chat
  Body: {
    "session_id": "uuid",
    "message": "...",
    "model": "...",
    "idempotency_key": "..."
  }
  → Forwards to PAS Root @ 6100

# Stream response (SSE)
GET /chat/stream/{session_id}
  → SSE stream of tokens
```

#### PAS Root (FastAPI @ 6100)
```python
# Process chat message
POST /pas/chat
  Body: {
    "session_id": "uuid",
    "message": "...",
    "model": "..."
  }
  → Returns: {
    "message_id": "uuid",
    "status": "processing"  # Will stream via SSE
  }

# Stream response tokens
GET /pas/chat/stream/{session_id}
  → SSE stream:
    data: {"type": "token", "content": "I'll"}
    data: {"type": "token", "content": " help"}
    data: {"type": "token", "content": " you"}
    ...
    data: {"type": "done", "metadata": {"tokens": 1234}}
```

### 4.5 Frontend Components (JavaScript)

#### ChatInterface.js
```javascript
class ChatInterface {
  constructor(containerId, apiBaseUrl) {
    this.container = document.getElementById(containerId);
    this.apiUrl = apiBaseUrl;
    this.sessionId = null;
    this.model = null;
    this.eventSource = null;
    this.init();
  }

  async init() {
    // Load model list
    this.models = await this.fetchModels();
    // Render UI
    this.render();
    // Setup event listeners
    this.bindEvents();
    // Restore last session (from localStorage)
    this.restoreSession();
  }

  async sendMessage(message) {
    // Append user message to UI
    this.appendMessage('user', message);

    // Send to backend
    const response = await fetch(`${this.apiUrl}/api/chat/message`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: this.sessionId,
        message: message,
        model: this.model
      })
    });

    const data = await response.json();
    this.sessionId = data.session_id;

    // Start streaming response
    this.streamResponse(data.message_id);
  }

  streamResponse(messageId) {
    // Create SSE connection
    this.eventSource = new EventSource(
      `${this.apiUrl}/api/chat/stream/${this.sessionId}`
    );

    let currentMessage = '';
    let messageDiv = this.appendMessage('assistant', '', {streaming: true});

    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'token') {
        currentMessage += data.content;
        this.updateMessage(messageDiv, currentMessage);
      } else if (data.type === 'done') {
        this.eventSource.close();
        this.finalizeMessage(messageDiv, data.metadata);
      }
    };
  }

  appendMessage(role, content, options = {}) {
    // Create message bubble with markdown rendering
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;

    // Add timestamp, model name, status badge
    // Render markdown with syntax highlighting
    // Return div for streaming updates

    this.container.appendChild(messageDiv);
    this.scrollToBottom();
    return messageDiv;
  }

  // ... more methods for export, clear, model selection, etc.
}
```

#### MarkdownRenderer.js
```javascript
class MarkdownRenderer {
  constructor() {
    // Use marked.js for markdown parsing
    this.marked = marked;
    // Use Prism.js or Highlight.js for syntax highlighting
    this.highlighter = Prism;
  }

  render(markdown) {
    // Convert markdown to HTML
    const html = this.marked.parse(markdown);
    // Apply syntax highlighting to code blocks
    return this.highlightCode(html);
  }

  renderStreaming(partialMarkdown) {
    // Handle incomplete code blocks gracefully
    // Don't highlight until code block is complete
  }
}
```

### 4.6 Database Schema

```sql
-- Conversation sessions
CREATE TABLE llm_conversations (
  session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status TEXT NOT NULL DEFAULT 'active',  -- active, archived, deleted
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_conversations_user ON llm_conversations(user_id);
CREATE INDEX idx_conversations_status ON llm_conversations(status);
CREATE INDEX idx_conversations_created ON llm_conversations(created_at DESC);

-- Messages
CREATE TABLE llm_messages (
  message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES llm_conversations(session_id) ON DELETE CASCADE,
  role TEXT NOT NULL,  -- user, assistant, system
  content TEXT NOT NULL,
  model_name TEXT,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_messages_session ON llm_messages(session_id, timestamp);
CREATE INDEX idx_messages_timestamp ON llm_messages(timestamp DESC);

-- Full-text search on message content (V2)
CREATE INDEX idx_messages_content_fts ON llm_messages USING gin(to_tsvector('english', content));
```

---

## 5. User Workflows

### 5.1 Simple Task Submission

1. User navigates to HMI → LLM tab
2. Types: "Add a health check endpoint to the API"
3. Clicks Send (or presses Enter)
4. PAS Root responds with clarifying questions:
   - "Which API? (Gateway, PAS Root, Aider-LCO?)"
   - "What should the health check return?"
5. User answers: "Gateway API, return status and uptime"
6. PAS Root generates plan, shows estimation, asks for approval
7. User clicks "Approve & Execute"
8. PAS Root delegates to Dir-Code → Mgr-API → Programmer
9. Progress updates stream to chat interface
10. Final result: "Health check endpoint added to Gateway at `/health`"
11. User can click "View in Tree" to see execution hierarchy

### 5.2 Iterative Plan Refinement

1. User: "Optimize the database queries"
2. PAS Root: "I found 12 slow queries. Here's my optimization plan..."
   - [Shows detailed plan with 3 phases]
   - Estimated tokens: 50,000 | Time: 25 minutes
3. User: "That's too much. Just optimize the top 3 slowest queries."
4. PAS Root: "Updated plan for top 3 queries only..."
   - Estimated tokens: 12,000 | Time: 8 minutes
5. User: "Perfect, go ahead"
6. PAS Root executes, streams progress updates
7. User sees completion notification with performance metrics

### 5.3 Model Switching Mid-Conversation

1. User starts conversation with "Claude Sonnet 4.5"
2. Asks complex architectural question
3. PAS Root provides detailed answer
4. User switches model to "GPT-4o" (dropdown)
5. Asks follow-up question
6. GPT-4o responds in same conversation thread
7. Message bubbles show which model generated each response

### 5.4 Export and Share

1. User completes complex task through conversation
2. Clicks "Export" button → selects "Markdown"
3. Downloads `pas_conversation_2025-11-12_10-23-45.md`
4. File contains:
   - Full conversation history
   - Syntax-highlighted code blocks
   - Timestamps and model names
   - Task status updates
5. User shares exported file with team for review

---

## 6. UI/UX Design Specifications

### 6.1 Color Scheme (Consistent with HMI)

```css
/* Match existing HMI dark theme */
--bg-primary: #0f1423;
--bg-secondary: #1a1f3a;
--bg-tertiary: #2a3558;
--border-color: #3a4568;

--text-primary: #ffffff;
--text-secondary: #a0aec0;
--text-muted: #6b7280;

--accent-blue: #3b82f6;
--accent-green: #10b981;
--accent-yellow: #f59e0b;
--accent-red: #ef4444;
--accent-purple: #8b5cf6;

/* Message bubbles */
--user-bg: #3b82f6;       /* Blue */
--user-text: #ffffff;
--assistant-bg: #2a3558;  /* Dark gray */
--assistant-text: #ffffff;

/* Status badges */
--status-planning: #f59e0b;   /* Yellow */
--status-executing: #3b82f6;  /* Blue */
--status-complete: #10b981;   /* Green */
--status-error: #ef4444;      /* Red */
--status-approval: #8b5cf6;   /* Purple */
```

### 6.2 Typography

```css
/* Message content */
.message-content {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-primary);
}

/* Code blocks */
.message-content code {
  font-family: 'SF Mono', Monaco, Menlo, 'Courier New', monospace;
  font-size: 13px;
  background: rgba(0, 0, 0, 0.3);
  padding: 2px 6px;
  border-radius: 4px;
}

/* Code blocks (multi-line) */
.message-content pre {
  background: #1e1e1e;
  padding: 16px;
  border-radius: 8px;
  overflow-x: auto;
  border: 1px solid var(--border-color);
}

/* Timestamps */
.message-timestamp {
  font-size: 12px;
  color: var(--text-muted);
  font-weight: 500;
}
```

### 6.3 Responsive Design

```css
/* Desktop (>= 1024px) */
@media (min-width: 1024px) {
  .chat-container {
    max-width: 900px;
    margin: 0 auto;
  }

  .message {
    max-width: 75%;
  }
}

/* Tablet (768px - 1023px) */
@media (min-width: 768px) and (max-width: 1023px) {
  .chat-container {
    padding: 1rem;
  }

  .message {
    max-width: 85%;
  }
}

/* Mobile (< 768px) */
@media (max-width: 767px) {
  .toolbar {
    flex-direction: column;
    gap: 0.5rem;
  }

  .message {
    max-width: 95%;
  }

  .input-area {
    padding: 0.75rem;
  }
}
```

### 6.4 Animations

```css
/* Typing indicator animation */
@keyframes typing {
  0%, 60%, 100% { opacity: 0.3; }
  30% { opacity: 1; }
}

.typing-indicator span {
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

/* Message fade-in */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message {
  animation: fadeInUp 0.3s ease-out;
}

/* Streaming text cursor */
@keyframes blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}

.streaming-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background: var(--accent-blue);
  animation: blink 1s step-start infinite;
}
```

---

## 7. Security & Safety

### 7.1 Authentication & Authorization
- User sessions validated via Flask session cookies
- Each user gets isolated conversation history (via `user_id`)
- No cross-user conversation access

### 7.2 Input Validation
- Message length limit: 50,000 characters (prevent abuse)
- Rate limiting: 10 messages per minute per user
- XSS prevention: All user input sanitized before rendering
- SQL injection prevention: Parameterized queries only

### 7.3 Model Access Control
- Only models marked `is_active=true` appear in dropdown
- Model endpoint credentials stored securely (env vars, not in DB)
- API keys never exposed to frontend

### 7.4 Content Safety
- Profanity filter optional (configurable in Settings)
- Prompt injection detection (basic heuristics)
- Task approval gates for high-risk operations (file deletion, git push --force, etc.)

### 7.5 Audit Trail
- All messages logged with timestamps, user_id, model, tokens used
- Task submissions linked to conversation message (via `metadata.task_id`)
- Full auditability: Who requested what, when, using which model

---

## 8. Performance & Scalability

### 8.1 Response Time Targets
- Message submission (user → server): < 200ms
- First token received (streaming start): < 1 second
- Median token latency: < 50ms
- Page load time: < 1 second

### 8.2 Streaming Optimization
- Use Server-Sent Events (SSE) for low-latency streaming
- Backpressure handling: Pause streaming if client falls behind
- Compression: Gzip for SSE stream (if supported by client)

### 8.3 Database Performance
- Conversation history paginated (20 messages per load)
- Infinite scroll with lazy loading
- Indexed queries (session_id, timestamp)
- Periodic archival of old conversations (>90 days → cold storage)

### 8.4 Caching
- Model list cached (5 minute TTL)
- User session cached in Redis (optional, for multi-instance deployments)

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Message rendering (markdown → HTML)
- Syntax highlighting (code blocks)
- Input validation (XSS, length limits)
- Model selection logic

### 9.2 Integration Tests
- End-to-end message flow (HMI → Gateway → PAS Root → HMI)
- Streaming response handling
- Conversation persistence (SQLite)
- Export functionality (markdown, JSON)

### 9.3 Load Tests
- 10 concurrent users sending messages
- 100 messages per minute across all users
- Verify no dropped SSE connections
- Verify database write performance

### 9.4 UI/UX Tests
- Mobile responsiveness (tablet, phone)
- Keyboard shortcuts (Enter to send, Shift+Enter for newline)
- Accessibility (screen reader compatibility, ARIA labels)

---

## 10. Rollout Plan

### Phase 1: Core Chat Interface (Week 1)
- [ ] Basic chat UI (message bubbles, input area)
- [ ] Model selection dropdown
- [ ] Message persistence (SQLite)
- [ ] Markdown rendering

### Phase 2: Streaming & Integration (Week 2)
- [ ] SSE streaming implementation
- [ ] Integration with Gateway/PAS Root
- [ ] Task status badges
- [ ] Typing indicators

### Phase 3: History & Export (Week 3)
- [ ] Conversation history view
- [ ] Export to markdown/JSON
- [ ] Session management (clear, archive)
- [ ] Search functionality (V2 stretch goal)

### Phase 4: Polish & Optimization (Week 4)
- [ ] Syntax highlighting (Prism.js)
- [ ] Mobile responsiveness
- [ ] Keyboard shortcuts
- [ ] Performance optimization
- [ ] Load testing
- [ ] Documentation

---

## 11. Open Questions

1. **Visibility Toggle**: Should users see only PAS Root messages (simple), or all Director/Manager/Programmer messages (detailed)?
   - **Recommendation**: Default to simple, add "Show Details" toggle

2. **Voice Input/Output**: Priority for V1 or defer to V2?
   - **Recommendation**: Defer to V2 (requires audio service integration)

3. **Task Approval in Chat**: Should approvals happen inline (chat buttons) or redirect to separate approval page?
   - **Recommendation**: Inline approval buttons ("✓ Approve", "✎ Refine", "✗ Cancel")

4. **Model Switching Mid-Task**: If user switches model while task is executing, should it:
   - Cancel current task and restart with new model?
   - Queue new model for next message only?
   - **Recommendation**: Queue for next message, don't interrupt running tasks

5. **Cost Tracking**: Show running token cost in chat interface?
   - **Recommendation**: Yes, show cumulative cost in toolbar ("Tokens: 12,456 | Cost: $0.18")

---

## 12. Success Metrics (Post-Launch)

### Usage Metrics
- Daily active users (DAU) on LLM page
- Messages per user per session (engagement)
- Average conversation length (messages per session)
- Model selection distribution (which models are most popular)

### Performance Metrics
- P50/P95/P99 latency for message submission
- P50/P95/P99 time-to-first-token (streaming)
- SSE connection success rate
- Export success rate

### Quality Metrics
- Task success rate (submitted via chat vs. completed successfully)
- User satisfaction score (optional feedback widget)
- Error rate (failed API calls, dropped streams)

### Business Metrics
- Reduction in support tickets (easier task submission)
- Increase in PAS usage (more accessible interface)
- Token cost per conversation (cost efficiency)

---

## 13. Future Enhancements (V2+)

### V2: Advanced Features

#### Voice Interface (Whisper STT + Open-Source TTS)
- [ ] **Speech-to-Text (STT)**: OpenAI Whisper (local model via Faster-Whisper or whisper.cpp)
  - Push-to-talk button (🎤) or voice activity detection (VAD)
  - Real-time transcription displayed in input area before sending
  - Support for multiple languages (English, Spanish, French, etc.)
  - Local processing (no cloud API calls for privacy)
- [ ] **Text-to-Speech (TTS)**: Open-source TTS engines
  - **Option 1**: Piper TTS (fast, low-latency, high-quality)
  - **Option 2**: Coqui TTS (formerly Mozilla TTS, voice cloning support)
  - **Option 3**: Festival TTS (classic, lightweight)
- [ ] **Voice Settings**:
  - Enable/disable voice input/output independently
  - Voice speed control (0.5x - 2.0x)
  - Voice selection (male/female, different accents)
  - Auto-play responses (on/off)
- [ ] **Integration**:
  - Microphone access via WebRTC (browser permission required)
  - Audio playback via Web Audio API
  - Background audio service (Port 6103) handles STT/TTS processing
  - Streaming audio playback (start playing TTS before full response completes)

#### Other V2 Features
- [ ] Multi-modal inputs (upload images, PDFs, diagrams)
- [ ] Branching conversations (explore multiple solutions)
- [ ] Conversation templates (pre-filled prompts for common tasks)
- [ ] Real-time collaboration (multiple users in same chat)

### V3: Agentic Features
- [ ] Agentic loop visualization (show Director/Manager sub-conversations in tree view)
- [ ] Interactive plan approval (edit plan steps before execution)
- [ ] What-if analysis ("What if I change this requirement?")
- [ ] Side-by-side comparison (compare outputs from different models)

### V4: Enterprise Features
- [ ] Team workspaces (shared conversation history)
- [ ] Role-based access control (read-only vs. execute)
- [ ] Conversation analytics dashboard (usage patterns, cost breakdown)
- [ ] Integration with external tools (Jira, GitHub, Slack)

---

## 14. Appendices

### A. Markdown Support

#### Supported Markdown Syntax
- **Headers**: `# H1`, `## H2`, `### H3`, etc.
- **Bold/Italic**: `**bold**`, `*italic*`, `***both***`
- **Lists**: Unordered (`-`, `*`, `+`), Ordered (`1.`, `2.`)
- **Links**: `[text](url)`
- **Images**: `![alt](url)` (V2: inline preview)
- **Code**: Inline `` `code` ``, Blocks ` ```lang ... ``` `
- **Blockquotes**: `> quote`
- **Tables**: Markdown tables with alignment
- **Task Lists**: `- [ ]`, `- [x]` (V2: interactive checkboxes)

#### Syntax Highlighting Languages
- Python, JavaScript, TypeScript, Go, Rust, Java, C++, C#
- SQL, JSON, YAML, TOML, Markdown, HTML, CSS
- Bash/Shell, Dockerfile, Makefile
- (Full list via Prism.js supported languages)

### B. Keyboard Shortcuts

| Shortcut          | Action                          |
|-------------------|---------------------------------|
| `Enter`           | Send message                    |
| `Shift + Enter`   | New line (don't send)           |
| `Ctrl + K`        | Clear conversation              |
| `Ctrl + E`        | Export conversation (markdown)  |
| `Ctrl + /`        | Focus model dropdown            |
| `Esc`             | Cancel streaming response       |
| `Ctrl + ↑`        | Load previous message (history) |
| `Ctrl + ↓`        | Load next message (history)     |

### C. API Rate Limits

| Endpoint                  | Rate Limit           |
|---------------------------|----------------------|
| `POST /api/chat/message`  | 10 req/min per user  |
| `GET /api/chat/sessions`  | 60 req/min per user  |
| `GET /api/chat/stream/*`  | 5 concurrent streams |
| `GET /api/models`         | 120 req/min per user |

### D. Error Messages

| Error Code | Message                                  | User Action                     |
|------------|------------------------------------------|---------------------------------|
| 400        | "Message too long (max 50,000 chars)"   | Shorten message                 |
| 401        | "Unauthorized. Please log in."          | Log in                          |
| 429        | "Rate limit exceeded. Wait 1 minute."   | Wait before sending more        |
| 500        | "Server error. Please try again."       | Retry or contact support        |
| 503        | "PAS Root unavailable. Check status."   | Check service health (Dashboard)|

---

**End of PRD**

---

## Review Checklist

- [ ] User workflows cover common use cases?
- [ ] API endpoints clearly defined?
- [ ] Security considerations addressed?
- [ ] Performance targets realistic?
- [ ] UI/UX mockups clear?
- [ ] Integration with existing HMI feasible?
- [ ] Testing strategy comprehensive?
- [ ] Rollout plan achievable in 4 weeks?

**Ready for implementation!** 🚀
