# PRD: PAS Agent Chat Interface (Phase 2)

**Status:** 🔴 Not Started
**Priority:** P0 (Critical for LLM Chat UI functionality)
**Created:** 2025-11-13
**Owner:** TBD

---

## Problem Statement

PAS agents (Architect, Directors) currently only accept structured job cards via `/job_card` endpoint. Users cannot have conversational interactions with these agents through the LLM Chat UI, even though the agents have Aider RPC integration for filesystem access.

When users select a PAS agent in the LLM Chat UI (e.g., "🏛️ Architect 🔧"), they receive a **placeholder message** instead of an interactive chat experience:

```
**ARCHITECT Agent Ready** 🔧

I'm a PAS agent with filesystem access via Aider. However, the direct
chat interface for PAS agents is not yet implemented.

To use me:
1. Submit a task via Verdict CLI: `./bin/verdict send --title "Task" ...`
2. Monitor progress in TRON (http://localhost:6101/tron)
3. View results in artifacts/runs/
```

**Current Code:** `services/gateway/gateway.py:528-556` (TODO Phase 2)

---

## Solution: Conversational Chat Endpoints for PAS Agents

Add `/chat` endpoints to all PAS agents that enable:

1. **LLM-powered conversational interface** with streaming responses
2. **Tool calling for Aider RPC** (filesystem read/write, code editing)
3. **System prompts** that advertise filesystem capabilities
4. **SSE streaming** compatible with existing Gateway routing

---

## Architecture

### Current Flow (Job Cards Only)
```
User → Verdict CLI → Gateway → PAS Root → Architect → Job Card → ManagerExecutor → Aider RPC
```

### New Flow (Interactive Chat)
```
User → LLM Chat UI → Gateway → PAS Agent /chat → Agent LLM + Tool Calls → Aider RPC
                                                   ↓
                                            Stream responses via SSE
```

---

## Technical Design

### 1. Add `/chat` Endpoint to PAS Agents

**Affected Files:**
- `services/pas/architect/app.py`
- `services/pas/director_code/app.py`
- `services/pas/director_models/app.py`
- `services/pas/director_data/app.py`
- `services/pas/director_devsecops/app.py`
- `services/pas/director_docs/app.py`

**New Endpoint Signature:**
```python
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Interactive chat endpoint with Aider tool calling.

    Streams SSE events:
    - status_update: Planning, executing, complete
    - token: Streaming text from LLM
    - tool_call: When calling Aider RPC
    - usage: Token/cost tracking
    - done: Stream complete
    """
    return StreamingResponse(
        _stream_chat_response(request),
        media_type="text/event-stream"
    )
```

### 2. System Prompts with Tool Calling

Each agent needs a **system prompt** that:

**Example (Dir-Code):**
```
You are Dir-Code, a PAS Director responsible for the Code Lane.

CAPABILITIES:
- You have FULL FILESYSTEM ACCESS via Aider
- You can read, write, and modify files
- You can create new files and directories
- You can execute git operations (via Aider)

TOOLS AVAILABLE:
1. aider_edit(message: str, files: list[str]) -> str
   - Use this to make code changes, create files, refactor code
   - Example: aider_edit("Add docstring to main()", ["src/main.py"])

2. read_file(path: str) -> str
   - Read file contents
   - Example: read_file("src/main.py")

3. list_directory(path: str) -> list[str]
   - Browse directory contents
   - Example: list_directory("src/")

WORKFLOW:
1. Understand user request
2. If filesystem access needed → Use tools
3. Stream results back to user
4. Provide clear status updates

You are conversational and helpful. Always explain what you're doing.
```

### 3. LLM Integration with Tool Calling

**Use Ollama or Anthropic with function calling:**

```python
async def _stream_chat_response(request: ChatRequest):
    """Stream chat response with tool calling support"""

    # System prompt with tool definitions
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_id]
    tools = [
        {"name": "aider_edit", "description": "...", "parameters": {...}},
        {"name": "read_file", "description": "...", "parameters": {...}},
        {"name": "list_directory", "description": "...", "parameters": {...}},
    ]

    # Call LLM with tool support
    async for event in llm_stream_with_tools(
        messages=request.messages,
        system=system_prompt,
        tools=tools
    ):
        if event.type == "tool_call":
            # Execute tool (call Aider RPC)
            result = await execute_tool(event.tool_name, event.args)
            yield sse_event("tool_call", result)
        elif event.type == "text":
            yield sse_event("token", event.content)
        elif event.type == "done":
            yield sse_event("done", {})
```

### 4. Aider RPC Tool Execution

**Wrapper functions for tool calls:**

```python
async def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute tool call by routing to appropriate handler"""

    if tool_name == "aider_edit":
        return await call_aider_rpc(
            message=args["message"],
            files=args["files"]
        )
    elif tool_name == "read_file":
        return await read_file_safe(args["path"])
    elif tool_name == "list_directory":
        return await list_directory_safe(args["path"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

async def call_aider_rpc(message: str, files: list[str]) -> dict:
    """Call Aider RPC service (port 6130)"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:6130/edit",
            json={
                "message": message,
                "files": files,
                "run_id": current_run_id,
                "parent_log_id": current_log_id
            },
            timeout=300.0
        )
        return response.json()
```

### 5. Gateway Update

**Update `services/gateway/gateway.py:527-573`:**

Replace placeholder with actual agent call:

```python
async def _stream_pas_agent_response(request: ChatStreamRequest):
    """Stream chat response from PAS agent with Aider access"""

    agent_url = agent_endpoints.get(request.agent_id)

    # Route to agent's /chat endpoint
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{agent_url}/chat",
            json={
                "session_id": request.session_id,
                "message_id": request.message_id,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "model": request.model
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"  # Forward SSE events
```

---

## Implementation Plan

### Phase 2A: Proof of Concept (1 Agent)
**Goal:** Get one agent working end-to-end
**Agent:** `dir-code` (simplest, most common use case)

**Tasks:**
1. ✅ Add `/chat` endpoint to `services/pas/director_code/app.py`
2. ✅ Create system prompt with tool definitions
3. ✅ Implement LLM streaming with tool calling (Ollama + qwen2.5-coder)
4. ✅ Add tool execution wrappers (aider_edit, read_file, list_directory)
5. ✅ Update Gateway to route to actual endpoint (remove placeholder)
6. ✅ Test end-to-end: User message → Tool call → Aider RPC → Response

**Success Criteria:**
- User can ask: "What's in src/main.py?"
- Agent calls `read_file("src/main.py")`
- Agent streams file contents back
- User can ask: "Add a docstring to main()"
- Agent calls `aider_edit("Add docstring", ["src/main.py"])`
- Agent streams Aider's changes back

### Phase 2B: Replicate to All Agents
**Goal:** Enable chat for all 6 PAS agents

**Tasks:**
1. Copy `/chat` endpoint pattern to remaining 5 agents
2. Customize system prompts for each role:
   - Architect: "Coordinate across lanes"
   - Dir-Models: "Model training, evaluation"
   - Dir-Data: "Data ingestion, quality"
   - Dir-DevSecOps: "CI/CD, security"
   - Dir-Docs: "Documentation, reports"
3. Test each agent independently

### Phase 2C: Advanced Features (Optional)
**Goal:** Enhanced user experience

**Tasks:**
- [ ] Conversation history persistence (DB)
- [ ] Multi-turn tool calling (agent can chain tools)
- [ ] Rich media responses (code diffs, file trees)
- [ ] Approval gates for destructive operations
- [ ] Cost tracking integration (via Gateway)

---

## Data Models

### ChatRequest
```python
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message_id: str
    messages: list[ChatMessage]
    model: str = "ollama/qwen2.5-coder:7b-instruct"
```

### SSE Event Types
```python
{
    "type": "status_update",
    "status": "planning" | "executing" | "complete" | "error",
    "detail": "Human-readable message"
}

{
    "type": "token",
    "content": "Text chunk from LLM"
}

{
    "type": "tool_call",
    "tool": "aider_edit",
    "args": {"message": "...", "files": [...]},
    "result": "Tool execution result"
}

{
    "type": "usage",
    "usage": {"prompt_tokens": 123, "completion_tokens": 456},
    "cost_usd": 0.0
}

{
    "type": "done"
}
```

---

## Dependencies

### Required Services
- ✅ Aider RPC (port 6130) - Already running
- ✅ Ollama (port 11434) - Already running
- ✅ Model Pool Manager (port 8050) - Already running
- ✅ Gateway (port 6120) - Already running (correct version now)

### Required Libraries
- ✅ `httpx` - Already installed (async HTTP client)
- ✅ `fastapi` - Already installed
- ⚠️ LLM SDK with tool calling support:
  - Ollama: Native support (use `/api/chat` with `tools` parameter)
  - Anthropic: `pip install anthropic` (if using Claude)

---

## Testing Strategy

### Unit Tests
```bash
# Test individual tool functions
pytest tests/test_pas_agent_tools.py -v

# Test /chat endpoint
pytest tests/test_pas_agent_chat.py -v
```

### Integration Tests
```bash
# Test full flow: UI → Gateway → Agent → Aider RPC
pytest tests/test_chat_integration.py -v
```

### Manual Testing
1. Open LLM Chat UI: http://localhost:6101/llm
2. Select "📝 Dir-Code 🔧"
3. Send: "What files are in src/?"
4. Verify: Agent calls `list_directory("src/")`
5. Send: "Show me src/main.py"
6. Verify: Agent calls `read_file("src/main.py")`
7. Send: "Add a TODO comment at line 10"
8. Verify: Agent calls `aider_edit(...)`

---

## Risks & Mitigations

### Risk 1: LLM Tool Calling Quality
**Problem:** LLM may not reliably call tools correctly
**Mitigation:**
- Use Ollama qwen2.5-coder (good at code/tools)
- Provide clear tool descriptions and examples
- Add validation for tool arguments

### Risk 2: Aider RPC Timeout
**Problem:** Large edits may timeout (300s limit)
**Mitigation:**
- Stream progress updates during Aider execution
- Add timeout configuration per operation
- Implement cancellation support

### Risk 3: Security (Filesystem Access)
**Problem:** User could request dangerous operations
**Mitigation:**
- ✅ Aider RPC already has allowlist enforcement
- ✅ Command allowlist blocks dangerous ops
- Add approval gates for destructive operations (Phase 2C)

---

## Success Metrics

**Phase 2A (POC):**
- [ ] Dir-Code `/chat` endpoint responds to messages
- [ ] At least 1 successful tool call (read_file)
- [ ] At least 1 successful Aider edit
- [ ] Response streams correctly in UI

**Phase 2B (All Agents):**
- [ ] All 6 agents have working `/chat` endpoints
- [ ] Gateway routes to correct agent based on agent_id
- [ ] Users can switch between agents seamlessly

**Phase 2C (Advanced):**
- [ ] Conversation history persists across sessions
- [ ] Multi-turn tool calling works
- [ ] Cost tracking integrated

---

## References

- **Current Placeholder:** `services/gateway/gateway.py:528-556`
- **Aider RPC Docs:** `services/tools/aider_rpc/app.py`
- **Gateway Routing:** `services/gateway/gateway.py:374-409`
- **HMI Agent Selector:** `services/webui/templates/llm.html:1053-1080`
- **Ollama Tool Calling:** https://ollama.com/blog/tool-support
- **Anthropic Function Calling:** https://docs.anthropic.com/en/docs/build-with-claude/tool-use

---

## Next Steps

1. **Review this PRD** - Get approval from team
2. **Start Phase 2A** - Implement Dir-Code POC
3. **Test thoroughly** - Ensure quality before replication
4. **Replicate** - Copy pattern to remaining 5 agents
5. **Document** - Update user guides and session summaries
