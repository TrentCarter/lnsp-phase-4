# Next Session Question

## User Question (from 2025-11-13 Session)

**Q:** How do we use LLM Chat through Aider so it can see the file system? Do we need another dropdown showing the connections? i.e. CHAT -> LLM or something like CHAT -> AIDER -> LLM?

## Context

This question came up during the session where we:
1. Fixed LLM Chat UI layout issues
2. Started PAS tier agents (Architect + 5 Directors)
3. Registered agents in Service Registry
4. Made all 6 agents available in the LLM Chat dropdown

The user is now asking about integrating Aider into the LLM Chat interface so that agents can interact with the filesystem through Aider instead of just pure LLM chat.

## Related Systems

- **LLM Chat Interface**: `services/webui/templates/llm.html` - Direct chat with agents
- **Aider-LCO**: Port 6130 - Aider RPC wrapper for code editing with filesystem access
- **Gateway**: Port 6120 - Prime Directive submission endpoint
- **PAS Root**: Port 6100 - Orchestrator that routes to agents

## Possible Approaches to Consider

1. **New Dropdown**: Add a "Mode" selector with options like:
   - "Chat Only" (current behavior - pure LLM conversation)
   - "Chat + Aider" (LLM with filesystem/git access via Aider)
   - "Prime Directive" (submit structured task via Gateway)

2. **Agent Capabilities**: Check agent capabilities from Registry and show Aider option only for agents that support code editing

3. **Unified Interface**: Always route through Aider when sending messages to code-related agents (Director-Code, Managers, Programmers)

4. **Separate Interface**: Keep LLM Chat as-is for conversations, and use /pas-task or Verdict CLI for filesystem operations

## Questions to Answer

- Should LLM Chat support filesystem operations, or keep it conversation-only?
- How do we indicate to the user whether they're in "chat mode" vs "code mode"?
- Do we need to show the file context/working directory when Aider is active?
- Should the Prime Directive submission flow be integrated into LLM Chat or kept separate?
