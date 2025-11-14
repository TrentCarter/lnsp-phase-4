#!/usr/bin/env python3
"""
Ask Parent Tool - LLM Tool for Agent Communication

This tool allows Child agents (Directors, Managers) to ask clarifying
questions to their Parent agents when tasks are ambiguous.

Used in Phase 3 of Parent-Child Agent Chat Communications.
"""
from typing import Dict, Any, List


# Tool definition for Claude API (Anthropic format)
ASK_PARENT_TOOL_ANTHROPIC = {
    "name": "ask_parent",
    "description": """Ask a clarifying question to your parent agent and wait for an answer.

Use this tool when:
- The task is ambiguous or underspecified
- Multiple valid implementation approaches exist and you need user/parent preference
- You need additional context that only the parent has (budget details, policy constraints, etc.)
- Technical decisions that could affect architecture or other lanes

Do NOT use this when:
- The task has clear, standard best practices (use them)
- The answer is trivial or obvious from context
- You have explicit instructions in the metadata
- Asking would cause unnecessary delay for a non-critical decision

The parent agent has full visibility into:
- Original user requirements (PRD)
- Project-wide budget and timeline
- Cross-lane dependencies
- Policy constraints (security, compliance, etc.)

Your question will be sent to the parent, and this function will wait up to 30 seconds for a response.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The specific question to ask. Be clear, concise, and include context about why you're asking. Example: 'Should I refactor all authentication code or just the OAuth flow? The task says \"improve auth\" but doesn\\'t specify scope.'"
            },
            "urgency": {
                "type": "string",
                "enum": ["blocking", "important", "informational"],
                "description": "How critical is this answer? 'blocking' = cannot proceed without answer, 'important' = can proceed but answer affects approach, 'informational' = nice to know but not critical"
            },
            "context": {
                "type": "string",
                "description": "Additional context about your analysis so far. What have you already considered? What are the trade-offs you see? This helps the parent give a better answer."
            }
        },
        "required": ["question", "urgency"]
    }
}


# Tool definition for Google Gemini (function calling format)
ASK_PARENT_TOOL_GOOGLE = {
    "name": "ask_parent",
    "description": "Ask a clarifying question to your parent agent and wait for an answer. Use when the task is ambiguous or you need parent's guidance on approach, scope, or technical decisions.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The specific question to ask. Be clear and include context."
            },
            "urgency": {
                "type": "string",
                "enum": ["blocking", "important", "informational"],
                "description": "How critical is this answer?"
            },
            "context": {
                "type": "string",
                "description": "Additional context about your analysis so far and trade-offs you see."
            }
        },
        "required": ["question", "urgency"]
    }
}


# Tool definition for Ollama (simplified, follows Claude format)
ASK_PARENT_TOOL_OLLAMA = ASK_PARENT_TOOL_ANTHROPIC  # Ollama uses similar format


def get_ask_parent_tool(provider: str = "anthropic") -> Dict[str, Any]:
    """
    Get ask_parent tool definition for specific LLM provider

    Args:
        provider: One of "anthropic", "google", "ollama"

    Returns:
        Tool definition dict in provider-specific format
    """
    tools = {
        "anthropic": ASK_PARENT_TOOL_ANTHROPIC,
        "google": ASK_PARENT_TOOL_GOOGLE,
        "ollama": ASK_PARENT_TOOL_OLLAMA
    }

    if provider not in tools:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(tools.keys())}")

    return tools[provider]


def validate_ask_parent_args(args: Dict[str, Any]) -> bool:
    """
    Validate arguments for ask_parent tool call

    Args:
        args: Tool call arguments from LLM

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Required fields
    if "question" not in args:
        raise ValueError("ask_parent requires 'question' argument")

    if not args["question"].strip():
        raise ValueError("ask_parent 'question' cannot be empty")

    if "urgency" not in args:
        raise ValueError("ask_parent requires 'urgency' argument")

    # Validate urgency
    valid_urgency = ["blocking", "important", "informational"]
    if args["urgency"] not in valid_urgency:
        raise ValueError(f"ask_parent 'urgency' must be one of {valid_urgency}, got: {args['urgency']}")

    # Optional: validate question length (avoid token waste)
    if len(args["question"]) > 500:
        raise ValueError("ask_parent 'question' too long (max 500 chars)")

    return True


# System prompt addition for agents that have ask_parent tool
ASK_PARENT_SYSTEM_PROMPT_ADDITION = """

## Available Tool: ask_parent

You have access to the `ask_parent` tool which allows you to ask clarifying questions to your parent agent.

**When to use:**
- Task is ambiguous (e.g., "refactor auth" without specifying which files/scope)
- Multiple valid approaches exist and you need parent's preference
- Need budget/policy clarification that affects your approach
- Technical decision that could impact other lanes or architecture

**When NOT to use:**
- Clear task with standard best practices → just follow best practices
- Trivial or obvious answers → proceed with common sense
- You have explicit instructions in metadata → follow them
- Non-critical decisions that don't affect outcomes → make reasonable choice

**How to use:**
```
ask_parent(
    question="Which files should I refactor? Task says 'improve auth' but no specific scope.",
    urgency="blocking",  # Cannot proceed without answer
    context="I see auth code in src/auth.py, src/oauth.py, and src/sessions.py. Should I refactor all or focus on specific areas?"
)
```

The parent will respond within ~30 seconds. If no response, proceed with your best judgment and document assumptions.
"""


def get_system_prompt_with_ask_parent(base_prompt: str, provider: str = "anthropic") -> str:
    """
    Add ask_parent tool documentation to agent's system prompt

    Args:
        base_prompt: Existing system prompt
        provider: LLM provider (affects formatting)

    Returns:
        Enhanced system prompt with ask_parent documentation
    """
    return base_prompt + ASK_PARENT_SYSTEM_PROMPT_ADDITION
