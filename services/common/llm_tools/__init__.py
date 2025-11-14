"""LLM Tools for Agent Communication"""
from .ask_parent_tool import (
    ASK_PARENT_TOOL_ANTHROPIC,
    ASK_PARENT_TOOL_GOOGLE,
    ASK_PARENT_TOOL_OLLAMA,
    get_ask_parent_tool,
    validate_ask_parent_args,
    get_system_prompt_with_ask_parent
)

__all__ = [
    "ASK_PARENT_TOOL_ANTHROPIC",
    "ASK_PARENT_TOOL_GOOGLE",
    "ASK_PARENT_TOOL_OLLAMA",
    "get_ask_parent_tool",
    "validate_ask_parent_args",
    "get_system_prompt_with_ask_parent"
]
