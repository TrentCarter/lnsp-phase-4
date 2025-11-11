#!/usr/bin/env python3
"""
LLM-powered title generator for Prime Directives.
Converts long goal descriptions into short, human-readable names.

Example:
  Input:  "Add **last updated** comment with today's date time to the top of readme.txt"
  Output: "Add Last Updated Header"
"""
import httpx
import json
from typing import Optional


def generate_short_title(goal: str, llm_endpoint: str = "http://localhost:11434") -> str:
    """
    Generate a short, human-readable title from a Prime Directive goal.

    Args:
        goal: The full Prime Directive goal text
        llm_endpoint: Ollama endpoint (default: localhost:11434)

    Returns:
        Short title (max 6 words, title case)
        Falls back to truncated goal if LLM unavailable

    Examples:
        >>> generate_short_title("Add docstrings to both functions in test_simple.py")
        "Add Function Docstrings"

        >>> generate_short_title("Add **last updated** comment with today's date time to the top of readme.txt")
        "Add Last Updated Header"
    """
    # Fallback: truncate goal if too long
    def fallback_title() -> str:
        if len(goal) <= 50:
            return goal
        return goal[:47] + "..."

    # Build LLM prompt
    prompt = f"""Generate a short, human-readable title (max 6 words) for this software task:

Task: {goal}

Title (6 words max, title case, no quotes):"""

    try:
        # Call Ollama API
        response = httpx.post(
            f"{llm_endpoint}/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": "You generate concise, professional titles for software tasks. Respond with ONLY the title, no quotes, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 20  # Short output
                }
            },
            timeout=10.0
        )

        if response.status_code != 200:
            return fallback_title()

        result = response.json()
        title_raw = result.get("message", {}).get("content", "").strip()

        # Clean up title (remove quotes, extra whitespace)
        title = title_raw.replace('"', '').replace("'", '').strip()

        # Validate: not empty, reasonable length
        if not title or len(title) > 100:
            return fallback_title()

        return title

    except Exception:
        # Silently fall back if LLM unavailable
        return fallback_title()


async def generate_short_title_async(goal: str, llm_endpoint: str = "http://localhost:11434") -> str:
    """
    Async version of generate_short_title.

    Args:
        goal: The full Prime Directive goal text
        llm_endpoint: Ollama endpoint (default: localhost:11434)

    Returns:
        Short title (max 6 words, title case)
        Falls back to truncated goal if LLM unavailable
    """
    # Fallback: truncate goal if too long
    def fallback_title() -> str:
        if len(goal) <= 50:
            return goal
        return goal[:47] + "..."

    # Build LLM prompt
    prompt = f"""Generate a short, human-readable title (max 6 words) for this software task:

Task: {goal}

Title (6 words max, title case, no quotes):"""

    try:
        # Call Ollama API (async)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{llm_endpoint}/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [
                        {"role": "system", "content": "You generate concise, professional titles for software tasks. Respond with ONLY the title, no quotes, no explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 20  # Short output
                    }
                }
            )

        if response.status_code != 200:
            return fallback_title()

        result = response.json()
        title_raw = result.get("message", {}).get("content", "").strip()

        # Clean up title (remove quotes, extra whitespace)
        title = title_raw.replace('"', '').replace("'", '').strip()

        # Validate: not empty, reasonable length
        if not title or len(title) > 100:
            return fallback_title()

        return title

    except Exception:
        # Silently fall back if LLM unavailable
        return fallback_title()
