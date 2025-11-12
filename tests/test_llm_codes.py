#!/usr/bin/env python3
"""
Test LLM code extraction from model names
"""
import sys
import pathlib

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from services.common.comms_logger import get_llm_code


def test_llm_codes():
    """Test LLM code extraction for various models"""
    test_cases = [
        # Claude models
        ("anthropic/claude-4.5-sonnet", "CLD450"),
        ("anthropic/claude-3-opus", "CLD30O"),
        ("anthropic/claude-3-sonnet", "CLD30S"),
        ("anthropic/claude-3-haiku", "CLD30H"),
        ("anthropic/claude-3.5-sonnet", "CLD350"),
        ("anthropic/claude-3.7-sonnet", "CLD370"),

        # GPT models
        ("openai/gpt-5", "GPT500"),
        ("openai/gpt-4.5-turbo", "GPT450"),
        ("openai/gpt-4", "GPT400"),
        ("openai/gpt-3.5-turbo", "GPT350"),

        # Gemini models
        ("google/gemini-2.5-flash", "GMI250"),
        ("google/gemini-2.0-flash", "GMI200"),
        ("google/gemini-1.5-pro", "GMI150"),

        # Qwen models
        ("ollama/qwen2.5-coder:7b-instruct", "QWE250"),
        ("ollama/qwen2-coder", "QWE200"),

        # Llama models
        ("ollama/llama3.1:8b", "LMA310"),
        ("ollama/llama3", "LMA300"),
        ("ollama/llama2", "LMA200"),

        # Edge cases
        (None, "------"),
        ("-", "------"),
        ("", "------"),
    ]

    print("Testing LLM code extraction:")
    print("=" * 80)

    all_passed = True
    for model, expected in test_cases:
        result = get_llm_code(model)
        status = "✓" if result == expected else "✗"

        model_display = str(model) if model is not None else "None"
        if result != expected:
            all_passed = False
            print(f"{status} FAIL: {model_display:50} → {result:5} (expected {expected})")
        else:
            print(f"{status} PASS: {model_display:50} → {result:5}")

    print("=" * 80)

    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(test_llm_codes())
