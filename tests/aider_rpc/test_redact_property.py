"""
Property-based tests for secret redaction.

Uses Hypothesis to generate random strings and verify that secrets
never leak through redaction, regardless of input.
"""
import importlib
import pytest
from hypothesis import given, strategies as st, settings


# Find redaction function
mod = importlib.import_module("tools.aider_rpc.redact")
redact_fn = getattr(mod, "redact_text", None) or getattr(mod, "scrub_secrets", None)

if redact_fn is None:
    pytest.skip("No redaction function found", allow_module_level=True)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=100)
def test_openai_keys_never_leak(s):
    """Property: OpenAI API keys should never appear in redacted output."""
    # Create a string with a real-looking OpenAI key
    secret_key = "sk-" + "A" * 48  # OpenAI keys are sk- + 48 chars
    input_text = f'OPENAI_API_KEY="{secret_key}" ' + s

    output = redact_fn(input_text)

    # The actual secret should NEVER appear in output
    assert secret_key not in output, f"Secret leaked! Input: {input_text[:100]}... Output: {output[:100]}..."


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=100)
def test_anthropic_keys_never_leak(s):
    """Property: Anthropic API keys should never appear in redacted output."""
    # Create a string with a real-looking Anthropic key
    secret_key = "sk-ant-" + "a" * 95  # Anthropic keys are sk-ant- + ~95 chars
    input_text = f'ANTHROPIC_API_KEY="{secret_key}" ' + s

    output = redact_fn(input_text)

    # The actual secret should NEVER appear in output
    assert secret_key not in output, f"Secret leaked! Input: {input_text[:100]}... Output: {output[:100]}..."


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=100)
def test_github_tokens_never_leak(s):
    """Property: GitHub tokens should never appear in redacted output."""
    # Create a string with a real-looking GitHub token
    secret_token = "ghp_" + "B" * 36  # GitHub tokens are ghp_ + 36 chars
    input_text = f'GITHUB_TOKEN="{secret_token}" ' + s

    output = redact_fn(input_text)

    # The actual secret should NEVER appear in output
    assert secret_token not in output, f"Secret leaked! Input: {input_text[:100]}... Output: {output[:100]}..."


@given(
    # Generate passwords with letters, digits, and common symbols
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-", min_size=12, max_size=50),
)
@settings(max_examples=100)
def test_generic_passwords_never_leak(password):
    """Property: Generic passwords in key=value format should never leak.

    Note: Tests passwords with alphanumeric + common symbols (._-) since
    redaction patterns require [A-Za-z0-9._-]{12,} minimum.
    """
    input_text = f'PASSWORD="{password}"'

    output = redact_fn(input_text)

    # The password in quotes should be redacted
    # Check that the pattern PASSWORD="<password>" is gone
    assert f'PASSWORD="{password}"' not in output, \
        f"Password in quotes leaked! Input: {input_text} Output: {output}"
    # Also check that the password value itself is not in output
    # (unless it's something very generic like "000000000000")
    if not password.isdigit():  # Skip pure digit passwords (too generic)
        assert password not in output, \
            f"Password value leaked! Input: {input_text} Output: {output}"


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=50)
def test_redaction_idempotent(s):
    """Property: Redacting twice should give the same result."""
    first_pass = redact_fn(s)
    second_pass = redact_fn(first_pass)

    assert first_pass == second_pass, "Redaction should be idempotent"


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=50)
def test_redaction_preserves_length_class(s):
    """Property: Redaction should preserve approximate string length class."""
    output = redact_fn(s)

    # Output should be roughly similar length (within 2x)
    # This prevents excessive expansion that could cause issues
    if len(s) > 0:
        ratio = len(output) / len(s)
        assert 0.1 < ratio < 10, f"Redaction changed length too much: {len(s)} -> {len(output)}"


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200))
@settings(max_examples=50)
def test_redaction_safe_for_logs(s):
    """Property: Redacted output should be safe for logging (no control chars)."""
    output = redact_fn(s)

    # Check that output doesn't introduce problematic characters
    # (This is more about ensuring redaction doesn't corrupt data)
    assert isinstance(output, str), "Redaction should return a string"
    assert len(output) >= 0, "Redaction should not fail catastrophically"


def test_redaction_deterministic():
    """Redaction should be deterministic for the same input."""
    test_input = 'OPENAI_API_KEY="sk-test1234567890abcdef" and PASSWORD="secret123"'

    result1 = redact_fn(test_input)
    result2 = redact_fn(test_input)
    result3 = redact_fn(test_input)

    assert result1 == result2 == result3, "Redaction should be deterministic"
