#!/usr/bin/env python3
"""
Secrets Redaction for Aider RPC Logs and Diffs

Scrubs sensitive information from transcripts, diffs, and receipts before
persistence to prevent credential leakage.

Based on PAS PRD security requirements (Phase 2 hardening).
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern


# Common secret patterns (ordered by specificity)
SECRET_PATTERNS: List[tuple[str, Pattern]] = [
    # API Keys & Tokens
    ("OPENAI_API_KEY", re.compile(r"sk-[a-zA-Z0-9]{48}", re.IGNORECASE)),
    ("ANTHROPIC_API_KEY", re.compile(r"sk-ant-[a-zA-Z0-9\-]{95,}", re.IGNORECASE)),
    ("GITHUB_TOKEN", re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE)),
    ("GITLAB_TOKEN", re.compile(r"glpat-[a-zA-Z0-9\-_]{20}", re.IGNORECASE)),
    ("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE)),
    ("AWS_SECRET_KEY", re.compile(r"[A-Za-z0-9/+=]{40}", re.IGNORECASE)),
    ("SLACK_TOKEN", re.compile(r"xox[baprs]-[0-9a-zA-Z\-]+", re.IGNORECASE)),
    ("STRIPE_KEY", re.compile(r"sk_live_[a-zA-Z0-9]{24,}", re.IGNORECASE)),
    ("STRIPE_TEST_KEY", re.compile(r"sk_test_[a-zA-Z0-9]{24,}", re.IGNORECASE)),

    # Generic patterns (broader)
    ("BEARER_TOKEN", re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+", re.IGNORECASE)),
    ("JWT", re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", re.IGNORECASE)),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----", re.MULTILINE)),

    # Environment variable assignments (key=value with secret keywords)
    ("ENV_SECRET", re.compile(r"(?:API|SECRET|TOKEN|KEY|PASSWORD|PASSWD|CREDENTIAL)[\w]*\s*[:=]\s*['\"]?([A-Za-z0-9._\-]{12,})", re.IGNORECASE)),

    # Environment variables with secret suffixes (catches _KEY, _TOKEN, _SECRET)
    ("ENV_VAR_SECRET", re.compile(r"\w+(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_PASSWD)\s*[:=]\s*['\"]?([A-Za-z0-9._\-]{8,})", re.IGNORECASE)),

    # Connection strings
    ("DB_CONNECTION", re.compile(r"(postgres|mysql|mongodb)://[^:]+:([^@]+)@", re.IGNORECASE)),

    # Emails (borderline - maybe keep for debugging?)
    # ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
]

# File patterns that commonly contain secrets
SENSITIVE_FILE_PATTERNS = [
    r"\.env$",
    r"\.env\.",
    r"credentials\.json$",
    r"secrets\.",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"config/database\.yml$",
    r"\.aws/credentials$",
]

# Replacement masks
REDACTED_MASK = "[REDACTED]"
PARTIAL_MASK_CHARS = 4  # Show first N chars for debugging


def redact_text(text: str, partial: bool = False) -> str:
    """
    Redact secrets from arbitrary text.

    Args:
        text: Input text (log line, diff, etc.)
        partial: If True, show first PARTIAL_MASK_CHARS of secrets for debugging

    Returns:
        Redacted text
    """
    result = text

    for label, pattern in SECRET_PATTERNS:
        def replace_match(m: re.Match) -> str:
            # Capture the full match or the secret group if available
            secret = m.group(2) if m.groups() and len(m.groups()) >= 2 else m.group(0)

            if partial and len(secret) > PARTIAL_MASK_CHARS:
                return secret[:PARTIAL_MASK_CHARS] + REDACTED_MASK
            return REDACTED_MASK

        result = pattern.sub(replace_match, result)

    return result


def redact_dict(data: Dict, partial: bool = False) -> Dict:
    """
    Recursively redact secrets from a dictionary (e.g., JSON receipt).

    Args:
        data: Input dictionary
        partial: If True, show partial secrets

    Returns:
        New dictionary with secrets redacted
    """
    if not isinstance(data, dict):
        if isinstance(data, str):
            return redact_text(data, partial=partial)
        return data

    result = {}
    for key, value in data.items():
        # Redact known secret keys
        if any(pattern in key.lower() for pattern in ["password", "secret", "token", "key", "credential"]):
            if isinstance(value, str) and value:
                if partial:
                    result[key] = value[:PARTIAL_MASK_CHARS] + REDACTED_MASK
                else:
                    result[key] = REDACTED_MASK
            else:
                result[key] = REDACTED_MASK
        elif isinstance(value, dict):
            result[key] = redact_dict(value, partial=partial)
        elif isinstance(value, list):
            result[key] = [redact_dict(item, partial=partial) if isinstance(item, dict) else redact_text(str(item), partial=partial) for item in value]
        elif isinstance(value, str):
            result[key] = redact_text(value, partial=partial)
        else:
            result[key] = value

    return result


def is_sensitive_file(path: str) -> bool:
    """
    Check if a file path matches sensitive file patterns.

    Args:
        path: File path to check

    Returns:
        True if file is likely to contain secrets
    """
    for pattern in SENSITIVE_FILE_PATTERNS:
        if re.search(pattern, path, re.IGNORECASE):
            return True
    return False


def redact_diff(diff: str, partial: bool = False) -> str:
    """
    Redact secrets from a git diff.

    Args:
        diff: Git diff text
        partial: If True, show partial secrets

    Returns:
        Redacted diff
    """
    lines = diff.split("\n")
    result_lines = []

    for line in lines:
        # Keep diff headers intact
        if line.startswith("diff --git") or line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            result_lines.append(line)
        elif line.startswith("+") or line.startswith("-"):
            # Redact added/removed lines
            prefix = line[0]
            content = line[1:]
            redacted_content = redact_text(content, partial=partial)
            result_lines.append(prefix + redacted_content)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def redact_env_file(content: str) -> str:
    """
    Redact secrets from .env file content while preserving structure.

    Args:
        content: .env file content

    Returns:
        Redacted content with variable names intact
    """
    lines = content.split("\n")
    result_lines = []

    for line in lines:
        stripped = line.strip()

        # Preserve comments and empty lines
        if not stripped or stripped.startswith("#"):
            result_lines.append(line)
            continue

        # Redact variable values
        if "=" in line:
            key, value = line.split("=", 1)
            result_lines.append(f"{key}={REDACTED_MASK}")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


if __name__ == "__main__":
    # Self-test
    test_cases = [
        ("OPENAI_API_KEY=sk-abc123def456ghi789jkl012mno345pqr678stu901vwx234", "OPENAI_API_KEY"),
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "JWT"),
        ("postgres://user:secretpass123@localhost:5432/db", "DB_CONNECTION"),
        ("Just some regular text without secrets", None),
    ]

    print("Redaction Self-Test:")
    print("\nFull redaction:")
    for text, expected_label in test_cases:
        redacted = redact_text(text, partial=False)
        has_redacted = REDACTED_MASK in redacted
        expected = expected_label is not None
        status = "✓" if has_redacted == expected else "✗"
        print(f"  {status} {text[:50]:<52} => {redacted[:80]}")

    print("\nPartial redaction (debugging):")
    for text, expected_label in test_cases:
        redacted = redact_text(text, partial=True)
        print(f"  {text[:50]:<52} => {redacted[:80]}")
