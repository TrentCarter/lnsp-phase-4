
import importlib
import pytest

CANDIDATES = ("redact_text","scrub_secrets","sanitize","redact")

def _find_callable(mod):
    for name in CANDIDATES:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None

def test_redaction_masks_keys():
    try:
        mod = importlib.import_module("tools.aider_rpc.redact")
    except ModuleNotFoundError:
        pytest.skip("redact module not found")
    fn = _find_callable(mod)
    if fn is None:
        pytest.skip("No redaction function found")
    sample = 'OPENAI_API_KEY="sk-ABCDEF1234567890" and password=SuperSecretPass123'
    out = fn(sample)
    # Verify secrets are redacted
    assert "ABCDEF1234567890" not in out, "Secret API key should be redacted"
    assert "SuperSecretPass123" not in out, "Password should be redacted"
    # Verify key prefix is preserved for debugging (pattern: "OPENAI_[REDACTED]")
    assert "OPENAI" in out, "Key name prefix should be preserved for debugging"
