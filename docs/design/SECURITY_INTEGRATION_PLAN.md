# Security Integration Plan: Auth, Secrets, Sandboxing

**Status**: Design Document (Pre-Phase-4)
**Version**: 2025-11-07-001
**Owner**: PAS/PEX Integration Team

---

## 1. Authentication & Authorization (AuthN/AuthZ)

### 1.1 Service Account Model

**Core Principle**: Every component (PEX, PAS Executors, PLMS, KPI Validators) operates as a **service account** with explicit scopes.

**Scopes (JWT claims)**:
- `pex.start` - Can initiate project execution
- `pas.submit` - Can submit jobcards to PAS
- `pas.pause` - Can pause/resume runs
- `pas.read_status` - Can read run status
- `plms.approve` - Can approve budget/estimates
- `plms.read_metrics` - Can read metrics
- `kpi.override` - Can override lane KPI thresholds
- `artifact.read` - Can read artifacts
- `artifact.write` - Can write artifacts

### 1.2 Token Format (JWT)

```json
{
  "iss": "lnsp-auth",
  "sub": "service:pex",
  "aud": "lnsp-services",
  "exp": 1699564800,
  "iat": 1699478400,
  "scopes": ["pex.start", "pas.submit", "pas.read_status", "artifact.write"],
  "run_id": "abc123",
  "project_id": "42"
}
```

**Token Issuance**:
- Service accounts get 24-hour tokens
- Run-scoped tokens expire when run completes + 1 hour
- Tokens signed with RS256 (asymmetric, rotate keys quarterly)

### 1.3 JWKS (JSON Web Key Set)

**Location**: `artifacts/jwks/public_keys.json`

```json
{
  "keys": [
    {
      "kty": "RSA",
      "kid": "2025-11-07-001",
      "use": "sig",
      "alg": "RS256",
      "n": "...",
      "e": "AQAB"
    }
  ]
}
```

**Key Rotation**:
- Generate new key pair quarterly
- Keep previous key for 7 days (grace period)
- All services fetch JWKS on startup and cache with TTL=1h

### 1.4 Implementation Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| Phase 4, Week 1 | Generate JWKS key pair | `artifacts/jwks/` |
| Phase 4, Week 1 | Implement token issuer | `services/auth/token_issuer.py` |
| Phase 4, Week 2 | Add JWT middleware to PAS/PLMS | `@require_scopes()` decorator |
| Phase 4, Week 2 | Wire service accounts | Each service gets token on startup |

**No API Keys**: All auth is JWT-based. No hardcoded credentials in env files.

---

## 2. Secrets Handling

### 2.1 Threat Model

**Risks**:
1. Secrets in PRD/prompts sent to LLMs → leaked in provider logs
2. Secrets in artifacts uploaded to artifact store → leaked to unauthorized users
3. Secrets in receipts/telemetry → leaked in observability dashboards
4. Secrets in git commits → leaked in version control

### 2.2 Defense-in-Depth Strategy

**Layer 1: Secret Detection (Pre-Flight)**
- Run regex patterns on all outbound prompts/artifacts
- Patterns: API keys, passwords, tokens, private keys, connection strings
- Denylist: `.env`, `secrets/`, `*.pem`, `*.key`, `credentials.json`

**Layer 2: Vault Integration**
- Store secrets in HashiCorp Vault or AWS Secrets Manager
- PEX/PAS fetch secrets at runtime via scoped tokens
- Secrets never written to disk or included in prompts

**Layer 3: Redaction (Last Resort)**
- If secret detected post-hoc, redact using `[REDACTED:ENV_VAR_NAME]`
- Log redaction event to audit log
- Halt outbound transmission until human approves

**Layer 4: Audit Trail**
- All secret access logged to append-only audit log
- Includes: timestamp, service account, secret name, context (run_id, task_id)
- Weekly review for anomalies

### 2.3 Redaction Patterns (Regex)

```python
REDACTION_PATTERNS = [
    r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
    r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]+)['\"]?",
    r"(?i)(token|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})['\"]?",
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"postgres://[^:]+:([^@]+)@",  # PostgreSQL connection string with password
    r"mongodb://[^:]+:([^@]+)@",   # MongoDB connection string
]
```

**Usage**:
```python
def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Redact secrets and return (redacted_text, secret_types)."""
    found_secrets = []
    for pattern in REDACTION_PATTERNS:
        if re.search(pattern, text):
            text = re.sub(pattern, lambda m: f"[REDACTED:{m.group(1).upper()}]", text)
            found_secrets.append(pattern)
    return text, found_secrets
```

### 2.4 Vault Integration (Phase 4)

**Vault Paths**:
- `secret/lnsp/postgres` → PostgreSQL connection string
- `secret/lnsp/neo4j` → Neo4j credentials
- `secret/lnsp/anthropic` → Anthropic API key
- `secret/lnsp/openai` → OpenAI API key

**Access Pattern**:
```python
import hvac

client = hvac.Client(url="http://localhost:8200", token=SERVICE_TOKEN)
secret = client.secrets.kv.v2.read_secret_version(path="lnsp/postgres")
pg_conn_str = secret["data"]["data"]["connection_string"]
```

**Service Token Scopes**: Read-only, path-restricted (e.g., `secret/lnsp/*`)

### 2.5 Implementation Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| Phase 4, Week 3 | Deploy Vault (local dev) | `docker-compose.vault.yml` |
| Phase 4, Week 3 | Implement secret scanner | `services/pas/security/secret_scanner.py` |
| Phase 4, Week 4 | Wire vault to PAS/PEX | All secrets fetched from vault |
| Phase 4, Week 4 | Add audit logging | `artifacts/logs/secret_access.log` |

---

## 3. Sandboxing & Command Allowlists

### 3.1 Execution Sandboxing

**Goal**: Isolate PAS executors so a compromised task cannot:
- Access files outside workspace roots
- Execute arbitrary commands (only allowlisted)
- Make outbound network connections (unless policy permits)
- Escalate privileges or modify system configuration

### 3.2 Linux Sandboxing (via Bubblewrap)

**Bubblewrap** (`bwrap`) provides lightweight containerization without Docker overhead.

**Example Sandbox Profile (Code-Impl lane)**:
```bash
#!/bin/bash
bwrap \
  --ro-bind /usr /usr \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /bin /bin \
  --ro-bind /sbin /sbin \
  --bind "$WORKSPACE" /workspace \
  --tmpfs /tmp \
  --proc /proc \
  --dev /dev \
  --unshare-all \
  --share-net \
  --die-with-parent \
  --setenv PATH /usr/bin:/bin \
  --chdir /workspace \
  /usr/bin/python3 "$@"
```

**Key Properties**:
- Read-only system directories (`/usr`, `/bin`, etc.)
- Read-write workspace bind mount (`$WORKSPACE`)
- No network by default (use `--unshare-net` for stricter lanes)
- Process dies if parent dies (prevents orphans)

### 3.3 macOS Sandboxing (via sandbox-exec)

**macOS Profile** (`code-impl.sb`):
```scheme
(version 1)
(deny default)
(allow file-read* (subpath "/usr"))
(allow file-read* (subpath "/System"))
(allow file-read* (subpath "/Library"))
(allow file-read-write* (subpath "/workspace"))
(allow process-exec (literal "/usr/bin/python3"))
(allow sysctl-read)
(allow mach-lookup (global-name "com.apple.system.logger"))
```

**Usage**:
```bash
sandbox-exec -f code-impl.sb python3 task.py
```

### 3.4 Command Allowlist Enforcement

**Pre-Execution Check**:
```python
def enforce_allowlist(command: list[str], lane: str) -> bool:
    """Check if command is allowed for lane."""
    allowlist = load_allowlist(f"services/pas/executors/allowlists/{lane}.yaml")

    # Check if command matches any allow pattern
    for allowed_pattern in allowlist["commands"]["allow"]:
        if matches_pattern(command, allowed_pattern):
            # Check deny list (deny takes precedence)
            for denied_pattern in allowlist["commands"]["deny"]:
                if matches_pattern(command, denied_pattern):
                    raise SecurityError(f"Command denied: {command} (matches deny pattern)")
            return True

    raise SecurityError(f"Command not in allowlist for lane {lane}: {command}")
```

**Pattern Matching**:
- `["pytest", "-q", "*"]` → allows `pytest -q tests/`
- `["bash", "-c", "*"]` → **denied** (no raw shell)
- `["git", "push"]` → allows `git push` (exact match)

### 3.5 File Access Controls

**Workspace Roots** (from allowlist):
```yaml
workspace_roots:
  - "./"
  - "src/"
  - "apps/"
```

**Enforcement**:
```python
def validate_file_access(path: str, lane: str) -> bool:
    """Check if file access is allowed for lane."""
    allowlist = load_allowlist(f"services/pas/executors/allowlists/{lane}.yaml")

    # Resolve to absolute path
    abs_path = Path(path).resolve()

    # Check if within any workspace root
    for root in allowlist["workspace_roots"]:
        if abs_path.is_relative_to(Path(root).resolve()):
            # Check deny globs
            for deny_glob in allowlist["file_globs"]["deny"]:
                if abs_path.match(deny_glob):
                    raise SecurityError(f"File access denied: {path} (matches deny glob)")
            # Check allow globs
            for allow_glob in allowlist["file_globs"]["allow"]:
                if abs_path.match(allow_glob):
                    return True

    raise SecurityError(f"File outside workspace roots: {path}")
```

### 3.6 Network Isolation

**Per-Lane Network Policy**:
```yaml
network:
  outbound: false  # Block all outbound connections
```

**Enforcement (Linux)**:
```bash
# Use network namespace isolation
bwrap --unshare-net ...  # No network access
bwrap --share-net ...    # Host network access (if policy allows)
```

**Enforcement (macOS)**:
```scheme
(deny network-outbound)  # Block all outbound
(allow network-outbound (remote ip "127.0.0.1:*"))  # Allow localhost only
```

### 3.7 Resource Limits (cgroups v2)

**Per-Task Resource Caps**:
```yaml
limits:
  max_cpu_pct: 80       # 80% CPU cap
  max_mem_mb: 8192      # 8GB RAM cap
  max_exec_seconds: 900 # 15 min timeout
  max_parallel: 2       # Max 2 concurrent tasks per lane
```

**Enforcement (Linux cgroups v2)**:
```python
import subprocess

def run_with_limits(command: list[str], limits: dict):
    """Run command with resource limits."""
    # Create cgroup
    cgroup_path = f"/sys/fs/cgroup/lnsp/{task_id}"
    os.makedirs(cgroup_path, exist_ok=True)

    # Set CPU limit
    cpu_max = int(limits["max_cpu_pct"] * 1000)  # Convert to microseconds
    with open(f"{cgroup_path}/cpu.max", "w") as f:
        f.write(f"{cpu_max} 100000")

    # Set memory limit
    mem_bytes = limits["max_mem_mb"] * 1024 * 1024
    with open(f"{cgroup_path}/memory.max", "w") as f:
        f.write(str(mem_bytes))

    # Run command in cgroup
    subprocess.run(
        ["cgexec", "-g", f"cpu,memory:lnsp/{task_id}", *command],
        timeout=limits["max_exec_seconds"]
    )
```

### 3.8 Implementation Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| Phase 4, Week 2 | Implement allowlist validator | `services/pas/executors/allowlist_validator.py` |
| Phase 4, Week 2 | Add bwrap profiles (Linux) | `services/pas/executors/sandbox/*.bwrap` |
| Phase 4, Week 3 | Add sandbox-exec profiles (macOS) | `services/pas/executors/sandbox/*.sb` |
| Phase 4, Week 3 | Wire sandboxing to PAS | All tasks run sandboxed |
| Phase 4, Week 4 | Add cgroups v2 resource limits | CPU/mem limits enforced |

---

## 4. Disaster Recovery & Replay

### 4.1 Replay from Passport

**Passport Contents**:
```json
{
  "run_id": "abc123",
  "project_id": "42",
  "provider_matrix_json": { /* model versions */ },
  "env_snapshot": { /* critical env vars */ },
  "prd_sha256": "...",
  "git_commit": "...",
  "allowlist_policy_version": "2025-11-07-001",
  "artifact_manifest": ["path/to/file1", "path/to/file2"]
}
```

**Recovery Script** (`scripts/replay_from_passport.sh`):
```bash
#!/bin/bash
set -euo pipefail

RUN_ID="$1"

echo "Replaying run: $RUN_ID"

# 1. Fetch passport from registry
PASSPORT=$(sqlite3 artifacts/registry/registry.db \
  "SELECT passport_json FROM project_runs WHERE run_id='$RUN_ID';")

# 2. Restore git state
GIT_COMMIT=$(echo "$PASSPORT" | jq -r '.git_commit')
git checkout "$GIT_COMMIT"

# 3. Restore env snapshot
ENV_SNAPSHOT=$(echo "$PASSPORT" | jq -r '.env_snapshot')
echo "$ENV_SNAPSHOT" | jq -r 'to_entries[] | "\(.key)=\(.value)"' > .env.replay

# 4. Restore artifacts (from backup)
ARTIFACT_MANIFEST=$(echo "$PASSPORT" | jq -r '.artifact_manifest[]')
for artifact in $ARTIFACT_MANIFEST; do
  cp "artifacts/backups/$RUN_ID/$artifact" "$artifact"
done

# 5. Re-submit to PAS with same passport
curl -X POST http://localhost:6100/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: replay-$RUN_ID" \
  -d "$PASSPORT"

echo "✅ Replay submitted: run_id=$RUN_ID"
```

### 4.2 Nightly Backups

**Backup Script** (`scripts/backup_artifacts.sh`):
```bash
#!/bin/bash
DATE=$(date +%Y-%m-%d)
tar -czf "backups/artifacts-$DATE.tar.gz" artifacts/
tar -czf "backups/registry-$DATE.tar.gz" artifacts/registry/registry.db
# Retain 30 days
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

**Cron Job**:
```cron
0 2 * * * /path/to/scripts/backup_artifacts.sh
```

---

## 5. Acceptance Tests (Before Phase 4)

**Run these tests to verify security foundations**:

1. **Token Validation**
   ```bash
   # Generate test token
   python services/auth/token_issuer.py --sub service:test --scopes pex.start

   # Verify signature
   python services/auth/token_validator.py --token <TOKEN>
   ```

2. **Secret Scanner**
   ```bash
   # Create test file with secret
   echo "api_key=sk-1234567890abcdef" > /tmp/test.txt

   # Run scanner
   python services/pas/security/secret_scanner.py /tmp/test.txt
   # Expected: ❌ Secret detected: API_KEY
   ```

3. **Sandbox Escape Test**
   ```bash
   # Try to escape sandbox
   bwrap --ro-bind /usr /usr --bind /tmp /tmp \
     --unshare-all /bin/bash -c "ls /"
   # Expected: Only /usr and /tmp visible
   ```

4. **Allowlist Enforcement**
   ```python
   # Try blocked command
   enforce_allowlist(["curl", "https://evil.com"], "Code-Impl")
   # Expected: SecurityError (curl not in allowlist)
   ```

5. **Replay Test**
   ```bash
   # Replay completed run
   bash scripts/replay_from_passport.sh abc123
   # Expected: Run resubmitted with same passport
   ```

---

## 6. Security Checklist (Before Production)

- [ ] JWT signing keys generated and stored securely
- [ ] JWKS endpoint accessible to all services
- [ ] Service accounts created with minimal scopes
- [ ] Token expiration enforced (24h for service accounts)
- [ ] Secret scanner integrated into PAS submission pipeline
- [ ] Vault deployed and all secrets migrated
- [ ] Audit log configured (append-only, weekly review)
- [ ] Allowlists validated for all 7 lanes
- [ ] Sandbox profiles tested on Linux and macOS
- [ ] Resource limits (cgroups v2) working correctly
- [ ] Network isolation tested (no outbound when policy=false)
- [ ] Replay script tested on 3 historical runs
- [ ] Nightly backups running and retention verified
- [ ] Security review completed (external audit recommended)

---

## 7. References

- PEX System Prompt: `docs/contracts/PEX_SYSTEM_PROMPT.md`
- Allowlists: `services/pas/executors/allowlists/*.yaml`
- Model Broker Policy: `services/pas/policy/model_broker.schema.json`
- JWKS: `artifacts/jwks/public_keys.json`
- Audit Log: `artifacts/logs/secret_access.log`
