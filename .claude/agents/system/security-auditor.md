---
name: security-auditor
display_name: Security Auditor
role: system
tier: 1
mode: long
parent: director-devsecops
capabilities:
  - vulnerability_scanning
  - dependency_checking
  - policy_audit
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6114
  status: not_implemented
---

# Security Auditor
Scans for vulnerabilities, audits dependencies, enforces security policies.
