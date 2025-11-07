---
name: compliance-checker
display_name: Compliance Checker
role: exec
tier: 1
mode: task
parent: director-devsecops
capabilities:
  - pii_verification
  - regulatory_compliance
transports:
  - rpc
rights:
  filesystem: read
  python: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Compliance Checker
Verifies outputs against regulatory standards (PII, etc).
