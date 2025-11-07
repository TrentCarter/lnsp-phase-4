---
name: corpus-auditor
display_name: Corpus Auditor
role: exec
tier: 1
mode: task
parent: manager-data
capabilities:
  - source_checking
  - licensing_validation
  - stats_generation
transports:
  - rpc
rights:
  filesystem: rw
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

# Corpus Auditor
Validates data sources, checks licensing, generates dataset statistics.
