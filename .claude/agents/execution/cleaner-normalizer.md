---
name: cleaner-normalizer
display_name: Cleaner/Normalizer
role: exec
tier: 1
mode: task
parent: manager-data
capabilities:
  - deduplication
  - normalization
  - encoding_fixes
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

# Cleaner/Normalizer
Deduplicates, normalizes, fixes encoding issues in data.
