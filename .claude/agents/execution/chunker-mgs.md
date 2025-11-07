---
name: chunker-mgs
display_name: Chunker-MGS
role: exec
tier: 1
mode: task
parent: manager-data
capabilities:
  - sentence_banking
  - paragraph_banking
  - chunk_metadata
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

# Chunker-MGS
Creates sentence/paragraph banks with chunk metadata.
