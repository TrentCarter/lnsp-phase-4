---
name: graph-builder
display_name: Graph Builder
role: exec
tier: 1
mode: task
parent: manager-data
capabilities:
  - kg_construction
  - link_generation
transports:
  - rpc
rights:
  filesystem: rw
  python: true
  sql: true
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

# Graph Builder
Builds knowledge graph from chunks, generates links.
