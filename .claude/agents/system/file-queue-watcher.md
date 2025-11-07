---
name: file-queue-watcher
display_name: File Queue Watcher
role: system
tier: 1
mode: long
parent: registry
capabilities:
  - atomic_ldjson
  - inbox_outbox_management
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 512
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6110
  status: not_implemented
---

# File Queue Watcher
Monitors atomic LDJSON file queues for durable task persistence.
