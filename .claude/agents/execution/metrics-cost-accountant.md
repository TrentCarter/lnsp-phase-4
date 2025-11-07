---
name: metrics-cost-accountant
display_name: Metrics/Cost Accountant
role: exec
tier: 1
mode: task
parent: architect
capabilities:
  - receipt_generation
  - cost_tracking
  - metrics_collection
transports:
  - rpc
rights:
  filesystem: rw
  python: true
resources:
  token_budget:
    target_ratio: 0.20
    hard_max_ratio: 0.40
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Metrics/Cost Accountant
Generates cost receipts, tracks token/latency/$ metrics.
