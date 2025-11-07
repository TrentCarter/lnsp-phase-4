---
name: gateway
display_name: Gateway
role: system
tier: 1
mode: long
parent: null
capabilities:
  - routing
  - cost_tracking
  - receipt_generation
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6120
  service_type: fastapi
---

# Gateway
Central routing hub with cost tracking. Already implemented at port 6120.
