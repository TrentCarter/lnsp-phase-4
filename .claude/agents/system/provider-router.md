---
name: provider-router
display_name: Provider Router
role: system
tier: 1
mode: long
parent: null
capabilities:
  - capability_matching
  - provider_selection
  - cost_optimization
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
  port: 6103
  service_type: fastapi
---

# Provider Router
Model provider selection and routing. Already implemented at port 6103.
