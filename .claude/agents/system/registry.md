---
name: registry
display_name: Registry
role: system
tier: 1
mode: long
parent: null
capabilities:
  - service_discovery
  - heartbeat_tracking
  - ttl_management
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
  port: 6121
  service_type: fastapi
---

# Registry
Service registration and discovery. Already implemented at port 6121.
