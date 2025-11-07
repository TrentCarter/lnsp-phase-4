---
name: heartbeat-monitor
display_name: Heartbeat Monitor
role: system
tier: 1
mode: long
parent: registry
capabilities:
  - health_checking
  - alert_generation
  - auto_recovery
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 30
metadata:
  version: 1.0.0
  port: 6109
  service_type: fastapi
---

# Heartbeat Monitor
Service health monitoring and auto-recovery. Already implemented at port 6109.
