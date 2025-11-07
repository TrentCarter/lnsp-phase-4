---
name: error-tracking-recovery
display_name: Error Tracking & Recovery
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - error_classification
  - recovery_trigger
  - rollback
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
  port: 6112
  status: not_implemented
---

# Error Tracking & Recovery
Classifies errors (transient/permanent), triggers retries and rollbacks.
