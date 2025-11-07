---
name: model-version-manager
display_name: Model Version Manager
role: system
tier: 1
mode: long
parent: director-models
capabilities:
  - model_versioning
  - deployment_management
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6118
  status: not_implemented
---

# Model Version Manager
Manages model versions and deployment lifecycle.
