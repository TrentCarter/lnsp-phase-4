---
name: directional-adapter-fitter
display_name: Directional Adapter Fitter
role: exec
tier: 1
mode: task
parent: manager-models
capabilities:
  - adapter_fitting
  - domain_specialization
transports:
  - rpc
rights:
  filesystem: rw
  python: true
  docker: true
resources:
  token_budget:
    target_ratio: 0.40
    hard_max_ratio: 0.70
  cpu_cores: 2
  memory_mb: 4096
  gpu_count: 1
  gpu_memory_mb: 4096
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Directional Adapter Fitter
Fits domain-specific directional adapters.
