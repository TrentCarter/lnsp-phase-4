---
name: qtower-trainer
display_name: Q-Tower Trainer
role: exec
tier: 1
mode: task
parent: manager-models
capabilities:
  - query_tower_training
  - model_training
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
  cpu_cores: 4
  memory_mb: 8192
  gpu_count: 1
  gpu_memory_mb: 8192
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Q-Tower Trainer
Trains retriever query tower models.
