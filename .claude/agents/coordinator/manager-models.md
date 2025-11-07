---
name: manager-models
display_name: Manager (Models Lane)
role: coord
tier: 1
mode: long
parent: director-models
children:
  - qtower-trainer
  - reranker-trainer
  - directional-adapter-fitter
  - hard-negative-miner
capabilities:
  - task_breakdown
  - training_coordination
  - gpu_resource_management
resources:
  token_budget:
    target_ratio: 0.50
    hard_max_ratio: 0.75
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Manager (Models Lane)
Coordinates model training agents, manages GPU resources, tracks experiments.
