---
name: cost-optimizer
display_name: Cost Optimizer
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - cost_analysis
  - resource_optimization
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
  port: 6115
  status: not_implemented
---

# Cost Optimizer
Analyzes spending patterns and optimizes resource usage.
