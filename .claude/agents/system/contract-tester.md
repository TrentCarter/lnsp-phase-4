---
name: contract-tester
display_name: Contract Tester
role: system
tier: 1
mode: task
parent: director-models
capabilities:
  - schema_validation
  - mini_replay
resources:
  token_budget:
    target_ratio: 0.20
    hard_max_ratio: 0.40
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6106
  status: not_implemented
---

# Contract Tester
Validates JSON schemas and performs mini-replay tests.
