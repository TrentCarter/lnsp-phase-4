---
name: evaluator-gatekeeper
display_name: Evaluator & Gatekeeper
role: exec
tier: 1
mode: task
parent: director-models
capabilities:
  - evaluation
  - scoring
  - threshold_gating
transports:
  - rpc
rights:
  filesystem: rw
  python: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 2
  memory_mb: 4096
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Evaluator & Gatekeeper
Evaluates models, scores performance, gates releases.
