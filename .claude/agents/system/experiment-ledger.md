---
name: experiment-ledger
display_name: Experiment Ledger
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - run_tracking
  - artifact_management
  - reproducibility
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
  port: 6107
  status: not_implemented
---

# Experiment Ledger
Tracks runs, artifacts, seeds, costs for reproducibility. SQLite-backed registry.
