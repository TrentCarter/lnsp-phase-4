---
name: change-control
display_name: Change Control (CM)
role: exec
tier: 1
mode: task
parent: manager-devsecops
capabilities:
  - pr_lifecycle
  - label_management
  - changelog_generation
transports:
  - rpc
rights:
  filesystem: rw
  git: true
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
---

# Change Control (CM)
Manages PR lifecycle (labels, merging), generates changelogs.
