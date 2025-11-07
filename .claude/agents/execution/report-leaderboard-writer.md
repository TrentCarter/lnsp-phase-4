---
name: report-leaderboard-writer
display_name: Report & Leaderboard Writer
role: exec
tier: 1
mode: task
parent: director-docs
capabilities:
  - report_generation
  - dashboard_creation
  - documentation
transports:
  - rpc
rights:
  filesystem: rw
  python: true
  git: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
approvals:
  required:
    - git_push
metadata:
  version: 1.0.0
---

# Report & Leaderboard Writer
Generates evaluation reports, dashboards, and documentation.
