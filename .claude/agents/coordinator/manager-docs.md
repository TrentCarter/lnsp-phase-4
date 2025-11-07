---
name: manager-docs
display_name: Manager (Docs Lane)
role: coord
tier: 1
mode: long
parent: director-docs
children:
  - report-leaderboard-writer
  - documentation-generator
capabilities:
  - task_breakdown
  - report_coordination
  - documentation_management
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

# Manager (Docs Lane)
Coordinates documentation and reporting agents.
