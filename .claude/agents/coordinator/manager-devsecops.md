---
name: manager-devsecops
display_name: Manager (DevSecOps Lane)
role: coord
tier: 1
mode: long
parent: director-devsecops
children:
  - devsecops-agent
  - change-control
  - compliance-checker
  - deployment-orchestrator
capabilities:
  - task_breakdown
  - ci_coordination
  - deployment_management
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

# Manager (DevSecOps Lane)
Coordinates CI/CD, security, and deployment agents.
