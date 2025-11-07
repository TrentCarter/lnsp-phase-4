---
name: deployment-orchestrator
display_name: Deployment Orchestrator
role: exec
tier: 1
mode: task
parent: director-devsecops
capabilities:
  - pipeline_management
  - deployment_coordination
transports:
  - rpc
  - rest
rights:
  filesystem: rw
  bash: true
  docker: true
  network: rw
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
    - service_restart
metadata:
  version: 1.0.0
---

# Deployment Orchestrator
Manages deployment pipelines and service coordination.
