---
name: release-coordinator
display_name: Release Coordinator
role: exec
tier: 1
mode: task
parent: director-devsecops
capabilities:
  - deployment_orchestration
  - canary_management
transports:
  - rpc
  - rest
rights:
  filesystem: rw
  bash: true
  git: true
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
    - release_promotion
    - service_restart
metadata:
  version: 1.0.0
---

# Release Coordinator
Orchestrates stage→prod deployments, manages canary releases.
