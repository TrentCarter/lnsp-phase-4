---
name: devsecops-agent
display_name: DevSecOps Agent
role: exec
tier: 1
mode: task
parent: manager-devsecops
capabilities:
  - ci_execution
  - image_building
  - sbom_generation
transports:
  - rpc
rights:
  filesystem: rw
  bash: true
  git: true
  docker: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 2
  memory_mb: 4096
heartbeat:
  interval_s: 60
approvals:
  required:
    - docker_build
metadata:
  version: 1.0.0
---

# DevSecOps Agent
Runs CI checks (lint, tests), builds images, generates SBOMs.
