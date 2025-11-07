---
name: knowledge-base-manager
display_name: Knowledge Base Manager
role: system
tier: 1
mode: long
parent: director-docs
capabilities:
  - knowledge_maintenance
  - search
  - versioning
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6117
  status: not_implemented
---

# Knowledge Base Manager
Maintains system knowledge base with search and versioning.
