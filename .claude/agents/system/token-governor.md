---
name: token-governor
display_name: Token Governor
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - context_tracking
  - budget_enforcement
  - summarization
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6105
  service_type: fastapi
---

# Token Governor
Context usage tracking and enforcement. Already implemented at port 6105.
