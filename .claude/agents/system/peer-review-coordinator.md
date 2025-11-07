---
name: peer-review-coordinator
display_name: Peer Review Coordinator
role: system
tier: 1
mode: task
parent: director-code
capabilities:
  - cross_vendor_review
  - pr_enforcement
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
  port: 6108
  status: not_implemented
---

# Peer Review Coordinator
Enforces cross-vendor PR reviews (e.g., Gemini reviews Claude PRs).
