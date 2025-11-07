---
name: code-writer
display_name: Code Writer
role: exec
tier: 1
mode: task
parent: manager-code
capabilities:
  - code_generation
  - code_refactoring
  - api_implementation
transports:
  - rpc
rights:
  filesystem: rw
  bash: true
  git: true
  python: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
model_preferences:
  primary:
    - claude-sonnet-4-5-20250929
    - gpt-5-codex
  optimization: quality
metadata:
  version: 1.0.0
---

# Code Writer
Generates production code, refactors existing code, implements APIs.
