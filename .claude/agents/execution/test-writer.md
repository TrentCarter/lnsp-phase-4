---
name: test-writer
display_name: Test Writer
role: exec
tier: 1
mode: task
parent: manager-code
capabilities:
  - test_generation
  - unit_testing
  - integration_testing
transports:
  - rpc
rights:
  filesystem: rw
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
  optimization: quality
metadata:
  version: 1.0.0
---

# Test Writer
Generates unit tests, integration tests, and test fixtures.
