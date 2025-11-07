---
name: tlc-domain-classifier
display_name: TLC Domain Classifier
role: exec
tier: 2
mode: task
parent: manager-data
capabilities:
  - domain_classification
  - l0_tagging
  - confidence_scoring
transports:
  - rpc
rights:
  filesystem: read
  python: true
resources:
  token_budget:
    target_ratio: 0.20
    hard_max_ratio: 0.40
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 60
model_preferences:
  primary:
    - llama-3.1-8b-instruct
    - TinyLlama-1.1B-chat
  optimization: cost
metadata:
  version: 1.0.0
---

# TLC Domain Classifier
Lightweight domain classification using local LLM (L0/Lpath tags).
