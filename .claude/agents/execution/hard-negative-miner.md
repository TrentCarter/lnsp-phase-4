---
name: hard-negative-miner
display_name: Hard-Negative Miner
role: exec
tier: 1
mode: task
parent: manager-models
capabilities:
  - hard_negative_mining
  - corpus_sampling
transports:
  - rpc
rights:
  filesystem: rw
  python: true
resources:
  token_budget:
    target_ratio: 0.30
    hard_max_ratio: 0.50
  cpu_cores: 2
  memory_mb: 4096
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Hard-Negative Miner
Mines hard negatives from corpus for training.
