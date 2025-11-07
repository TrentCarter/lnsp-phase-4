---
name: embed-indexer
display_name: Embed/Indexer
role: exec
tier: 1
mode: task
parent: manager-data
capabilities:
  - embedding_generation
  - faiss_indexing
  - cache_management
transports:
  - rpc
rights:
  filesystem: rw
  python: true
  sql: true
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

# Embed/Indexer
Generates embeddings, builds FAISS indices, manages caches.
