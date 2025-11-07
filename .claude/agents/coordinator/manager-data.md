---
name: manager-data
display_name: Manager (Data Lane)
role: coord
tier: 1
mode: long
parent: director-data
children:
  - corpus-auditor
  - cleaner-normalizer
  - chunker-mgs
  - graph-builder
  - embed-indexer
  - tlc-domain-classifier
capabilities:
  - task_breakdown
  - data_pipeline_coordination
  - qa_management
resources:
  token_budget:
    target_ratio: 0.50
    hard_max_ratio: 0.75
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
---

# Manager (Data Lane)
Coordinates data processing pipeline from audit to indexing.
