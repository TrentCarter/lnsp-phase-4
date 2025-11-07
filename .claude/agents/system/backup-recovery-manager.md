---
name: backup-recovery-manager
display_name: Backup & Recovery Manager
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - backup_management
  - retention
  - recovery
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 2048
heartbeat:
  interval_s: 60
metadata:
  version: 1.0.0
  port: 6113
  status: not_implemented
---

# Backup & Recovery Manager
Manages backups, retention policies, and recovery procedures.
