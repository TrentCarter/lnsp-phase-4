# Agent Hierarchy — Complete Structure

**Date:** 2025-11-06
**Status:** Phase 4 - In Progress
**Total Agents:** 42 (23 Coordinators/System + 19 Execution)

---

## Overview

This document defines the complete agent hierarchy for the Polyglot Agent Swarm (PAS). All agents follow a tree structure with clear parent-child relationships and capability-based routing.

---

## Hierarchy Visualization

```
                              ┌─────────────────┐
                              │   Architect     │ ← Top-level coordinator
                              │   (coord)       │
                              └────────┬────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │ Director-Code │        │Director-Models│        │ Director-Data  │
    │   (coord)     │        │   (coord)     │        │   (coord)      │
    └───────┬───────┘        └───────┬───────┘        └───────┬────────┘
            │                        │                        │
            │                        │                        │
    ┌───────┴────────┐       ┌───────┴────────┐       ┌──────┴────────┐
    │  Manager(Code) │       │Manager(Models) │       │ Manager(Data) │
    │    (coord)     │       │    (coord)     │       │   (coord)     │
    └───────┬────────┘       └───────┬────────┘       └──────┬────────┘
            │                        │                        │
    ┌───────┴────────┐       ┌───────┴────────┐       ┌──────┴────────┐
    │                │       │                │       │               │
    ▼                ▼       ▼                ▼       ▼               ▼
┌─────────┐   ┌─────────┐ ┌─────────┐   ┌─────────┐ ┌─────────┐ ┌─────────┐
│  Code   │   │  Test   │ │Q-Tower  │   │Reranker │ │ Corpus  │ │ Graph   │
│ Writer  │   │ Writer  │ │Trainer  │   │Trainer  │ │ Auditor │ │ Builder │
│ (exec)  │   │ (exec)  │ │ (exec)  │   │ (exec)  │ │ (exec)  │ │ (exec)  │
└─────────┘   └─────────┘ └─────────┘   └─────────┘ └─────────┘ └─────────┘


    ┌───────────────┐        ┌───────────────┐
    │Director-      │        │ Director-Docs │
    │ DevSecOps     │        │   (coord)     │
    │   (coord)     │        └───────┬───────┘
    └───────┬───────┘                │
            │                ┌───────┴────────┐
            │                │  Manager(Docs) │
    ┌───────┴────────┐       │    (coord)     │
    │Manager         │       └───────┬────────┘
    │(DevSecOps)     │               │
    │   (coord)      │       ┌───────┴────────┐
    └───────┬────────┘       │                │
            │                ▼                ▼
    ┌───────┴────────┐   ┌─────────┐   ┌─────────┐
    │                │   │ Report  │   │  Docs   │
    ▼                ▼   │ Writer  │   │Generator│
┌─────────┐   ┌─────────┐ │ (exec)  │   │ (exec)  │
│DevSecOps│   │ Change  │ └─────────┘   └─────────┘
│ Agent   │   │ Control │
│ (exec)  │   │ (exec)  │
└─────────┘   └─────────┘


        SYSTEM AGENTS (parallel to hierarchy, provide services)

┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Gateway     │  │   Registry    │  │  Resource Mgr │  │ Token Governor│
│   (system)    │  │   (system)    │  │   (system)    │  │   (system)    │
│   Port 6120   │  │   Port 6121   │  │   Port 6104   │  │   Port 6105   │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘

┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Event Stream │  │  Heartbeat    │  │Provider Router│  │ Experiment    │
│   (system)    │  │   Monitor     │  │   (system)    │  │   Ledger      │
│   Port 6102   │  │   (system)    │  │   Port 6103   │  │   (system)    │
│               │  │   Port 6109   │  │               │  │   Port 6107   │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘

┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ File Queue    │  │ Peer Review   │  │ Contract      │  │ Error Track & │
│  Watcher      │  │  Coordinator  │  │  Tester       │  │  Recovery     │
│  (system)     │  │   (system)    │  │   (system)    │  │   (system)    │
│  Port 6110    │  │   Port 6108   │  │   Port 6106   │  │   Port 6112   │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘

┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Backup &      │  │  Security     │  │  Cost         │  │ Performance   │
│ Recovery Mgr  │  │  Auditor      │  │  Optimizer    │  │  Monitor      │
│  (system)     │  │  (system)     │  │  (system)     │  │  (system)     │
│  Port 6113    │  │  Port 6114    │  │  Port 6115    │  │  Port 6116    │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘

┌───────────────┐  ┌───────────────┐
│ Knowledge Base│  │ Model Version │
│     Mgr       │  │     Mgr       │
│  (system)     │  │  (system)     │
│  Port 6117    │  │  Port 6118    │
└───────────────┘  └───────────────┘
```

---

## Complete Agent List

### Coordinator Tier (7 agents)

| Name | Role | Parent | Children | Capabilities |
|------|------|--------|----------|--------------|
| Architect | coord | null | Directors | planning, task_decomposition, approval_routing |
| Director-Code | coord | Architect | Manager(Code) | code_lane_management, review_scheduling, build_management |
| Director-Models | coord | Architect | Manager(Models) | training_management, eval_management |
| Director-Data | coord | Architect | Manager(Data) | data_intake, qa_management, split_management |
| Director-DevSecOps | coord | Architect | Manager(DevSecOps) | ci_cd_management, security_oversight, supply_chain |
| Director-Docs | coord | Architect | Manager(Docs) | documentation_management, report_generation, leaderboard_management |
| Manager (generic) | coord | Director(*) | Executors | task_breakdown, approval_handling, rollback_management |

### System Agents (18 agents)

| Name | Role | Port | Parent | Capabilities |
|------|------|------|--------|--------------|
| Gateway | system | 6120 | null | routing, cost_tracking, receipt_generation |
| Registry | system | 6121 | null | service_discovery, heartbeat_tracking, ttl_management |
| Resource Manager | system | 6104 | Architect | resource_allocation, quota_enforcement, cleanup |
| Token Governor | system | 6105 | Architect | context_tracking, budget_enforcement, summarization |
| Event Stream | system | 6102 | null | event_broadcasting, websocket_management, buffering |
| Heartbeat Monitor | system | 6109 | Registry | health_checking, alert_generation, auto_recovery |
| Provider Router | system | 6103 | null | capability_matching, provider_selection, cost_optimization |
| Experiment Ledger | system | 6107 | Architect | run_tracking, artifact_management, reproducibility |
| File Queue Watcher | system | 6110 | Registry | atomic_ldjson, inbox_outbox_management |
| Peer Review Coord | system | 6108 | Director-Code | cross_vendor_review, pr_enforcement |
| Contract Tester | system | 6106 | Director-Models | schema_validation, mini_replay |
| Error Track & Recovery | system | 6112 | Architect | error_classification, recovery_trigger, rollback |
| Backup & Recovery Mgr | system | 6113 | Architect | backup_management, retention, recovery |
| Security Auditor | system | 6114 | Director-DevSecOps | vulnerability_scanning, dependency_checking, policy_audit |
| Cost Optimizer | system | 6115 | Architect | cost_analysis, resource_optimization |
| Performance Monitor | system | 6116 | Architect | metrics_tracking, performance_analysis |
| Knowledge Base Mgr | system | 6117 | Director-Docs | knowledge_maintenance, search, versioning |
| Model Version Mgr | system | 6118 | Director-Models | model_versioning, deployment_management |

### Execution Agents (19 agents)

| Name | Role | Parent | Capabilities |
|------|------|--------|--------------|
| Corpus Auditor | exec | Manager(Data) | source_checking, licensing_validation, stats_generation |
| Cleaner/Normalizer | exec | Manager(Data) | deduplication, normalization, encoding_fixes |
| Chunker-MGS | exec | Manager(Data) | sentence_banking, paragraph_banking, chunk_metadata |
| Graph Builder | exec | Manager(Data) | kg_construction, link_generation |
| Embed/Indexer | exec | Manager(Data) | embedding_generation, faiss_indexing, cache_management |
| Hard-Negative Miner | exec | Manager(Models) | hard_negative_mining, corpus_sampling |
| Q-Tower Trainer | exec | Manager(Models) | query_tower_training, model_training |
| Reranker Trainer | exec | Manager(Models) | reranker_training, model_training |
| Directional Adapter Fitter | exec | Manager(Models) | adapter_fitting, domain_specialization |
| Evaluator & Gatekeeper | exec | Director-Models | evaluation, scoring, threshold_gating |
| Release Coordinator | exec | Director-DevSecOps | deployment_orchestration, canary_management |
| Metrics/Cost Accountant | exec | Architect | receipt_generation, cost_tracking, metrics_collection |
| Report & Leaderboard Writer | exec | Director-Docs | report_generation, dashboard_creation, documentation |
| DevSecOps Agent | exec | Manager(DevSecOps) | ci_execution, image_building, sbom_generation |
| Change Control (CM) | exec | Manager(DevSecOps) | pr_lifecycle, label_management, changelog_generation |
| TLC Domain Classifier | exec | Manager(Data) | domain_classification, l0_tagging, confidence_scoring |
| Documentation Generator | exec | Director-Docs | auto_documentation, schema_documentation |
| Compliance Checker | exec | Director-DevSecOps | pii_verification, regulatory_compliance |
| Deployment Orchestrator | exec | Director-DevSecOps | pipeline_management, deployment_coordination |

---

## Routing Rules

### Capability-Based Routing

**From Architect:**
- `planning` tasks → stays with Architect
- `code_*` tasks → Director-Code
- `model_*` tasks → Director-Models
- `data_*` tasks → Director-Data
- `security_*` tasks → Director-DevSecOps
- `docs_*` tasks → Director-Docs

**From Directors:**
- Decompose tasks → assign to Manager
- Manager breaks down further → assign to Executors

**From Managers:**
- Sequential pipeline tasks → chain Executors
- Parallel tasks → spawn multiple Executors

---

## Agent Tiers

### Tier 1: Claude Code Sub-Agents
- **All Coordinators** (Architect, Directors, Managers)
- **Tool-heavy Executors** (Corpus Auditor, Graph Builder, Report Writer, etc.)
- **Cost:** Free
- **Strengths:** Full tool access, context-aware, local execution

### Tier 2: Local LLM Services
- **Classification tasks** (TLC Domain Classifier)
- **Batch operations** (if scaled beyond Claude capacity)
- **Cost:** Zero API cost
- **Strengths:** Fast inference, privacy, good for bulk

### Tier 3: External LLM APIs
- **Complex reasoning** (Evaluator & Gatekeeper when needed)
- **Cross-vendor reviews** (Peer Review Coordinator)
- **High-quality generation** (when local models insufficient)
- **Cost:** Per-token pricing
- **Strengths:** Highest quality, large context, specialized capabilities

---

## Resource Allocation Defaults

### Coordinator Agents
- Token budget: 0.50/0.75 (target/hard)
- CPU: 2 cores
- Memory: 4096 MB
- Heartbeat: 60s

### Execution Agents
- Token budget: 0.30/0.50 (target/hard)
- CPU: 1 core
- Memory: 2048 MB
- Heartbeat: 60s

### System Agents
- Token budget: n/a (stateless services)
- CPU: 1 core
- Memory: 1024-2048 MB
- Heartbeat: 30-60s

---

## Rights & Permissions

### Common Patterns

**Code-focused agents:**
- [F:rw] Filesystem read/write
- [B:x] Bash execution
- [G:x] Git operations
- [P:x] Python execution

**Data-focused agents:**
- [F:rw] Filesystem read/write
- [P:x] Python execution
- [S:x] SQL/PostgreSQL access

**DevOps-focused agents:**
- [F:rw] Filesystem read/write
- [B:x] Bash execution
- [G:x] Git operations
- [D:x] Docker execution
- [N:rw] Network read/write

**Docs-focused agents:**
- [F:rw] Filesystem read/write
- [P:x] Python execution
- [G:x] Git operations (for commit)

---

## Approval Requirements

### Always Requires Approval:
- `git push` (all agents with [G:x])
- File/directory deletions (all agents with [F:rw])
- DB destructive ops (agents with [S:x])
- External network POSTs (agents with [N:rw])

### Optional (Manager-configurable):
- Release promotion (Release Coordinator)
- Docker build (DevSecOps Agent)
- Service restart (Deployment Orchestrator)

---

## Implementation Notes

1. **Agent definitions stored in:** `.claude/agents/coordinator/`, `.claude/agents/execution/`, `.claude/agents/system/`
2. **Each agent gets:** Markdown file with YAML frontmatter following `agent_definition.schema.json`
3. **Registration:** On startup, all agents register with Registry (6121)
4. **Discovery:** Clients query Registry by capability, role, or name
5. **Invocation:** Via Gateway (6120) which routes to appropriate agent
6. **Monitoring:** All agents send heartbeats to Heartbeat Monitor (6109)

---

## Next Steps

1. ✅ Schema defined (`contracts/agent_definition.schema.json`)
2. ✅ Hierarchy documented (this file)
3. ⏳ Create 42 agent definition files (`.claude/agents/**/*.md`)
4. ⏳ Create registration script (`tools/register_agents.py`)
5. ⏳ Integration tests (`scripts/test_phase4.sh`)
6. ⏳ Agent invocation framework
7. ⏳ HMI integration for agent visibility

---

**END OF DOCUMENT**
