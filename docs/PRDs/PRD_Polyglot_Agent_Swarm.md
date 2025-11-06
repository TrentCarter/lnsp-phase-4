# PRD — Polyglot Agent Swarm (PAS) for Windsurf / Cascade Code

**Owner:** Trent Carter
**Repo root:** `/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4`
**Scope:** Production‑grade, local‑first agent swarm that orchestrates OpenAI, Anthropic, Google Gemini, Grok/xAI, and local LLMs (llama‑3.1‑8b, TinyLlama‑1.1B, deepseek‑r1 variants) for LVM pipeline operations and design‑time project bootstrap.
**Non‑goals (Phase‑1):** Air‑gapped mode (Phase‑2), distributed multi‑host scheduling beyond single box.

---

## 1) Objectives & Success Criteria

**Primary workflows (runtime):**
A. Chunk datasets → DBs
B. DB QA (field completeness) + lightweight Domain tagging via local LLMs
C. Model training (Query Tower, reranker, directional adapters)
D. I&T evaluation + reports
E. Deployment to internal/external services (8999/9007, etc.)
F. Metrics + cost accounting
G. Documentation, leaderboards, tracking
H. Architect (top‑level coordinator)
I. Manager (multi‑agent tasking)
J. DevSecOps
K. Change Control (Git, CM)
L. Resource Manager (CPU/GPU/RAM/Disk)

**Design‑time bootstrap:** Conductor spawns 5–6 sub‑agents to read PRD, derive tasks, request approvals, schedule with Resource Manager.

**Hard gates / KPIs (v1):**

* ≤ 75% context usage hard‑max per agent; target ≤ 50% (enforced by Token Governor).
* Heartbeat from every active agent at **≤ 60s** intervals; missing two beats = escalation.
* CI: mypy, pytest; cross‑vendor PR review required (e.g., Gemini reviews Claude PRs).
* Cost Router: prove provider selection receipts with latency + estimated $/1k‑tokens.

---

## 2) System Overview

**Topology:** Local orchestrator + modular agents. Agents communicate via pluggable transports (priority order below), share a job/manifest contract, and report status via heartbeats.

**Default transport order (fallback chain):**

1. **FastAPI RPC** (low latency, structured)
2. **File‑queue (atomic JSONL)** with watchdog (durable, debuggable)
3. **MCP** (Windsurf/Cascade interop)
4. **REST** (external services when needed)

> Rationale: RPC for speed; file queue for forensics; MCP optional; REST for externalization.

**Core services (ports start at 6100 as requested):**

* **6100** PAS Orchestrator API (FastAPI)
* **6101** PAS Web UI (Flask) — dashboard, runs, costs, approvals
* **6102** Event Stream (WS/SSE) — pub/sub to UI and CLIs
* **6103** Provider Router (model selection + receipts)
* **6104** Resource Manager (quotas, reservations)
* **6105** Token Governor (context policy + summarization)
* **6106** Contract Tester (schema + mini‑replay)
* **6107** Experiment Ledger (SQLite/JSON registry)
* **6108** Peer Review Coordinator (cross‑vendor PR reviews)
* **6109** Heartbeat Monitor (SLOs, alerts)
* **6110** File Queue Watcher (atomic JSONL inbox/outbox)

> Existing ports respected: 7001/7002 enc/dec; 8999 eval UI; 9000‑9007 model services.

---

## 3) Providers & Models

**Environment:**

* `GEMINI_API_KEY`, `GEMINI_PROJECT_ID`
* `GEMINI_MODEL_NAME_LOW='gemini-2.5-flash-lite'`
* `GEMINI_MODEL_NAME_MEDIUM='gemini-2.5-flash'`
* `GEMINI_MODEL_NAME_HIGH='gemini-2.5-pro'`
* `OPENAI_MODEL_NAME='gpt-5-codex'`
* `ANTHROPIC_MODEL_NAME_HIGH='claude-sonnet-4-5-20250929'`
* `ANTHROPIC_MODEL_NAME_MEDIUM='claude-sonnet-4-5-20250929'` (*Note: Same as HIGH, intentional for routing flexibility*)
* `ANTHROPIC_MODEL_NAME_LOW='claude-haiku-4-5'`
* Local primary: `deepseek-r1:7b-q4_k_m` (ctx 32k) @ `localhost:11434`
* Local secondary: `deepseek-r1:1.5b-q4_k_m` (ctx 16k) @ `localhost:11435`
* Optional locals for micro‑tasks: `llama-3.1-8b-instruct`, `TinyLlama-1.1B-chat`

**Provider Matrix (template):** `config/providers.matrix.json`

```json
{
  "capabilities": {
    "plan": [
      {"provider":"openai","model":"gpt-5-codex","p95_ms":null,"usd_per_1k":null},
      {"provider":"anthropic","model":"claude-sonnet-4-5-20250929"},
      {"provider":"gemini","model":"gemini-2.5-pro"},
      {"provider":"local","model":"deepseek-r1:7b-q4_k_m"}
    ],
    "code_write": [
      {"provider":"anthropic","model":"claude-sonnet-4-5-20250929"},
      {"provider":"openai","model":"gpt-5-codex"},
      {"provider":"gemini","model":"gemini-2.5-flash"}
    ],
    "classify_light": [
      {"provider":"local","model":"llama-3.1-8b-instruct"},
      {"provider":"local","model":"TinyLlama-1.1B-chat"}
    ],
    "eval_review": [
      {"provider":"gemini","model":"gemini-2.5-pro"},
      {"provider":"anthropic","model":"claude-sonnet-4-5-20250929"}
    ]
  },
  "routing_policy": {"prefer_cheapest_that_meets_sla": true, "sla_ms_p95": 1500}
}
```

---

## 4) Agent Roster (v1)

**Coordinator tier**

* **Architect** (Planner/Foreman) — decomposes PRD, allocates to Directors.
* **Directors (by lane)** — Code/Models/Data/DevSecOps/Docs.
* **Managers** — split work to executors; enforce approvals.

**Execution tier (examples)**

* Corpus Auditor → Cleaner/Normalizer → Chunker‑MGS → Graph Builder → Embed/Indexer → Hard‑Negative Miner → Trainers (Query Tower, Reranker, Directional Adapters) → Evaluator & Gatekeeper → Release/Canary → Metrics/Cost Accountant → Report & Leaderboard Writer.

**System agents (new)**

* **Token Governor (TG)**
* **Provider Router (PR)**
* **Contract Tester (CT)**
* **Experiment Ledger (XL)**
* **Peer Review Coordinator (PRC)**
* **Air‑Gap Compiler (AGC)** (Phase‑2)
* **Token‑Light Classifier (TLC)** for Domain tagging
* **Resource Manager (RM)**

**Default model mapping:**

* Planning/reasoning: OpenAI/Anthropic/Gemini (router chooses).
* Code write/repair: Anthropic (primary), OpenAI (alt), Gemini (alt).
* Reviews/PRs: Cross‑vendor enforced by PRC.
* Lightweight tasks (domain tagging, quick PII/license flags): local `llama-3.1-8b` → fallback `TinyLlama-1.1B`.

---

## 5) Contracts (Schemas)

All JSON contracts live under `contracts/` with JSON‑Schema validation and pytest contract tests.

### 5.1 agent_config.schema.json

```json
{
  "$id": "agent_config.schema.json",
  "type": "object",
  "required": ["agent","role","transports","paths","rights","heartbeat","token_budget"],
  "properties": {
    "agent": {"type":"string"},
    "role": {"type":"string"},
    "model": {"type":["object","null"]},
    "transports": {"type":"array","items":{"enum":["rpc","file","mcp","rest"]}},
    "paths": {"type":"object","properties":{"root":{"type":"string"}}},
    "rights": {"type":"object","properties":{"bash":{"type":"boolean"},"git":{"type":"boolean"},"python":{"type":"boolean"},"psql":{"type":"boolean"},"network":{"type":"string"}}},
    "sandbox": {"type":"object","additionalProperties":true},
    "approvals": {"type":"object","properties":{"required":{"type":"array","items":{"type":"string"}}}},
    "heartbeat": {"type":"object","properties":{"interval_s":{"type":"integer"},"timeout_s":{"type":"integer"}}},
    "token_budget": {"type":"object","properties":{"target_ratio":{"type":"number"},"hard_max_ratio":{"type":"number"}}},
    "timeouts": {"type":"object","properties":{"soft_s":{"type":"integer"},"hard_s":{"type":"integer"}}},
    "routing": {"type":"object","additionalProperties":true},
    "air_gapped": {"type":"boolean"}
  }
}
```

### 5.2 job_card.schema.json (LDJSON records)

```json
{
  "$id": "job_card.schema.json",
  "type":"object",
  "required":["job_id","agent","task","inputs","outputs","priority","created_at"],
  "properties":{
    "job_id":{"type":"string"},
    "parent_job_id":{"type":["string","null"]},
    "agent":{"type":"string"},
    "task":{"type":"string"},
    "capability":{"type":["string","null"]},
    "inputs":{"type":"object"},
    "outputs":{"type":"object"},
    "artifacts":{"type":"array","items":{"type":"string"}},
    "priority":{"type":"integer","minimum":0,"maximum":9},
    "resource_request":{"$ref":"resource_request.schema.json"},
    "approvals_required":{"type":"array","items":{"type":"string"}},
    "created_at":{"type":"string"},
    "deadline_at":{"type":["string","null"]}
  }
}
```

### 5.3 manifest.schema.json (per run)

```json
{
  "$id": "manifest.schema.json",
  "type":"object",
  "required":["run_id","agent","env","files"],
  "properties":{
    "run_id":{"type":"string"},
    "agent":{"type":"string"},
    "env":{"type":"object"},
    "files":{"type":"array","items":{"type":"string"}},
    "hashes":{"type":"object"},
    "cost_receipt":{"type":"object"},
    "metrics":{"type":"object"}
  }
}
```

### 5.4 heartbeat.schema.json

```json
{
  "$id": "heartbeat.schema.json",
  "type":"object",
  "required":["run_id","agent","ts","progress","status"],
  "properties":{
    "run_id":{"type":"string"},
    "agent":{"type":"string"},
    "ts":{"type":"string"},
    "progress":{"type":"number","minimum":0,"maximum":1},
    "status":{"enum":["queued","running","blocked","waiting_approval","paused","error","done"]},
    "message":{"type":"string"},
    "token_usage":{"type":"object","properties":{"ctx_used":{"type":"integer"},"ctx_limit":{"type":"integer"}}},
    "resources":{"type":"object","properties":{"cpu":{"type":"number"},"mem_mb":{"type":"number"},"gpu_mem_mb":{"type":"number"}}}
  }
}
```

### 5.5 status_update.schema.json

```json
{
  "$id": "status_update.schema.json",
  "type":"object",
  "required":["job_id","agent","event","ts"],
  "properties":{
    "job_id":{"type":"string"},
    "agent":{"type":"string"},
    "event":{"enum":["accepted","started","awaiting_approval","approved","rejected","soft_timeout","hard_timeout","escalated","completed"]},
    "ts":{"type":"string"},
    "details":{"type":"object"}
  }
}
```

### 5.6 resource_request.schema.json

```json
{
  "$id": "resource_request.schema.json",
  "type":"object",
  "properties":{
    "cpu":{"type":"number","minimum":0},
    "mem_mb":{"type":"integer","minimum":0},
    "gpu":{"type":"integer","minimum":0},
    "gpu_mem_mb":{"type":"integer","minimum":0},
    "disk_gb":{"type":"integer","minimum":0},
    "ports":{"type":"array","items":{"type":"integer"}}
  }
}
```

---

## 6) Approval Policy (initial)

**Always requires approval:**

* `git push` (guard: large file detector)
* Deletions of files/directories (move to `archive/` is auto‑allowed)
* DB destructive ops (e.g., `DROP`, `TRUNCATE`)
* External network POSTs outside vendor allowlist

**Optional approvals (manager‑set):** release promotion, docker build, service restart.

Approvals logged in `artifacts/approvals/` and surfaced in UI.

---

## 7) Resource Manager

* Maintains quotas; allocates reservations from job cards.
* Enforces concurrency by role (trainers=1, embedder=2 by default).
* Reports to Heartbeat Monitor; kills on hard timeouts; writes cleanup plans to `artifacts/cleanup/`.
* Heartbeat interval is configurable per-agent; not required for tasks expected to complete faster than the interval.

**Reservation protocol (request → grant → release):**

```json
// Request
{"type":"reserve","job_id":"J123","cpu":4,"mem_mb":8192,"gpu":1,"gpu_mem_mb":8192,"ports":[6100]}

// Grant
{"type":"grant", "job_id":"J123", "reservation_id": "RXYZ", "status": "ok"}

// Release
{"type":"release", "reservation_id": "RXYZ"}
```

---

## 8) Token Governor

* Target context ratio ≤ 0.50; hard max 0.75.
* Auto‑summarizes threads into 1–2k token state cards using local model first.
* If approaching hard max: _Save-State -> Clear -> Resume_ workflow emits a full summary to `docs/runs/<run_id>_summary.md` and restarts with trimmed context.

---

## 9) Domain Classification (DB fields) — pick from list

**Fields to emit:** `domain_l0`, `domain_path` (e.g., `science.physics.quantum`), `domain_conf` (0–1), `pii_flag`, `license_flag`.

**Proposed L0 label set (16 top‑level):**

1. `math`
2. `physics`
3. `chemistry`
4. `biology`
5. `medicine`
6. `cs`
7. `engineering`
8. `earth_env`
9. `space_astro`
10. `history`
11. `geography`
12. `culture_arts`
13. `law_policy`
14. `econ_business`
15. `social_science`
16. `language_linguistics`

> Extendable with `domain_path` (e.g., `engineering.aero.structures`). TLC Agent runs on local `llama-3.1-8b` with TinyLlama fallback.

---

## 10) Observability

* **Web UI (Flask @ 6101):** View runs, job status, costs, heartbeats, and manage approvals. Visualizes agent hierarchy and resource usage.
* **API (FastAPI @ 6100):** Submit jobs, query status, grant approvals, retrieve artifacts.
* **TUI:** `pas tui`: provides a real-time view of the job queue, allows tailing logs, and managing approvals from the terminal.

Artifacts registry under `artifacts/{runs,evals,releases,costs,approvals}` with symlinks per latest.

---

## 11) CI/CD & PR Flow

* **Gates:** mypy, pytest, coverage threshold (e.g., ≥85%), style (ruff/black).
* **Cross‑vendor review enforced** by PRC: PR author vendor ≠ reviewer vendor.
* **Protected paths:** `app/`, `contracts/`, `scripts/`, `docs/PRDs/`.

---

## 12) Contract Tests (Pytest names) — answer to A13

Use these canonical test names; the suite ships with template fixtures.

* `tests/contracts/test_agent_config.py::test_valid_agent_config_schema`
* `tests/contracts/test_job_card.py::test_job_card_schema_valid`
* `tests/contracts/test_manifest.py::test_manifest_schema_valid`
* `tests/contracts/test_heartbeat.py::test_heartbeat_interval_and_schema`
* `tests/contracts/test_status_updates.py::test_status_event_flow`
* `tests/contracts/test_resource_request.py::test_reservation_lifecycle`
* `tests/contracts/test_file_queue.py::test_atomic_ldjson_roundtrip`
* `tests/contracts/test_router_receipts.py::test_cost_latency_receipt_fields`
* `tests/contracts/test_token_governor.py::test_context_budget_enforced`
* `tests/contracts/test_domain_classifier.py::test_l0_labels_and_confidence_bounds`

---

## 13) Pricing/Capability Matrix

`config/providers.matrix.json` defines per‑capability candidate models with optional `p95_ms` and `usd_per_1k`. Provider Router chooses **cheapest that meets SLA**, falling back to locals on cost/quota breaches. Every call emits a `routing_receipt.json` to `artifacts/costs/<run_id>/`.

---

## 14) Security & Secrets

* Secrets in repo root `.env` (never committed).
* Vendor allowlist by directory (design‑time): `config/vendor.allow.json` and `config/vendor.deny.json`.
* Commands requiring approval (initial): `git push`, delete ops, DB destructive ops, external POSTs.
* Rights are enforced by a wrapper around shell/execution environments that validates the agent's grants before proceeding.

---

## 15) Error Handling, Timeouts, and Recovery

This section expands on the system's resilience strategy. The `Error Tracking & Recovery` agent is central to this process.

*   **Soft Timeout:** Triggers a `soft_timeout` status update. The responsible `Manager` agent is notified and may decide to either grant an extension or escalate to a human for approval.
*   **Hard Timeout:** Triggers the `Resource Manager` to kill the offending process to prevent resource starvation. The `Manager` agent is notified and will initiate a rollback.
*   **Rollbacks:** The system aims for atomic operations where possible. For multi-step tasks, a failure triggers a rollback to the last known-good state, typically the output of the previously successful agent. The `Experiment Ledger` is used to identify the correct artifacts for rollback.
*   **Retry Logic:** Upon task failure, the `Manager` consults the error type.
    *   **Transient Errors** (e.g., network timeouts, API rate limits): The task is automatically retried up to 3 times with exponential backoff.
    *   **Permanent Errors** (e.g., code bugs, data corruption): The task is marked as `error`, and the `Manager` escalates to the parent `Director` for replanning. This may involve assigning the task to a different agent or flagging it for human intervention.
*   **Escalation:** An escalation involves creating a high-priority approval request, surfaced in the Web UI and TUI, clearly stating the error and the failed job.

---

## 16) File Layout (v1)

```
lnsp-phase-4/
  app/
    agents/
    cli/
    pipeline/
    utils/
    mamba/
    nemotron_vmmoe/
  contracts/
    agent_config.schema.json
    job_card.schema.json
    manifest.schema.json
    heartbeat.schema.json
    resource_request.schema.json
  config/
    providers.matrix.json
    vendor.allow.json
    vendor.deny.json
    domains.l0.json
  artifacts/
    runs/
    evals/
    releases/
    costs/
    approvals/
    cleanup/
  docs/
    PRDs/
      PRD_Polyglot_Agent_Swarm.md
  scripts/
  tests/
    contracts/
```

---

## 17) Phase Plan

* **Phase‑1 (This PRD):** Single‑host orchestrator; transports (RPC+file+MCP); core agents; heartbeats; token governor; cross‑vendor PR; provider matrix; minimal UI.
* **Phase‑2:** Air‑Gap Compiler + local substitutes; distributed scheduling; richer metrics.

---

## 18) Open Decisions (defaults proposed)

* **Transport order:** RPC → file → MCP → REST (adopt).
* **Concurrency:** governed by Resource Manager; trainers=1, embedder=2 initial.
* **Domain L0 set:** adopt proposed 16 unless revised.
* **UI stack:** Flask (UI) + FastAPI (APIs) as requested.
* **Ports:** 6100–6110 as mapped; more available 6111+.
* **Local micro‑models:** enable `llama-3.1-8b` + `TinyLlama-1.1B` for TLC Agent.

---

## 19) Acceptance Checklist

* [ ] Contracts compiled; pytest suite green.
* [ ] Heartbeats arriving ≤60s; drop detection working.
* [ ] Token budgets enforced; Save-State -> Clear -> Resume verified.
* [ ] Provider Router receipts generated.
* [ ] Cross‑vendor PR flow enforced on protected paths.
* [ ] UI shows jobs, approvals, costs, and heartbeats.
* [ ] Resource Manager alloc→kill→cleanup lifecycle proven.

---

## 20) Agent Catalog (Roles, Hierarchy, Rights, Budgets)

**Legend (compact fields used below)**

*   **Role:** `coord` (coordinator), `exec` (executor), `system` (infra/service)
*   **Mode:** `long` (long‑lived), `task` (ephemeral per job)
*   **Primary transport:** `rpc` | `file` | `mcp` | `rest`
*   **Rights (perm‑code):** `[F:rw]` filesystem, `[B:x]` bash, `[G:x]` git, `[P:x]` python, `[N:rw]` network, `[S:x]` sql/psql, `[D:x]` docker. Include only granted rights.
*   **Token budget:** `target/hard` as context‑ratio (e.g., `0.50/0.75`).
*   **HB:** heartbeat interval seconds (timeout = 2×HB by Registry).

### 20.1 Coordinator & System Agents

| Agent                              | Role   | Parent          | Children (key)      | Primary tasks                              | Mode | Primary transport | Rights                           | Token budget | HB |
| ---------------------------------- | ------ | --------------- | ------------------- | ------------------------------------------ | ---- | ----------------- | -------------------------------- | ------------ | -- |
| **Architect**                      | coord  | —               | Directors           | Decompose PRD, planning, approvals routing | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw]      | 0.50/0.75    | 60 |
| **Director‑Code**                  | coord  | Architect       | Managers(Code)      | Own code lane, schedule reviews & builds   | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw][D:x] | 0.50/0.75    | 60 |
| **Director‑Models**                | coord  | Architect       | Managers(Models)    | Own training/eval lanes                    | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw]      | 0.50/0.75    | 60 |
| **Director‑Data**                  | coord  | Architect       | Managers(Data)      | Data intake, QA, splits                    | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw][S:x] | 0.50/0.75    | 60 |
| **Director‑DevSecOps**             | coord  | Architect       | Managers(DevSecOps) | CI/CD gates, security, supply‑chain        | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw][D:x] | 0.50/0.75    | 60 |
| **Director‑Docs**                  | coord  | Architect       | Managers(Docs)      | Docs, reports, leaderboards                | long | rpc               | [F:rw][G:x][P:x]                 | 0.50/0.75    | 60 |
| **Manager (per lane)**             | coord  | Director(*)     | Executors           | Break down tasks, approvals, rollback      | long | rpc               | [F:rw][B:x][G:x][P:x][N:rw]      | 0.50/0.75    | 60 |
| **Gateway (6120)**                 | system | —               | —                   | Single entrypoint; route & receipt         | long | rpc/rest          | [F:rw][N:rw]                     | n/a          | 60 |
| **Registry (6121)**                | system | —               | —                   | Service CRUD, TTL, discovery               | long | rpc               | [F:rw]                           | n/a          | 60 |
| **Resource Manager (6104)**        | system | Architect       | Directors/Managers  | Reservations, quotas, cleanup              | long | rpc               | [F:rw][B:x]                      | n/a          | 60 |
| **Token Governor (6105)**          | system | Architect       | All agents          | Enforce context budget; summarize          | long | rpc               | [F:rw][P:x]                      | n/a          | 60 |
| **Contract Tester (6106)**         | system | Director‑Models | Trainers/Evaluator  | Schema validate + mini‑replay              | task | rpc               | [F:rw][P:x]                      | 0.20/0.40    | 60 |
| **Experiment Ledger (6107)**       | system | Architect       | All                 | Run cards, seeds, costs, repro             | long | rpc               | [F:rw]                           | n/a          | 60 |
| **Peer Review Coordinator (6108)** | system | Director‑Code   | Reviewers           | Cross‑vendor PR enforcement                | task | rpc/rest          | [F:rw][G:x][N:rw]                | 0.30/0.50    | 60 |
| **Heartbeat Monitor (6109)**       | system | Registry        | —                   | Detect missed beats, escalate              | long | rpc               | [F:rw]                           | n/a          | 30 |
| **File Queue Watcher (6110)**      | system | Registry        | —                   | Atomic LDJSON inbox/outbox                 | long | file              | [F:rw]                           | n/a          | 60 |
| **Error Tracking & Recovery(6112)**| system | Architect       | All agents          | Classify errors (transient/perm), trigger recovery/rollback | long | rpc | [F:rw][P:x][N:rw] | n/a | 30 |
| **Backup & Recovery Mgr (6113)**   | system | Architect       | —                   | Manage backups, retention, recovery        | long | rpc               | [F:rw][B:x][P:x][S:x]            | n/a          | 60 |
| **Security Auditor (6114)**        | system | Director-DevSecOps | —                | Scan for vulnerabilities, check dependencies, audit policies | long | rpc | [F:rw][P:x][N:rw] | n/a | 60 |
| **Cost Optimizer (6115)**          | system | Architect       | —                   | Optimize resource usage and costs          | long | rpc               | [F:rw][P:x]                      | n/a          | 60 |
| **Performance Monitor (6116)**     | system | Architect       | —                   | Track and analyze performance metrics      | long | rpc               | [F:rw][P:x]                      | n/a          | 30 |
| **Knowledge Base Mgr (6117)**      | system | Director-Docs   | —                   | Maintain system knowledge base             | long | rpc               | [F:rw][P:x]                      | 0.30/0.50    | 60 |
| **Model Version Mgr (6118)**       | system | Director-Models | —                   | Manage model versions and deployments      | long | rpc               | [F:rw][P:x][D:x]                 | n/a          | 60 |

### 20.2 Execution Agents

| Agent                           | Role | Parent             | Children (key)      | Primary tasks                   | Mode | Primary transport | Rights                      | Token budget | HB |
| ------------------------------- | ---- | ------------------ | ------------------- | ------------------------------- | ---- | ----------------- | --------------------------- | ------------ | -- |
| **Corpus Auditor**              | exec | Manager(Data)      | Cleaner/Normalizer  | Source checks, licensing, stats | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Cleaner/Normalizer**          | exec | Manager(Data)      | Chunker‑MGS         | Dedup, normalize, fix encoding  | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Chunker‑MGS**                 | exec | Manager(Data)      | Graph Builder       | Sentence/para banks, chunk meta | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Graph Builder**               | exec | Manager(Data)      | Embed/Indexer       | Build KG/links from chunks      | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Embed/Indexer**               | exec | Manager(Data)      | Hard‑Neg Miner      | Embeddings; FAISS/IVF + caches  | task | rpc               | [F:rw][P:x][S:x]            | 0.30/0.50    | 60 |
| **Hard‑Negative Miner**         | exec | Manager(Models)    | Trainers            | Mine hards from corpus          | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Q‑Tower Trainer**             | exec | Manager(Models)    | —                   | Train retriever Q‑tower         | task | rpc               | [F:rw][P:x][D:x]            | 0.40/0.70    | 60 |
| **Reranker Trainer**            | exec | Manager(Models)    | —                   | Train reranker                  | task | rpc               | [F:rw][P:x][D:x]            | 0.40/0.70    | 60 |
| **Directional Adapter Fitter**  | exec | Manager(Models)    | —                   | Fit adapters on domains         | task | rpc               | [F:rw][P:x][D:x]            | 0.40/0.70    | 60 |
| **Evaluator & Gatekeeper**      | exec | Director‑Models    | Release Coordinator | Eval, score, gate thresholds    | task | rpc               | [F:rw][P:x]                 | 0.30/0.50    | 60 |
| **Release Coordinator**         | exec | Director‑DevSecOps | —                   | Stage → prod deploy via Gateway | task | rpc/rest          | [F:rw][G:x][B:x][D:x][N:rw] | 0.30/0.50    | 60 |
| **Metrics/Cost Accountant**     | exec | Architect          | —                   | Token/latency/$ receipts        | task | rpc               | [F:rw][P:x]                 | 0.20/0.40    | 60 |
| **Report & Leaderboard Writer** | exec | Director‑Docs      | —                   | Reports, dashboards, docs       | task | rpc               | [F:rw][P:x][G:x]            | 0.30/0.50    | 60 |
| **DevSecOps Agent**             | exec | Director‑DevSecOps | —                   | Run CI checks (lint, tests), build images, generate SBOMs | task | rpc | [F:rw][B:x][G:x][D:x] | 0.30/0.50 | 60 |
| **Change Control (CM)**         | exec | Director‑DevSecOps | —                   | Manage PR lifecycle (labels, merging), generate changelogs | task | rpc | [F:rw][G:x] | 0.20/0.40 | 60 |
| **TLC Domain Classifier**       | exec | Manager(Data)      | —                   | L0/Lpath tags via local LLM     | task | rpc               | [F:rw][P:x]                 | 0.20/0.40    | 60 |
| **Documentation Generator**      | exec | Director-Docs      | —                   | Auto-generate documentation from code and schemas | task | rpc | [F:rw][P:x][G:x] | 0.30/0.50 | 60 |
| **Compliance Checker**          | exec | Director-DevSecOps | —                   | Verify outputs against regulatory standards (PII, etc) | task | rpc | [F:rw][P:x] | 0.30/0.50 | 60 |
| **Deployment Orchestrator**     | exec | Director-DevSecOps | —                   | Manage deployment pipelines     | task | rpc/rest          | [F:rw][B:x][D:x][N:rw]       | 0.30/0.50    | 60 |

> Notes: All agents inherit vendor allow/deny and approvals from the PRD’s Security & Approvals section unless overridden here.

### 20.3 Common Defaults & Field Legend

| Field             | Default                                              | Notes                                                                                              |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Primary transport | `rpc`                                                | Fallbacks: `file → mcp → rest` per PRD §2.                                                         |
| Rights            | `[F:rw][P:x]`                                        | Minimal; coordinators add `[B:x][G:x][N:rw]`; trainers may add `[D:x]`; DB actors may add `[S:x]`. |
| Token budget      | `0.50/0.75`                                          | Enforced by Token Governor; Save-State -> Clear -> Resume on breach.                               |
| Heartbeat (HB)    | `60`                                                 | Registry TTL = `HB + 30`; evict after two misses. Not required for short-lived tasks.              |
| Approvals         | `git push`, deletions, DB destructive, external POST | Per PRD §8.                                                                                        |
| Logging           | `artifacts/runs/*` + routing receipts                | Experiment Ledger writes run cards; Gateway writes receipts.                                       |
| Scheduling        | Resource Manager quotas                              | Trainers=1, Embedder=2 initial (override per job card).                                            |
| Discovery         | Registry(6121)                                       | Clients use Gateway(6120); agents never hard‑code model ports.                                     |
| Secrets           | `.env`                                               | Loaded by services; never logged.                                                                  |
| Vendor review     | Cross‑vendor PR required                             | PRC enforces on protected paths.                                                                   |

## 21) Agent Interaction Example: Dataset Chunking

To clarify how agents collaborate, this sequence outlines the "Chunk a new dataset" workflow.

1.  **Architect** receives a high-level goal, e.g., "Process the new `arxiv_v2.jsonl` dataset."
2.  **Architect** creates a plan and issues a `job_card` to the `Director-Data`.
    *   `"task": "process_new_corpus"`, `"inputs": {"path": "data/new/arxiv_v2.jsonl"}`
3.  **Director-Data** accepts the job and breaks it down. It issues a `job_card` to `Manager(Data)`.
    *   `"task": "full_ingestion_pipeline"`
4.  **Manager(Data)** accepts and orchestrates the execution tier. It issues the first `job_card` to `Corpus Auditor`.
    *   `"task": "audit_corpus"`, `"inputs": {"path": "..."}`
    *   `Corpus Auditor` requests a resource reservation from the `Resource Manager (RM)`.
5.  **Corpus Auditor** runs, checking for licensing, basic stats, and PII flags. It writes its output artifact (e.g., `audit_report.json`) and sends a `completed` status update.
6.  **Manager(Data)** receives the completion, verifies the output, and issues a new `job_card` to `Cleaner/Normalizer`.
    *   `"task": "clean_and_normalize"`, `"inputs": {"audit_report": "..."}`
7.  The process continues sequentially down the execution chain: `Chunker-MGS` → `Graph Builder` → `Embed/Indexer`. Each agent:
    *   Receives a `job_card` from the `Manager(Data)`.
    *   Requests resources from `RM`.
    *   Performs its task.
    *   Writes output artifacts.
    *   Releases resources.
    *   Reports `completed`.
8.  Throughout the process, all agents send heartbeats to the **Heartbeat Monitor**. The **Token Governor** observes context usage, and the **Provider Router** selects models for any sub-tasks requiring an LLM.
9.  Upon final completion from `Embed/Indexer`, **Manager(Data)** reports `completed` to **Director-Data**, who in turn reports to the **Architect**.

---

## 22) State Management

Agents are designed to be as stateless as possible, receiving all necessary information via their `job_card`. State is primarily managed externally rather than internally.

*   **Default State:** Agents are ephemeral by default (`Mode: task`) and do not maintain state across jobs.
*   **Long-Lived Agents:** Coordinator agents (`Mode: long`) maintain state in memory (e.g., active jobs, child agent status). This state can be rebuilt from the Experiment Ledger upon restart.
*   **State Persistence:** When state must be persisted durably, it is written to an artifact file or a record in the Experiment Ledger (SQLite). This follows the _Save-State -> Clear -> Resume_ pattern, ensuring that a killed agent can be restarted by another agent with the correct context.
*   **No Internal Databases:** Agents do not manage their own internal databases for state.

---

## 23) Testing Strategy

The system's reliability is ensured through a multi-layered testing strategy, executed by the `Contract Tester` and `DevSecOps` agents as part of the CI/CD pipeline.

*   **Unit Tests:** Each agent's internal logic, helper functions, and classes are tested in isolation. Mocks are used for external dependencies like APIs or the filesystem.
*   **Contract Tests:** As defined in §12, these tests validate the integrity of the JSON schemas and ensure that agents can correctly produce and consume messages that adhere to the defined contracts.
*   **Integration Tests:** These tests verify the interaction between two or more agents. For example, ensuring a `Manager` can correctly dispatch a job to an `Executor` and process the response. This includes testing the transport layers (RPC, file queue).
*   **End-to-End (E2E) Tests:** These are comprehensive tests that simulate a full workflow, such as the "Dataset Chunking" example. They are designed to catch issues in the overall system orchestration, resource management, and error handling logic.

---

## 24) Configuration Management

*   All configuration files reside in the `config/` directory.
*   Configuration is loaded at service startup.
*   Dynamic reloading is not supported in Phase-1 to ensure stability. Services must be restarted to apply configuration changes. This can be orchestrated by the `Director-DevSecOps` agent upon approval.

---

## 25) Risks & Mitigations

- **Provider outage**: Gateway falls back to locally hosted models and queues non-urgent jobs until remote providers recover; required action is rehearsing the failover playbook each sprint.
- **Local model degradation**: Health checks run on `deepseek-r1` and `llama-3.1-8b`; if latency or quality breaches thresholds, Provider Router escalates to remote vendors and flags the Resource Manager to free GPU memory.
- **Cost overrun**: Cost Optimizer watches rolling spend per provider; breaching daily caps triggers automatic routing to cheaper tiers and a Manager-level approval requirement for premium models.
- **Transport failure**: If FastAPI RPC fails the orchestrator explicitly promotes the file queue transport and emits an alert so DevSecOps can inspect the RPC stack before switching back.
- **Heartbeat gaps**: Missing two heartbeats raises a Priority-1 alert; the Manager pauses downstream jobs, invokes Save-State -> Clear -> Resume, and files an escalation if restart fails.
- **Open question — audit trail retention**: Need confirmation on minimum retention window for ledger artifacts and approvals to align with compliance requirements before Phase-1 ship.
