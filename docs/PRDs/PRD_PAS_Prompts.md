# PRD: PAS Prompt Catalog & Operating Index

**Doc ID:** PRD_PAS_Prompts
**Version:** 1.0.0
**Status:** Stable (Ready to use)
**Last Updated:** 2025-11-10
**Owners:** PAS Architect, Dir-Docs

---

## 1) Purpose

This document is the **single source of truth** for all PAS system prompts. It catalogs each agent’s prompt file, role, authority, I/O contracts, budgets, heartbeats, transport fallbacks, required approvals, and acceptance checks. Use this index when creating or updating prompts, auditing agent behavior, or onboarding new lanes.

---

## 2) PAS Hierarchy & Naming (reference)

**Execution chain (down → up):**
Gateway → PAS Root → **Architect** → **Directors** (Code, Models, Data, DevSecOps, Docs) → **Managers** (per lane) → **Programmers** (per LLM)

**Naming:**

* Director: `Dir-{Domain}` (e.g., Dir-Code, Dir-Data)
* Manager: `Mgr-{Lane}-{Instance}` (e.g., Mgr-Code-01)
* Programmer: `Prog-{LLM}-{Instance:03d}` (e.g., Prog-Qwen-001)

**Transport preference:** RPC → File → MCP → REST
**Heartbeat:** 60s while active; two-miss rule triggers escalate/recover
**Token budget:** target ≤ 0.50 context ratio, hard ceiling 0.75; use Token Governor Save-State → Clear → Resume
**Approvals (always required before):** git push, deletions, DB destructive ops, external POSTs outside allowlist

---

## 3) Prompt Catalog (master table)

| File Path                                            | Tier / Role                    | Canonical Agent(s)   | Parent                | Children        | Purpose / Scope                                                            | Key Inputs                           | Key Outputs & Artifacts                             | Approvals Required                                  | KPIs / Acceptance                            | Transport         | HB / Tokens      | Owner               |
| ---------------------------------------------------- | ------------------------------ | -------------------- | --------------------- | --------------- | -------------------------------------------------------------------------- | ------------------------------------ | --------------------------------------------------- | --------------------------------------------------- | -------------------------------------------- | ----------------- | ---------------- | ------------------- |
| `docs/contracts/PEX_SYSTEM_PROMPT.md`                | Exec / Customer-Facing Lead    | PEX                  | Gateway               | Architect       | Human-facing executive interface; budget guards; orchestration framing     | Prime Directive, PRD, budget, policy | Executive plan, signed approvals, routing decisions | External POSTs, destructive ops                     | SLA adherence, budget compliance             | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Architect           |
| `docs/contracts/DIRENG_SYSTEM_PROMPT.md`             | Exec / Conversational Director | DirEng               | PAS Root or Architect | Directors       | Human-facing DirEng interface; exploration & micro-edits; delegation rules | User briefs, PRDs                    | Delegation notes, small code edits, follow-ups      | Protected-path PRs                                  | Review completeness, edit limits             | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-Docs            |
| `docs/contracts/ARCHITECT_SYSTEM_PROMPT.md`          | Coord / Top-Level              | Architect            | PAS Root              | Directors       | Decompose PRD into lane job_cards; allocate resources; enforce SLOs        | PRD, run config, receipts            | `architect_plan.md`, job_cards, status              | Git push, deletions, DB destructive, external POSTs | Plan published; lane acceptance gates set    | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Architect           |
| `docs/contracts/DIRECTOR_CODE_SYSTEM_PROMPT.md`      | Coord / Director               | Dir-Code             | Architect             | Mgr-Code-*      | Convert job_cards into code tasks; PRs, builds, reviews                    | job_card, repo state                 | mgr job_cards, PR requests, release drafts          | Protected-path PRs, pushes                          | Tests pass, lint=0, cov≥85%, PR cross-vendor | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-Code            |
| `docs/contracts/DIRECTOR_MODELS_SYSTEM_PROMPT.md`    | Coord / Director               | Dir-Models           | Architect             | Mgr-Models-*    | Training & eval orchestration; splits, seeds, KPIs                         | job_card, datasets, schemas          | run cards, metrics, gating decision                 | Cluster use, data export                            | KPI gates met (e.g., Echo-Loop ≥ target)     | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-Models          |
| `docs/contracts/DIRECTOR_DATA_SYSTEM_PROMPT.md`      | Coord / Director               | Dir-Data             | Architect             | Mgr-Data-*      | Data intake, QA, splits, tagging                                           | job_card, sources                    | audit report, manifests, chunks, graphs, embeddings | Ingest to external endpoints                        | Schema_diff==0; row_delta≤5%; tags present   | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-Data            |
| `docs/contracts/DIRECTOR_DEVSECOPS_SYSTEM_PROMPT.md` | Coord / Director               | Dir-DevSecOps        | Architect             | Mgr-DevSecOps-* | CI/CD gates, supply chain, deploys                                         | job_card, repo, infra policy         | build artifacts, SBOM, scan report, deploy receipts | Deployments, image publish                          | Gates pass; SBOM+scan clean; rollback plan   | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-DevSecOps       |
| `docs/contracts/DIRECTOR_DOCS_SYSTEM_PROMPT.md`      | Coord / Director               | Dir-Docs             | Architect             | Mgr-Docs-*      | Docs, reports, leaderboards                                                | job_card, artifacts                  | docs site, run report, leaderboard                  | Protected-path docs PRs                             | Docs completeness; review pass               | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Dir-Docs            |
| `docs/contracts/MANAGER_SYSTEM_PROMPT.md`            | Coord / Manager Template       | Mgr-{{Lane}}-{{N}}   | Dir-{Domain}          | Execs in lane   | Break job_card into executor tasks; schedule & unblock                     | job_card                             | executor job_cards, status, acceptance checks       | Approvals on risky ops                              | Lane KPIs met; artifacts valid               | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Respective Director |
| `docs/contracts/PROGRAMMER_SYSTEM_PROMPT.md`         | Exec / Implementer Template    | Prog-{{LLM}}-{{###}} | Mgr-{Lane}-{N}        | —               | Surgical code changes with tests/docs                                      | manager job_card                     | diffs/patches, test results, summary                | Protected-path PRs via Manager                      | Tests pass; lint=0; minimal docs updated     | RPC→File→MCP→REST | HB60 / 0.50–0.75 | Manager             |

> **Note:** “Protected paths” typically include `app/`, `contracts/`, `scripts/`, and `docs/PRDs/`. Cross‑vendor review is required on these paths.

---

## 4) I/O Contract Glossary

* **job_card**: JSON/YAML instruction packet including `task`, `inputs`, `expected_artifacts`, `acceptance`, `risks`, `budget`, `ids`.
* **artifacts**: Files produced under `artifacts/runs/{RUN_ID}/...` with receipts and metrics.
* **routing receipts**: Provider Router evidence of model/route selection for SLA/cost accounting.
* **acceptance gates**: Lane-specific checks (tests, coverage, schema diffs, KPIs). Failing a gate blocks promotion.

---

## 5) Operating Policies (applies to all prompts)

1. **Transport**: Prefer RPC; fall back to File → MCP → REST.
2. **Heartbeats**: Emit every 60s while active. On two consecutive misses, supervisors escalate.
3. **Token Budgets**: Keep ≤ 0.50; never exceed 0.75. If approaching, call Token Governor to Save-State → Clear → Resume.
4. **Approvals**: Required before pushing to git, deleting files, destructive DB ops, or external POSTs beyond allowlist.
5. **Quotas & Concurrency**: Reserve via Resource Manager; never assume unlimited capacity.
6. **Error Handling**: Retry transient (max 3, backoff). On hard timeout/permanent error, rollback or re-plan and escalate.
7. **Cross‑Vendor Review**: Enforced for Protected Paths.

---

## 6) Acceptance Matrix (quick audit)

| Tier        | Must Produce                                     | Must Verify                            | Fail-Safe / Recovery                              |
| ----------- | ------------------------------------------------ | -------------------------------------- | ------------------------------------------------- |
| Architect   | `architect_plan.md`, lane job_cards              | SLOs & budgets per lane set            | Re-plan fan-out; rollback conflicting allocations |
| Directors   | mgr job_cards; lane reports                      | Lane-specific acceptance gates         | Substitute/parallelize managers; pause lane       |
| Managers    | executor job_cards; acceptance checklist results | KPIs met; artifacts valid & linked     | Replace blocked executors; escalate to Director   |
| Programmers | diffs/patches; tests; docs updates               | Tests pass; lint=0; coverage threshold | Rollback patch; request help/clarification        |

---

## 7) Versioning & Change Control

* Any prompt change requires a PR referencing this index and updating the Catalog table.
* For protected paths, include a cross‑vendor reviewer in the PR.
* Update the **Acceptance Matrix** if a prompt’s required outputs or fail‑safes change.
* Bump **Version** in doc header and annotate the change log below.

---

## 8) Change Log

* **1.0.0 (2025‑11‑10):** Initial consolidation. Added Architect, all Directors, Manager template, Programmer template. Linked existing PEX & DirEng.

---

## 9) Quick Templates

**job_card (YAML)**

```yaml
id: jc-{{run}}-{{lane}}-{{seq}}
parent_id: {{parent_jc}}
role: {{architect|director|manager}}
lane: {{Code|Models|Data|DevSecOps|Docs}}
task: "..."
inputs:
  - path: "..."
expected_artifacts:
  - path: "artifacts/runs/{{RUN_ID}}/..."
acceptance:
  - check: "pytest>=0.90" # example per lane
risks:
  - "..."
budget:
  tokens_target_ratio: 0.50
  tokens_hard_ratio: 0.75
notes: "..."
```

**status update (JSON)**

```json
{
  "agent": "Mgr-Code-01",
  "run_id": "{{RUN_ID}}",
  "job_card_id": "jc-...",
  "state": "running|awaiting_approval|blocked|soft_timeout|hard_timeout|completed",
  "heartbeat": 1731264000,
  "message": "Starting tests",
  "links": ["artifacts/runs/{{RUN_ID}}/logs/test.out"]
}
```

---

## 10) File Locations (authoritative)

```
docs/contracts/
  PEX_SYSTEM_PROMPT.md
  DIRENG_SYSTEM_PROMPT.md
  ARCHITECT_SYSTEM_PROMPT.md
  DIRECTOR_CODE_SYSTEM_PROMPT.md
  DIRECTOR_MODELS_SYSTEM_PROMPT.md
  DIRECTOR_DATA_SYSTEM_PROMPT.md
  DIRECTOR_DEVSECOPS_SYSTEM_PROMPT.md
  DIRECTOR_DOCS_SYSTEM_PROMPT.md
  MANAGER_SYSTEM_PROMPT.md
  PROGRAMMER_SYSTEM_PROMPT.md
```

**End of PRD**
