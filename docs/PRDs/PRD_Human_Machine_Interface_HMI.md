# PRD — Human–Machine Interface (HMI) for PAS Agent Swarm

**Owner:** Trent Carter
**Repo Root:** `/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4`
**Related PRDs:** PRD_Agent_Swarm_v1 (PAS / Windsurf)
**Date:** 2025‑11‑06
**Status:** Draft (v1)

---

## 1) Purpose & Outcomes

**Goal:** Provide a multi‑modal, hierarchical interface for observing, directing, and auditing a complex agent swarm. The HMI must make it easy to:

* See *what* is happening (tasks, flows, dependencies), *where* (which agent/tier), *how far* (progress), and *at what cost* (time, tokens, $).
* Detect problems early (stalls, deadlocks, context overrun, heartbeat misses) and intervene safely (pause, reassign, rollback).
* Scale from tiny jobs (1–2 agents) to large swarms (40+ agents) without losing clarity.

**Primary outcomes**

* Real‑time situational awareness (≤1s perceived lag).
* Hierarchical roll‑ups from worker → manager → director → VP.
* Cost and token usage visibility, with budget alerts.
* Voice and sonification channels for eyes‑busy/voice‑only use (driving), and silent visual modes for office/lab use.

---

## 2) Personas & Scenarios

* **VP Engineering (you):** Needs the 10,000‑ft view and quick drill‑downs. Cares about progress, risk, cost, and time‑to‑green.
* **Directors (Code/Models/Data/DevSecOps/Docs):** Track lane health, approve gates, resolve blockers, enforce policy.
* **Managers:** Triage work, restart/kill stuck tasks, keep heartbeats green, maintain quality gates.
* **Operators/ICs:** Observe their agent instance, confirm instructions, and provide human approvals.

**Key scenarios**

1. Refactor‑wide change across repo and docs; watch work stream out to workers then roll up to completion.
2. Detect a stuck worker (2 missed heartbeats); manager restarts; VP sees the bubble clear.
3. Token budget breach at a worker; Token Governor triggers Save‑State → Clear → Resume; HMI shows cause and recovery.

---

## 3) Architecture Integration (Ports & Services)

* **PAS Web UI:** Flask @ **6101** (this HMI).
* **Event Stream:** WS/SSE @ **6102** (live updates).
* **PAS Orchestrator API:** FastAPI @ **6100** (job submit/approve).
* **Gateway:** @ **6120** (single client entrypoint).
* **Registry:** @ **6121** (service discovery & heartbeats).

HMI consumes:

* Heartbeats (`heartbeat.schema.json`) and status updates (`status_update.schema.json`).
* Routing receipts from Gateway (`artifacts/costs/<run_id>.json`).
* Job cards and manifests for provenance.

---

## 4) Views (UI Surfaces)

### 4.1 Hierarchical Dashboards (per‑agent Flask mini‑UIs, consistent look)

* **Agent card:** name, role, parent, children, rights (perm‑codes), ctx limit/usage, current job, ETA, artifacts, last heartbeat.
* **Roll‑up summaries:** parent dashboards aggregate children (counts, progress, alerts).

### 4.2 Global Tree (Message Flow Graph)

* Dynamic, zoomable DAG/tree.
* **Nodes:** agents; **size** ∝ live token usage / load; **color** = status; **ring** indicates role (coord/exec/system).
* **Edges:** animated “light bars” for messages; brightness = throughput.

### 4.3 Sequencer (Activity Timeline)

* Rows = tiers (Workers bottom → Managers → Directors → VP top).
* Time on X‑axis; **glyph length** = task duration; **thickness** = live token usage; **color** = state.
* A moving playhead shows “now”.

### 4.4 Sonification (Musical Notes)

* Optional audio stream: short, rate‑limited notes for task events (assign, start, progress, done).
* **Pitch:** higher = lower tier; **instrument:** differentiates agents; **note length:** task duration hint.

### 4.5 Spoken Summaries (Narration)

* One‑sentence summaries on assign/complete events.
* **Voice depth:** maps to hierarchy (VP = deep; workers = light).
* Mute per‑tier and per‑agent; batch mode for “digest” every N minutes.

### 4.6 Timeline Scrubber (Replay)

* Scrub backward to replay the last 24–72h with the sequencer/graph synchronized; export as MP4/GIF for post‑mortems.

---

## 5) Signals, Encodings, and Alerts

### 5.1 Status → Color Map

* queued=gray, running=blue, waiting_approval=purple, blocked=orange, paused=teal, error=red, done=green.

### 5.2 Audio Map (defaults; user‑configurable)

* **Tier → pitch range:** VP (C2–G2), Directors (C3–G3), Managers (C4–G4), Workers (C5–G5).
* **Event → note:** assign=staccato, start=attack, progress=short sustain, complete=resolve cadence.
* **Instrument families:** VP=contrabass, Directors=cello, Managers=viola, Workers=violin; alternates per agent to avoid collisions.

### 5.3 Narration Map

* Per‑event templates:

  * *Assign:* “Director‑Data assigned ‘Chunk wiki batch‑7’ to Manager‑North (ETA 14m).”
  * *Heartbeat:* “Manager‑North healthy, 35% load, tokens 12.3k/32k.”
  * *Complete:* “Worker‑CPE‑12 finished 184 files; 0 errors; 12m 41s.”

### 5.4 Alerts & Policies

* **Heartbeat miss:** >2 intervals ⇒ red alert at node + banner; auto‑action: notify parent and Manager.
* **Context breach:** ≥75% window ⇒ Token Governor Save‑State → Clear → Resume; HMI shows link to summary artifact.
* **Cost spike:** rolling p95 > threshold or $/min > budget ⇒ yellow alert with top contributors.

---

## 6) Controls (Operator Actions)

* Pause/Resume agent or subtree.
* Reassign task to different agent (via Gateway target filters).
* Restart worker (kills run, preserves artifacts).
* Approve/Reject pending gates (PRs, destructive ops).
* Mute/solo tiers for audio; set narration cadence (live vs digest).
* Toggle views (Tree/Sequencer), tie playhead to real time or replay mode.

---

## 7) Data & Contracts (Consumed by HMI)

* **heartbeat.schema.json** → `run_id, agent, ts, progress[0..1], status, token_usage{ctx_used,ctx_limit}, resources{cpu,mem,gpu_mem}`.
* **status_update.schema.json** → `event: {accepted|started|awaiting_approval|approved|rejected|soft_timeout|hard_timeout|escalated|completed}`.
* **job_card.schema.json** → task metadata, resource requests, parents/children, approvals_required.
* **routing_receipt.schema.json** → source, resolved target, timings, cost_estimate, ctx info.

Retention:

* `artifacts/hmi/events/` LDJSON for 30d (configurable).
* Audio/narration logs (JSON sidecar) for audit (optional).

---

## 8) Performance & SLOs

* UI render latency: **≤250ms** P95.
* Event propagation: **≤1s** end‑to‑end P95.
* Graph layout stability: node jitter < 20px/frame at 60fps.
* Audio rate limit: ≤8 notes/sec global; per‑agent ≤2 notes/sec.

---

## 9) Security, Privacy, and Approvals

* Respect PRD approvals: `git push`, deletions, DB destructive, external POSTs require explicit approval.
* Mute narration/notes in sensitive spaces; privacy mode hides file paths and vendor names.
* All actions audited to `artifacts/hmi/audit/` with user, ts, before/after.

---

## 10) Accessibility & UX

* Color‑blind safe palette; redundant shapes/labels for status.
* Full keyboard/TUI parity for core actions.
* Captions/real‑time transcript for narration; per‑tier volume sliders.
* Mobile layout for quick checks; voice‑only control set (basic commands).

---

## 11) Extensibility & Integrations

* **Plugins:** add visualization panels, custom alert rules, new audio instruments/packs.
* **Claude /agents:** ship a *HMI Operator* subagent with Gateway tool for approvals and view toggles.
* **AR Prototype:** optional “Holographic Task Cube” (WebXR) in Phase‑3.

---

## 12) API (HMI Facade)

* `GET /hmi/agents` → list with hierarchy, status, ctx, rights.
* `GET /hmi/tree` → graph JSON (nodes, edges, metrics).
* `GET /hmi/seq?from..to` → sequencer strips.
* `GET /hmi/metrics` → roll‑ups (tokens/min, $/min, errors).
* `WS /hmi/stream` → unified event bus (beats, status, receipts).
* `POST /hmi/action` → pause/resume/reassign/approve with reason.

---

## 13) Visual Encodings (Spec)

* **Node size:** linear map of `ctx_used/ctx_limit` (min 8px, max 48px).
* **Edge pulse speed:** proportional to message rate (cap at 4 pulses/s).
* **Sequencer thickness:** token rate (tokens/s); cap to preserve legibility.
* **Legends:** hoverable, always visible in top‑right.

---

## 14) Audio Encodings (Spec)

* **Pitch ladder per tier** (VP low → Workers high).
* **Event envelope** (ADSR): assign (short attack), start (attack+short decay), progress (staccato ticks), complete (resolve).
* **Ducking** during narration to avoid overlap.

---

## 15) Milestones & Deliverables

**P0 (1–2 days):**

* Agent list, per‑agent cards, roll‑up counts; basic Tree.
* WS/SSE stream; log to `artifacts/hmi/events/`.
* Status colors & heartbeat alerts.

**P1:**

* Sequencer view + timeline scrubber.
* Cost/token roll‑ups; receipts ingested.
* Operator actions (pause/resume/reassign/approve).

**P2:**

* Sonification + spoken summaries (mute/solo, digest mode).
* Replay export; anomaly flags (stall, token breaches).

**P3:**

* AR “Holographic Task Cube” prototype (WebXR).
* Plugin API + presets; mobile quick‑status.

---

## 16) Acceptance Criteria

* Live Tree and Sequencer reflect events with ≤1s lag (P95).
* Node size, color, and thickness encode load/status/token usage clearly.
* Two missed heartbeats raise red alert and visible breadcrumb to parent/VP.
* Token Governor actions surface in HMI with link to Save‑State summary.
* Costs visible by agent, tier, and run; top N contributors listed.
* Audio and narration can be toggled per‑tier and per‑agent; rate‑limited.
* P0–P2 features demoed on a real multi‑agent run; logs preserved.

---

## 17) Open Questions

* Preferred TTS backend for narration (local vs vendor).
* Default audio pack/instrument set.
* Minimum retention for events vs audio sidecars (30/90 days?).
* Whether to auto‑suggest scaling ("add N workers") from HMI based on saturation.

---

## 18) Appendix — Quick Mappings & Tables

### 18.1 Event Types

| Event             | Source           | Triggers                                                |
| ----------------- | ---------------- | ------------------------------------------------------- |
| job_created       | Manager/Director | Sequencer block appears, graph edge from parent → child |
| accepted/started  | Worker           | Note: start; narration (optional)                       |
| progress          | Worker           | Sequencer thickness update                              |
| awaiting_approval | Any              | VP/Director banner + action button                      |
| soft_timeout      | Manager          | Orange alert; suggest restart                           |
| hard_timeout      | Manager          | Red alert; kill + requeue                               |
| completed         | Worker/Manager   | Resolve chord; roll‑up progress ↑                       |
| heartbeat         | All              | Node glow refresh; missed x2 ⇒ alert                    |
| cost_receipt      | Gateway          | Costs/latency charts update                             |

### 18.2 Rights (perm‑codes)

`[F:rw]` filesystem, `[B:x]` bash, `[G:x]` git, `[P:x]` python, `[N:rw]` network, `[S:x]` sql/psql, `[D:x]` docker.

### 18.3 Status Colors

queued gray · running blue · waiting_approval purple · blocked orange · paused teal · error red · done green

---

## 19) One‑Screen Summary (for Execs)

* **Where are we?** Tree + roll‑up bar (done/running/blocked).
* **What’s it costing?** $/min and tokens/min with top spenders.
* **What’s risky?** Alerts panel (missed beats, stalled tasks, budget breaches).
* **What can I do?** Approve / Reassign / Pause‑Resume / Open Report.
