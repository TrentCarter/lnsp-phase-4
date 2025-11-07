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
* **Audio Service:** FastAPI @ **6103** (TTS, MIDI notes, tones).
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

### 4.3 Sequencer (Activity Timeline) — MIDI-Style Task Visualizer

**Layout:**
* **Horizontal axis:** Time (scrolling left-to-right, like DAW/MIDI sequencer).
* **Vertical axis:** Each agent is a row (grouped by tier: VP → Directors → Managers → Workers).
* **Playhead:** Moving vertical line showing "now" (auto-scroll or manual scrub).

**Visual Encoding (Task "Notes"):**
* **Note blocks:** Rectangles representing agent tasks/actions.
* **Length:** Task duration (start → end time).
* **Height:** Row height (consistent per agent).
* **Color:** Task status/progress:
  * 🟦 **Blue:** Running (0-25% complete)
  * 🟨 **Yellow:** Running (25-75% complete)
  * 🟩 **Green:** Running (75-99% complete) or Done (100%)
  * 🟧 **Orange:** Blocked/Waiting
  * 🟪 **Purple:** Awaiting Approval
  * 🟥 **RED:** Stuck/Error (no progress for >2 heartbeat intervals)
* **Opacity:** Progress percentage (0% = 0.4 opacity, 100% = 1.0 opacity).
* **Border:** Thick border if task is currently active/selected.

**Interactions:**
* **Click note:** Show task details tooltip (task ID, agent, duration, status, tokens used).
* **Hover:** Highlight corresponding agent in tree view (if visible).
* **Zoom:** Mouse wheel to zoom in/out on time axis.
* **Pan:** Drag to scroll timeline left/right (optional scrollbars can be hidden via settings).
* **Playhead:** Click timeline to jump to timestamp, or drag playhead handle directly.
* **Draggable Playhead:** Red circle handle on playhead for precise scrubbing (cursor: grab/grabbing).

**Playback Controls (Toolbar):**
* **Play/Pause/Stop:** Standard playback controls with animated playhead.
* **Playback Speed:** Dual sliders (top toolbar + bottom bar) for speed control:
  * Range: 0.1x to 100x (default: 1.0x)
  * Non-linear scaling for intuitive control:
    * 0-50%: 0.1x to 1.0x (slow to normal, linear)
    * 50-75%: 1.0x to 10x (normal to fast, exponential)
    * 75-100%: 10x to 100x (fast to ultra-fast, exponential)
  * Live display shows current speed (e.g., "1.5x", "25.0x")
  * Both sliders synchronized in real-time
  * Setting persists across page loads
* **Sound Mode:** Dropdown selector for audio playback:
  * **None** (default) — Silent mode
  * **Voice** — Text-to-speech announcements for task events
  * **Music Note** — Musical notes mapped to task events (pitch = tier)
  * **Random Sounds** — Random sound effects for variety
* **Time Range:** Dropdown to select visible time window (5min to 4hr).
* **Zoom:** Controls to zoom in/out on timeline (10%-1000%).
* **Refresh:** Manual data refresh button.

**Row Controls:**
* **Solo:** Mute all other agents (highlight only this agent's notes).
* **Mute:** Hide this agent's notes from view.
* **Color-code by tier:** VP (dark blue), Directors (blue), Managers (cyan), Workers (light blue).

### 4.4 Sonification (Musical Notes)

* Optional audio stream: short, rate‑limited notes for task events (assign, start, progress, done).
* **Pitch:** higher = lower tier; **instrument:** differentiates agents; **note length:** task duration hint.

### 4.5 Spoken Summaries (Narration)

* One‑sentence summaries on assign/complete events.
* **Voice depth:** maps to hierarchy (VP = deep; workers = light).
* Mute per‑tier and per‑agent; batch mode for “digest” every N minutes.

### 4.6 Timeline Scrubber (Replay)

* Scrub backward to replay the last 24–72h with the sequencer/graph synchronized; export as MP4/GIF for post‑mortems.

### 4.7 Task Status Indicator (Header)

**Location:** Header bar, between "PAS Agent Swarm (OK)" badge and navigation tabs.

**Purpose:** Provide at-a-glance visibility of current active task without consuming screen space.

**Visual Design:**
* **Compact Layout:** Max width 350px, does not push navigation tabs.
* **LED Indicator:** Animated status light (10px circle) with glow effects.
* **Task Name:** Current active task name (truncated with ellipsis if too long).
* **Status Label:** Uppercase status text (RUNNING, DONE, ERROR, etc.).

**Status Colors & Animations:**
* 🔵 **RUNNING** — Blue LED, pulsing animation (2s cycle).
* 🟢 **DONE/COMPLETED** — Green LED, steady glow.
* 🔴 **ERROR/STUCK/FAILED** — Red LED, fast pulsing (1s cycle).
* 🟠 **BLOCKED/WAITING** — Orange LED, steady glow.
* 🟣 **AWAITING APPROVAL** — Purple LED, steady glow.
* ⚪ **IDLE** — Gray LED, no animation (hidden by default).

**Behavior:**
* **Auto-Hide:** Hidden when no active tasks detected.
* **Auto-Show:** Appears when task starts (job_started event).
* **Real-Time Updates:**
  * Polls `/api/current-task` every 5 seconds.
  * Updates on WebSocket events (heartbeat, completed, error, blocked).
  * Shows recently completed tasks for 10 seconds before hiding.
* **Graceful Degradation:** Falls back to polling if WebSocket unavailable.

**API Integration:**
* **Endpoint:** `GET /api/current-task`
* **Returns:** Most recent active task or `null` if none.
* **Sources:** Event Stream (recent 50 events) with status inference.

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

**Agent Management:**
* Pause/Resume agent or subtree.
* Reassign task to different agent (via Gateway target filters).
* Restart worker (kills run, preserves artifacts).
* Approve/Reject pending gates (PRs, destructive ops).

**View Navigation:**
* Toggle views: Dashboard, Tree View, Sequencer (navigation tabs).
* Settings panel (⚙️ button) — Global configuration with persistence.

**Sequencer Playback:**
* **Play/Pause/Stop:** Control playback animation.
* **Playback Speed:** Dual sliders (top toolbar + bottom bar):
  * Range: 0.1x to 100x (non-linear scaling)
  * Synchronized sliders update in real-time
  * Setting persists via localStorage
  * Allows rapid replay (100x) or slow-motion analysis (0.1x)
* **Draggable Playhead:** Click/drag red circle handle to scrub timeline.
* **Time Range:** Dropdown to select visible window (5min to 4hr).
* **Zoom:** Controls to zoom timeline (10%-1000%).
* **Sound Mode:** Dropdown to select audio output:
  * None, Voice, Music Note, Random Sounds
  * Integrates with master audio settings

**Audio Controls:**
* Mute/solo tiers for audio; set narration cadence (live vs digest).
* Master audio toggle (enable/disable all sound).
* Individual toggles for sequencer notes and agent voice.
* Volume slider (0-100%) affecting all audio output.

**Settings Panel Controls:**
* **Auto-Refresh:** Toggle + interval (5-300 seconds).
* **Display:** Tooltips, compact mode, time zone selection.
* **Sequencer:** Hide scrollbars, default playback speed, default sound mode.
* **Audio:** Master toggle, sequencer notes, agent voice, volume.
* **Performance:** Animation duration (0-2000ms).
* **Reset to Defaults:** Restore factory settings (with confirmation).

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

### 10.1 Sound Controls (Settings Panel)

**Audio Settings Section (🔊 Audio):**
* **Enable/Disable Master Audio:** Global toggle for all sound output (default: OFF).
* **Sequencer Notes:** Musical note sonification for task events (assign, start, progress, complete).
  * Individual toggle for sequencer notes (default: OFF)
  * Volume slider (0-100%, default: 70%)
  * Pitch mapping: VP (low) → Directors → Managers → Workers (high)
* **Agent Voice Status:** Text-to-speech narration of agent status updates.
  * Individual toggle for spoken status (default: OFF)
  * Volume slider (0-100%, default: 70%)
  * Voice depth mapping: VP (deep) → Workers (light)
  * Per-tier mute controls
* **Audio Volume:** Master volume slider (0-100%, default: 70%).
  * Real-time percentage display
  * Affects both notes and voice
* **Audio Rate Limiting:** ≤8 notes/sec global; ≤2 notes/sec per-agent (prevent audio chaos).
* **Ducking:** Auto-reduce music notes during voice narration to ensure clarity.

**Sequencer Settings Section (🎹 Sequencer):**
* **Hide Scrollbars:** Toggle to use draggable playhead instead of scrollbars (default: ON).
  * When enabled: Canvas wrapper has `overflow: hidden`
  * When disabled: Standard scrollbars appear for navigation
* **Default Playback Speed:** Initial playback speed multiplier (0.1x to 100x, default: 1.0x).
  * Applies when sequencer first loads
  * User can adjust live via dual sliders (top + bottom)
  * Non-linear scaling: 0-50% → 0.1x-1.0x, 50-75% → 1.0x-10x, 75-100% → 10x-100x
* **Default Sound Mode:** Initial sound output mode (default: None).
  * Options: None, Voice, Music Note, Random Sounds
  * User can change via toolbar dropdown during playback

**Display Settings:**
* **Time Zone:** Display time zone for all timestamps (default: EST / America/New_York).
  * Options: EST, CST, MST, PST, UTC, GMT, JST
  * Applies to Dashboard, Tree View, and Sequencer
* **Show Tooltips:** Display detailed info on hover (default: ON).
* **Compact Mode:** Reduce spacing for higher information density (default: OFF).

**Performance Settings:**
* **Animation Duration:** Transition speed for visual updates (0-2000ms, default: 750ms).
  * Affects tree transitions, sequencer updates, and UI animations

**Settings Persistence:**
* **localStorage Integration:** All settings automatically saved to browser storage.
  * Settings survive page reloads and browser restarts
  * Per-user, per-browser storage
* **Reset to Defaults:** Button in settings footer to restore factory defaults.
  * Confirmation dialog before resetting
  * Clears localStorage key: `pas_hmi_settings`

---

## 11) Extensibility & Integrations

* **Plugins:** add visualization panels, custom alert rules, new audio instruments/packs.
* **Claude /agents:** ship a *HMI Operator* subagent with Gateway tool for approvals and view toggles.
* **AR Prototype:** optional “Holographic Task Cube” (WebXR) in Phase‑3.

---

## 12) API (HMI Facade)

**Service Discovery & Status:**
* `GET /api/services` → List all registered services from Registry.
* `GET /api/tree` → Agent hierarchy tree (D3.js-compatible JSON).
* `GET /api/metrics` → Aggregated metrics from all services.
* `GET /api/alerts` → Current alerts from Heartbeat Monitor.

**Task & Timeline:**
* `GET /api/current-task` → Most recent active task (for header indicator).
  * Returns: `{task: {...}}` or `{task: null}`
  * Sources: Event Stream (recent 50 events)
* `GET /api/sequencer` → Sequencer timeline data (agents + tasks).
  * Returns: `{agents: [...], tasks: [...], timestamp}`
  * Tasks include: task_id, agent_id, name, status, progress, start_time, end_time

**Cost & Budget:**
* `GET /api/costs?window=minute` → Cost metrics from Gateway.
* `GET /api/costs/receipts/:run_id` → Cost receipts for specific run.
* `GET /api/costs/budget/:run_id` → Budget status for specific run.

**Real-Time Events:**
* `WS ws://localhost:6102` → WebSocket stream (via Event Stream service).
  * Events: heartbeat, job_started, job_completed, error, blocked, etc.
  * Client subscribes on connect, receives event history + live updates.

**Control Actions (Future):**
* `POST /api/action` → pause/resume/reassign/approve with reason.
  * Body: `{action, target, reason}`
  * Returns: `{status, message}`

**Health Checks:**
* `GET /health` → HMI app health status.
* Returns: `{status: "ok", service: "hmi_app", port: 6101, timestamp}`

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

* ✅ ~~Preferred TTS backend for narration~~ → **RESOLVED**: f5_tts_mlx (local, Apple Silicon optimized)
* Default audio pack/instrument set → Currently using generated tones, could add custom samples
* Minimum retention for events vs audio sidecars (30/90 days?).
* Whether to auto‑suggest scaling ("add N workers") from HMI based on saturation.
* How to handle concurrent audio (mixing strategy, priority queue)?

---

## 18) Implementation Status (as of 2025-11-07)

### ✅ Completed Features

#### Dashboard View
- ✅ Service cards with live status (running/error/idle)
- ✅ Real-time metrics (latency, throughput, success rate)
- ✅ Cost metrics visualization (per-minute window)
- ✅ Auto-refresh with configurable interval (5-300 seconds)
- ✅ Persistent settings via localStorage

#### Tree View
- ✅ D3.js hierarchical tree visualization
- ✅ Parent-child agent relationships
- ✅ Node color coding by status
- ✅ State preservation during refresh (expanded nodes, zoom/pan)
- ✅ Smart refresh (updates colors/stats without moving tree)
- ✅ Interactive node collapse/expand
- ✅ Auto-refresh with settings integration
- ✅ **Orientation dropdown** (Top ⬇️, Left ➡️, Right ⬅️, Bottom ⬆️):
  - Real-time layout switching
  - Persistent orientation saved to settings
  - Proper text alignment and link paths for each orientation
  - Available in toolbar and Settings modal
- ✅ **Auto-refresh bug fix**: Now respects "Enable Auto-Refresh" setting for WebSocket events

#### Sequencer View (MIDI-Style Timeline)
- ✅ Canvas-based task timeline rendering
- ✅ Horizontal time axis with auto-scaling grid
- ✅ Agent rows sorted by tier (VP → Directors → Managers → Workers)
- ✅ Color-coded task blocks by status/progress
- ✅ Interactive playhead with draggable red circle handle
- ✅ Play/Pause/Stop playback controls
- ✅ **Non-linear playback speed (0.1x-100x)**:
  - Slider range: 0-100 (percentage position)
  - 0-50%: 0.1x to 1.0x (linear scaling)
  - 50-75%: 1.0x to 10x (exponential, t²)
  - 75-100%: 10x to 100x (exponential, t²)
  - Round-trip accuracy: <0.0001 error
  - Smart formatting: 0.00x, 0.0x, or 0x based on magnitude
- ✅ Dual synchronized playback sliders (top toolbar + bottom bar)
- ✅ Sound mode dropdown (None/Voice/Music/Random) — UI complete
- ✅ Zoom controls (10%-1000%)
- ✅ Time range selector (5min to 4hr)
- ✅ Task tooltips on hover (name, agent, status, progress, duration)
- ✅ Click-to-scrub timeline
- ✅ Scrollbar visibility toggle via settings

#### Actions View (Hierarchical Task Flow Log) — **2025-11-06**
- ✅ **Two-panel layout**: Tasks sidebar + Action tree main view
- ✅ **Hierarchical action tree** showing agent-to-agent communication flows:
  - Parent-child relationships via `parent_log_id`
  - Multi-level nesting (unlimited depth)
  - Example: `VP_ENG → Dir_SW → SW-MGR_1 → Programmer_1 → work → responses back up`
- ✅ **Task list with metadata**:
  - Action count, agent involvement, timestamps
  - Search/filter by task ID
  - Click to load hierarchical action tree
- ✅ **Expandable tree nodes**:
  - Individual node expand/collapse (click arrow icon)
  - "Expand All" button (⬇️) — Expands entire task tree instantly
  - "Collapse All" button (⬆️) — Collapses entire tree
  - State preserved during auto-refresh (30s interval)
- ✅ **Agent flow visualization**:
  - From/to agent badges with arrow (→) indicator
  - Action name and type display
  - Timestamp (relative: "just now", "5m ago", etc.)
- ✅ **Status indicators** (color-coded):
  - ✅ Completed (green)
  - 🔵 Running (blue)
  - ⚠️ Blocked (orange)
  - ❌ Error (red)
- ✅ **Token-based metrics** (AI agent system):
  - `estimated_tokens` — Estimated token usage
  - `estimated_task_points` — Task complexity (story points)
  - `tokens_used` — Actual tokens consumed
  - `task_duration` — Actual time taken
  - `total_cost_usd` — Total cost in USD
- ✅ **Action data display** (JSON):
  - Expandable action details
  - File changes, test results, blockers, etc.
- ✅ **Auto-refresh** with state preservation:
  - Refreshes every 30 seconds
  - Preserves expanded/collapsed state
  - Cleared when switching tasks
- ✅ **Empty state handling**: Helpful messages for no tasks/actions

#### Settings System
- ✅ Persistent settings with localStorage
- ✅ Reset to defaults functionality
- ✅ Auto-refresh toggle and interval control (5-300 seconds)
- ✅ Display settings (tooltips, compact mode, timezone)
- ✅ Tree View settings (orientation: top/left/right/bottom)
- ✅ Sequencer settings (hide scrollbars, default speed, default sound)
- ✅ Audio settings (master toggle, sequencer notes, agent voice, volume)
- ✅ Performance settings (animation duration 0-2000ms)
- ✅ Timezone selector (EST/PST/UTC/etc., default: EST)
- ✅ Settings validation and bounds checking

#### Task Status Indicator
- ✅ Compact LED indicator in header bar
- ✅ Shows current active task name
- ✅ Animated status LED (RUNNING/DONE/ERROR/IDLE/etc.)
- ✅ Color-coded states:
  - 🟢 Green (done)
  - 🔵 Blue (running, pulsing animation)
  - 🟡 Yellow (waiting/blocked)
  - 🔴 Red (error/stuck)
  - ⚪ Gray (idle)
- ✅ Auto-hides when no active task
- ✅ 5-second polling + WebSocket updates

#### API Endpoints

**HMI Service (Port 6101)**:
- ✅ `/api/services` — Service registry data
- ✅ `/api/metrics` — Performance metrics
- ✅ `/api/costs` — Cost tracking with time windows
- ✅ `/api/tree` — Hierarchical agent tree
- ✅ `/api/sequencer` — Timeline data (agents + tasks)
- ✅ `/api/current-task` — Active task status
- ✅ `/api/actions/tasks` — List all tasks from action logs
- ✅ `/api/actions/task/<task_id>` — Get hierarchical actions for specific task
- ✅ `/api/actions/log` — Log new action (proxy to Registry)
- ✅ `/health` — Service health check

**Registry Service (Port 6121)**:
- ✅ `POST /action_logs` — Log new action/message
- ✅ `GET /action_logs/tasks` — List all tasks with summary metadata
- ✅ `GET /action_logs/task/{task_id}` — Get hierarchical action tree for task

**Audio Service (Port 6103)**:
- ✅ `POST /audio/tts` — Text-to-speech synthesis (f5_tts_mlx)
- ✅ `POST /audio/note` — MIDI note playback (21-108)
- ✅ `POST /audio/tone` — Tone/beep generation
- ✅ `POST /audio/play` — Audio file playback
- ✅ `POST /audio/volume` — Master volume control
- ✅ `POST /audio/enable` — Enable/disable features
- ✅ `GET /health` — Audio service health
- ✅ `GET /status` — Current playback status

#### Technical Infrastructure
- ✅ Flask backend @ port 6101
- ✅ WebSocket integration @ port 6102
- ✅ **Audio Service @ port 6103** (FastAPI with f5_tts_mlx)
- ✅ D3.js for tree visualization
- ✅ HTML5 Canvas for sequencer rendering
- ✅ Real-time event processing from Event Stream
- ✅ Service Registry integration @ port 6121
- ✅ JavaScript audio integration in base.html (TTS, notes, tones)

#### Audio Playback (NEW - 2025-11-07)
- ✅ **Unified Audio Service** @ port 6103 (FastAPI)
- ✅ **Text-to-Speech (TTS)** using f5_tts_mlx:
  - Reference voice: Sophia3.wav (352KB)
  - Speed control (0.5x-2.0x)
  - Generation methods (midpoint, euler, rk4)
  - Auto-play option
  - ~1-3 seconds per sentence (Apple Silicon MLX)
- ✅ **MIDI Note Playback**:
  - Full MIDI range (21-108, A4=440Hz)
  - Event-to-note mapping (task_assigned=C4, completed=C5, error=C3)
  - Duration and velocity control
  - Multiple waveforms (piano, sine, square, sawtooth)
- ✅ **Tone/Beep Generation**:
  - Frequency range (20Hz-20kHz)
  - Multiple waveforms
  - Fade in/out (anti-click)
  - Alert type mapping (success=800Hz, error=200Hz)
- ✅ **Volume Control**:
  - Master volume (0.0-1.0)
  - Per-sound volume override
  - Synced with HMI settings (0-100%)
- ✅ **Frontend Integration**:
  - `speakStatus(text, speed)` — TTS helper
  - `playNoteForEvent(eventType)` — Sequencer notes
  - `playAlert(type)` — Alert tones
  - `checkAudioService()` — Health check
  - Settings-aware (respects Master Audio, TTS, Notes toggles)
- ✅ **Startup Script**: `./scripts/start_audio_service.sh`
- ✅ **Documentation**: `docs/AUDIO_SERVICE_API.md` (comprehensive guide)
- ✅ **Concurrent Playback**: Multiple sounds can overlap

**Performance**:
- TTS: ~1-3s generation time (MLX optimized)
- Tones: <100ms
- Notes: <100ms

**Output Directory**: `/tmp/pas_audio/` (temporary WAV files)

### 🔲 Not Yet Implemented

#### Tree View Enhancements
- 🔲 Edge animations (message throughput)
- 🔲 Node size encoding (load/tokens)
- 🔲 3D orientation mode

#### Cost Visualization
- 🔲 Detailed cost breakdown by agent/tier
- 🔲 Budget alerts and thresholds
- 🔲 Top N spenders list

#### Agent Interaction
- 🔲 Approval workflow UI
- 🔲 Task reassignment controls
- 🔲 Pause/Resume/Kill actions

#### Advanced Features (P2+)
- 🔲 Log viewer with filtering
- 🔲 Save-State UI integration
- 🔲 Dependency graph overlay
- 🔲 AR/VR holographic task cube

### 📊 Test Coverage

#### Non-Linear Playback Speed
- ✅ Boundary tests: 0→0.1x, 50→1.0x, 75→10x, 100→100x
- ✅ Round-trip tests: Slider→Speed→Slider (0.0000 error)
- ✅ Round-trip tests: Speed→Slider→Speed (0.0000 error)
- ✅ Exponential growth verified across all ranges

#### Service Integration
- ✅ Health checks passing (port 6101)
- ✅ API endpoints returning valid data
- ✅ WebSocket connection stable
- ✅ Settings persistence across page reloads

### 📝 Documentation

- ✅ PRD updated with implementation status
- ✅ Code comments for all major functions
- ✅ Inline documentation for scaling algorithms
- ✅ API response format examples

### 🎯 Next Priorities

1. **Sequencer Audio Integration** — Connect audio service to sequencer events (play notes on task start/complete)
2. **Cost Dashboard Enhancement** — Detailed breakdown and budget alerts
3. **Approval Workflow UI** — Interactive approval interface
4. **Tree View Edge Animations** — Animate message flow between agents
5. **Advanced Audio Features**:
   - Pitch mapping by agent tier (VP=low, Workers=high)
   - Rate limiting (≤8 notes/sec global)
   - Custom voice samples for different agent types

---

## 19) Appendix — Quick Mappings & Tables

### 19.1 Event Types

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

### 19.2 Rights (perm‑codes)

`[F:rw]` filesystem, `[B:x]` bash, `[G:x]` git, `[P:x]` python, `[N:rw]` network, `[S:x]` sql/psql, `[D:x]` docker.

### 19.3 Status Colors

queued gray · running blue · waiting_approval purple · blocked orange · paused teal · error red · done green

---

## 20) One‑Screen Summary (for Execs)

* **Where are we?** Tree + roll‑up bar (done/running/blocked).
* **What’s it costing?** $/min and tokens/min with top spenders.
* **What’s risky?** Alerts panel (missed beats, stalled tasks, budget breaches).
* **What can I do?** Approve / Reassign / Pause‑Resume / Open Report.
