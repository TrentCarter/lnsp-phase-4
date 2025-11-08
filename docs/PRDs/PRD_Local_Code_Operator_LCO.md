# PRD Local Code Operator LCO

## PRD_Local_Code_Operator_LCO.md

_(a.k.a. “VP of Engineering” primary agent with local file-system access + multi-kickoff interfaces)_

11/7/2025
Trent Carter

## 1) Summary (what & why)

We’re shipping a **local, sandboxed code operator** that acts as the user’s entry point (“VP of Engineering” agent). It runs on the developer’s machine (terminal-first), exposes a small command API (JSON-RPC over stdio/WebSocket), and can **read/write/edit files, run tests, call git, and talk to PAS/PLMS/HMI**. It can delegate to **external LLMs** but never requires them to have direct repo access—the agent is the bridge. This gives us **one kickoff surface** that works from Terminal, VS Code, Cursor, and Windsurf (all can drive a terminal or a small localhost port).

We’ll ship with two **open-source code-model options** pre-configured (for fully local execution):

- **Qwen-Coder** (code-tuned, strong general coding)
    
- **Phi-code (latest Phi release with code weights)** (compact, fast local inference)  
    (You can swap these for any local or API model via the model broker.)
    

## 2) Users & primary jobs

- **Human (engineer/manager):** declares an initiative, approves plans/budgets, watches HMI, steps in only when needed.
    
- **VP Agent (this component):** plans kickoff, collects context, files tasks with PLMS, streams execution via PAS, performs local code ops (edit, run, test, commit).
    
- **PLMS/PAS:** planning gates, estimates, execution, receipts, KPIs, calibration (already live).
    

## 3) Goals & Success metrics

**Goals**

- Kick off projects from terminal/IDE with minimal friction.
    
- Safely give the agent **local file access & command execution** in a sandbox.
    
- Broker across **multiple LLMs** (local/API) by cost/capability policies.
    
- Integrate tightly with **PLMS/PAS/HMI** (idempotency, receipts, KPIs).
    

**V1 success**

- Start a project with `vp start` and complete end-to-end through PAS, with artifacts committed to git.
    
- > 95% of operations require **no manual file edits** by the human.
    
- CI passes; KPI gates enforce completion.
    

## 4) Scope (V1) / Non-goals

**In scope (V1)**

- Terminal client + local service (single binary or Python entrypoint)
    
- Minimal **JSON-RPC** API (stdio & optional localhost WebSocket)
    
- Core file ops (read/patch/diff/write), code search, test run, build, git
    
- Model broker (choose local vs API provider by policy)
    
- PAS/PLMS integration (create runs, receipts, rehearsal, budget)
    
- Safety: path allowlist, command allowlist, dry-run, approval prompts
    
- Observability: token/time/cost/energy tracking, run passports
    

**Non-goals (V1)**

- IDE-native UI widgets (we rely on terminal/port; VS Code/Cursor/Windsurf can attach)
    
- Fine-grained corporate policy engines (basic RBAC and allowlists only)
    

## 5) Architecture (high-level)

`[Human]   │  (Terminal/IDE/Web)   ▼ [VP Agent: Local Code Operator]   ├─ File System Sandbox (allowlisted workspace)   ├─ Command Runner (build/test/lint)   ├─ Git Client (branch/commit/PR)   ├─ Model Broker:   │    ├─ Local: Qwen-Coder, Phi-code   │    └─ API: OpenAI/Gemini/Claude/… (context packaged by VP)   ├─ PAS Client (job cards, receipts, KPI emit)   └─ PLMS Client (plan/estimate/simulate/start/metrics)`

## 6) Kickoff surfaces (5 ways)

1. **Terminal-first (primary)**  
    `vp new`, `vp plan`, `vp simulate --rehearsal 0.01`, `vp start`, `vp status`, `vp logs`, `vp validate`, `vp complete`.
    
2. **IDE bridge** (VS Code / Cursor / Windsurf)  
    Extension invokes the terminal client under the hood; no duplicate logic.
    
3. **HTTP local port (optional)**  
    `--serve 127.0.0.1:7011` provides JSON-RPC over WebSocket for simple web UI buttons.
    
4. **CI kickoff**  
    Git hook or CI step hits local/runner to start a PLMS run with the current commit.
    
5. **API relay**  
    External systems post a PRD to PLMS; PLMS notifies VP Agent on the developer workstation to initialize the local workspace.
    

## 7) Command/API specification (essentials)

**CLI**

- `vp new --name <proj> [--template <id>]` → initializes local state, registers with PLMS.
    
- `vp plan [--from-prd docs/PRD.md]` → orchestrates PLMS clarify/PRD gen.
    
- `vp estimate` → PLMS estimates; prints CI bands.
    
- `vp simulate --rehearsal 0.01` → 1% canary via PAS; shows risk & runway.
    
- `vp start [--idempotency <uuid>]` → baseline run, writes replay passport.
    
- `vp status` / `vp logs` → tails PAS receipts, KPI results.
    
- `vp validate` → Echo + KPI gates; prints violations.
    
- `vp complete` → marks completion if gates pass.
    
- `vp model set <policy.json>` → set model broker policy (cost/capability/latency).
    
- `vp fs patch <diff.patch>` → apply unified diff (audit-safe edits).
    
- `vp git {branch|commit|push|pr}` → signed commits, PR metadata.
    

**JSON-RPC (subset)**

- `fs.read(path)` → `{content, sha}`
    
- `fs.patch(unified_diff)` → `{applied, hunks}`
    
- `search.code(query)` → `{matches:[{file, line, snippet}]}`
    
- `run.test(args)` → `{pass_rate, report_path}`
    
- `git.commit(message, signoff)` → `{sha}`
    
- `pas.submit(jobcard)` / `plms.*` → thin wrappers with idempotency header
    

## 8) Model broker (cost/capability policy)

- Policy file specifies **tiers** (Local-Small, Local-Medium, API-Premium) with:
    
    - max cost/min, max tokens/min, expected latency, context limits
        
    - required tools (e.g., code-edit, tool-use)
        
- Broker chooses cheapest tier that passes **capability tests** for the task (e.g., multi-file refactor requires tool-use + patch planning).
    
- **Provider snapshot** captured to the run passport for deterministic replay.
    

## 9) Security & safety

- Workspace **allowlist**; no traversal outside repo root.
    
- Command allowlist (`pytest`, `ruff`, `make`, `npm`, etc.), **no raw shell** by default.
    
- Dry-run for file edits; always store **unified diff** + backup.
    
- Secrets filter on outbound LLM context (env/keys masked).
    
- RBAC: “start/approve/budget override” require scopes (aligned with PLMS).
    

## 10) Telemetry & metrics

- Track **time**, **tokens** (input/output/tool-use or “thinking”), **cost**, **energy (estimated)** per task.
    
- Emit KPI receipts post-task (test_pass_rate, linter_pass, schema_diff, …).
    
- All runs produce a **replay_passport.json** and an artifact manifest.
    

## 11) Rollout plan

- **Week 1:** Terminal client + JSON-RPC; Qwen-Coder + Phi-code local; PAS/PLMS wiring.
    
- **Week 2:** IDE bridges (VS Code simple extension), HMI action buttons hitting local port.
    
- **Week 3:** Policy-driven model broker + red-team safety pass.
    

## 12) Risks & mitigations

- **Accidental destructive edits** → diff-only patches, auto-backups, dry-run default.
    
- **Provider drift** → provider snapshot & deterministic replay.
    
- **Context blowups** → VP agent performs **fractional window packing** and uses RAG (see addendum).