# Research: Claude Code‑Like Open Source Options

Prepared for the PAS/LCO program to identify terminal-first coding agents that can be embedded into the polyglot swarm without license friction or closed-source constraints.

---

## Top 5 Open Source Terminal-Based Coding Agents

| Tool     | License | GitHub Stars | Language | Key Strengths for PAS/LCO | Cross-Platform | Local Model Support |
|----------|---------|--------------|----------|---------------------------|----------------|---------------------|
| **Aider** | Apache 2.0 | ~35.2K | Python | • Git-native auto-commit workflow<br>• Repository map for large monorepos<br>• Proven SWE-Bench track record<br>• Multi-file edit planning<br>• Terminal-first UX | ✅ Windows/macOS/Linux | ✅ Ollama, LM Studio, local HTTP |
| **Cline** | Apache 2.0 | ~48K | TypeScript | • IDE + terminal combo<br>• Browser automation (Computer Use)<br>• Snapshot/restore of task state<br>• Model-agnostic via adapters<br>• Real-time cost dashboard | ✅ Windows/macOS/Linux | ✅ LM Studio, Ollama |
| **OpenCode** | MIT | ~32K | Go | • Terminal-native Bubble Tea TUI<br>• Client/server separation<br>• Provider-agnostic pipeline<br>• Non-interactive scripting mode<br>• Fast static binary distribution | ✅ Windows/macOS/Linux | ✅ Ollama, Bedrock, local HTTP |
| **Goose** | Apache 2.0 | ~10K | Python | • Enterprise Block/Square heritage<br>• MCP (Model Context Protocol) native<br>• CLI and desktop shells<br>• Extensible agent graph<br>• Privacy-first execution | ✅ Windows/macOS/Linux | ✅ Any LLM (local/remote) |
| **Continue** | Apache 2.0 | ~29.6K | TypeScript | • IDE extension + CLI<br>• Customizable agent workflows<br>• Background worker mode<br>• Mission Control automation<br>• Config sharing hub | ✅ Windows/macOS/Linux | ✅ Ollama, local HTTP |

### Alignment with PRD Requirements

- **Commercial use:** All five tools are Apache 2.0 or MIT, allowing redistribution, modification, and resale.
- **Cross-platform:** Native support for Windows, macOS, and Linux; Android/iOS reachable via SSH flows.
- **Local model readiness:** Ollama, LM Studio, Bedrock-local, and generic OpenAI-compatible endpoints are supported across the board.
- **Filesystem + shell access:** All expose sandboxed workspace operations, git commands, and test runners through either CLI verbs or tool definitions.
- **Model-agnostic routing:** Each integrates multiple providers, so PAS’s Provider Router can swap model backends without patching the agent core.

### Hardware Readiness (M4 Max, 128 GB RAM)

All candidates can drive local Qwen 2.5 Coder (7B–34B), Llama 3.1 (70B quantized), and Phi-4 (14B) using Ollama/LM Studio on the M4 platform. The ample unified memory allows concurrent runs (e.g., PAS Architect + LCO executor) without swapping.

---

## Why Aider Leads for LCO

1. **Terminal-first architecture** mirrors the PRD’s “terminal client + local service” requirement.
2. **JSON-RPC friendly:** Python codebase makes it straightforward to wrap stdin/stdout with a JSON-RPC or WebSocket layer for PAS job cards.
3. **Git excellence:** Auto-commit, branch control, and diff-packing line up with `git.commit`, `git.branch`, `git.push`, `git.pr` actions in the PRD.
4. **Repository mapping:** Tree-sitter/ctags ingest provides the LightRAG hook you already spec’d for context retrieval.
5. **Model brokerage:** Built-in LiteLLM integration already tracks per-model cost/performance, feeding directly into PAS receipts.

### Integration Path for PAS / PLMS

- Fork Aider and add:
  - JSON-RPC stdio/WebSocket shim that speaks LCO API semantics.
  - PAS client wrapper (`pas.submit(job_card)`, `plms.*` calls).
  - Provider snapshot capture (model+temperature) for replay passports.
  - KPI/cost receipts emitted to PAS Token Governor + Experiment Ledger.
  - Budget middleware (enforce per-job token caps).
- Alternatively, fork OpenCode if you want a Go binary—but you must implement Aider-grade git automation before it meets the acceptance criteria.

---

## Deep Dive: Aider vs OpenCode

### Table 1 – PRD Feature Alignment Matrix

| PRD Requirement | Aider (Python) | OpenCode (Go) | Notes |
|-----------------|----------------|---------------|-------|
| JSON-RPC API | ❌ Not native; wrap via custom layer | ✅ HTTP + WebSocket server built-in | OpenCode ships client/server out of the box |
| File operations | ✅ Unified diff, SEARCH/REPLACE blocks, whole-file edits | ✅ View/edit/write tools, diff generation | Aider’s diff heuristics reduce unnecessary rewrites |
| Git integration | ✅ Auto-commit, conventional commits, pre-commit hooks, diff-in-context | ⚠️ Basic git; no auto-commit or summaries | Git rigor is a must-have for LCO |
| Model broker | ✅ 75+ models via LiteLLM, Architect/Editor split | ✅ 75+ providers, agent-specific models | Both strong; Aider has better delegation prompts |
| Test runner | ✅ `/test` command with auto-fix loop | ✅ Shell tool; manual orchestration | Aider embeds lint/test guardrails |
| Command allowlist | ⚠️ `/run` can execute anything; no guard | ✅ Permissioned tool invocations | OpenCode aligns with PAS security stance |
| Dry-run/sandbox | ✅ `/undo` via git; no preview mode | ⚠️ No dry-run; session snapshots | Both need Save-State previews |
| Secrets scrubbing | ❌ Not built-in | ❌ Not built-in | Needs PAS middleware regardless |
| Budget tracking | ✅ Token + $ analytics, exportable | ✅ Token tracking; coarse cost calc | Aider more mature but both acceptable |
| Telemetry | ✅ Input/output/thinking token breakdown | ⚠️ Aggregate counts only | PAS requires breakdown for receipts |
| Provider snapshots | ✅ Model versions logged per session | ⚠️ Config only | Aider easier to plug into replay passports |
| Codebase mapping | ✅ Tree-sitter map + ranking | ✅ LSP (gopls, tsserver) integration | Aider better for static repo map, OpenCode better for live diagnostics |
| Local models | ✅ Ollama, LM Studio, custom endpoints | ✅ Ollama, Bedrock-local, adapters | Tie |
| Multi-file editing | ✅ Coordinated change plans | ✅ Multi-file ops, less automation | Aider handles refactors better |
| Voice input | ✅ Whisper/voice commands | ❌ None | Nice-to-have unique to Aider |

**Score:** Aider 12 ✅, OpenCode 8 ✅, 3 ties.

### Table 2 – Architecture Comparison

| Dimension | Aider | OpenCode | Edge |
|-----------|-------|----------|------|
| Language | Python 3.8+ | Go 1.21+ | Go (single binary) |
| Pattern | Monolithic CLI | Client/server + TUI | OpenCode |
| Distribution | `pip install` scripts | ~30 MB static binary | OpenCode |
| Startup | ~0.5 s (Python VM) | ~0.1 s (Go binary) | OpenCode |
| Memory | ~150 MB | ~50 MB | OpenCode |
| Extensibility | Python plugin hooks | MCP servers + Go plugins | OpenCode |
| API surface | CLI stdin/stdout | REST + WebSocket | OpenCode |
| State | File-based analytics | SQLite session DB | OpenCode |
| Concurrency | Asyncio single-process | Goroutines | OpenCode |
| Deployment | Requires venv | Copy binary | OpenCode |
| Dev setup | Python toolchain | Go toolchain | Go simpler overall |
| Hot reload | ❌ Restart | ✅ Dev mode with Bun | OpenCode |
| Testing | `pytest` | `go test` | Tie |
| CI/CD | GitHub Actions | Turbo + Actions | Tie |
| Code maturity | 35K+ stars, SWE-Bench pedigree | 32K stars, younger | Aider |

**Verdict:** OpenCode wins on architecture & deployment ergonomics; Aider wins on battle-tested workflows and git-heavy features.

### Table 3 – Platform & Device Support Matrix

| Platform | Aider | OpenCode | Notes |
|----------|-------|----------|-------|
| Windows 11 (x64) | ✅ Python + WSL2 | ✅ Native binary + PowerShell | OpenCode smoother on stock Windows |
| Windows 11 (ARM) | ✅ Python universal | ✅ Cross-compiled Go | Tie |
| macOS Intel | ✅ Native | ✅ Native | Tie |
| macOS Apple Silicon | ✅ Native | ✅ Native | Tie |
| Linux distros | ✅ Ubuntu/Debian/Fedora/Arch | ✅ Same incl. static builds | Tie |
| FreeBSD | ⚠️ Python ports | ✅ Go build works | OpenCode edge |
| iOS/iPadOS | ⚠️ SSH via Blink/a-Shell | ⚠️ SSH (no native app) | Tie |
| Android | ⚠️ Termux + Python | ⚠️ Termux + Go binary | OpenCode slightly better (single binary) |
| Browsers (Chrome/Edge/Firefox/Safari) | ⚠️ Terminal via SSH or copy/paste | ✅ Native Web UI (opencode.ai) & Workers | OpenCode advantage |
| IDEs (VS Code, JetBrains) | ✅ Terminal plugins + aider.el | ✅ Official VS Code extension + shortcuts | OpenCode more integrated |
| Neovim/Emacs | ✅ Terminal friendly, aider.el | ✅ Terminal friendly | Tie |
| Remote dev (SSH/Docker/K8s) | ✅ Works; needs Python layer | ✅ Works; copy binary | OpenCode easier in containers |

---

## Additional Considerations

### Table 4 – Three Things Forgotten

| # | Gap | Why It Matters | Aider Status | OpenCode Status | Recommendation |
|---|-----|----------------|--------------|-----------------|----------------|
| 1 | Collaborative multi-user sessions | Pair programming, remote reviews, training scenarios | ❌ Single-user | ✅ Web UI sharing, Durable Objects | Add `vp share <run_id>` flow with read/observe mode |
| 2 | Offline / air-gapped mode | Government, finance, healthcare deployments | ✅ Works offline, no phone-home | ✅ Local models ok, but disable Cloudflare UI | Add `--airgap` flag that enforces local-only endpoints and certificate pinning |
| 3 | Immutable audit trails | Needed for SOC2, FDA, GDPR | ⚠️ Git history only | ⚠️ SQLite logs (mutable) | Add audit log with HMAC signatures + `vp audit export --signed` command |

### Table 5 – Three Novel Ideas

| # | Innovation | Physics Analogy | Implementation Sketch | Value |
|---|------------|-----------------|-----------------------|-------|
| 1 | **Quantum Superposition Branching** | Multiple solution branches collapse to best answer | `vp start --branches 3 --models "claude,gpt4,deepseek"` runs parallel PAS jobs scored on code quality, tests, cost | Higher success rate via parallel exploration |
| 2 | **Thermodynamic Energy Budget** | Minimize free energy, control entropy | Energy budget `E = tokens × cost`; Boltzmann distribution selects models; temperature drops as budget depletes | Automatic shift from premium to local models as funds shrink |
| 3 | **Holographic Context Projection** | 3D info encoded on 2D surface | Diffusion-map embedding of repo; LightRAG returns geodesic path fed as compressed instructions | 20× context compression with interpretable navigation hints |

---

## Recommendation & Next Steps

1. **Primary path:** Fork Aider.
   - Implement JSON-RPC/WebSocket shim, PAS/PLMS clients, secrets scrubber, and tamper-proof audit logging (leveraging Novel Idea #3).
2. **Secondary path:** Maintain an OpenCode fork for scenarios where a single Go binary and native API are mandatory; port Aider’s git discipline into it.
3. **Shared next steps:**
   - Wire both agents into PAS Token Governor for heartbeat + budget enforcement.
   - Add MCP client/server adapters so they can consume PAS transports (file queue, MCP, REST).
   - Stress-test local-model pipelines (Qwen 2.5 Coder, Llama 3.1, Phi-4) on the M4 Max to finalize quantization profiles.

With these adaptations, PAS gains a Claude Code–like operator that is fully auditable, air-gappable, and aligned with the PRD’s git-anchored workflow. Let me know if you want a follow-up note focused purely on the Aider fork architecture. 
