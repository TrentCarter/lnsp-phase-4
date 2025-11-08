# # PRD_Addendum_LightRAG_LearningLLM_Enhanced_Metrics.md

_(three enhancements aligned with PLMS/PAS/HMI)_
## Trent Carter
## 11/7/2025

## A) LightRAG for Codebase (with Vector Manager agent)

**Objective:** maintain a continuously updated **code+graph index** for precise retrieval and dependency-aware planning.

**Scope (V1):**


- Ingest repo on `vp new` and **on every commit** (post-commit hook).
    
- Build both **semantic vectors** and a **call/import graph** (tree-sitter/ctags).
    
- Expose queries: `where_defined(symbol)`, `who_calls(fn)`, `impact_set(file|symbol)`, `nearest_neighbors(snippet)`.
    
- **Vector Manager agent** owns refresh cadence, backfills, and integrity checks; surfaces drift warnings in HMI.
    

**APIs:**

- `rag.refresh(scope)` → repo or subpath.
    
- `rag.query(kind, payload)` → returns code locations + graph paths.
    
- `rag.snapshot()` → writes `rag_snapshot.json` bound to git SHA for reproducibility.
    

**KPIs / SLOs:**

- Index freshness ≤ 2 min from commit.
    
- Query latency P95 ≤ 300 ms (local).
    
- Coverage: ≥ 98% of files indexed; graph edges on par with static analyzer.
    

**Risks:**

- Large monorepos → shard index per submodule; lazy load on demand.
    

---

## B) Planner Learning LLM (project-experience model)

**Objective:** reduce cost/time by training a **planner-facing LLM** on **what worked** (per lane, per provider, per topology), both **locally (per project)** and **globally (portfolio)**.

**Data to learn from:**

- Task tree + assignments, lane ids, provider matrix, rehearsal outcomes, KPI passes/fails, budget runways, violations, rework counts.
    

**Pipeline:**

1. **After completion:** PLMS emits a **planner_training_pack.json** (sanitized).
    
2. **Trainer agent** fine-tunes or LoRA-adapts the Planner model (or updates a retrieval memory) with **dual partitions**: LOCAL(project) and GLOBAL(portfolio).
    
3. **A/B validation:** Re-run the **same project template** with the updated Planner (no human), compare units (time, tokens by type, cost, energy). Target ≥15% median improvement after 10 projects.
    

**Serving:**

- Planner uses **GLOBAL first**, overlays **LOCAL deltas** if the repo/team matches.
    
- **Cold-start:** fallback to default priors + CI bands.
    

**KPIs / SLOs:**

- Estimation MAE% drops over time (goal: ≤20% at 10 projects).
    
- Rework rate ↓, KPI violations ↓, budget overruns ↓.
    

**Risks:**

- Calibration poisoning → include only `(baseline|hotfix) ∧ validation_pass ∧ !sandbox`.
    
- Privacy → anonymize task text; keep only structured features when required.
    

---

## C) Multi-Metric Telemetry & Visualization (incl. Energy/Carbon)

**Objective:** track and visualize **time**, **tokens** (input/output/**tool-use/think**), **cost**, and **energy** (estimated) separately; allow custom **roll-ups** per stakeholder.

**Data model (already compatible):**

- Extend receipts to log token breakdown per step.
    
- Add energy estimator: `E ≈ (GPU_kW × active_time) + (CPU_kW × active_time)` with model-specific coefficients; store in receipts.
    

**HMI**

- **Stacked bars** per task and per lane (time / token types / cost / energy).
    
- **Budget runway** (already added) + **carbon overlay** for “green” stakeholders.
    
- **Compare runs**: show percent deltas to prior baselines for the same project template.
    

**APIs**

- `GET /metrics?with_ci=1&breakdown=all` → returns mean + CI for each metric and token subtype.
    
- `GET /compare?runA=…&runB=…` → structured diff with significance flags.
    

**KPIs / SLOs**

- Visualization latency ≤ 1s for recent projects.
    
- Metrics completeness ≥ 99% of steps report all four classes.
    

**Risks**

- Energy estimates imperfect → clearly label as **estimated** and show coefficient sources; allow per-cluster overrides.
    

---

## Cross-cutting: how these three fit PLMS/PAS/HMI

- **PLMS**: planning uses LightRAG to localize work; estimates borrow historical priors; rehearsal uses RAG to sample representative strata.
    
- **PAS**: Vector Manager triggers re-index after artifact writes; KPI receipts attach RAG lookups used.
    
- **HMI**: adds **RAG search widget**, **dependency graph view** (2D/3D), **learning gains** panel (units saved vs last baseline), and **multi-metric bars** with carbon overlay.
    

---

## Open questions (call these in stand-up)

1. **Rename “VP of Engineering”?** Proposal: **“Project Executive (PEX) Agent”** to avoid org-role confusion.
    
2. **Local model pack defaults?** Decide exact SKUs and VRAM targets for Mac/PC.
    
3. **Energy coefficients:** Which GPU/CPU profiles to ship by default?
    
4. **RAG indexer:** prefer tree-sitter or ctags+custom? (I recommend tree-sitter for rich edges; ctags as fallback.)
    
5. **Planner training cadence:** every run vs nightly batch (I recommend nightly).