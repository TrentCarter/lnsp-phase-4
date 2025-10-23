PRD — VecRAG + LVM “Dual-Path Next-Vector Generation”
Purpose

Trent Carter
10/22/2025

1) Problem & Goal
Our LVM is a generative, autoregressive next-vector predictor. It can produce a valid next concept that may not exist in the 771k bank (unlike token LLMs that must pick from a fixed vocab). However, when the next concept does exist (or is close) in the bank, we want to ground efficiently. Today’s blocker is Stage-1 recall (0.65% Hit@5 end-to-end) due to poor query formation. We will add a Two-Tower retriever to form good queries for FAISS, and then decide per step whether to (a) snap/blend to a nearby bank vector (grounded) or (b) keep the novel generated vector. Result: strong Recall@K for grounding when useful, without sacrificing novel generation.

2) Context: LLMs vs. Our LVM
* Token LLMs: fixed vocab (~50k–100k). They predict logits over that fixed set; no retrieval inside the model.
* Our LVM: no token layer; it predicts a 768-D vector directly. The correct next concept may be off-bank.
Implication: We need fast external retrieval to search a huge “dynamic dictionary” (bank) when grounding helps; but we must not force picks from the bank when novel is better.

3) Scope & Non-Goals
In scope
* Two-Tower retriever (query tower + doc tower identity) for high-recall candidate generation.
* Dual-Path decoder: per step choose snap/blend (grounded) vs novel (free vector).
* TMD policy to set thresholds per lane (e.g., legal = more grounding; creative = more novelty).
* Training/eval to report both grounded and novel quality.
Not in scope
* Replacing LVM with a token LLM.
* Forcing bank-only decoding.
* Rewriting vec2text.

4) Users & Stories
* Architect/CEO: “I need a system that can both invent new sentences and ground to facts when available.”
* Engineer: “Give me a clean module: retriever → shortlist; LVM outputs a vector; switch/blend picks the final vector; metrics show when/why we snapped vs. stayed novel.”
* Evaluator: “I want clear reports: Recall@K, %Novel, grounded quality, novel quality.”

5) System Overview
flowchart LR
  U[User Text] --> E[GTR-T5 Embedder]
  E --> C[Context: 5–100 × 768D]
  C --> Q[Two-Tower Query\nGRU(context 100×768) → 768D]
  D[(Doc Tower = identity on 768D bank)\nPre-indexed in FAISS]
  Q --> R[FAISS on 771k\nTop-500 • Recall@500 ≈ 55–60%]
  D -. informs .- R
  C --> L[LVM: autoregressive\nnext-vector predictor (768D)]
  R --> S[Snap/Blend Decision]
  L --> S
  S -->|if snap/blend| G[Grounded/Blended 768D]
  S -->|if novel| N[Novel 768D]
  G --> V[vec2text] --> O[Text]
  N --> V
Legend:LVM always generates a free next vector.Two-Tower+FAISS returns a shortlist only.Decision: snap/blend to bank when nearby, otherwise keep the novel vector.

6) Detailed Behavior — Dual-Path Decision
At each step t:
LVM generation:  v̂ₜ = f_LVM(v₁:ₜ₋₁) ∈ ℝ⁷⁶⁸ (unit-norm)
Retriever shortlist (optional but default-on):  {nᵢ} = FAISS(f_q(context)), K=500, each nᵢ ∈ ℝ⁷⁶⁸ (unit-norm)
Decision: snap / blend / novel
* Compute c = maxᵢ cos(v̂ₜ, nᵢ); let n* = argmax
* Snap if c ≥ τ_snap (e.g., 0.92)
* Novel if c ≤ τ_novel (e.g., 0.85)
* Blend if τ_novel < c < τ_snap:  vₜ = α(c) · v̂ₜ + (1 − α(c)) · n*
α(c) increases with c
TMD can override thresholds per lane.Decode vₜ via vec2text.
Notes:Retrieval is advisory: it proposes candidates; it never blocks novelty.Snap/blend keeps us on-manifold when the bank already contains a great vector; novelty lets us create new content.

7) Two-Tower Retriever
* Query tower f_q: GRU (or LSTM) + pooling → 768D, trained with InfoNCE + curriculum hard negatives.
* Doc tower f_d: identity (use bank vectors as-is).
* Index: FAISS (IVF/HNSW/Flat), async batched mining.
* Target: Recall@500 ≥ 55–60% on full bank.
Why: Phase-3 optimized small candidate reranking, not global search. Two-Tower learns query formation so we can find good neighbors when grounding is helpful.

8) Functional Requirements
* LVM output is primary: system must always produce v̂ₜ (novel allowed).
* Retriever candidate pool: top-K (default K=500; tunable 200–1000).
* Decision module: snap / blend / novel, with thresholds τ_snap, τ_novel, and α(c) schedule.
* TMD policy: per-lane thresholds and blending rules.
* Async mining: retrieval runs overlapped; training must not block.
* Telemetry: log per step: c_max, decision (SNAP/BLEND/NOVEL), lane, and nearest neighbor ID.

9) Non-Functional Requirements
* Latency: decision ≤ 5 ms avg (excluding FAISS query).
* Stability: retriever failure → default to NOVEL path.
* Determinism: seedable evaluation mode.
* Scalability: bank up to billions (via sharding); recall metrics must remain meaningful.

10) Metrics & Acceptance Criteria
* Retriever: Recall@{10,100,500,1000}; gate: ≥55% Recall@500.
* Decision behavior: %SNAP / %BLEND / %NOVEL overall and by TMD lane.
* Grounded quality: cosine to reference vector + vec2text semantic sim.
* Novel quality: vec2text BLEU/ROUGE + embedding sim to reference or next authored vec.
* End-to-end: Global Hit@5 (when groundable): 10–20% expected (vs. current 0.65%).
Pass if: Retriever gate met and mixed report shows coherent %NOVEL vs %SNAP/BLEND with quality at/above Phase-3 baselines.

11) Design Details
* Decision defaults:  τ_snap = 0.92  τ_novel = 0.85  α(c): linear from 0.3 at 0.86 to 0.7 at 0.91 (cap 0.9 at 0.95+)
* TMD overrides:  Legal: snap=0.94, novel=0.88  Creative: snap=0.90, novel=0.82
* Async retriever:  Batched FAISS (qbatch=1024–2048), prefetch queue (depth 2–3), TTL cache(=3–5)  Indices-only return; gather vectors via index_select from CPU bank (fp16 optional)
* Failure modes:  Timeout or empty queue → NOVEL  If c_max > 0.98 and near duplicate → drop or NOVEL

12) Plan & Milestones
MVP (2–3 days)
* Implement decision module + TMD hooks
* Integrate Two-Tower (v4) and async mining
* Report %SNAP/BLEND/NOVEL + Recall@K
Gate Review
* Pass if Recall@500 ≥ 55% and telemetry sane by lane
Production hardening (1–2 days)
* Tune thresholds per lane
* Add config profiles (Conservative/Neutral/Creative)
* Add guardrails (fallback to NOVEL)
* Ship dashboards

13) Risks & Mitigations
* Over-snapping harms novelty → Per-lane thresholds, widen novel band, log %NOVEL target
* Retriever stalls → Async + TTL + fallback
* Bank bias → Keep blend path; periodic novel-only ablation
