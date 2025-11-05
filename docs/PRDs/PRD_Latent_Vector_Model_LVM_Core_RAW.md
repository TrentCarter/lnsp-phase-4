Here is a Product Requirements Document (PRD) for the Latent Vector Model (LVM) concept, structured around your goals and incorporating the insights from our previous discussion.

-----

# Product Requirements Document: Latent Vector Model (LVM) Core

  * **Author**: Gemini
  * **Version**: 1.0
  * **Date**: 2025-11-03
  * **Status**: Draft
  * **Target**: LVM Core v1.0, LNSP (Latent Neurolese Semantic Processor) v1.0

## 1\. 🌍 Introduction & Vision

This document outlines the requirements for the **Latent Vector Model (LVM) Core**, the reasoning engine for the **Latent Neurolese Semantic Processor (LNSP)**.

The LNSP's vision is to create a reasoning system that overcomes the limitations of token-based processing. By operating directly in a high-dimensional conceptual (vector) space, the LNSP will perform "thinking" as a series of vector transformations, unburdened by linguistic syntax. This allows for a chain of thought that is purely semantic and logical. The final, synthesized vector-based "thought" is only translated into human-readable text at the final step.

The LVM is the heart of this system: a vector-in, vector-out model responsible for this internal conceptual "thinking" loop.

## 2\. 🎯 The Core Problem: LVM vs. LLM Parity

The primary challenge is achieving performance parity with token-based Large Language Models (LLMs). Our initial attempts to train an LVM on encyclopedic data (like Wikipedia) failed, not due to a model bug, but due to a fundamental **Data-Objective Mismatch**.

  * **LLM (Token-based):** An LLM's objective is simple: **predict the next token**. This task is atomic, local, and *always* structurally true for any text, regardless of its high-level semantic structure.
  * **LVM (Vector-based):** An LVM's objective must be: **predict the next concept**. This task is holistic, non-local, and, as we've proven, *not* structurally true for explanatory, backward-referential data like Wikipedia.

The LVM's failure was a *symptom* of being trained on data that is optimized for explanation (hub-and-spoke references), not for prediction (causal chains).

This PRD re-aligns the LVM's objective with a new, requisite data structure. **The LVM must be trained on data where the next concept is a logical consequence of the previous one.**

## 3\. 🏛️ LNSP System Architecture

The LNSP is a three-stage pipeline. The LVM Core (Stage 2) is the primary focus of this document.

### Stage 1: Encoder (Text-to-Concept)

  * **Function:** Ingests a text-based prompt and translates it into an initial context of concept vectors.
  * **Implementation:** A pre-trained text-to-vector model (e.g., GTR-T5) combined with our LLM-based TMD (Domain-Task-Modifier) extractor.
  * [cite\_start]**Output:** A sequence of 784-dimensional vectors (16D TMD + 768D embedding)[cite: 1], representing the initial state: `[c_1, c_2, ..., c_n]`.

### Stage 2: LVM Core (Concept-to-Concept "Thinking")

  * **Function:** The primary reasoning loop. It takes the current vector context and synthesizes a *new emergent concept* (a "thought"). This new vector is appended to the context, and the process repeats.
  * **Implementation:** The LVM (e.g., VMMoE, Transformer) trained on the Causal Corpus (see Section 4).
  * **Loop:**
    1.  `Context_k = [c_1, ..., c_n, e_1, ..., e_k]`
    2.  `Emergent_Concept_k+1 = LVM(Context_k)`
    3.  Append `e_k+1` to context.
    4.  Repeat until a stop condition is met.

### Stage 3: Decoder (Concept-to-Text)

  * **Function:** Translates the final sequence of emergent "thought" vectors into a coherent, human-readable text answer.
  * **Implementation:** A dedicated vector-sequence-to-text (V2T) model.

### LNSP Flowchart

```ascii
  [User Prompt (Text)]
           |
           v
  [cite_start][Stage 1: Encoder]  (GTR-T5 + LLM-based 16D TMD) [cite: 1]
           |
           v
  [Initial Context: (c_1, ..., c_n)]
           |
           +----------------------------------+
           | [Stage 2: LVM Core (Recursive)]  |
           |   |                              |
           |   v                              |
           | LVM(Context_k) -> [Emergent_Concept_e_k+1]
           |   |                              |
           |   +---(Append)-------------------+
           |   |                              |
           | [Context_k+1: (c_1, ..., c_n, e_1, ..., e_k+1)]
           |   |                              |
           |   +--(Loop until Stop Condition)--+
           |
           v
  [Final Thought Vectors: (e_1, ..., e_final)]
           |
           v
  [Stage 3: Decoder (V2T)]
           |
           v
  [Final Answer (Text)]
```

-----

## 4\. 📚 Requirement: The Causal Corpus

The LVM's success is entirely dependent on its training data. The optimal training data is not just *sequential*; it must be **semantically causal and developmental**.

The ideal dataset is a **"chain-of-thought" corpus** where each vector `c_n+1` represents the **logical consequence** or **next step** derived from `c_n`.

### Top 20 Ranked Data Sources for LVM Training

The following table ranks potential, downloadable, large-scale datasets based on their "LVM Suitability"—a measure of their causal/procedural/narrative flow. [cite\_start]Domains are mapped from the TMD Schema[cite: 1].

| Rank | Data Source | [cite\_start]Domain(s) [cite: 1] | Structure | LVM Suitability & Rationale |
|:---:|:---|:---|:---|:---|
| 1 | **Project Gutenberg** | 8, 6, 7 | Continuous/Book | **Excellent:** Strong, linear narrative and argumentative flow. The gold standard for causal chains in general knowledge. |
| 2 | **arXiv** (full text) | 0, 1, 2, 15 | Article | **Excellent:** Highly structured logical flow: *Intro* $\rightarrow$ *Methods* $\rightarrow$ *Results* $\rightarrow$ *Conclusion*. A perfect causal corpus for STEM. |
| 3 | **GitHub Code Repos** (Python) | 15 | Repo/File | **Excellent:** Purely causal. `import` $\rightarrow$ `use`; `class def` $\rightarrow$ `instance`; `function A` $\rightarrow$ `call from B`. |
| 4 | **ProofWiki** / Math Proofs | 1 | Article/Proof | **Excellent:** The purest form of logical dependency. `Axiom` $\rightarrow$ `Lemma` $\rightarrow$ `Theorem`. |
| 5 | **Stack Overflow** (Q\&A pairs) | 15, 2 | Q\&A Pair | **Excellent:** Direct causal link: *Problem (Question)* $\rightarrow$ *Solution (Answer)*. |
| 6 | **WikiHow** (filtered) | 13, 14, 2 | Article/Steps | **Very Good:** Pure procedural "how-to" data. *Step 1* $\rightarrow$ *Step 2*. |
| 7 | **RecipeDB / Cooking Sets** | 14 | Article/Steps | **Very Good:** Classic procedural data. *Ingredients* $\rightarrow$ *Prep* $\rightarrow$ *Cook*. |
| 8 | **Screenplay Datasets** (IMSDb) | 8, 9 | Script/Scene | **Very Good:** Strong causal and temporal flow. *Scene A* causes *Scene B*. |
| 9 | **PubMed Central** (full text) | 4 | Article | **Good:** Strong logical flow similar to arXiv, but for medicine. *Hypothesis* $\rightarrow$ *Study* $\rightarrow$ *Result*. |
| 10 | **Khan Academy** (transcripts) | 13, 0, 1 | Continuous/Lesson | **Good:** Excellent pedagogical flow. *Simple concept* $\rightarrow$ *builds to* $\rightarrow$ *Complex concept*. |
| 11 | **Caselaw Access Project** | 11 | Article/Case | **Good:** Strong logical and causal chains. *Facts* $\rightarrow$ *Precedent* $\rightarrow$ *Argument* $\rightarrow$ *Ruling*. |
| 12 | **Code Documentation** (ReadTheDocs) | 15 | Article | **Good:** Mixed explanatory and procedural. The *examples* and *API tutorials* are high-value causal chains. |
| 13 | **Git Commit Messages** | 15 | Continuous/Log | **Good:** Causal by nature. *State N* $\rightarrow$ *Change (commit)* $\rightarrow$ *State N+1*. Teaches semantic diffs. |
| 14 | **Political Speeches/Debates** | 12 | Continuous | **Good:** Strong argumentative flow. *Premise A* $\rightarrow$ *Argument B* $\rightarrow$ *Conclusion C*. |
| 15 | **Philosophy Texts** (S. Ebooks) | 6 | Continuous/Book | **Good:** Pure logical argument flow, but can be highly abstract. |
| 16 | **EDGAR SEC Filings** | 10 | Article/Report | **Good:** Strong temporal flow (quarter-to-quarter) and financial causality. |
| 17 | **YouTube Transcripts** (Tutorials) | 13, 15, 9 | Continuous | **Medium:** High potential but *requires heavy filtering* to isolate the procedural/causal content from conversational filler. |
| 18 | **Common Crawl** | All | Continuous/Web | **Low (Raw):** Massive, but 99% is not causal. Requires *extreme* filtering to extract narrative/procedural blogs, etc. |
| 19 | **Chess PGN Databases** | 5 (Reasoning) | Game/Moves | **Niche/High:** Perfect causal chain for a *specific* domain (strategy). Not general knowledge, but high-purity. |
| 20 | **Wikipedia** (as-is) | All | Article | **Very Low:** Proven to be backward-referential and explanatory. **Must not be used** as-is for forward-predictive training. |

-----

## 5\. 🛠️ Requirement: Training Methodology for Emergent Concepts

To support the LNSP's "thinking" loop, the LVM must be trained to synthesize new concepts. A simple `f(c_n) -> c_n+1` model is insufficient, even with a causal corpus, as it only learns a single-step transition.

We must train the LVM to synthesize a *conclusion* from a *context*.

### Proposed Training Objective: Causal Synthesis Loss

The model must be trained to predict an "emergent concept" vector `y_emergent` from a variable-length context `[c_1, ..., c_n]`.

We will create this `y_emergent` target in two ways:

1.  **"Causal Chain" Target (Local Consequence):**

      * **Context:** `[c_1, ..., c_n]` (e.g., *Recipe steps 1-3*)
      * **Target `y`:** `c_n+1` (e.g., *Recipe step 4*)
      * **Loss:** `Loss_Chain = CosineDistance(LVM(Context), c_n+1)`
      * **Why:** This teaches the model to predict the *immediate next logical step*. It's the baseline for causal flow.

2.  **"Section Synthesis" Target (Holistic Conclusion):**

      * **Context:** `[c_1, ..., c_n]` (e.g., *All "Methodology" chunks from an arXiv paper*)
      * **Target `y`:** `c_summary` (e.g., The *single vector* for the "Results" or "Conclusion" section's abstract/summary).
      * **Loss:** `Loss_Synth = CosineDistance(LVM(Context), c_summary)`
      * **Why:** This explicitly trains the model to *read a block of concepts and synthesize their implication/summary*. This is the core "thinking" task.

The final LVM training will use a **combined loss** to learn both local progression and holistic synthesis:

$$
Loss_{Total} = \alpha \cdot Loss_{Chain} + (1 - \alpha) \cdot Loss_{Synth}
$$This dual-objective ensures the LVM can both "take the next step" and "form a conclusion."

-----

## 6\. ⚙️ Requirement: Tunable Parameters

The LNSP architecture introduces new hyperparameters for both training the LVM and running the inference loop.

### Training Parameters

* **`context_window_size` (k):** The number of vectors `[c_n-k, ..., c_n]` used to make a prediction.
* **`synthesis_window_size` (s):** The number of vectors `[c_1, ..., c_s]` used in the `Loss_Synth` objective.
* **`loss_alpha`:** The weighting (0.0-1.0) between `Loss_Chain` and `Loss_Synth`.
* **`model_architecture`:** (e.g., VMMoE, Transformer), including number of layers, heads, and expert count.
* [cite\_start]**`tmd_weight`:** The degree to which the 16D TMD vector [cite: 1] influences the model's attention or gating mechanisms.

### Inference Parameters (The "Thinking Loop")

* **`max_depth` (int):** The maximum number of recursive "thought" steps (e.g., 50) the LVM can take before forcing a stop. Prevents infinite loops.
* **`stop_threshold` (float):** A cosine similarity threshold. The loop stops if an emergent concept `e_k+1` is highly similar to `e_k` (i.e., "conceptual convergence") or a pre-trained "stop" vector.
* **`context_management` (enum):**
* `FIFO`: Oldest vectors are dropped as new ones are added.
* `Summarize`: The LVM periodically synthesizes its *own context* into a new, single vector.
* `Full`: All vectors are retained (risk of context overflow).
* [cite\_start]**`TMD_bias_weights` (vector):** A 16D vector [cite: 1] applied at inference to *guide* the thinking loop. [cite\_start]A high weight on `Domain: 15 (Software)` [cite: 1] would bias the LVM to "think like a programmer."

## 7\. ❌ Out of Scope for this PRD

* **V2T Decoder:** The architecture and training of the Stage 3 (Vector-to-Text) model.
* **Text Encoder:** The selection and fine-tuning of the Stage 1 (Text-to-Vector) model.
* **Data Sourcing/ETL:** The physical downloading, cleaning, and vectorization of the Causal Corpus. This PRD *defines* the data, but a separate plan will be required to *acquire* it.
$$


# [Architect] Review

⏺ Excellent! You have 220 arXiv papers downloaded (210 cs.CL/LG/ML + 10 cs.AI). Now let me provide my deep critical analysis of the PRD.

  ---
  🔬 CRITICAL ANALYSIS: LVM Core PRD v1.0

  ✅ What the PRD Gets RIGHT

  1. Core Problem Diagnosis (Section 2) - EXCELLENT
  - Correctly identifies the data-objective mismatch as root cause
  - Clear distinction: LLM (token prediction = always structurally true) vs LVM (concept prediction = depends on data structure)
  - Critical insight: "The LVM must be trained on data where the next concept is a logical consequence of the previous one"
  - This is the EXACT lesson from our Wikipedia failure

  2. Data Source Rankings (Section 4) - OUTSTANDING
  - Top 5 are all excellent choices:
    a. Project Gutenberg (narrative flow)
    b. arXiv (logical flow: Intro→Methods→Results→Conclusion) ✅ YOU HAVE THIS!
    c. GitHub code (pure causality)
    d. ProofWiki (pure logic)
    e. Stack Overflow (problem→solution)
  - Wikipedia correctly ranked #20 (Very Low)
  - Rationales are sound and domain-mapped

  3. Three-Stage Architecture (Section 3) - CLEAN
  - Text→Vector→LVM Loop→Vector→Text is well-designed
  - Recursive thinking loop is ambitious but logical
  - Separation of concerns is clear

  4. Dual-Objective Training (Section 5) - SMART
  - Loss_Chain (predict immediate next) + Loss_Synth (synthesize conclusion)
  - Addresses both local and global reasoning
  - Alpha-weighted combination allows tuning

  ---
  ❌ What the PRD Gets WRONG or CRITICALLY MISSES

  🚨 CRITICAL GAP #1: NO DATA QUALITY GATES

  The PRD ranks data sources but has ZERO validation requirements. Based on our Wikipedia disaster, this is MANDATORY:

  Missing Requirements:
  # PRD says: "arXiv is Excellent" 
  # PRD SHOULD say:

  BEFORE training on ANY data source:
  1. Measure Δ (forward - backward bias): MUST be ≥ +0.05 for "Good", ≥ +0.10 for "Excellent"
  2. Verify temporal coherence ≥ 0.40
  3. Verify monotonic offset curve (k=0 < k=1 < ... < target)
  4. Run diagnose_data_direction.py on 5k samples minimum

  FAILURE CASE: arXiv abstracts may be forward (Δ > 0) but full papers may be backward (like Wikipedia)!

  Why this matters:
  - arXiv abstracts (intro→conclusion) likely have Δ > 0
  - arXiv full papers (with extensive related work, background sections) may have Δ < 0
  - We don't know until we measure!

  🚨 CRITICAL GAP #2: NO POST-TRAINING VALIDATION (5CAT)

  PRD has ZERO success criteria for trained models. Based on P6b v2.2's fake win (margin +0.002 but R@5 crashed to 12%), this is MANDATORY:

  Missing Requirements:
  # PRD says: Nothing about validation
  # PRD SHOULD require:

  AFTER training, BEFORE deployment:
  ✅ 5CAT Validation (pass minimum 3/5 gates):
    A. Offset Sweep: Margin ≥ +0.10 (forward > backward)
    B. Retrieval Rank: R@5 ≥ 70%, R@1 ≥ 40%
    C. Ablations: Shuffle delta ≤ -0.15 (order matters)
    D. Rollout: Multi-step coherence ≥ 0.42
    E. Generalization: |Val - OOD| ≤ 0.05

  FAILURE CASE: P6b v2.2 had margin +0.002 (passed!) but R@5 = 12% (COLLAPSED!)

  🚨 CRITICAL GAP #3: NO DIRECTIONAL TRAINING CONSTRAINTS

  PRD mentions Loss_Chain and Loss_Synth but has ZERO mention of directional loss. Based on P1-P6 failures, MSE alone is insufficient:

  Missing from Section 5:
  # PRD says: "Loss = α * Loss_Chain + (1-α) * Loss_Synth"
  # PRD SHOULD say:

  Loss_Total = α * Loss_Chain + (1-α) * Loss_Synth + λ_dir * Loss_Directional

  Where Loss_Directional includes:
  1. Margin Loss: max(0, margin - (cos(pred, next) - cos(pred, prev)))
     - Explicitly enforces forward > backward
     - Target margin: 0.04-0.06 by final epoch

  2. Positive Floor Penalty: max(0, τ - cos(pred, next))²
     - Prevents orthogonal escape (P6b v2.2 failure mode)
     - τ = 0.10 minimum

  3. Orthogonality Penalty: (cos(pred, prev))²
     - Prevents predicting vectors far from both next and prev
     - Critical for stability

  4. Directional-When-Confident Gate (P6b v2.3):
     - Scale directional loss by cos(pred, target)
     - If cos < 0.30: scale = 0 (OFF when misaligned)
     - If cos > 0.45: scale = 1 (FULL when aligned)
     - Prevents training collapse

  Why this matters: Without these, model can:
  - Learn backward prediction (P1-P5 failures)
  - Collapse to orthogonal vectors (P6b v2.2 failure)
  - Converge to negative cosines (P6b v1 collapse)

  🚨 CRITICAL GAP #4: DATA CONTAMINATION PREVENTION

  PRD doesn't mention train/val/OOD splits. Our Wikipedia training showed this is critical:

  Missing Requirements:
  # PRD says: Nothing about splits
  # PRD SHOULD require:

  1. Article-based splits (NO random_split on chunks)
     - Train: articles 0-70%
     - Val: articles 70-85%
     - OOD: articles 85-100% (truly held-out)

  2. Verify NO article overlap between splits
     - If paper 1234.5678 in train → CANNOT be in val or OOD
     - Critical for OOD generalization testing

  FAILURE CASE: Old Wikipedia 340k data used random_split → data leakage → fake val scores

  🚨 CRITICAL GAP #5: UNCLEAR TMD INTEGRATION

  PRD mentions 16D TMD throughout but doesn't specify:

  Critical Questions:
  1. Does LVM predict 784D (768+16) or 768D only?
  2. Is TMD input-only (context has TMD, prediction doesn't) or input+output?
  3. How is TMD created for emergent concepts in thinking loop?
    - If LVM([c1, c2, ...]) → e_new, what is e_new's TMD?
    - Is it predicted? Copied from context? Zeroed?

  Our current experiments: All use 768D only (no TMD in predictions)

  PRD needs to specify: Concrete TMD handling strategy

  ---
  ⚠️ STRATEGIC CONCERNS

  CONCERN #1: Scope Creep - Recursive Loop Before Single-Step Works

  The PRD focuses heavily on the "thinking loop" (Stage 2 recursive inference):
  - LVM(Context_k) → e_k+1
  - Append to context
  - Repeat until stop

  Problem: We haven't proven single-step prediction works yet!
  - P1-P6b all failed or had backward bias
  - P6b v2.3 is ready but untrained

  Risk: Building recursive loop on unproven foundation = house of cards

  Recommendation: Phase 1 = Prove single-step, Phase 2 = Add recursion

  CONCERN #2: Vec2Text Bottleneck Not Addressed

  PRD says V2T decoder is "out of scope" but:
  - Current vec2text: 10 seconds per vector (97% of total latency!)
  - Thinking loop with 50 steps = 500 seconds = 8+ minutes
  - This makes the "thinking loop" completely impractical

  Recommendation: V2T speedup should be a PREREQUISITE, not out-of-scope

  CONCERN #3: Loss_Synth Validation Strategy Unclear

  PRD proposes training on section summaries:
  - Context: [all chunks from "Methods" section]
  - Target: vector for "Results" section summary

  Critical questions:
  1. How do we create the "Results summary" vector?
    - LLM-generated summary → embed?
    - First chunk of Results section?
    - Average of all Results chunks?
  2. How do we validate that model learned synthesis vs just memorized section transitions?
  3. What if sections don't have clear summaries (GitHub code, Stack Overflow)?

  Recommendation: Start with Loss_Chain only (proven concept), add Loss_Synth after validation

  CONCERN #4: No Quantitative Targets

  PRD ranks data sources but doesn't specify:
  - How many arXiv papers is "enough"? 10k? 100k? 1M?
  - What's minimum corpus size for LVM→LLM parity?
  - How much compute/time for training?

  Recommendation: Add concrete targets (e.g., "50k arXiv papers minimum for v1.0")

  ---
  🎯 CRITICAL QUESTIONS BEFORE PROCEEDING

  These MUST be answered before implementing this PRD:

  Q1: Is recursive thinking loop a v1.0 requirement?
  - Option A: v1.0 = Prove single-step prediction works (arXiv Δ > 0, pass 5CAT)
  - Option B: v1.0 = Full thinking loop with 50-step recursion

  My vote: Option A. Prove foundation first, recursion is Phase 2.

  Q2: What is minimum Δ threshold for data sources?
  - PRD says "arXiv is Excellent" but we haven't measured Δ yet!
  - Proposal: Require Δ ≥ +0.05 for "Good", ≥ +0.10 for "Excellent"
  - Validate arXiv Δ BEFORE building training pipeline

  Q3: Should we train forward + backward models, or unified bi-directional?
  - Option A: Separate models (forward for generation, backward for knowledge graph traversal)
  - Option B: Single bi-directional model
  - My vote: Option A. Different use cases, clearer training objectives.

  Q4: Is TMD 784D end-to-end, or 16D input + 768D prediction?
  - Option A: 784D everywhere (consistent but complex)
  - Option B: 16D input context, 768D predictions (simpler, matches current experiments)
  - My vote: Option B for v1.0, Option A for v2.0 if needed.

  Q5: What's the plan for vec2text speedup?
  - Current: 10s per vector = impractical for thinking loop
  - Options:
    - Distill vec2text to smaller model?
    - Train custom V2T decoder on our data?
    - Accept slow inference for v1.0?
  - Recommendation: Optimize V2T before implementing thinking loop

  ---
  📋 MY PHASED RECOMMENDATIONS

  PHASE 1: Prove Single-Step Prediction (4-6 weeks)

  Goal: Train first successful forward LVM on arXiv data

  Tasks:
  1. ✅ Measure Δ on arXiv data (diagnose_data_direction.py on 210 papers you have)
  2. ✅ If Δ ≥ +0.05, proceed. If Δ < 0, analyze why (abstracts vs full papers?)
  3. ✅ Implement all directional losses from P6b v2.3 (margin, floor, orthogonality, gating)
  4. ✅ Train on Loss_Chain only (no Loss_Synth yet)
  5. ✅ Validate with full 5CAT (must pass 3/5 gates minimum)
  6. ✅ Target metrics:
    - Margin ≥ +0.10
    - R@5 ≥ 70%, R@1 ≥ 40%
    - Val cosine ≥ 0.48
  7. ✅ Document success/failure for each architecture (Transformer, VMMoE)

  Deliverables:
  - First LVM model with positive forward margin
  - Validated Δ measurements for arXiv corpus
  - Updated PRD with concrete data quality gates

  PHASE 2: Add Synthesis + Multi-Domain (6-8 weeks)

  Prerequisites: Phase 1 succeeded (margin > 0, passed 5CAT)

  Tasks:
  1. ✅ Implement Loss_Synth (section summary prediction)
  2. ✅ Validate synthesis capability (not just memorization)
  3. ✅ Expand to multi-domain:
    - arXiv (STEM reasoning)
    - ProofWiki (mathematical logic)
    - GitHub code (procedural causality)
  4. ✅ Train dual models (forward + backward)
  5. ✅ Measure cross-domain generalization

  Deliverables:
  - LVM with both local and global reasoning
  - Multi-domain training pipeline
  - Dual-model deployment (forward + backward)

  PHASE 3: Recursive Thinking Loop (8-12 weeks)

  Prerequisites: Phase 2 succeeded + V2T optimized to <1s per vector

  Tasks:
  1. ✅ Implement recursive inference loop
  2. ✅ Tune stop conditions (convergence, diversity, goal alignment)
  3. ✅ Implement context management (FIFO, Summarize, Full)
  4. ✅ Validate multi-step coherence
  5. ✅ Full LNSP pipeline integration

  Deliverables:
  - Working end-to-end LNSP (Text→Vector→LVM Loop→Vector→Text)
  - Benchmarked against LLM baselines
  - Production-ready deployment

  ---
  ✏️ REQUIRED ADDITIONS TO PRD

  Add Section 8: Success Criteria & Validation Gates

  ## 8. ✅ Success Criteria & Validation Gates

  ### 8.1 Pre-Training Data Quality Gates (MANDATORY)

  Before training on ANY data source, the following must be validated:

  | Metric | Minimum Threshold | Measurement Tool |
  |--------|------------------|------------------|
  | **Δ (Forward - Backward)** | ≥ +0.05 (Good), ≥ +0.10 (Excellent) | `diagnose_data_direction.py` |
  | **Temporal Coherence** | ≥ 0.40 | `diagnose_data_direction.py` |
  | **Offset Curve** | Monotonic increasing (k=0 < k=1 < ... < target) | `diagnose_data_direction.py` |
  | **Sample Size** | ≥ 5,000 sequences per data source | - |

  **Validation Protocol**:
  1. Extract 5k random sequences (context_size=5) from data source
  2. Run `diagnose_data_direction.py --n-samples 5000`
  3. If Δ < +0.05 → **DO NOT USE** for forward LVM training
  4. Document results in `artifacts/lvm/data_quality_reports/`

  **Failure Actions**:
  - If Δ ≈ 0: Data is neutral (not suitable for directional training)
  - If Δ < 0: Data is backward-biased (like Wikipedia, requires rewiring or rejection)
  - If Δ > 0 but coherence < 0.40: Data is noisy (improve chunking/preprocessing)

  ### 8.2 Post-Training Model Quality Gates (5CAT - MANDATORY)

  After training, BEFORE deployment, models must pass the 5→1 Causal Alignment Test (5CAT):

  | Gate | Metric | VAL Threshold | OOD Threshold | What It Tests |
  |------|--------|---------------|---------------|---------------|
  | **A: Offset Sweep** | Margin (forward - backward) | ≥ +0.12 | ≥ +0.10 | Directional preference |
  | **B: Retrieval Rank** | R@1, R@5 within article | R@1≥60%, R@5≥95% | R@1≥55%, R@5≥92% | Prediction accuracy |
  | **C: Ablations** | Shuffle delta | ≤ -0.15 | ≤ -0.15 | Context order matters |
  | **D: Rollout** | Multi-step coherence | ≥ 0.45 | ≥ 0.42 | Temporal stability |
  | **E: Generalization** | abs(Val - OOD bins) | ≤ 0.05 | ≤ 0.05 | OOD robustness |

  **Passing Criteria**: Must pass **minimum 3 out of 5 gates**

  **Critical Failure**: If margin is **NEGATIVE**, model learned backward prediction → **DO NOT DEPLOY**

  **Validation Tool**: `tools/tests/test_5to1_alignment.py`

  ### 8.3 v1.0 Deployment Criteria

  - [ ] Single-step prediction proven (Phase 1 5CAT passed)
  - [ ] Trained on minimum 50k documents from validated data sources (Δ ≥ +0.05)
  - [ ] Inference latency < 5ms (LVM only, excluding vec2text)
  - [ ] Production model: R@5 ≥ 80%, margin ≥ +0.10 on held-out test set
  - [ ] Documentation complete (architecture, training, validation results)

  Add Section 5.3: Directional Training Constraints

  ### 5.3 Directional Training Constraints (Critical for Forward Prediction)

  Based on empirical failures (P1-P6b), MSE/cosine loss alone is insufficient. The following directional constraints are **MANDATORY**:

  #### 5.3.1 Directional Margin Loss
  ```python
  L_margin = max(0, margin_target - (cos(pred, next) - cos(pred, prev)))
  - Explicitly enforces: cos(pred, next) > cos(pred, prev) + margin
  - Epoch schedule: margin_target ramps from 0.02 → 0.04 → 0.06
  - Weight: λ_dir ramps from 0.01 → 0.02 over training

  5.3.2 Positive Floor Penalty

  L_floor = β * max(0, τ - cos(pred, next))²
  - Prevents orthogonal escape (predicting vectors far from target)
  - τ = 0.10 (minimum acceptable similarity)
  - β = 1e-3 to 2e-3

  5.3.3 Orthogonality Penalty

  L_orth = κ * (cos(pred, prev))²
  - Prevents predicting vectors similar to prev (backward preference)
  - κ = 1e-4

  5.3.4 Directional-When-Confident Gate (P6b v2.3)

  alignment_quality = cos(pred, next)
  if alignment_quality < 0.30:
      scale = 0.0  # Directional OFF when misaligned
  elif alignment_quality > 0.45:
      scale = 1.0  # Directional FULL when aligned
  else:
      scale = (alignment_quality - 0.30) / 0.15  # Linear ramp

  L_dir_gated = scale * L_margin
  - Critical for training stability
  - Prevents directional pressure from pushing model into bad local minima

  5.3.5 Combined Loss

  L_total = (
      α * L_chain +
      (1 - α) * L_synth +
      λ_dir * L_dir_gated +
      β * L_floor +
      κ * L_orth
  )

  Why These Are Mandatory:
  - Without L_margin: Model learns backward prediction (P1-P5 failures, all had negative margin)
  - Without L_floor: Model escapes to orthogonal vectors (P6b v2.2 failure, cosine 0.44→0.18)
  - Without L_orth: Model predicts vectors similar to prev (backward preference)
  - Without gating: Training collapses (P6b v1 failure at epoch 3)

  **Add Section 4.1: Data Quality Validation Requirements**

  ```markdown
  ### 4.1 Data Quality Validation Protocol

  All data sources listed in Section 4 must be validated BEFORE training:

  **Step 1: Sample Extraction**
  - Extract 5k random sequences (context_size=5, stride=1)
  - Ensure representative sampling across corpus

  **Step 2: Δ Measurement**
  ```bash
  ./.venv/bin/python tools/tests/diagnose_data_direction.py \
    DATA_SOURCE.npz --n-samples 5000 --output DATA_SOURCE_quality_report.json

  Step 3: Decision Gate
  | Δ Range           | Quality   | Action                                  |
  |-------------------|-----------|-----------------------------------------|
  | Δ ≥ +0.10         | Excellent | Use for training (high priority)        |
  | +0.05 ≤ Δ < +0.10 | Good      | Use for training (medium priority)      |
  | +0.02 ≤ Δ < +0.05 | Marginal  | Use with caution, may need augmentation |
  | Δ < +0.02         | Poor      | DO NOT USE for forward LVM training     |

  Step 4: Documentation
  - Store quality reports in artifacts/lvm/data_quality_reports/
  - Update Section 4 table with measured Δ values
  - Track temporal coherence, offset curves, sample sizes

  CRITICAL: The rankings in Section 4 are hypotheses until measured. arXiv may be Excellent for abstracts but backward for full papers!

  ---

  ### 🎯 IMMEDIATE ACTION ITEMS

  **Before implementing this PRD, DO THIS FIRST:**

  1. **Measure arXiv Δ** (2-4 hours):
     ```bash
     # Extract sequences from your 210 papers
     ./.venv/bin/python tools/create_arxiv_training_sequences.py \
       --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz \
       --output artifacts/lvm/arxiv_sequences_ctx5.npz

     # Measure Δ
     ./.venv/bin/python tools/tests/diagnose_data_direction.py \
       artifacts/lvm/arxiv_sequences_ctx5.npz --n-samples 5000

     Expected: Δ ≥ +0.08 (papers follow Intro→Methods→Results flow)
     If Δ < 0: arXiv has same problem as Wikipedia! Need different data.

  2. Update PRD with measurements (1 hour):
    - Add actual Δ values to Section 4 table
    - Add Section 8 (Success Criteria)
    - Add Section 5.3 (Directional Constraints)
    - Add Section 4.1 (Validation Protocol)
  3. Decide on Phase 1 scope (discussion):
    - Is recursive thinking loop v1.0 or v2.0?
    - Is TMD 784D or 16D input + 768D prediction?
    - What's minimum corpus size (50k papers? 100k?)?

  ---
  💡 BOTTOM LINE

  The PRD is 80% excellent but missing the 20% that matters most: validation.

  What's RIGHT:
  - ✅ Correctly diagnoses data-objective mismatch
  - ✅ Excellent data source rankings (arXiv, Gutenberg, GitHub, ProofWiki)
  - ✅ Clean three-stage architecture
  - ✅ Dual-objective training (chain + synthesis)

  What's WRONG:
  - ❌ NO data quality gates (Δ measurement, temporal coherence)
  - ❌ NO post-training validation (5CAT)
  - ❌ NO directional loss constraints (margin, floor, orthogonality, gating)
  - ❌ NO train/val/OOD split requirements
  - ❌ Unclear TMD integration
  - ❌ Recursive loop before single-step proven (scope creep)
  - ❌ Vec2text bottleneck ignored (10s/vector kills thinking loop)

  RECOMMENDATION:
  1. PAUSE on implementing the PRD as-is
  2. MEASURE arXiv Δ first (validate the foundation!)
  3. ADD validation sections (8, 5.3, 4.1) to PRD
  4. DECIDE on Phase 1 scope (single-step only? Or full recursion?)
  5. THEN proceed with implementation

  This could save weeks/months by catching data problems BEFORE training, not after!

  ---
  Want me to:
  1. Draft the complete updated PRD with all additions?
  2. Write the arXiv Δ measurement script?
  3. Create a Phase 1 implementation plan?
  4. Something else?