# LNSP-RFC-004: Self-Rehearsal Phase for Vector-Native Reinforcement in Large Vector Models (LVMs)

**Author:** Trent Carter
**Date:** October 24, 2025
**Status:** Draft
**Version:** 0.1

---

## Abstract

*Self-Rehearsal* is a post-reinforcement training phase for **Large Vector Models (LVMs)** that strengthens conceptual coherence through **autonomous vector replay and refinement**.
After reinforcement fine-tuning (e.g., Echo-Loop or RL-based alignment), the model engages in a brief self-supervised rehearsal epoch in which it re-examines its own most recent concept transitions and re-derives their latent relationships.

Instead of relying on new external prompts, the model uses the **embeddings and responses generated during reinforcement** as its own rehearsal set.
During this process, the LVM re-computes internal similarity gradients between its predicted and target concept vectors, consolidating high-coherence patterns and attenuating unstable transitions.

The goal is to provide an analog of “sleep-phase consolidation” for concept-based systems—strengthening recently-reinforced pathways without new data or supervision.
Empirically, Self-Rehearsal is expected to:

1. Improve stability of latent vector dynamics across conversational turns.
2. Reduce catastrophic forgetting of prior conceptual alignments.
3. Increase coherence in long-range semantic reasoning.

---

## Functional Requirements

| ID       | Category                 | Description                                                                                                                                                                |   |                     |   |      |
| -------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | ------------------- | - | ---- |
| **SR-1** | **Triggering Condition** | Activated automatically after a reinforcement epoch (Echo-Loop, RLHF, or lane-specific reward optimization).                                                               |   |                     |   |      |
| **SR-2** | **Input Dataset**        | Uses cached *concept-response pairs* (`C_t`, `Ĉ_t+1`) and their cosine similarity deltas from the latest reinforcement run.                                               |   |                     |   |      |
| **SR-3** | **Objective Function**   | Minimize drift between replayed and original reinforced vectors:  (\mathcal{L}*{reh} = 1 - \cos(v*{orig}, v_{replay}) + \lambda                                            |   | v_{orig}-v_{replay} |   | ^2). |
| **SR-4** | **Gradient Flow**        | Gradients back-propagate only through *vector-mapping parameters* (not text decoders); weights in semantic-memory layers are updated with low-learning-rate consolidation. |   |                     |   |      |
| **SR-5** | **Sampling Policy**      | Rehearse top-K percent (e.g., K = 10–20%) of high-reward concept pairs per lane, plus a stochastic mix of low-reward outliers for diversity.                               |   |                     |   |      |
| **SR-6** | **Iteration Count**      | Default 1–3 mini-epochs or until cosine drift < 0.02 for 95% of replayed pairs.                                                                                            |   |                     |   |      |
| **SR-7** | **Integration Point**    | Implemented as an optional *P15.5* step between **P15 (LNSP Training)** and **P16 (Multi-RAG Query)** in the existing pipeline.                                            |   |                     |   |      |
| **SR-8** | **Persistence**          | Stores per-lane rehearsal metrics: mean cosine Δ, vector norm Δ, concept stability index; appended to `echo_validation.log`.                                               |   |                     |   |      |
| **SR-9** | **Safety Guard**         | Prevents mode collapse by imposing variance floor σ² ≥ baseline_σ²×0.85; aborts rehearsal if collapse detected.                                                            |   |                     |   |      |

---

## Implementation Notes

* **Location:** `app/pipeline/self_rehearsal.py`
* **Dependencies:** uses `torch.no_grad()` replay cache, FAISS or pgvector cosine retrieval, and the existing Echo validation metrics.
* **Invocation:**

  ```bash
  python -m app.pipeline.self_rehearsal --lane science-fact --epochs 2
  ```
* **Outputs:**

  * Updated vector weights in latent-space model checkpoint.
  * JSON metrics in `outputs/self_rehearsal/`.
  * Optional visualization: drift histograms via matplotlib.

---

## Evaluation Metrics

| Metric                  | Description                                                   | Target    |
| ----------------------- | ------------------------------------------------------------- | --------- |
| **Δ Cosine Stability**  | Mean cosine distance between original and replayed vectors    | ≤ 0.02    |
| **Recall@K Retention**  | Fraction of concepts retaining top-K neighborhood consistency | ≥ 0.95    |
| **Echo Score Drift**    | Change in Echo Score after rehearsal                          | ≤ ±1%     |
| **Entropy Floor Check** | Ensure latent diversity not reduced > 15%                     | Pass/Fail |

---

## Rationale

Self-Rehearsal acts as **vector-space consolidation**, analogous to hippocampal replay in biological systems.
It provides the model a chance to internalize reinforced behaviors without new input data, leveraging its own activity traces as training stimuli.
This not only improves stability and semantic recall but also yields smoother downstream inference by aligning newly-reinforced concept mappings with long-term latent geometry.

---

**End of Document**


# LNSP-RFC-004: Self-Rehearsal Phase for Vector-Native Reinforcement in Large Vector Models (LVMs)

**Author:** Trent Carter
**Date:** October 24, 2025
**Status:** Draft v0.2 (Deferred Activation)
**Version:** 0.2

---

## Abstract

*Self-Rehearsal* is a post-reinforcement training phase for **Large Vector Models (LVMs)** that strengthens conceptual coherence through **autonomous vector replay and refinement**.
After reinforcement fine-tuning (e.g., Echo-Loop or RL-based alignment), the model engages in a brief self-supervised rehearsal epoch in which it re-examines its own most recent concept transitions and re-derives their latent relationships.

Instead of relying on new external prompts, the model uses the **embeddings and responses generated during reinforcement** as its own rehearsal set.
During this process, the LVM re-computes internal similarity gradients between its predicted and target concept vectors, consolidating high-coherence patterns and attenuating unstable transitions.

The goal is to provide an analog of “sleep-phase consolidation” for concept-based systems—strengthening recently-reinforced pathways without new data or supervision.
Empirically, Self-Rehearsal is expected to:

1. Improve stability of latent vector dynamics across conversational turns.
2. Reduce catastrophic forgetting of prior conceptual alignments.
3. Increase coherence in long-range semantic reasoning.

---

## Functional Requirements

| ID       | Category                 | Description                                                                                                                                                                |   |                     |   |      |
| -------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | ------------------- | - | ---- |
| **SR-1** | **Triggering Condition** | Activated automatically after a reinforcement epoch (Echo-Loop, RLHF, or lane-specific reward optimization).                                                               |   |                     |   |      |
| **SR-2** | **Input Dataset**        | Uses cached *concept-response pairs* (`C_t`, `Ĉ_t+1`) and their cosine similarity deltas from the latest reinforcement run.                                               |   |                     |   |      |
| **SR-3** | **Objective Function**   | Minimize drift between replayed and original reinforced vectors:  (\mathcal{L}*{reh} = 1 - \cos(v*{orig}, v_{replay}) + \lambda                                            |   | v_{orig}-v_{replay} |   | ^2). |
| **SR-4** | **Gradient Flow**        | Gradients back-propagate only through *vector-mapping parameters* (not text decoders); weights in semantic-memory layers are updated with low-learning-rate consolidation. |   |                     |   |      |
| **SR-5** | **Sampling Policy**      | Rehearse top-K percent (e.g., K = 10–20%) of high-reward concept pairs per lane, plus a stochastic mix of low-reward outliers for diversity.                               |   |                     |   |      |
| **SR-6** | **Iteration Count**      | Default 1–3 mini-epochs or until cosine drift < 0.02 for 95% of replayed pairs.                                                                                            |   |                     |   |      |
| **SR-7** | **Integration Point**    | Implemented as an optional *P15.5* step between **P15 (LNSP Training)** and **P16 (Multi-RAG Query)** in the existing pipeline.                                            |   |                     |   |      |
| **SR-8** | **Persistence**          | Stores per-lane rehearsal metrics: mean cosine Δ, vector norm Δ, concept stability index; appended to `echo_validation.log`.                                               |   |                     |   |      |
| **SR-9** | **Safety Guard**         | Prevents mode collapse by imposing variance floor σ² ≥ baseline_σ²×0.85; aborts rehearsal if collapse detected.                                                            |   |                     |   |      |

---

## Implementation Notes

* **Location:** `app/pipeline/self_rehearsal.py` *(to be created after Echo/Replay infrastructure exists)*
* **Dependencies:** will use `torch.no_grad()` replay cache, FAISS or pgvector cosine retrieval, and the existing Echo validation metrics.
* **Invocation (future):**

  ```bash
  python -m app.pipeline.self_rehearsal --lane science-fact --epochs 2
  ```
* **Outputs:**

  * Updated vector weights in latent-space model checkpoint.
  * JSON metrics in `outputs/self_rehearsal/`.
  * Optional visualization: drift histograms via matplotlib.

---

## Evaluation Metrics

| Metric                  | Description                                                   | Target    |
| ----------------------- | ------------------------------------------------------------- | --------- |
| **Δ Cosine Stability**  | Mean cosine distance between original and replayed vectors    | ≤ 0.02    |
| **Recall@K Retention**  | Fraction of concepts retaining top-K neighborhood consistency | ≥ 0.95    |
| **Echo Score Drift**    | Change in Echo Score after rehearsal                          | ≤ ±1%     |
| **Entropy Floor Check** | Ensure latent diversity not reduced > 15%                     | Pass/Fail |

---

## Implementation Readiness (v0.2)

The Self-Rehearsal phase is **deferred** until the Echo/Reward infrastructure (P13/P15) produces lane-scored replay caches.
Interim goal: develop replay-logging and selective-layer mechanisms required by SR-2, SR-4, and SR-5.
Once these exist, a prototype script (`tools/self_rehearsal_prototype.py`) will validate cosine-stability gains before full pipeline integration.

### Near-Term Milestones

| Step                                             | Focus                                                                                                     | Deliverable                        |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **1. Reinforcement Data Plumbing (P13 Upgrade)** | Modify echo validation to emit replay buffers `{lane_id, concept_vec, response_vec, reward, echo_score}`. | `replay_cache_writer.py`           |
| **2. Lane-Level Metrics**                        | Add `lane_metrics.jsonl` tracking reward distributions, cosine drift, and echo deltas.                    | Appended to `echo_validation.log`  |
| **3. Tagged Checkpoints (P15 Upgrade)**          | Extend training save logic to mark vector-mapping layers and store per-lane weights.                      | Checkpoint metadata schema         |
| **4. Prototype Isolated Rehearsal Script**       | Build `tools/self_rehearsal_prototype.py` to replay top-K sequences and re-optimize cosine drift.         | Proof-of-concept + stability chart |
| **5. Promote to Pipeline (Future P15.5)**        | Once replay data and selective layers exist, integrate into full pipeline and re-activate RFC v1.0.       | Merge into `app/pipeline/`         |

---

## Rationale

Self-Rehearsal acts as **vector-space consolidation**, analogous to hippocampal replay in biological systems.
It provides the model a chance to internalize reinforced behaviors without new input data, leveraging its own activity traces as training stimuli.
This not only improves stability and semantic recall but also yields smoother downstream inference by aligning newly-reinforced concept mappings with long-term latent geometry.

---

**End of Document (v0.2 Deferred Activation)**
