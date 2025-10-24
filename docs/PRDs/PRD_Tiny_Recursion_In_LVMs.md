# PRD: Tiny Recursion in Large Vector Models (LVMs)

## Overview

This Product Requirements Document (PRD) defines the purpose, implementation strategy, and design structure for introducing **Tiny Recursion (TR)** into Large Vector Model (LVM) inference pipelines. Tiny Recursion is a lightweight feedback mechanism that enables vector-native models to refine or validate their outputs via a single additional pass.

---

## Objective

Introduce a minimal, low-cost recursion loop that operates **after initial vector generation** but **before final output consumption**, allowing the model to verify or refine the inferred vector for higher confidence, reduced drift, or improved semantic stability.

---

## Motivation

Traditional LVM pipelines rely on single-pass vector generation and immediate downstream consumption (e.g., retrieval, vec2text decoding). This approach:

* Lacks a verification mechanism
* May propagate small semantic drifts
* Has no built-in “confidence recheck” before emitting outputs

Tiny Recursion solves this by:

* Refeeding the output vector into the model
* Checking for stabilization or refinement
* Allowing rejection or re-routing of unstable outputs

---

## Architecture Diagram

```
                       ┌────────────────────┐
                       │  User Query Vector │ ◄───┐
                       └────────┬───────────┘     │
                                ▼                 │
                      ┌────────────────────┐      │
                      │   TMD Pre-routing   │      │
                      └────────┬───────────┘      │
                                ▼                 │
                 ┌─────────────────────────────┐  │
                 │  Vector Search (FAISS etc.) │  │
                 └────────┬────────────────────┘  │
                          ▼                       │
             ┌─────────────────────────────┐      │
             │ Top-K Concept Candidates    │      │
             └────────┬───────────────┬────┘      │
                      ▼               ▼           │
            ┌────────────────┐   ┌────────────┐    │
            │ Vec→Vec Decode │   │ Graph RAG  │    │
            └──────┬─────────┘   └────┬───────┘    │
                   ▼                  ▼            │
                ┌───────────────────────────────┐  │
                │    LVM Core Vector Inference  │◄─┘
                └────────┬──────────────────────┘
                         ▼
                ┌────────────────────────────┐
                │ 🔁 Tiny Recursion (TR) Step │   ◄──── Optional recheck
                └────────┬───────────────────┘
                         ▼
                ┌────────────────────────────┐
                │  Final Vector Refinement   │
                └────────┬───────────────────┘
                         ▼
            ┌──────────────────────────────┐
            │  Final Decision (Text or Vec)│
            │  → vec2text, or return vec   │
            └──────────────────────────────┘
```

---

## Design

### Where TR fits:

* Location: **After initial LVM vector output**, before final consumption (e.g. ranking, decoding, echo loop)
* Format: **Same architecture**, reuses the model for a single-pass re-evaluation

### What is passed:

* Original output vector (768D or 784D fused)
* Original mission/prompt context
* Optionally: Top-K nearby concepts (as supporting memory)

### How it operates:

* Refeeds the vector into the LVM as a pseudo-input
* Compares output vector (V2) to initial (V1)
* Measures cosine delta between V1 and V2
* If delta < threshold (e.g., 0.05), accept V2; else, flag or reroute

---

## Benefits

* ✅ Improves semantic stability of vector outputs
* ✅ Enables lightweight self-validation inside the LVM
* ✅ Reduces hallucination or vector drift without costly multi-pass systems

---

## Optional Enhancements

* Echo Score Integration: Use post-TR vector in final cosine check
* Graph Expansion Retry: If TR fails, expand graph neighborhood and rerun
* Temperature Annealing: Apply slightly different attention mask or temp in TR step for diversity check

---

## Metrics

| Metric                 | Baseline | With TR | Goal      |
| ---------------------- | -------- | ------- | --------- |
| Output Drift (ΔCosine) | —        | < 0.05  | ≤ 0.03    |
| Rejection Rate         | —        | 5–15%   | Adaptable |
| Echo Match Rate        | —        | +3–7%   | ≥ +5%     |

---

## Implementation Notes

* TR is implemented as a modular hook in the LVM pipeline
* Can be toggled per TMD-lane or per batch
* Compatible with batched inference (e.g., vLLM)
* Adds ~1 extra forward pass per inference, no additional training cost

---

## Next Steps

* [ ] Integrate TR module into `lnsp_lvm_infer.py`
* [ ] Add config flag: `--use_tiny_recursion`
* [ ] Log vector deltas (pre vs post TR)
* [ ] Evaluate across 1000 samples: with and without TR
* [ ] Deploy into full CPESH-RAG+MoE pipeline
