# P9: Sentence-Aware Retrieval - Findings Report

**Date**: 2025-11-05
**Status**: ✅ COMPLETE - Tested and evaluated

---

## Executive Summary

**Finding**: Sentence-level granularity does NOT improve retrieval for self-contained technical content (arXiv papers).

**Key Results**:
- **Baseline (paragraph-only)**: R@5 = 0.280, MRR = 0.116
- **Best sentence config**: R@5 = 0.280 (tied), MRR = 0.121 (+0.005)
- **Sentence reranking with any weight > 0.3**: R@5 drops to 0.260 (-7%)

**Conclusion**: Paragraph-only retrieval is optimal for this dataset. Sentence-aware retrieval did NOT meet acceptance gates (+7pp R@5, +0.04 MRR).

---

## Background

### Motivation

After abandoning AR-LVM (Δ ≈ 0 at both paragraph and sentence level), we investigated whether sentence-level granularity could improve **retrieval/ranking** performance, even though it doesn't provide temporal directionality signals.

### Hypothesis

Sentence-level matching should help retrieval by:
1. Reducing topic smearing (paragraphs mix multiple concepts)
2. Enabling finer-grained semantic matching
3. Supporting local entity/verb continuity

### Test Setup

**Dataset**: 159 arXiv paper paragraphs, 253 sentences
**Queries**: 50 paragraph-to-paragraph retrieval tasks
**Metrics**: R@1, R@5, R@10, MRR
**Configurations**: 5 fusion weight combinations (a=sent, b=para)

---

## Results

### Configuration Performance

| Config | a (sent) | b (para) | R@1 | R@5 | R@10 | MRR |
|--------|----------|----------|-----|-----|------|-----|
| **Baseline (para-only)** | 0.00 | 1.00 | 0.000 | **0.280** | 0.380 | 0.116 |
| Conservative (favor para) | 0.30 | 0.60 | 0.000 | **0.280** | 0.380 | **0.121** |
| Balanced | 0.50 | 0.40 | 0.000 | 0.260 | 0.380 | 0.122 |
| Slight sent favor | 0.60 | 0.35 | 0.000 | 0.260 | 0.380 | 0.122 |
| Original sent-heavy | 0.75 | 0.15 | 0.000 | 0.260 | 0.380 | 0.116 |

### Key Observations

1. **R@5 degrades with sentence weight > 0.3**:
   - Baseline/conservative: 0.280
   - Balanced/sent-favor: 0.260 (-7%)

2. **MRR slightly improves with light sentence mixing** (a=0.3):
   - Baseline: 0.116
   - Conservative: 0.121 (+0.005)
   - But R@5 stays flat

3. **R@1 is zero across all configs**:
   - Dataset is too hard for top-1 retrieval
   - Indicates paragraph embeddings don't capture next-paragraph signal

4. **Directional adapter hurt performance** (tested separately):
   - Baseline R@5: 0.280
   - With adapter: 0.180 (-36%)
   - Confirmed: adapter doesn't help retrieval

---

## Why Sentence-Aware Retrieval Failed Here

### 1. arXiv Paragraphs Are Self-Contained

arXiv papers have **paragraph-level coherence**, not sentence-level temporal flow:
- Each paragraph discusses a complete sub-topic
- Sentences within paragraphs are mutually reinforcing (not sequential)
- Next paragraph often jumps to a new topic (weak local continuity)

**Example**:
```
Para 1: "We propose a novel attention mechanism..."
Para 2: "Related work in neural architectures includes..."
```
→ Weak next-para signal, strong within-para coherence

### 2. Sentence Splitting Breaks Technical Concepts

Technical writing uses complex sentences with multiple clauses:
```
Original: "The transformer architecture (Vaswani et al., 2017) uses
           self-attention to compute contextual representations..."
```

Split into sentences:
```
Sent 1: "The transformer architecture (Vaswani et al., 2017) uses self-attention..."
Sent 2: "to compute contextual representations..."
```
→ Second sentence loses context, hurts semantic matching

### 3. Small Dataset (159 Paragraphs)

- Only 253 sentences total
- Not enough data to see statistical benefits
- Paragraph-level ANN already captures most signal

### 4. No Temporal Flow in arXiv Content

- Papers are organized by logical sections (intro, methods, results)
- NOT temporal sequences (setup → payoff)
- Confirmed by Δ ≈ 0 at both paragraph and sentence level

---

## Sentence Delta Validation

Confirmed that sentence-level granularity does NOT create forward temporal signal:

| Data Source | N Sequences | Δ (mean) | Result |
|-------------|-------------|----------|--------|
| Narrative (5 novels) | 2,775 | +0.0001 | Zero signal |
| arXiv (10 papers) | 7,041 | -0.00004 | Zero signal |

**Conclusion**: Sentence splitting doesn't rescue temporal directionality. GTR-T5 embeddings are symmetric at all scales.

---

## Acceptance Gates

**Gates** (required for production):
1. R@5: +7pp vs baseline → ❌ **FAILED** (0pp improvement)
2. MRR: +0.04 vs baseline → ❌ **FAILED** (+0.005)
3. Latency: ≤+15% → ⚠️ **N/A** (not measured, but likely OK)

**Verdict**: Did NOT meet gates. Sentence-aware retrieval is not production-ready for this dataset.

---

## When Might Sentence-Aware Retrieval Help?

### Suitable Datasets (Hypothetical)

1. **Wikipedia articles** (explanatory flow with local references)
2. **Programming tutorials** (step-by-step instructions)
3. **Narrative fiction** (character/plot continuity)
4. **Recipe/how-to guides** (procedural sequences)

### Key Requirements

- **Sequential content**: Next sentence/paragraph builds on previous
- **Local continuity**: Entities/verbs carry across boundaries
- **Large scale**: 10k+ paragraphs to see statistical gains
- **Fine-grained queries**: Sentence-level retrieval tasks

### Recommended Test

If retrying sentence-aware retrieval:
1. Use Wikipedia or tutorial dataset (>10k paragraphs)
2. Create sentence-level retrieval tasks (not paragraph-level)
3. Measure Δ > 0.01 at sentence level before investing in infrastructure
4. Use conservative fusion weights (a=0.2-0.3, b=0.7-0.8)

---

## Components Delivered

### Scripts Created

| File | Purpose | Status |
|------|---------|--------|
| `tools/sentence_delta_check.py` | Validate Δ at sentence level | ✅ Complete |
| `tools/build_sentence_bank.py` | Build sentence-level vector bank | ✅ Complete |
| `tools/npz_to_paragraph_jsonl.py` | Convert NPZ to JSONL | ✅ Complete |
| `app/retriever/sent_para_rerank.py` | Two-stage retrieval (para→sent) | ✅ Complete |
| `tools/fit_directional_adapter.py` | Fit linear adapter for ranking | ✅ Complete |
| `tools/test_sentence_retrieval.py` | Test 3 configs (baseline, sent, adapter) | ✅ Complete |
| `tools/test_fusion_weights.py` | Grid search over fusion weights | ✅ Complete |

### Artifacts Created

| File | Description | Size |
|------|-------------|------|
| `artifacts/arxiv_sentence_bank.npz` | 253 sentences, 768D vectors | 0.7 MB |
| `artifacts/directional_adapter.npz` | 768×768 linear transform (W) | 2.1 MB |
| `artifacts/sentence_retrieval_results.json` | 3-config test results | 1 KB |
| `artifacts/fusion_weight_results.json` | 5-config fusion weight grid | 2 KB |

---

## Recommendations

### 1. **Keep Paragraph-Only Retrieval** (Current System)

- R@5 = 0.280 is the best we can do on this dataset
- Adding sentence information doesn't help
- Simpler system, lower latency

### 2. **Do NOT Deploy Sentence-Aware Retrieval**

- Failed acceptance gates (-7% R@5 at high sent weight)
- Only marginal MRR gain (+0.005) at conservative weight
- Not worth the added complexity

### 3. **Archive Directional Adapter**

- Δ got WORSE after adapter (-0.0004)
- Hurt retrieval performance (-36% R@5)
- Confirms symmetric embedding geometry

### 4. **Do NOT Train Q-Tower for arXiv**

- No temporal signal to learn (Δ ≈ 0)
- Q-tower would just learn paragraph-level similarity (already have that)
- Save compute for other experiments

### 5. **Possible Future Work** (If Desired)

Try sentence-aware retrieval on:
- **Wikipedia articles** (better local continuity)
- **Programming tutorials** (procedural sequences)
- **Larger datasets** (10k+ paragraphs)

But **gate first**: Measure Δ > 0.01 at sentence level before investing.

---

## Session Timeline

| Time | Task | Result |
|------|------|--------|
| 09:30 | Created `sentence_delta_check.py` | ✅ Script working |
| 09:45 | Ran sentence Δ probe (narrative) | Δ = +0.0001 (zero signal) |
| 10:00 | Ran sentence Δ probe (arXiv) | Δ = -0.00004 (zero signal) |
| 10:15 | Built sentence bank (253 sentences) | ✅ NPZ created |
| 10:30 | Created reranker + test scripts | ✅ Code complete |
| 10:45 | Fit directional adapter | ⚠️ Δ got worse (-0.0004) |
| 11:00 | Ran 3-config test | ❌ Sent rerank hurt R@5 |
| 11:15 | Ran fusion weight grid | ✅ Baseline best (a=0, b=1) |
| 11:30 | Documented findings | ✅ This report |

**Total time**: ~2 hours from concept to decision

---

## Conclusion

**Sentence-level granularity does NOT improve retrieval on self-contained technical content (arXiv papers).**

- **Best performance**: Paragraph-only (R@5 = 0.280)
- **Sentence reranking**: No gain at conservative weight, -7% at high weight
- **Directional adapter**: Hurt performance (-36%)
- **Acceptance gates**: ❌ Failed (need +7pp R@5, got 0pp)

**Decision**: Keep paragraph-only retrieval. Archive sentence-aware components.

**Next steps**: If revisiting sentence-aware retrieval, test on Wikipedia or tutorial datasets with confirmed Δ > 0.01 at sentence level.

---

## Files to Archive

```bash
# Archive sentence-aware components (don't delete, just move out of active path)
mkdir -p artifacts/lvm/archived_p9_sentence_retrieval
mv artifacts/arxiv_sentence_bank.npz artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/directional_adapter.npz artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/sentence_retrieval_results.json artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/fusion_weight_results.json artifacts/lvm/archived_p9_sentence_retrieval/
```

Scripts remain in `tools/` for future use on different datasets.

---

**Report complete**: 2025-11-05
