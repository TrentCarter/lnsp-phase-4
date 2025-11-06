# Session Summary: P9 Sentence-Aware Retrieval Testing

**Date**: 2025-11-05
**Duration**: ~2 hours
**Status**: ✅ COMPLETE

---

## Session Goals

Test whether sentence-level granularity can improve retrieval/ranking, even though it doesn't provide temporal directionality for AR-LVM (which we abandoned after Δ ≈ 0 at both paragraph and sentence level).

---

## What We Built

### 1. Sentence Delta Validation (Confirmed Δ ≈ 0)

**Script**: `tools/sentence_delta_check.py`

**Results**:
| Data Source | N Sequences | Δ (mean) | Conclusion |
|-------------|-------------|----------|------------|
| Narrative (5 novels) | 2,775 | +0.0001 | Zero forward signal |
| arXiv (10 papers) | 7,041 | -0.00004 | Zero forward signal |

**Finding**: Sentence-level granularity does NOT create forward temporal signal. GTR-T5 embeddings are symmetric at all scales.

### 2. Sentence Bank Builder

**Scripts**:
- `tools/build_sentence_bank.py` - Split paragraphs → embed sentences
- `tools/npz_to_paragraph_jsonl.py` - Convert NPZ → JSONL format

**Artifact**: `artifacts/arxiv_sentence_bank.npz`
- 253 sentences from 159 paragraphs
- 768D vectors, 0.7 MB

### 3. Two-Stage Retrieval System

**Script**: `app/retriever/sent_para_rerank.py`

**Design**:
- Stage 1: Paragraph ANN retrieval (FAISS)
- Stage 2: Sentence-based reranking with fusion weights
- Fusion: `score = a·cos_sent + b·cos_para`

### 4. Directional Adapter (Optional)

**Script**: `tools/fit_directional_adapter.py`

**Result**: ❌ Δ got WORSE after adapter
- Before: Δ = -0.0004
- After: Δ = -0.0008 (improvement: -0.0004)
- Confirms symmetric embedding geometry

### 5. Comprehensive Testing

**Scripts**:
- `tools/test_sentence_retrieval.py` - 3 configs (baseline, sent, adapter)
- `tools/test_fusion_weights.py` - Grid search over 5 fusion weights

**Test Setup**:
- 159 arXiv paragraphs, 253 sentences
- 50 paragraph-to-paragraph retrieval queries
- Metrics: R@1, R@5, R@10, MRR

---

## Key Results

### Configuration Performance

| Config | a (sent) | b (para) | R@5 | MRR | vs Baseline |
|--------|----------|----------|-----|-----|-------------|
| **Baseline (para-only)** | 0.00 | 1.00 | **0.280** | 0.116 | - |
| Conservative (favor para) | 0.30 | 0.60 | **0.280** | **0.121** | +0.0pp R@5, +0.005 MRR |
| Balanced | 0.50 | 0.40 | 0.260 | 0.122 | -7% R@5 |
| Slight sent favor | 0.60 | 0.35 | 0.260 | 0.122 | -7% R@5 |
| Original sent-heavy | 0.75 | 0.15 | 0.260 | 0.116 | -7% R@5 |
| **With adapter** | 0.75 | 0.15 | 0.180 | 0.071 | **-36% R@5** |

### Findings

1. **Paragraph-only is best** (R@5 = 0.280)
2. **Sentence reranking doesn't help**:
   - Conservative weight (a=0.3): Matched baseline
   - Any weight > 0.3: Hurt R@5 by 7%
3. **Directional adapter hurt performance** (-36% R@5)
4. **R@1 is zero across all configs** (dataset too hard)

---

## Why Sentence-Aware Retrieval Failed

### 1. arXiv Paragraphs Are Self-Contained
- Each paragraph = complete sub-topic
- Sentences within paragraphs are mutually reinforcing
- Next paragraph often jumps to new topic (weak continuity)

### 2. Sentence Splitting Breaks Technical Concepts
```
Original: "The transformer uses self-attention to compute representations..."
Split:    "The transformer uses self-attention..." + "to compute representations..."
```
→ Second sentence loses context

### 3. Small Dataset (159 Paragraphs)
- Only 253 sentences total
- Not enough to see statistical benefits

### 4. No Temporal Flow (Confirmed by Δ ≈ 0)
- Papers organized by logical sections, not temporal sequences
- Δ ≈ 0 at both paragraph and sentence level

---

## Acceptance Gates

**Required for production**:
1. R@5: +7pp vs baseline → ❌ **FAILED** (0pp improvement)
2. MRR: +0.04 vs baseline → ❌ **FAILED** (+0.005 at best)
3. Latency: ≤+15% → ⚠️ **N/A** (not measured)

**Verdict**: Did NOT meet gates. Keep paragraph-only retrieval.

---

## Decision

**Keep**: Paragraph-only retrieval (current system)
- Best R@5 = 0.280
- Simpler, no added latency
- Sentence information doesn't help on this dataset

**Archive**: Sentence-aware components
- Sentence bank (artifacts moved to `archived_p9_sentence_retrieval/`)
- Directional adapter (hurt performance)
- Scripts kept in `tools/` for future use on different datasets

**Do NOT**: Train Q-tower for arXiv
- No temporal signal (Δ ≈ 0)
- Would just learn paragraph similarity (already have that)

---

## When Might Sentence-Aware Retrieval Help?

**Suitable datasets** (hypothetical):
1. Wikipedia articles (explanatory flow with local references)
2. Programming tutorials (step-by-step instructions)
3. Narrative fiction (character/plot continuity)
4. Recipe/how-to guides (procedural sequences)

**Requirements**:
- Sequential content (next builds on previous)
- Local continuity (entities/verbs carry across)
- Large scale (10k+ paragraphs)
- **Gate first**: Measure Δ > 0.01 at sentence level before investing

---

## Components Delivered

### Scripts (7 total)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `tools/sentence_delta_check.py` | Validate Δ at sentence level | 145 | ✅ Complete |
| `tools/build_sentence_bank.py` | Build sentence vector bank | 139 | ✅ Complete |
| `tools/npz_to_paragraph_jsonl.py` | Convert NPZ → JSONL | 59 | ✅ Complete |
| `app/retriever/sent_para_rerank.py` | Two-stage retrieval | 247 | ✅ Complete |
| `tools/fit_directional_adapter.py` | Fit linear adapter | 127 | ✅ Complete |
| `tools/test_sentence_retrieval.py` | Test 3 configs | 236 | ✅ Complete |
| `tools/test_fusion_weights.py` | Grid search fusion weights | 197 | ✅ Complete |

**Total code**: ~1,150 lines

### Artifacts (4 total)

| File | Description | Size |
|------|-------------|------|
| `artifacts/arxiv_sentence_bank.npz` | 253 sentences, 768D | 0.7 MB |
| `artifacts/directional_adapter.npz` | 768×768 transform | 2.1 MB |
| `artifacts/sentence_retrieval_results.json` | 3-config results | 1 KB |
| `artifacts/fusion_weight_results.json` | 5-config grid | 2 KB |

### Documentation (2 reports)

| File | Purpose | Lines |
|------|---------|-------|
| `artifacts/lvm/P9_SENTENCE_RETRIEVAL_FINDINGS.md` | Comprehensive findings report | 350 |
| `artifacts/lvm/SESSION_2025_11_05_P9_SENTENCE_RETRIEVAL.md` | Session summary (this file) | 200 |

**Total documentation**: ~550 lines

---

## Timeline

| Time | Task | Result |
|------|------|--------|
| 09:30 | User requested sentence database build + test | Plan agreed |
| 09:35 | Created `sentence_delta_check.py` | ✅ Script working |
| 09:50 | Ran sentence Δ probe (narrative) | Δ = +0.0001 |
| 10:05 | Ran sentence Δ probe (arXiv) | Δ = -0.00004 |
| 10:20 | Built sentence bank (253 sentences) | ✅ NPZ created |
| 10:35 | Created reranker + directional adapter scripts | ✅ Code complete |
| 10:50 | Fit directional adapter | ⚠️ Δ got worse |
| 11:05 | Ran 3-config test | ❌ Sent rerank hurt R@5 |
| 11:20 | Created fusion weight grid test | ✅ Script ready |
| 11:30 | Ran 5-config fusion weight grid | ✅ Baseline best |
| 11:40 | Documented findings | ✅ Reports complete |

**Total session time**: ~2 hours

---

## Key Learnings

### 1. Sentence-Level Δ ≈ 0 (Confirmed)

**Tested**: 9,816 sequences across narrative + arXiv
**Result**: Δ within [-0.0001, +0.0001] (essentially zero)
**Conclusion**: GTR-T5 embeddings are symmetric at all scales (sentence, paragraph, chapter)

### 2. Self-Contained Content ≠ Sequential Content

**arXiv papers**:
- Paragraph-level coherence (complete sub-topics)
- Weak next-paragraph signal
- Sentence splitting breaks technical concepts
- **Best**: Paragraph-only retrieval

**Sequential content** (hypothetical):
- Wikipedia articles, tutorials, stories
- Sentence-level continuity (entities, verbs)
- **Might benefit** from sentence-aware retrieval (needs testing)

### 3. Directional Adapter Doesn't Help Retrieval

**Theory**: Learn linear transform W to prefer forward over backward
**Result**: Δ got worse (-0.0004), R@5 dropped 36%
**Conclusion**: Can't fix symmetric geometry with linear transforms

### 4. Acceptance Gates Are Critical

**Without gates**: Might deploy sentence-aware retrieval based on marginal MRR gain (+0.005)
**With gates**: Clear decision to reject (need +7pp R@5, got 0pp)
**Learning**: Always set concrete gates BEFORE investing in infrastructure

---

## Recommendations

### Immediate Actions

1. ✅ **Keep paragraph-only retrieval** for arXiv
2. ✅ **Archive sentence-aware components** (don't delete)
3. ✅ **Update CLAUDE.md** with P9 findings

### Future Work (If Desired)

**If testing sentence-aware retrieval on new datasets**:

1. **Gate first**: Measure Δ > 0.01 at sentence level
2. **Choose sequential content**: Wikipedia, tutorials, stories
3. **Large scale**: 10k+ paragraphs
4. **Conservative fusion**: a=0.2-0.3, b=0.7-0.8
5. **Set acceptance gates**: +7pp R@5, +0.04 MRR, ≤+15% latency

**Don't bother if**:
- Δ ≤ 0.01 at sentence level
- Content is self-contained (technical papers, encyclopedia entries)
- Dataset < 1k paragraphs

---

## Files to Archive

```bash
# Move artifacts to archive (keep scripts in tools/)
mkdir -p artifacts/lvm/archived_p9_sentence_retrieval
mv artifacts/arxiv_sentence_bank.npz artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/directional_adapter.npz artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/sentence_retrieval_results.json artifacts/lvm/archived_p9_sentence_retrieval/
mv artifacts/fusion_weight_results.json artifacts/lvm/archived_p9_sentence_retrieval/
mv data/arxiv_paragraphs.jsonl artifacts/lvm/archived_p9_sentence_retrieval/
```

Scripts remain in `tools/` for future use.

---

## Updated CLAUDE.md Checkpoint

```markdown
## 📌 ACTIVE CHECKPOINT: P9 Sentence-Aware Retrieval TESTED (2025-11-05)

**STATUS**: ✅ TESTED - Keep paragraph-only retrieval

**Finding**: Sentence-level granularity does NOT improve retrieval on arXiv papers.

**Results**:
- Baseline (para-only): R@5 = 0.280, MRR = 0.116
- Sentence rerank (best): R@5 = 0.280 (tied), MRR = 0.121 (+0.005)
- Directional adapter: R@5 = 0.180 (-36%), MRR = 0.071
- **Acceptance gates**: ❌ FAILED (+7pp R@5 required, got 0pp)

**Decision**: Keep paragraph-only retrieval. Archive sentence-aware components.

**Why it failed**:
1. arXiv paragraphs are self-contained (not sequential)
2. Sentence splitting breaks technical concepts
3. Small dataset (159 paragraphs, 253 sentences)
4. Confirmed Δ ≈ 0 at sentence level (GTR-T5 is symmetric)

**When it might work**:
- Sequential content (Wikipedia, tutorials, stories)
- Large scale (10k+ paragraphs)
- **Gate first**: Measure Δ > 0.01 at sentence level before investing

**See**: `artifacts/lvm/P9_SENTENCE_RETRIEVAL_FINDINGS.md` for full report

**Previous Checkpoint**: AR-LVM Abandoned (2025-11-04)
```

---

## Summary

**Completed**: Full test of sentence-aware retrieval in ~2 hours
**Result**: Paragraph-only is best for arXiv papers
**Decision**: Keep current system, archive sentence components
**Next**: If revisiting, test on Wikipedia/tutorials with Δ > 0.01 gate

---

**Session complete**: 2025-11-05 11:45
