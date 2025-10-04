# TMD Alpha Parameter Tuning Guide

## Quick Start

```bash
# Run alpha tuning (tests 5 values: 0.2, 0.3, 0.4, 0.5, 0.6)
bash tune_alpha.sh

# Compare results
./.venv/bin/python compare_alpha_results.py
```

## What is Alpha?

Alpha controls the balance between vector similarity and TMD alignment:

```
final_score = (1-alpha) * vec_score + alpha * tmd_score
```

- **alpha = 0.0**: Pure vector similarity (no TMD)
- **alpha = 0.3**: 70% vector, 30% TMD (current default)
- **alpha = 0.5**: Equal weight
- **alpha = 1.0**: Pure TMD matching

## Expected Results

Based on initial testing (alpha=0.3):
- Baseline vec: P@1=91.5%, P@5=95.6%
- TMD re-rank: P@1=94.5%, P@5=97.5% (+2-3%)

## Tuning Strategy

1. **Start conservative** (alpha=0.2-0.3)
   - Preserves strong vector similarity signal
   - TMD adds refinement at the margins

2. **Increase for specialized domains** (alpha=0.4-0.6)
   - If your queries have strong task/domain structure
   - If TMD metadata is highly accurate

3. **Never go too high** (alpha > 0.7)
   - Risks over-relying on TMD metadata
   - Vector similarity is still the primary signal

## Time Estimates

- **Per alpha value**: ~5 minutes (200 queries × 1.5s LLM call)
- **Full sweep (5 values)**: ~25 minutes

## Files Created

- `RAG/results/tmd_alpha_0.2_oct4.jsonl`
- `RAG/results/tmd_alpha_0.3_oct4.jsonl`
- `RAG/results/tmd_alpha_0.4_oct4.jsonl`
- `RAG/results/tmd_alpha_0.5_oct4.jsonl`
- `RAG/results/tmd_alpha_0.6_oct4.jsonl`

## After Tuning

Once you find the optimal alpha:

1. **Update default** in `RAG/bench.py` line 439
2. **Consider re-ingestion** with LLM-based TMD if improvement > 5%
3. **Document choice** in project documentation

## Notes

- Alpha tuning uses query-time LLM TMD extraction
- Corpus still uses pattern-based TMD (unless re-ingested)
- Results may improve further with full LLM-based corpus re-ingestion
