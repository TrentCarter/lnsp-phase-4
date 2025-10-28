# LEAKED EVAL SETS - DO NOT USE

**Status**: ❌ COMPROMISED
**Date Quarantined**: 2025-10-27

## Leak Details

All files in this directory have significant train/eval overlap:

| File | Article Overlap | Chunk Overlap | Status |
|------|-----------------|---------------|---------|
| `eval_v2_payload_aligned.npz` | 91.5% | 74% | Oracle (cos=1.0) |
| `eval_v2_ready.npz` | ~90% | ~70% | Leaked |
| `eval_v2_ready_aligned.npz` | ~90% | ~70% | Leaked |

## Why This Happened

1. Training data uses articles 0-6100 (6101 articles)
2. Eval sets reused these articles with different chunk splits
3. Result: Model saw same article patterns during training

## Impact

- **AMN baseline**: 100% R@5 (invalid - oracle scenario)
- **Evaluation**: Does NOT test generalization to unseen articles
- **Decision**: Can only test within-article generalization (Level 1)

## What To Use Instead

**For Level 1 (within-article)**: Create new split with disjoint chunks from same articles
**For Level 2 (cross-article)**: Ingest fresh Wikipedia articles (10000+)

## CI Guard

Never use these files without explicit warning:

```python
LEAKED_EVAL_FILES = {
    'eval_v2_payload_aligned.npz',
    'eval_v2_ready.npz',
    'eval_v2_ready_aligned.npz',
}

if eval_file.name in LEAKED_EVAL_FILES:
    raise ValueError(f"LEAK: {eval_file} is compromised! Use disjoint eval set.")
```

## Resolution

See: `docs/EVAL_STRATEGY_UPDATE.md` for multi-level evaluation approach.
