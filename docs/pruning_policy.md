# CPESH Cache Pruning Policy
_Version: 1.0 (S3 Sprint - 2025-09-25)_

## Overview

The CPESH (Concept-Probe-Expected with Soft/Hard negatives) cache pruning system maintains optimal cache quality by removing low-value entries based on quality metrics, access patterns, and curator overrides.

## Pruning Rules

### Default Criteria

Entries are marked for pruning if they meet ANY of the following conditions:

1. **Low Quality Score**
   - `echo_score < 0.82`
   - Indicates poor semantic alignment between concept and probe

2. **Zero Access Pattern**
   - `access_count == 0` over 14 days since creation
   - Identifies entries that provide no value to queries

3. **Curator Blacklist**
   - Entry ID appears in curator manifest
   - Manual override for known problematic entries

### Lane-Aware Overrides

Different content lanes have different quality thresholds:

#### L1_FACTOID (Strict)
- `echo_score < 0.85` (higher threshold)
- `access_count == 0` over 7 days (shorter window)
- Priority: Accuracy over coverage

#### L2_NARRATIVE (Balanced)
- `echo_score < 0.82` (default threshold)
- `access_count == 0` over 14 days
- Priority: Balance accuracy and diversity

#### L3_INSTRUCTION (Permissive)
- `echo_score < 0.78` (lower threshold)
- `access_count == 0` over 21 days
- Priority: Coverage over strict accuracy

## Manifest Schema

### JSON Format
```json
{
  "version": "1.0",
  "generated_at": "2025-09-25T10:00:00Z",
  "policy": {
    "echo_threshold": 0.82,
    "access_window_days": 14,
    "lane_overrides": {
      "L1_FACTOID": {"echo_threshold": 0.85, "access_window_days": 7},
      "L3_INSTRUCTION": {"echo_threshold": 0.78, "access_window_days": 21}
    }
  },
  "entries": [
    {
      "id": "uuid-1234-5678",
      "reason": "low_echo_score",
      "score": 0.75,
      "timestamp": "2025-09-25T09:30:00Z",
      "author": "auto_policy"
    },
    {
      "id": "uuid-2345-6789",
      "reason": "zero_access",
      "last_access": null,
      "created_at": "2025-09-01T00:00:00Z",
      "author": "auto_policy"
    },
    {
      "id": "uuid-3456-7890",
      "reason": "curator_override",
      "note": "Hallucinated content detected",
      "timestamp": "2025-09-24T15:00:00Z",
      "author": "human_curator"
    }
  ],
  "statistics": {
    "total_entries": 10000,
    "marked_for_pruning": 1523,
    "by_reason": {
      "low_echo_score": 892,
      "zero_access": 581,
      "curator_override": 50
    },
    "estimated_quality_lift": 0.073
  }
}
```

## Implementation

### Pruning Script Usage
```bash
# Generate manifest (dry run)
python scripts/prune_cache.py \
  --input artifacts/cpesh_cache.jsonl \
  --policy default \
  --dry-run \
  --output eval/prune_manifest.json

# Apply manifest (actual pruning)
python scripts/prune_cache.py \
  --manifest eval/prune_manifest.json \
  --input artifacts/cpesh_cache.jsonl \
  --output artifacts/cpesh_cache_pruned.jsonl \
  --backup

# Verify results
python scripts/prune_cache.py \
  --verify \
  --before artifacts/cpesh_cache_backup.jsonl \
  --after artifacts/cpesh_cache_pruned.jsonl \
  --report eval/prune_report.md
```

### Makefile Targets
```bash
# Generate pruning manifest
make prune-manifest

# Execute pruning with backup
make prune

# Generate pruning report
make prune-report
```

## Quality Metrics

### Pre-Pruning Baseline
- Cache size: ~10,000 entries
- Average echo_score: 0.84
- Hit@1: 43%
- P50 latency: 85ms

### Expected Post-Pruning
- Cache size: ~8,500 entries (-15%)
- Average echo_score: 0.87 (+3.6%)
- Hit@1: 47% (+4%)
- P50 latency: 75ms (-12%)

## Safety Measures

### Automatic Backups
- Original cache backed up to `artifacts/cpesh_cache_backup_{timestamp}.jsonl`
- Backup retention: 30 days
- Rollback command: `make prune-rollback`

### Atomic Operations
- Pruning creates new file, then atomically swaps
- No in-place modifications
- Process is resumable if interrupted

### Validation Checks
- Entry count verification
- Schema validation
- Echo score distribution analysis
- Access pattern histogram

## Monitoring

### Key Metrics to Track
1. **Cache Quality**
   - Average echo_score
   - Score distribution percentiles (P25, P50, P75, P95)

2. **Cache Efficiency**
   - Hit rate
   - Average access_count
   - Zero-access percentage

3. **System Impact**
   - Query latency changes
   - Memory usage reduction
   - Index rebuild time

### Alert Thresholds
- Pruning removes > 25% of cache → Manual review required
- Average echo_score drops post-prune → Rollback and investigate
- Hit@1 degrades > 5% → Rollback to backup

## Scheduled Maintenance

### Weekly (Automated)
- Generate pruning candidates report
- Update access statistics
- No actual pruning

### Bi-weekly (Semi-automated)
- Run pruning with dry-run
- Review manifest
- Execute if metrics acceptable

### Monthly (Manual)
- Full cache quality audit
- Curator review of edge cases
- Policy threshold adjustments

## Future Enhancements

### v1.1 (Planned)
- Adaptive thresholds based on query patterns
- Temporal decay for older entries
- Multi-stage pruning (soft → hard delete)

### v2.0 (Research)
- ML-based quality prediction
- Cross-lane transfer instead of deletion
- Active learning from pruning outcomes