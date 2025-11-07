"""
PLMS Stratified Sampler - Representative Canary Coverage

Ensures 1% rehearsal doesn't under-sample rare, expensive tasks.

Key design:
- Stratify by: task_kind × complexity × lane
- Minimum: ≥1 sample per stratum
- Auto-bump: if strata < 100, increase rehearsal_pct until all covered (cap 5%)
- Returns: sample + strata_coverage metrics
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random


def stratified_sample(
    plan: List[Dict[str, Any]],
    rehearsal_pct: float,
    min_per_stratum: int = 1,
    max_pct: float = 0.05
) -> Tuple[List[Dict], Dict]:
    """
    Sample tasks with stratified coverage guarantee.

    Args:
        plan: Full task list with task_kind, complexity, lane_id
        rehearsal_pct: Requested rehearsal percentage (e.g., 0.01 for 1%)
        min_per_stratum: Minimum samples per stratum (default: 1)
        max_pct: Maximum allowed rehearsal_pct (default: 5%)

    Returns:
        (sampled_tasks, metrics)
        metrics = {
            "n_total": int,
            "n_sample": int,
            "actual_pct": float,
            "requested_pct": float,
            "n_strata": int,
            "strata_coverage": float,  # 0.0-1.0
            "auto_bumped": bool
        }
    """
    n_total = len(plan)

    # Group by stratum: (task_kind, complexity, lane_id)
    strata = defaultdict(list)
    for task in plan:
        stratum_key = (
            task.get("task_kind", "unknown"),
            task.get("complexity", "medium"),
            task.get("lane_id", 0)
        )
        strata[stratum_key].append(task)

    n_strata = len(strata)

    # Compute initial sample size
    n_sample_requested = max(1, int(n_total * rehearsal_pct))

    # Check if we can cover all strata with requested sample size
    min_required = n_strata * min_per_stratum

    if n_sample_requested < min_required:
        # Auto-bump to ensure coverage
        n_sample = min_required
        actual_pct = n_sample / n_total
        auto_bumped = True

        # Cap at max_pct
        if actual_pct > max_pct:
            n_sample = max(1, int(n_total * max_pct))
            actual_pct = n_sample / n_total
    else:
        n_sample = n_sample_requested
        actual_pct = rehearsal_pct
        auto_bumped = False

    # Perform stratified sampling
    sampled = []
    samples_per_stratum = defaultdict(int)

    # Phase 1: Ensure min_per_stratum samples from each stratum
    for stratum_key, tasks in strata.items():
        n_from_stratum = min(min_per_stratum, len(tasks))
        sampled_from_stratum = random.sample(tasks, n_from_stratum)
        sampled.extend(sampled_from_stratum)
        samples_per_stratum[stratum_key] = n_from_stratum

    # Phase 2: Fill remaining quota proportionally
    remaining_quota = n_sample - len(sampled)

    if remaining_quota > 0:
        # Allocate remaining samples proportional to stratum size
        for stratum_key, tasks in strata.items():
            already_sampled = samples_per_stratum[stratum_key]
            remaining_in_stratum = len(tasks) - already_sampled

            if remaining_in_stratum > 0:
                # Proportional allocation
                proportion = len(tasks) / n_total
                additional = max(0, int(remaining_quota * proportion))
                additional = min(additional, remaining_in_stratum)

                if additional > 0:
                    # Sample additional tasks (exclude already sampled)
                    already_sampled_ids = {id(t) for t in sampled if (
                        t.get("task_kind", "unknown"),
                        t.get("complexity", "medium"),
                        t.get("lane_id", 0)
                    ) == stratum_key}

                    available = [t for t in tasks if id(t) not in already_sampled_ids]
                    additional_sample = random.sample(available, min(additional, len(available)))
                    sampled.extend(additional_sample)
                    samples_per_stratum[stratum_key] += len(additional_sample)

    # Trim to exact quota if oversampled
    if len(sampled) > n_sample:
        sampled = random.sample(sampled, n_sample)

    # Compute strata coverage
    strata_with_samples = sum(1 for count in samples_per_stratum.values() if count >= min_per_stratum)
    strata_coverage = strata_with_samples / max(1, n_strata)

    metrics = {
        "n_total": n_total,
        "n_sample": len(sampled),
        "actual_pct": len(sampled) / n_total,
        "requested_pct": rehearsal_pct,
        "n_strata": n_strata,
        "strata_coverage": strata_coverage,
        "auto_bumped": auto_bumped,
        "samples_per_stratum": dict(samples_per_stratum)
    }

    return sampled, metrics


def validate_strata_coverage(metrics: Dict) -> List[str]:
    """
    Validate strata coverage and return warnings.

    Args:
        metrics: Output from stratified_sample()

    Returns:
        List of warning messages (empty if all good)
    """
    warnings = []

    if metrics["strata_coverage"] < 1.0:
        missing_pct = (1.0 - metrics["strata_coverage"]) * 100
        warnings.append(
            f"⚠️  {missing_pct:.1f}% of strata have no samples. "
            f"Consider increasing rehearsal_pct to {metrics['actual_pct'] * 1.5:.3f}"
        )

    if metrics["auto_bumped"]:
        warnings.append(
            f"✓ Auto-bumped rehearsal from {metrics['requested_pct']:.1%} to "
            f"{metrics['actual_pct']:.1%} for full strata coverage"
        )

    if metrics["actual_pct"] > 0.05:
        warnings.append(
            f"⚠️  Rehearsal percentage {metrics['actual_pct']:.1%} exceeds 5% cap. "
            f"Project has {metrics['n_strata']} strata, consider simplifying."
        )

    return warnings
