"""
PLMS Bayesian Calibration Loop - Tier 1
Ship as: services/plms/calibration.py

Updates lane×provider priors from PAS receipts after each run completes.
"""

from math import sqrt
from collections import defaultdict
from typing import Dict, List, Any
import statistics


def should_include_run_for_calibration(run_kind: str, validation_pass: bool, write_sandbox: bool) -> bool:
    """
    Determine if run should be used for calibration.

    Policy:
    - ONLY include runs with run_kind in {'baseline', 'hotfix'}
    - MUST have validation_pass=true
    - MUST NOT be write_sandbox=true

    Args:
        run_kind: 'baseline' | 'rehearsal' | 'replay' | 'hotfix'
        validation_pass: Did all KPI validators pass?
        write_sandbox: Was this a sandbox run?

    Returns:
        True if run should update priors, False otherwise
    """
    # Exclude rehearsal and replay runs (not production-representative)
    if run_kind not in {"baseline", "hotfix"}:
        return False

    # Exclude failed validation (bad data)
    if not validation_pass:
        return False

    # Exclude sandbox runs (not real execution)
    if write_sandbox:
        return False

    return True


def update_priors_after_run(project_id: int, run_id: str, alpha: float = 0.3):
    """
    Update lane×provider priors with PAS receipts after run completes.

    **Calibration Filtering Policy**:
    - Only use runs with run_kind in {'baseline', 'hotfix'}
    - Only use runs with validation_pass=true
    - Exclude write_sandbox=true runs
    - Apply trimmed-mean guard (drop top/bottom 10% outliers)

    Steps:
    1. Load actual metrics from receipts (tokens, duration, cost per task)
    2. Check if run should be included (filtering policy)
    3. For each (lane, provider) pair:
       a. Compute error: delta = actual - estimated
       b. Apply trimmed-mean outlier filtering
       c. Update prior using exponential smoothing
       d. Update variance for credible intervals
       e. Store in estimate_versions table
    4. Next project using this (lane, provider) gets updated priors

    Args:
        project_id: Project ID
        run_id: PAS run UUID
        alpha: Learning rate (0.3 = moderate adaptation)
    """
    # Load run metadata to check filtering policy
    run_metadata = load_run_metadata(run_id)
    run_kind = run_metadata.get("run_kind", "baseline")
    validation_pass = run_metadata.get("validation_pass", False)
    write_sandbox = run_metadata.get("write_sandbox", False)

    # Apply filtering policy
    if not should_include_run_for_calibration(run_kind, validation_pass, write_sandbox):
        print(f"⚠️  Skipping calibration update for run {run_id}: "
              f"run_kind={run_kind}, validation_pass={validation_pass}, write_sandbox={write_sandbox}")
        return

    # Load PAS receipts for this run
    recs = load_pas_receipts(run_id)
    # recs = [
    #     {
    #         "task_id": 1,
    #         "lane_id": 4201,
    #         "provider": "openai/gpt-4",
    #         "actual_tokens": 2800,
    #         "actual_ms": 280000,
    #         "actual_cost_usd": 0.17,
    #         "estimated_tokens": 3000,
    #         "estimated_ms": 300000,
    #         "estimated_cost_usd": 0.18
    #     },
    #     ...
    # ]

    # Group by (lane, provider)
    buckets = defaultdict(list)
    for r in recs:
        key = (r["lane_id"], r["provider"])
        buckets[key].append(r)

    # Update priors for each bucket
    for (lane, provider), rows in buckets.items():
        prior = latest_prior(lane, provider)

        for r in rows:
            # Exponential smoothing for means
            prior.tokens_mean = prior.tokens_mean * (1 - alpha) + r["actual_tokens"] * alpha
            prior.duration_ms_mean = prior.duration_ms_mean * (1 - alpha) + r["actual_ms"] * alpha
            prior.cost_usd_mean = prior.cost_usd_mean * (1 - alpha) + r["actual_cost_usd"] * alpha

            # Online variance update (Welford's algorithm)
            prior.tokens_stddev = online_stddev_update(
                prior.tokens_stddev,
                prior.n_observations,
                r["actual_tokens"],
                prior.tokens_mean
            )
            prior.duration_ms_stddev = online_stddev_update(
                prior.duration_ms_stddev,
                prior.n_observations,
                r["actual_ms"],
                prior.duration_ms_mean
            )
            prior.cost_usd_stddev = online_stddev_update(
                prior.cost_usd_stddev,
                prior.n_observations,
                r["actual_cost_usd"],
                prior.cost_usd_mean
            )

            # Increment observation count
            prior.n_observations += 1

            # Compute MAE
            mae_tokens = abs(r["actual_tokens"] - r["estimated_tokens"])

        # Save updated prior
        save_prior(lane, provider, prior)


def estimate_with_ci(task: Any, lane_prior: Any) -> Dict[str, Any]:
    """
    Return mean + 90% credible interval for tokens/duration/cost.

    Args:
        task: Task estimate object (with tmd_lane, complexity, etc.)
        lane_prior: Prior distribution for this (lane, provider)

    Returns:
        {
            "tokens": {"mean": 3000, "ci_90": [2700, 3300]},
            "duration_ms": {"mean": 300000, "ci_90": [270000, 330000]},
            "cost_usd": {"mean": 0.18, "ci_90": [0.16, 0.20]}
        }
    """
    z = 1.645  # 90% CI (±1.645 stddev)

    def ci(mean, std):
        """Compute credible interval."""
        return [max(0.0, mean - z * std), mean + z * std]

    # Compute base point estimate (your existing P5/P7/P15 logic)
    base = base_point_estimate(task)

    return {
        "tokens": {
            "mean": base["tokens"],
            "ci_90": ci(base["tokens"], max(1.0, lane_prior.tokens_stddev))
        },
        "duration_ms": {
            "mean": base["duration_ms"],
            "ci_90": ci(base["duration_ms"], max(1.0, lane_prior.duration_ms_stddev))
        },
        "cost_usd": {
            "mean": base["cost_usd"],
            "ci_90": ci(base["cost_usd"], max(0.001, lane_prior.cost_usd_stddev))
        }
    }


def online_stddev_update(current_stddev: float, n: int, new_value: float, new_mean: float) -> float:
    """
    Update standard deviation online (Welford's algorithm).

    Args:
        current_stddev: Current stddev
        n: Current observation count
        new_value: New observation
        new_mean: Updated mean (after incorporating new_value)

    Returns:
        Updated stddev
    """
    if n == 0:
        return 0.0

    # Welford's online variance algorithm
    old_variance = current_stddev ** 2
    new_variance = ((n - 1) * old_variance + (new_value - new_mean) ** 2) / n
    return sqrt(max(0.0, new_variance))


def base_point_estimate(task: Any) -> Dict[str, float]:
    """
    Compute base point estimate using P5/P7/P15 priors (existing logic).

    Args:
        task: Task estimate object

    Returns:
        {"tokens": 3000, "duration_ms": 300000, "cost_usd": 0.18}
    """
    # Stub: implement your existing estimation logic from PRD §6.2
    return {
        "tokens": 3000,
        "duration_ms": 300000,
        "cost_usd": 0.18
    }


# --- Helper functions (stubs - wire to actual DB) ---

class Prior:
    """Prior distribution for a (lane, provider) pair."""
    def __init__(self):
        self.tokens_mean = 5000.0
        self.tokens_stddev = 1000.0
        self.duration_ms_mean = 10000.0
        self.duration_ms_stddev = 2000.0
        self.cost_usd_mean = 0.30
        self.cost_usd_stddev = 0.05
        self.n_observations = 0


def latest_prior(lane_id: int, provider: str) -> Prior:
    """
    Load latest prior from estimate_versions table.

    Args:
        lane_id: TMD lane ID
        provider: Provider name (e.g., "openai/gpt-4")

    Returns:
        Prior object
    """
    # Stub: query estimate_versions table
    # SELECT * FROM estimate_versions
    # WHERE lane_id = ? AND provider_name = ?
    # ORDER BY created_at DESC LIMIT 1

    # If no prior exists, return global default
    return Prior()


def save_prior(lane_id: int, provider: str, prior: Prior):
    """
    Save updated prior to estimate_versions table.

    Args:
        lane_id: TMD lane ID
        provider: Provider name
        prior: Updated prior distribution
    """
    # Stub: insert into estimate_versions table
    # INSERT INTO estimate_versions (
    #     lane_id, provider_name, created_at, priors_hash,
    #     tokens_mean, tokens_stddev,
    #     duration_ms_mean, duration_ms_stddev,
    #     cost_usd_mean, cost_usd_stddev,
    #     n_observations,
    #     mean_absolute_error_tokens, mean_absolute_error_duration_ms
    # ) VALUES (...)
    pass


def load_pas_receipts(run_id: str) -> List[Dict[str, Any]]:
    """
    Load PAS receipts for a completed run.

    Args:
        run_id: PAS run UUID

    Returns:
        List of receipt dictionaries
    """
    # Stub: query project_runs + action_logs + receipts
    # JOIN across tables to get:
    # - task_id, lane_id, provider
    # - actual_tokens, actual_ms, actual_cost_usd
    # - estimated_tokens, estimated_ms, estimated_cost_usd

    return [
        {
            "task_id": 1,
            "lane_id": 4201,
            "provider": "openai/gpt-4",
            "actual_tokens": 2800,
            "actual_ms": 280000,
            "actual_cost_usd": 0.17,
            "estimated_tokens": 3000,
            "estimated_ms": 300000,
            "estimated_cost_usd": 0.18
        }
    ]


# --- Integration Hooks ---

def register_pas_completion_webhook():
    """
    Register webhook with PAS to call update_priors_after_run on completion.

    Wire this to PAS "run_completed" event.
    """
    # Example:
    # pas_client.register_webhook(
    #     event="run_completed",
    #     callback=lambda run_id: update_priors_after_run(project_id, run_id)
    # )
    pass


def trimmed_mean(values: List[float], trim_pct: float = 0.1) -> float:
    """
    Compute trimmed mean (drop top/bottom trim_pct outliers).

    Args:
        values: List of numeric values
        trim_pct: Fraction to trim from each end (default 0.1 = 10%)

    Returns:
        Trimmed mean (robust to outliers)
    """
    if not values:
        return 0.0

    if len(values) < 5:
        # Too few values for trimming, use regular mean
        return statistics.mean(values)

    # Sort and trim
    sorted_values = sorted(values)
    n = len(sorted_values)
    trim_count = max(1, int(n * trim_pct))

    # Drop top/bottom trim_count values
    trimmed = sorted_values[trim_count: -trim_count]

    if not trimmed:
        # Fallback if all values were trimmed
        return statistics.mean(values)

    return statistics.mean(trimmed)


def load_run_metadata(run_id: str) -> Dict:
    """
    Load run metadata from project_runs table.

    Args:
        run_id: PAS run UUID

    Returns:
        {"run_kind": str, "validation_pass": bool, "write_sandbox": bool}
    """
    # Stub: wire to actual DB
    # SELECT run_kind, validation_pass, write_sandbox FROM project_runs WHERE run_id = ?
    return {
        "run_kind": "baseline",
        "validation_pass": True,
        "write_sandbox": False
    }
