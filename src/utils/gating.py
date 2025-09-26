"""
CPESH Gating Configuration and Logic

This module implements two-stage gating for CPESH (Concept-Probe-Expected-Soft-Hard)
entries to determine whether to use CPESH-assisted search or fallback to standard search.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CPESHGateConfig:
    """Configuration for CPESH gating decisions.

    Attributes:
        q_min: Minimum quality score required for CPESH usage (default: 0.82)
        cos_min: Minimum cosine similarity required for CPESH usage (default: 0.55)
        nprobe_cpesh: nprobe value to use when CPESH is selected (default: 8)
        nprobe_fallback: nprobe value to use when falling back (default: 16)
        lane_overrides: Lane-specific overrides for gating parameters
    """
    q_min: float = 0.82
    cos_min: float = 0.55
    nprobe_cpesh: int = 8
    nprobe_fallback: int = 16
    lane_overrides: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self):
        """Initialize lane_overrides if None."""
        if self.lane_overrides is None:
            self.lane_overrides = {"L1_FACTOID": {"q_min": 0.85}}


def apply_lane_overrides(cfg: CPESHGateConfig, lane: Optional[str]) -> CPESHGateConfig:
    """Apply lane-specific overrides to the base configuration.

    Args:
        cfg: Base CPESH gate configuration
        lane: Lane identifier (e.g., "L1_FACTOID", "L2_GRAPH", "L3_SYNTH")

    Returns:
        Configuration with lane-specific overrides applied
    """
    if lane and cfg.lane_overrides and lane in cfg.lane_overrides:
        overrides = cfg.lane_overrides[lane]
        return CPESHGateConfig(
            q_min=overrides.get("q_min", cfg.q_min),
            cos_min=overrides.get("cos_min", cfg.cos_min),
            nprobe_cpesh=overrides.get("nprobe_cpesh", cfg.nprobe_cpesh),
            nprobe_fallback=overrides.get("nprobe_fallback", cfg.nprobe_fallback),
            lane_overrides=cfg.lane_overrides
        )
    return cfg


def should_use_cpesh(cpesh_entry: Optional[dict], cfg: CPESHGateConfig) -> bool:
    """Determine whether to use CPESH based on entry quality and configuration.

    Args:
        cpesh_entry: CPESH entry from cache containing quality, cosine, etc.
        cfg: CPESH gate configuration

    Returns:
        True if CPESH should be used, False otherwise
    """
    if not cpesh_entry:
        return False

    # Check for insufficient evidence flag
    if cpesh_entry.get("insufficient_evidence"):
        return False

    # Check quality and cosine similarity thresholds
    quality = cpesh_entry.get("quality", 0)
    cosine = cpesh_entry.get("cosine", 0)

    return (quality >= cfg.q_min) and (cosine >= cfg.cos_min)


def create_gate_config_from_env() -> CPESHGateConfig:
    """Create CPESH gate configuration from environment variables.

    Environment variables:
        LNSP_CPESH_Q_MIN: Minimum quality threshold (default: 0.82)
        LNSP_CPESH_COS_MIN: Minimum cosine similarity threshold (default: 0.55)
        LNSP_NPROBE_CPESH: nprobe for CPESH-assisted search (default: 8)
        LNSP_NPROBE_DEFAULT: nprobe for fallback search (default: 16)

    Returns:
        Configured CPESHGateConfig instance
    """
    import os

    return CPESHGateConfig(
        q_min=float(os.getenv("LNSP_CPESH_Q_MIN", "0.82")),
        cos_min=float(os.getenv("LNSP_CPESH_COS_MIN", "0.55")),
        nprobe_cpesh=int(os.getenv("LNSP_NPROBE_CPESH", "8")),
        nprobe_fallback=int(os.getenv("LNSP_NPROBE_DEFAULT", "16")),
        lane_overrides={"L1_FACTOID": {"q_min": 0.85}}  # L1 stricter by default
    )


def log_gating_decision(
    query_id: str,
    lane: str,
    used_cpesh: bool,
    quality: Optional[float],
    cosine: Optional[float],
    chosen_nprobe: int,
    latency_ms: float,
    log_path: str = "artifacts/gating_decisions.jsonl"
) -> None:
    """Log a gating decision for analysis and debugging.

    Args:
        query_id: Unique identifier for the query
        lane: Lane used for the search
        used_cpesh: Whether CPESH was used
        quality: Quality score from CPESH entry (if available)
        cosine: Cosine similarity from CPESH entry (if available)
        chosen_nprobe: nprobe value that was used
        latency_ms: Search latency in milliseconds
        log_path: Path to log file (default: artifacts/gating_decisions.jsonl)
    """
    import json
    import time
    import os

    decision = {
        "query_id": query_id,
        "lane": lane,
        "used_cpesh": used_cpesh,
        "quality": quality,
        "cosine": cosine,
        "chosen_nprobe": chosen_nprobe,
        "latency_ms": round(latency_ms, 3),
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    except Exception as e:
        # Log to stderr if file logging fails
        print(f"[gating] Failed to log decision: {e}", file=sys.stderr)


def get_gating_metrics(log_path: str = "artifacts/gating_decisions.jsonl") -> Dict[str, int]:
    """Get gating metrics from the decision log.

    Args:
        log_path: Path to gating decisions log file

    Returns:
        Dictionary with total queries and CPESH usage count
    """
    import json
    import os

    counts = {"total": 0, "used_cpesh": 0}

    if not os.path.exists(log_path):
        return counts

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                counts["total"] += 1
                try:
                    decision = json.loads(line)
                    if decision.get("used_cpesh"):
                        counts["used_cpesh"] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {"total": 0, "used_cpesh": 0}

    return counts
