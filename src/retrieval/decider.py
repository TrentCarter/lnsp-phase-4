from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np

@dataclass
class LaneConfig:
    tau_snap: float = 0.92
    tau_novel: float = 0.85
    lane_name: str = "neutral"

@dataclass
class DecisionRecord:
    c_max: float
    decision: str
    neighbor_id: Optional[str]
    alpha: Optional[float]
    lane: str
    near_dup_drop: bool = False


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def alpha_from_cos(c: float) -> float:
    # Default schedule: 0.3 at 0.86 → 0.7 at 0.91 → 0.9 at ≥0.95
    if c <= 0.86:
        return 0.3
    if c < 0.91:
        return 0.3 + 0.4 * (c - 0.86) / 0.05
    if c < 0.95:
        return 0.7 + 0.2 * (c - 0.91) / 0.04
    return 0.9


def choose_next_vector(
    v_hat: np.ndarray,
    neighbors: Iterable[Tuple[str, np.ndarray, float]],
    lane_cfg: LaneConfig,
    recent_ids: Iterable[str] = (),
    near_dup_cos: float = 0.98,
    near_dup_window: int = 8,
) -> Tuple[np.ndarray, DecisionRecord]:
    """Dual‑path decision: SNAP / BLEND / NOVEL with near‑duplicate guard.

    neighbors: iterable of (id, vec, cosine) with vec unit‑norm
    returns: (v_out unit‑norm, DecisionRecord)
    """
    v_hat = l2norm(v_hat)
    nbrs = list(neighbors)
    if not nbrs:
        return v_hat, DecisionRecord(0.0, "NOVEL", None, None, lane_cfg.lane_name)

    # Pick argmax by cosine
    n_id, n_vec, c = max(nbrs, key=lambda x: x[2])

    # Decide
    if c >= lane_cfg.tau_snap:
        v_out = n_vec
        decision = "SNAP"
        alpha = None
    elif c <= lane_cfg.tau_novel:
        v_out = v_hat
        decision = "NOVEL"
        alpha = None
    else:
        a = alpha_from_cos(c)
        v_out = l2norm(a * v_hat + (1.0 - a) * n_vec)
        decision = "BLEND"
        alpha = a

    near_dup = False
    if c > near_dup_cos and n_id in set(list(recent_ids)[-near_dup_window:]) and lane_cfg.lane_name != "legal":
        # Prefer novel to avoid degenerate repeats (except legal lane)
        v_out = v_hat
        near_dup = True
        decision = "NOVEL_DUP_DROP"
        alpha = None

    rec = DecisionRecord(c, decision, n_id, alpha, lane_cfg.lane_name, near_dup)
    return v_out, rec
