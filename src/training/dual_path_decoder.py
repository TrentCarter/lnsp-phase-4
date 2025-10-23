from __future__ import annotations
import numpy as np
from typing import Dict, Any, Iterable, Tuple
from src.retrieval.decider import choose_next_vector, LaneConfig

class DualPathDecoder:
    def __init__(self, lane: str, tau_snap: float, tau_novel: float, near_dup_cos: float = 0.98, near_dup_window: int = 8):
        self.cfg = LaneConfig(tau_snap=tau_snap, tau_novel=tau_novel, lane_name=lane)
        self.near_dup_cos = near_dup_cos
        self.near_dup_window = near_dup_window
        self.recent_ids = []

    def step(self, v_hat: np.ndarray, neighbors: Iterable[Tuple[str, np.ndarray, float]]):
        v_out, rec = choose_next_vector(v_hat, neighbors, self.cfg, self.recent_ids, self.near_dup_cos, self.near_dup_window)
        if rec.neighbor_id:
            self.recent_ids.append(rec.neighbor_id)
            if len(self.recent_ids) > 64:
                self.recent_ids = self.recent_ids[-64:]
        return v_out, rec
