import numpy as np
from src.retrieval.decider import LaneConfig, choose_next_vector

v_hat = np.eye(1, 768, 0).astype(np.float32).ravel()

# Synthetic neighbors at specific cosines
n_083 = ("n083", np.eye(1, 768, 0).astype(np.float32).ravel() * 0 + 1/np.sqrt(2), 0.83)
# Build exact cosines by constructing dot products directly

def mk_neighbor(id, cosine):
    # Build a neighbor with exact cosine to v_hat along first dim
    vec = np.zeros(768, dtype=np.float32)
    vec[0] = cosine
    # Fill orthogonal component to keep unit norm
    ortho = np.sqrt(max(1e-8, 1 - cosine**2))
    vec[1] = ortho
    return (id, vec, cosine)

n_0875 = mk_neighbor("n875", 0.875)
n_0925 = mk_neighbor("n925", 0.925)

lane = LaneConfig(tau_snap=0.92, tau_novel=0.85)

# 0.83 → NOVEL
_, rec = choose_next_vector(v_hat, [mk_neighbor("n830", 0.83)], lane)
assert rec.decision.startswith("NOVEL")

# 0.875 → BLEND
_, rec = choose_next_vector(v_hat, [n_0875], lane)
assert rec.decision == "BLEND"

# 0.925 → SNAP
_, rec = choose_next_vector(v_hat, [n_0925], lane)
assert rec.decision == "SNAP"

# near‑dup drop
_, rec = choose_next_vector(v_hat, [mk_neighbor("dup", 0.99)], lane, recent_ids=["x","y","dup"])
assert rec.decision == "NOVEL_DUP_DROP"
