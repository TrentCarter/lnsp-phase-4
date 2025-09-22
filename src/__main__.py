from __future__ import annotations
import numpy as np
from uuid import uuid4

from .tmd_encoder import pack_tmd, unpack_tmd, lane_index_from_bits, tmd16_deterministic
from .vectorizer import EmbeddingBackend
from .utils.norms import l2_normalize


def _smoke():
    # TMD
    bits = pack_tmd(1, 2, 3)
    assert unpack_tmd(bits) == (1, 2, 3)
    lane = lane_index_from_bits(bits)
    assert 0 <= lane <= 32767
    tmd16 = tmd16_deterministic(1, 2, 3)
    assert tmd16.shape == (16,), "tmd16 must be 16D"

    # Embeddings
    be = EmbeddingBackend()
    e = be.encode(["Light-dependent reactions split water", "Photolysis of water"])
    assert e.shape[1] == 768

    # Fused 784D
    fused = np.concatenate([tmd16.reshape(1, -1), e[:1]], axis=1)
    assert fused.shape[1] == 784
    fused = l2_normalize(fused)
    print("SMOKE OK: lane=", lane, " fused_dim=", fused.shape[1])


if __name__ == "__main__":
    _smoke()
