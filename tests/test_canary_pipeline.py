import os
import numpy as np
import pytest

NPZ_PATH = os.path.join("artifacts", "fw_canary.npz")


def _load_npz():
    if not os.path.exists(NPZ_PATH):
        pytest.skip(f"missing canary artifact: {NPZ_PATH}")
    return np.load(NPZ_PATH, allow_pickle=True)


def test_npz_shapes_and_counts():
    z = _load_npz()
    vectors = z["vectors"]
    concept = z["concept_vecs"] if "concept_vecs" in z else z["concept"]
    question = z["question_vecs"] if "question_vecs" in z else z["question"]
    lanes = z.get("lane_indices")
    doc_ids = z["doc_ids"]
    cpe_ids = z["cpe_ids"]

    assert vectors.ndim == 2 and vectors.shape[1] == 784
    assert concept.shape == (vectors.shape[0], 768)
    assert question.shape == (vectors.shape[0], 768)
    assert len(doc_ids) == len(cpe_ids) == vectors.shape[0]
    if lanes is not None:
        assert lanes.shape[0] == vectors.shape[0]

    norms = np.linalg.norm(vectors, axis=1)
    assert float(norms.min()) > 0.9 and float(norms.max()) <= 1.001

    if "tmd_dense" in z:
        tmd_dense = z["tmd_dense"]
        assert tmd_dense.shape == (vectors.shape[0], 16)
        tmd_norms = np.linalg.norm(tmd_dense, axis=1)
        assert np.allclose(tmd_norms, 1.0, atol=1e-5)


def test_tmd_lane_consistency():
    z = _load_npz()
    lanes = z.get("lane_indices")
    if lanes is None:
        pytest.skip("npz missing lane_indices")
    lanes = lanes.astype(int)
    assert lanes.min() >= 0
    assert lanes.max() < 32768

    if "tmd_dense" in z:
        tmd_dense = z["tmd_dense"]
        vectors = z["vectors"]
        norms = np.linalg.norm(tmd_dense, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

        fused_slice = vectors[:, :16]
        fused_norms = np.linalg.norm(fused_slice, axis=1)
        assert np.all(fused_norms > 0)

        # Fused vectors are global L2-normalized, so the first 16 dims should be a
        # scaled copy of the TMD projection. Check proportionality per-row.
        scale = (norms / fused_norms)[:, None]
        assert np.allclose(fused_slice * scale, tmd_dense, atol=1e-4)

    try:
        from src.utils.tmd import lane_index_from_bits  # type: ignore
    except Exception:
        return

    # When lane indices are stored, ensure the helper bounds remain realistic
    assert callable(lane_index_from_bits)
