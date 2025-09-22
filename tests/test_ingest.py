import numpy as np

from src.ingest_factoid import ingest, SAMPLE_ITEMS


def test_ingest_smoke(tmp_path):
    out = tmp_path / "vecs.npz"
    stats = ingest(SAMPLE_ITEMS, write_pg=False, write_neo4j=False, faiss_out=str(out))
    assert stats["count"] == 4
    data = np.load(out)
    assert data["fused"].shape[1] == 784
    assert data["concept"].shape[1] == 768
    assert data["question"].shape[1] == 768
    print("Smoke test passed.")
