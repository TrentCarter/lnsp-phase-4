import os, json, numpy as np, subprocess, sys, pathlib

# Ensure src is in path
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def test_vec_upsert_end_to_end(tmp_path):
    # minimal run
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE","1")
    env.setdefault("TRANSFORMERS_OFFLINE","1")
    env["PYTHONPATH"] = str(SRC_DIR)
    
    # Use a temporary NPZ path to avoid conflicts
    npz_path = tmp_path / "test_vectors.npz"
    env["LNSP_NPZ_PATH"] = str(npz_path)

    active_jsonl_path = tmp_path / "cpesh_active.jsonl"
    if active_jsonl_path.exists():
        active_jsonl_path.unlink()

    command = [
        sys.executable, str(ROOT_DIR / "tools" / "vec_upsert.py"),
        "--text", "Photosynthesis uses light to synthesize sugars.",
        "--doc-id", "test:demo:001"
    ]

    p = subprocess.Popen(
        command,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    out, err = p.communicate(timeout=60)
    assert p.returncode == 0, f"Script failed. STDERR: {err}"
    j = json.loads(out)
    assert j.get("ok") is True

    # check Active JSONL
    active = pathlib.Path("artifacts/cpesh_active.jsonl"); assert active.exists()
    hit = False
    with active.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("doc_id") == "test:demo:001":
                assert "cpe_id" in row and "created_at" in row and "access_count" in row
                assert "concept_text" in row and "probe_question" in row and "expected_answer" in row
                assert "tmd_bits" in row and "lane_index" in row
                hit = True; break
    assert hit, "upsert row not found in active jsonl"

    # check NPZ updated
    z = np.load(npz_path, allow_pickle=True)
    assert z["vectors"].shape[1] == 784
    assert len(z["doc_ids"]) == len(z["cpe_ids"]) == z["vectors"].shape[0]
