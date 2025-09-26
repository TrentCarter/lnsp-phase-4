import os, json, pytest

def require_file(path):
    assert os.path.exists(path), f"missing {path}"

def test_index_meta_keys():
    require_file("artifacts/index_meta.json")
    with open("artifacts/index_meta.json") as f:
        meta = json.load(f)

    # Handle both nested structure (current format) and flat structure
    if len(meta.keys()) == 1 and isinstance(list(meta.values())[0], dict):
        # Nested structure - get first index entry
        first_key = list(meta.keys())[0]
        index_meta = meta[first_key]
    else:
        # Flat structure
        index_meta = meta

    for k in ("nlist","max_safe_nlist","requested_nlist","count"):
        assert k in index_meta, f"Missing key '{k}' in index metadata"
    assert index_meta["nlist"] <= index_meta["max_safe_nlist"]

@pytest.mark.skipif("S7_API_URL" not in os.environ, reason="set S7_API_URL to run API checks")
def test_gating_metrics_endpoint():
    import urllib.request
    url = os.environ["S7_API_URL"].rstrip("/") + "/metrics/gating"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert "total" in data and "used_cpesh" in data