
import os, json, time, pathlib
import pytest

from starlette.testclient import TestClient

RUN_ID = "pytest-run-001"

def test_health_and_invoke_creates_receipt(rpc_app):
    client = TestClient(rpc_app)

    # /health
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"

    # /invoke (stub payload) - must include 'message' field for execute_aider
    payload = {
        "target": {"name":"Aider-LCO","type":"agent","role":"execution"},
        "payload": {
            "message": "Update documentation",  # Required by execute_aider
            "files": ["docs/README.md"],
            "dry_run": True
        },
        "policy": {"timeout_s": 30, "require_caps": ["git-edit"]},
        "run_id": RUN_ID
    }
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200, r.text
    resp = r.json()
    assert "routing_receipt" in resp
    receipt = resp["routing_receipt"]
    assert receipt["run_id"] == RUN_ID

    # Verify receipt structure (in-memory response)
    for key in ("run_id","resolved","timings_ms","status","ts"):
        assert key in receipt, f"Missing key '{key}' in routing_receipt"

    # TODO: Receipt file persistence
    # Currently the server returns receipts in-memory but doesn't persist to disk.
    # When background task persistence is added, uncomment this:
    #
    # _repo_root = pathlib.Path(__file__).parent.parent.parent
    # artifacts = pathlib.Path(os.getenv("PAS_COST_DIR", str(_repo_root / "artifacts" / "costs"))).resolve()
    # receipt_path = artifacts / f"{RUN_ID}.json"
    # for _ in range(10):
    #     if receipt_path.exists():
    #         break
    #     time.sleep(0.1)
    # assert receipt_path.exists(), f"Missing receipt at {receipt_path}"
