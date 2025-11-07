
import json
import pathlib

def test_required_schemas_exist():
    # Use project-relative path (macOS compatible)
    _repo_root = pathlib.Path(__file__).parent.parent.parent
    root = _repo_root / "schemas"
    expected = {
        "gateway_invoke_request.schema.json",
        "heartbeat.schema.json",
        "job_card.schema.json",
        "manifest.schema.json",
        "resource_request.schema.json",
        "routing_receipt.schema.json",
        "service_discovery_query.schema.json",
        "service_heartbeat.schema.json",
        "service_registration.schema.json",
        "status_update.schema.json",
    }
    present = {p.name for p in root.glob("*.json")}
    missing = expected - present
    assert not missing, f"Missing schemas: {missing}"

def test_routing_receipt_min_fields():
    # Use project-relative path (macOS compatible)
    _repo_root = pathlib.Path(__file__).parent.parent.parent
    root = _repo_root / "schemas"
    obj = json.loads((root / "routing_receipt.schema.json").read_text())
    for k in ("run_id","resolved","timings_ms","status","ts"):
        assert k in obj.get("properties", {}), f"Missing property {k}"
