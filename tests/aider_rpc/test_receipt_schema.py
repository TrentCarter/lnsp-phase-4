"""
Test that actual receipts/heartbeats conform to their JSON Schemas.

Prevents schema drift by validating real data against schemas/*.json.
"""
import json
import pathlib
import pytest
from jsonschema import validate, ValidationError


def test_routing_receipt_matches_schema():
    """Validate routing receipt structure against schema."""
    schemas_dir = pathlib.Path(__file__).parent.parent.parent / "schemas"
    schema_path = schemas_dir / "routing_receipt.schema.json"

    # Skip if schema doesn't exist
    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text())

    # Create a minimal valid receipt for testing
    receipt = {
        "run_id": "test-001",
        "resolved": {
            "service_id": "aider-lco-001",
            "name": "Aider-LCO",
            "role": "execution",
            "url": "http://localhost:6150"
        },
        "timings_ms": {
            "total": 1234.56
        },
        "status": "success",
        "cost_estimate": {
            "usd": 0.0042
        },
        "ts": "2025-11-07T12:34:56Z"
    }

    # Validate against schema
    try:
        validate(instance=receipt, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Receipt does not match schema: {e.message}")


def test_persisted_receipt_matches_schema():
    """Validate actual persisted receipt files against schema."""
    schemas_dir = pathlib.Path(__file__).parent.parent.parent / "schemas"
    schema_path = schemas_dir / "routing_receipt.schema.json"

    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text())

    # Find actual receipt files
    artifacts_dir = pathlib.Path(__file__).parent.parent.parent / "artifacts" / "costs"
    receipt_files = list(artifacts_dir.glob("*.json"))

    if not receipt_files:
        pytest.skip("No receipt files found in artifacts/costs/")

    # Validate each receipt
    for receipt_path in receipt_files:
        receipt = json.loads(receipt_path.read_text())
        try:
            validate(instance=receipt, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Receipt {receipt_path.name} invalid: {e.message}")


def test_heartbeat_matches_schema():
    """Validate heartbeat payload against schema."""
    schemas_dir = pathlib.Path(__file__).parent.parent.parent / "schemas"
    schema_path = schemas_dir / "heartbeat.schema.json"

    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text())

    # Create a minimal valid heartbeat (matches actual schema requirements)
    heartbeat = {
        "run_id": "task-001",
        "agent": "Aider-LCO",
        "ts": "2025-11-07T12:34:56Z",
        "progress": 0.5,  # 0.0-1.0
        "status": "running",  # queued|running|blocked|waiting_approval|paused|error|done
        "message": "Processing file edits",
        "token_usage": {
            "ctx_used": 1234,
            "ctx_limit": 200000
        },
        "resources": {
            "cpu": 45.6,
            "mem_mb": 512.0
        }
    }

    # Validate against schema
    try:
        validate(instance=heartbeat, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Heartbeat does not match schema: {e.message}")


def test_gateway_invoke_request_matches_schema():
    """Validate invoke request payload against schema."""
    schemas_dir = pathlib.Path(__file__).parent.parent.parent / "schemas"
    schema_path = schemas_dir / "gateway_invoke_request.schema.json"

    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text())

    # Create a minimal valid invoke request
    request = {
        "target": {
            "name": "Aider-LCO",
            "type": "agent",
            "role": "execution"
        },
        "payload": {
            "message": "Add docstrings",
            "files": ["src/example.py"]
        },
        "policy": {
            "timeout_s": 120,
            "prefer": "role"
        },
        "run_id": "test-001"
    }

    # Validate against schema
    try:
        validate(instance=request, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Invoke request does not match schema: {e.message}")
