"""
PLMS KPI Receipts Emitter

Collects task KPI measurements and emits structured receipts.
Called by PAS task template post-execution.

Usage:
    python -m services.plms.kpi_emit \
        --task-id 1287 \
        --lane 4202 \
        --artifacts-dir artifacts/t1287
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from services.plms.kpi_validators import (
    KPI_VALIDATORS,
    validate_kpi,
    code_tests,
    linter_pass,
    schema_diff,
    row_count_delta,
    graph_edge_count_delta,
    bleu_score,
    readability
)


# Lane-specific KPI formulas (what to check per lane)
LANE_KPI_FORMULAS = {
    4200: [  # Code-API
        {"name": "test_pass_rate", "threshold": 0.90, "operator": ">=", "validator": "code_tests"},
        {"name": "linter_pass", "threshold": True, "operator": "==", "validator": "linter_pass"}
    ],
    4201: [  # Code-Test
        {"name": "test_pass_rate", "threshold": 0.95, "operator": ">=", "validator": "code_tests"},
        {"name": "linter_pass", "threshold": True, "operator": "==", "validator": "linter_pass"}
    ],
    4202: [  # Code-Docs
        {"name": "readability", "threshold": 10.0, "operator": "<=", "validator": "readability"}
    ],
    5100: [  # Data-Schema
        {"name": "schema_diff", "threshold": 0, "operator": "==", "validator": "schema_diff"},
        {"name": "row_count_delta", "threshold": 0.05, "operator": "<=", "validator": "row_count_delta"}
    ],
    5101: [  # Data-Ingest
        {"name": "row_count_delta", "threshold": 0.10, "operator": "<=", "validator": "row_count_delta"}
    ],
    5102: [  # Data-Transform
        {"name": "row_count_delta", "threshold": 0.05, "operator": "<=", "validator": "row_count_delta"}
    ],
    6100: [  # Model-Train
        # Future: training loss convergence, validation accuracy
    ],
    6101: [  # Model-Eval
        # Future: BLEU/ROUGE for generative, accuracy/F1 for classification
    ],
}


def compute_echo_cos(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Compute echo cosine similarity from artifacts.

    Args:
        artifacts_dir: Path to task artifacts directory

    Returns:
        {"cos": float, "threshold": float, "pass": bool}
    """
    echo_file = artifacts_dir / "echo_result.json"
    if not echo_file.exists():
        print(f"Warning: Echo result not found at {echo_file}", file=sys.stderr)
        return {"cos": 0.0, "threshold": 0.82, "pass": False}

    try:
        with open(echo_file) as f:
            echo_data = json.load(f)

        cos_sim = echo_data.get("cosine_similarity", 0.0)
        threshold = echo_data.get("threshold", 0.82)
        passes = cos_sim >= threshold

        return {
            "cos": round(cos_sim, 4),
            "threshold": threshold,
            "pass": passes
        }
    except Exception as e:
        print(f"Error reading echo result: {e}", file=sys.stderr)
        return {"cos": 0.0, "threshold": 0.82, "pass": False}


def run_kpi_checks(lane_id: int, artifacts_dir: Path) -> List[Dict[str, Any]]:
    """
    Run all KPI checks for a given lane.

    Args:
        lane_id: Lane ID
        artifacts_dir: Path to task artifacts directory

    Returns:
        List of KPI results with pass/fail status
    """
    formulas = LANE_KPI_FORMULAS.get(lane_id, [])
    if not formulas:
        print(f"Warning: No KPI formulas defined for lane {lane_id}", file=sys.stderr)
        return []

    results = []
    for formula in formulas:
        kpi_name = formula["name"]
        threshold = formula["threshold"]
        operator = formula["operator"]
        validator_name = formula["validator"]

        # Get validator function
        validator_func = KPI_VALIDATORS.get(validator_name)
        if not validator_func:
            print(f"Error: Unknown validator '{validator_name}' for KPI '{kpi_name}'", file=sys.stderr)
            continue

        # Determine validator arguments based on KPI type
        try:
            if kpi_name == "test_pass_rate":
                # Look for pytest results
                pytest_path = artifacts_dir / "tests"
                value = validator_func(str(pytest_path)) if pytest_path.exists() else 0.0
                logs_path = str(artifacts_dir / "pytest.json")

            elif kpi_name == "linter_pass":
                # Look for repo root
                repo_root = artifacts_dir.parent.parent  # artifacts/t1287 -> lnsp-phase-4
                value = validator_func(str(repo_root))
                logs_path = str(artifacts_dir / "ruff.txt")

            elif kpi_name == "readability":
                # Look for generated docs
                docs_file = artifacts_dir / "generated_docs.md"
                if docs_file.exists():
                    with open(docs_file) as f:
                        text = f.read()
                    value = validator_func(text)
                else:
                    value = 12.0  # Default: college-level
                logs_path = str(docs_file)

            elif kpi_name == "schema_diff":
                # Load expected schema from artifacts
                schema_file = artifacts_dir / "expected_schema.json"
                if schema_file.exists():
                    with open(schema_file) as f:
                        expected = json.load(f)
                    table_fq = expected.get("table_fq", "public.unknown")
                    value = validator_func(expected.get("schema", {}), table_fq)
                else:
                    value = 999
                logs_path = str(schema_file)

            elif kpi_name == "row_count_delta":
                # Load expected row count
                count_file = artifacts_dir / "expected_row_count.json"
                if count_file.exists():
                    with open(count_file) as f:
                        expected = json.load(f)
                    table_fq = expected.get("table_fq", "public.unknown")
                    expected_rows = expected.get("row_count", 0)
                    value = validator_func(expected_rows, table_fq)
                else:
                    value = 1.0
                logs_path = str(count_file)

            else:
                # Generic validator (no args)
                value = 0.0
                logs_path = str(artifacts_dir / f"{kpi_name}.log")

        except Exception as e:
            print(f"Error running validator '{validator_name}': {e}", file=sys.stderr)
            value = 0.0
            logs_path = str(artifacts_dir / f"{kpi_name}.log")

        # Check if KPI passes
        passes = validate_kpi(kpi_name, threshold, operator, value)

        results.append({
            "name": kpi_name,
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "pass": passes,
            "logs_path": logs_path
        })

    return results


def emit_receipt(task_id: int, lane_id: int, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Generate KPI receipt for a task.

    Args:
        task_id: Task ID
        lane_id: Lane ID
        artifacts_dir: Path to task artifacts directory

    Returns:
        Receipt dict (ready for JSON serialization)
    """
    kpi_results = run_kpi_checks(lane_id, artifacts_dir)
    echo_result = compute_echo_cos(artifacts_dir)

    receipt = {
        "task_id": task_id,
        "lane": lane_id,
        "kpis": kpi_results,
        "echo": echo_result,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return receipt


def persist_receipt(receipt: Dict[str, Any], output_path: Path = None):
    """
    Save receipt to disk and optionally insert into database.

    Args:
        receipt: Receipt dict
        output_path: Path to save JSON file (optional)
    """
    # Save to disk
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"✓ Receipt saved to {output_path}")

    # TODO: Insert into kpi_receipts table (wire to actual DB)
    # Example:
    # conn = get_db_connection()
    # cursor = conn.cursor()
    # cursor.execute("""
    #     INSERT INTO kpi_receipts (task_id, lane_id, kpi_results, echo_result, created_at)
    #     VALUES (%s, %s, %s, %s, NOW())
    # """, (receipt["task_id"], receipt["lane"], json.dumps(receipt["kpis"]), json.dumps(receipt["echo"])))
    # conn.commit()
    print(f"✓ Receipt recorded for task {receipt['task_id']}")


def main():
    """CLI entrypoint for PAS task template."""
    parser = argparse.ArgumentParser(description="Emit KPI receipt for completed task")
    parser.add_argument("--task-id", type=int, required=True, help="Task ID")
    parser.add_argument("--lane", type=int, required=True, help="Lane ID")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Path to artifacts directory")
    parser.add_argument("--output", type=str, help="Output path for receipt JSON (optional)")

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate receipt
    receipt = emit_receipt(args.task_id, args.lane, artifacts_dir)

    # Persist receipt
    output_path = Path(args.output) if args.output else artifacts_dir / "kpi_receipt.json"
    persist_receipt(receipt, output_path)

    # Print summary
    total_kpis = len(receipt["kpis"])
    passed_kpis = sum(1 for kpi in receipt["kpis"] if kpi["pass"])
    echo_pass = receipt["echo"]["pass"]

    print(f"\n=== KPI Summary ===")
    print(f"Task ID: {receipt['task_id']}")
    print(f"Lane: {receipt['lane']}")
    print(f"KPIs: {passed_kpis}/{total_kpis} passed")
    print(f"Echo: {'✓' if echo_pass else '✗'} (cos={receipt['echo']['cos']:.4f})")

    # Exit with error code if any KPI failed
    all_pass = (passed_kpis == total_kpis) and echo_pass
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
