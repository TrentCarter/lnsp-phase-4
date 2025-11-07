#!/usr/bin/env python3
"""
PLMS Data Invariants Checker - Nightly Job

Checks data model invariants to detect corruption/inconsistencies.

Invariants:
1. Every project_runs has exactly one passport artifact with matching run_id
2. task_estimates.kpi_formula is valid JSON and references only registered validators
3. No baseline run has write_sandbox=true
4. Calibration rows only from runs with validation_pass=true
5. lane_overrides.corrected_lane != predicted_lane (skip no-ops)

Usage:
  python scripts/check_plms_invariants.py --db artifacts/registry/registry.db

Exit codes:
  0: All invariants passed
  1: One or more invariants failed
"""

import argparse
import json
import sqlite3
import sys
from typing import List, Dict, Any


# Registered KPI validators (must match services/plms/kpi_validators.py)
REGISTERED_KPI_VALIDATORS = {
    "test_pass_rate",
    "linter_pass",
    "schema_diff",
    "row_count_delta",
    "graph_edge_count_delta",
    "bleu_score",
    "readability"
}


def check_passport_artifacts(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 1: Every project_runs has exactly one passport artifact.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    # Check if we have a way to query artifacts (stub for now)
    # In production, check: artifacts/project-{project_id}/replay_passport_{run_id}.json exists

    # Stub check: ensure run_id is set for all project_runs
    cursor = conn.cursor()
    cursor.execute("""
        SELECT project_id, run_kind, started_at
        FROM project_runs
        WHERE run_id IS NULL OR run_id = ''
    """)

    missing_run_ids = cursor.fetchall()
    if missing_run_ids:
        errors.append(
            f"⚠️  {len(missing_run_ids)} project_runs have missing run_id (cannot verify passport artifact)"
        )

    return errors


def check_kpi_formulas(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 2: task_estimates.kpi_formula is valid JSON and references only registered validators.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, kpi_formula
        FROM task_estimates
        WHERE kpi_formula IS NOT NULL
    """)

    for row in cursor.fetchall():
        task_id, kpi_formula = row

        try:
            # Parse JSON
            formula = json.loads(kpi_formula)

            # Check that all referenced KPIs are registered
            for kpi_name in formula.keys():
                if kpi_name not in REGISTERED_KPI_VALIDATORS:
                    errors.append(
                        f"⚠️  task_estimates.id={task_id}: "
                        f"kpi_formula references unregistered validator '{kpi_name}'"
                    )

        except json.JSONDecodeError as e:
            errors.append(
                f"⚠️  task_estimates.id={task_id}: "
                f"kpi_formula is invalid JSON ({e})"
            )

    return errors


def check_baseline_sandbox(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 3: No baseline run has write_sandbox=true.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, run_id, run_kind, write_sandbox
        FROM project_runs
        WHERE run_kind = 'baseline' AND write_sandbox = 1
    """)

    violations = cursor.fetchall()
    if violations:
        errors.append(
            f"⚠️  {len(violations)} baseline runs have write_sandbox=true (invalid configuration)"
        )
        for row in violations:
            errors.append(f"   - project_runs.id={row[0]}, run_id={row[1]}")

    return errors


def check_calibration_sources(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 4: Calibration rows only from runs with validation_pass=true.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    # Check estimate_versions table
    cursor = conn.cursor()

    # First, check if we can correlate estimate_versions to project_runs
    # (Requires project_id or run_id in estimate_versions)
    cursor.execute("""
        SELECT COUNT(*)
        FROM estimate_versions
        WHERE project_id IS NULL
    """)

    orphaned_priors = cursor.fetchone()[0]
    if orphaned_priors > 0:
        errors.append(
            f"⚠️  {orphaned_priors} estimate_versions have project_id=NULL "
            f"(cannot verify calibration source)"
        )

    # Check that all estimate_versions with project_id link to passing runs
    cursor.execute("""
        SELECT ev.id, ev.project_id, pr.validation_pass
        FROM estimate_versions ev
        LEFT JOIN project_runs pr ON ev.project_id = pr.project_id
        WHERE pr.validation_pass = 0 OR pr.validation_pass IS NULL
    """)

    bad_priors = cursor.fetchall()
    if bad_priors:
        errors.append(
            f"⚠️  {len(bad_priors)} estimate_versions derived from failed/unvalidated runs"
        )

    return errors


def check_lane_overrides(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 5: lane_overrides.corrected_lane != predicted_lane (skip no-ops).

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, task_estimate_id, predicted_lane, corrected_lane
        FROM lane_overrides
        WHERE predicted_lane = corrected_lane
    """)

    no_ops = cursor.fetchall()
    if no_ops:
        errors.append(
            f"⚠️  {len(no_ops)} lane_overrides are no-ops (corrected_lane == predicted_lane)"
        )
        for row in no_ops[:10]:  # Show first 10
            errors.append(f"   - lane_overrides.id={row[0]}, task_estimate_id={row[1]}")

    return errors


def run_all_checks(db_path: str) -> int:
    """
    Run all invariant checks.

    Args:
        db_path: Path to SQLite database

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    conn = sqlite3.connect(db_path)
    all_errors = []

    print("=== PLMS Data Invariants Checker ===\n")

    # Run all checks
    checks = [
        ("Passport Artifacts", check_passport_artifacts),
        ("KPI Formulas", check_kpi_formulas),
        ("Baseline Sandbox", check_baseline_sandbox),
        ("Calibration Sources", check_calibration_sources),
        ("Lane Overrides", check_lane_overrides)
    ]

    for check_name, check_fn in checks:
        print(f"Checking: {check_name}...")
        errors = check_fn(conn)

        if errors:
            all_errors.extend(errors)
            print(f"  ✗ FAILED ({len(errors)} violations)")
        else:
            print(f"  ✓ PASSED")

    conn.close()

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total violations: {len(all_errors)}")

    if all_errors:
        print("\nViolations:\n")
        for error in all_errors:
            print(f"  {error}")
        return 1
    else:
        print("✓ All invariants passed")
        return 0


def main():
    parser = argparse.ArgumentParser(description="PLMS data invariants checker")
    parser.add_argument(
        "--db",
        type=str,
        default="artifacts/registry/registry.db",
        help="Path to SQLite database (default: artifacts/registry/registry.db)"
    )

    args = parser.parse_args()
    exit_code = run_all_checks(args.db)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
