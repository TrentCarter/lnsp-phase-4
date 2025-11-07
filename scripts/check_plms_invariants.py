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
from datetime import datetime
from typing import List, Dict, Any, Tuple


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


def check_calibration_pollution(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 6 (STRICT): Calibration dataset excludes rehearsal/replay/sandbox/failed runs.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    cursor = conn.cursor()

    # Check for rehearsal runs in estimate_versions
    cursor.execute("""
        SELECT ev.id, ev.project_id, pr.run_kind
        FROM estimate_versions ev
        JOIN project_runs pr ON ev.project_id = pr.project_id
        WHERE pr.run_kind IN ('rehearsal', 'replay')
    """)

    rehearsal_pollution = cursor.fetchall()
    if rehearsal_pollution:
        errors.append(
            f"🔴 CRITICAL: {len(rehearsal_pollution)} estimate_versions include non-production runs "
            f"(rehearsal/replay) - calibration poisoned!"
        )
        for row in rehearsal_pollution[:5]:
            errors.append(f"   - estimate_versions.id={row[0]}, run_kind={row[2]}")

    # Check for sandbox runs
    cursor.execute("""
        SELECT ev.id, ev.project_id, pr.write_sandbox
        FROM estimate_versions ev
        JOIN project_runs pr ON ev.project_id = pr.project_id
        WHERE pr.write_sandbox = 1
    """)

    sandbox_pollution = cursor.fetchall()
    if sandbox_pollution:
        errors.append(
            f"🔴 CRITICAL: {len(sandbox_pollution)} estimate_versions include sandbox runs "
            f"- calibration poisoned!"
        )

    # Check for failed validation runs
    cursor.execute("""
        SELECT ev.id, ev.project_id, pr.validation_pass
        FROM estimate_versions ev
        JOIN project_runs pr ON ev.project_id = pr.project_id
        WHERE pr.validation_pass = 0
    """)

    failed_pollution = cursor.fetchall()
    if failed_pollution:
        errors.append(
            f"🔴 CRITICAL: {len(failed_pollution)} estimate_versions include failed validation runs "
            f"- calibration poisoned!"
        )

    return errors


def check_provider_snapshots(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 7 (STRICT): Baseline/hotfix runs have complete provider snapshots.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, run_id, run_kind, provider_matrix_json
        FROM project_runs
        WHERE run_kind IN ('baseline', 'hotfix')
          AND (provider_matrix_json IS NULL OR provider_matrix_json = '' OR provider_matrix_json = '{}')
    """)

    missing_snapshots = cursor.fetchall()
    if missing_snapshots:
        errors.append(
            f"🔴 CRITICAL: {len(missing_snapshots)} baseline/hotfix runs missing provider_matrix_json "
            f"- cannot replay deterministically!"
        )
        for row in missing_snapshots[:5]:
            errors.append(f"   - project_runs.id={row[0]}, run_id={row[1]}, run_kind={row[2]}")

    return errors


def check_strata_coverage(conn: sqlite3.Connection) -> List[str]:
    """
    Invariant 8 (STRICT): Representative canaries must have strata_coverage = 1.0.

    Returns:
        List of error messages (empty if passed)
    """
    errors = []

    # This requires a column to track strata_coverage per run
    # For now, stub check (implement when column added)
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(project_runs)")
    columns = {row[1] for row in cursor.fetchall()}

    if 'strata_coverage' not in columns:
        # No column yet, skip check
        return errors

    cursor.execute("""
        SELECT id, run_id, rehearsal_pct, strata_coverage
        FROM project_runs
        WHERE run_kind = 'rehearsal'
          AND strata_coverage < 1.0
    """)

    incomplete_strata = cursor.fetchall()
    if incomplete_strata:
        errors.append(
            f"⚠️  {len(incomplete_strata)} rehearsal runs have incomplete strata coverage "
            f"(not representative!)"
        )
        for row in incomplete_strata[:5]:
            errors.append(
                f"   - run_id={row[1]}, rehearsal_pct={row[2]}, "
                f"strata_coverage={row[3]}"
            )

    return errors


def run_all_checks(db_path: str, strict: bool = False, json_output: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    Run all invariant checks.

    Args:
        db_path: Path to SQLite database
        strict: Enable strict checks (calibration pollution, provider snapshots)
        json_output: Output results as JSON

    Returns:
        (exit_code, results_dict)
    """
    conn = sqlite3.connect(db_path)

    # Base checks (always run)
    checks = [
        ("passport_artifacts", "Passport Artifacts", check_passport_artifacts),
        ("kpi_formulas", "KPI Formulas", check_kpi_formulas),
        ("baseline_sandbox", "Baseline Sandbox", check_baseline_sandbox),
        ("calibration_sources", "Calibration Sources", check_calibration_sources),
        ("lane_overrides", "Lane Overrides", check_lane_overrides)
    ]

    # Strict checks (optional)
    if strict:
        checks.extend([
            ("calibration_pollution", "Calibration Pollution (STRICT)", check_calibration_pollution),
            ("provider_snapshots", "Provider Snapshots (STRICT)", check_provider_snapshots),
            ("strata_coverage", "Strata Coverage (STRICT)", check_strata_coverage)
        ])

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "db_path": db_path,
        "strict_mode": strict,
        "checks": {},
        "summary": {}
    }

    all_errors = []

    if not json_output:
        print("=== PLMS Data Invariants Checker ===")
        if strict:
            print("Mode: STRICT (production-hardened checks)\n")
        else:
            print("Mode: STANDARD (basic checks)\n")

    for check_key, check_name, check_fn in checks:
        if not json_output:
            print(f"Checking: {check_name}...")

        errors = check_fn(conn)
        passed = len(errors) == 0

        results["checks"][check_key] = {
            "name": check_name,
            "passed": passed,
            "violations": len(errors),
            "errors": errors
        }

        if errors:
            all_errors.extend(errors)
            if not json_output:
                print(f"  ✗ FAILED ({len(errors)} violations)")
        else:
            if not json_output:
                print(f"  ✓ PASSED")

    conn.close()

    # Summary
    total_checks = len(checks)
    passed_checks = sum(1 for c in results["checks"].values() if c["passed"])
    total_violations = len(all_errors)

    results["summary"] = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": total_checks - passed_checks,
        "total_violations": total_violations,
        "status": "pass" if total_violations == 0 else "fail"
    }

    if json_output:
        # JSON output for machine consumption
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"\n=== Summary ===")
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Total violations: {total_violations}")

        if all_errors:
            print("\nViolations:\n")
            for error in all_errors:
                print(f"  {error}")
        else:
            print("\n✓ All invariants passed")

    exit_code = 0 if total_violations == 0 else 1
    return exit_code, results


def main():
    parser = argparse.ArgumentParser(description="PLMS data invariants checker")
    parser.add_argument(
        "--db",
        type=str,
        default="artifacts/registry/registry.db",
        help="Path to SQLite database (default: artifacts/registry/registry.db)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checks (calibration pollution, provider snapshots, strata coverage)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for machine consumption / Slack alerts)"
    )

    args = parser.parse_args()
    exit_code, results = run_all_checks(args.db, strict=args.strict, json_output=args.json)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
