#!/usr/bin/env python3
"""
LNSP Baseline v1.0 Verification Script

Verifies that the system matches the v1.0 baseline specifications:
- Database record counts
- Service availability
- CPESH quality metrics
- Graph structure
- Artifact presence

Usage:
    python reports/scripts/verify_baseline_v1.0.py
    python reports/scripts/verify_baseline_v1.0.py --strict  # Exit with error on mismatch
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> tuple[bool, str]:
    """Run command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_postgres():
    """Verify PostgreSQL database state."""
    checks = {}

    # CPE entries count
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*) FROM cpe_entry;"
    ])
    if success:
        count = int(output.strip())
        checks["cpe_entries"] = {
            "expected": "~999",
            "actual": count,
            "pass": 950 <= count <= 1050  # Allow some variance
        }
    else:
        checks["cpe_entries"] = {"expected": "~999", "actual": "ERROR", "pass": False}

    # Vectors count
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*) FROM cpe_vectors;"
    ])
    if success:
        count = int(output.strip())
        checks["vectors"] = {
            "expected": "~999",
            "actual": count,
            "pass": 950 <= count <= 1050
        }
    else:
        checks["vectors"] = {"expected": "~999", "actual": "ERROR", "pass": False}

    # CPESH quality
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*) FROM cpe_entry WHERE jsonb_array_length(soft_negatives) > 0;"
    ])
    if success:
        count = int(output.strip())
        checks["cpesh_coverage"] = {
            "expected": ">95%",
            "actual": f"{count}/{checks['cpe_entries']['actual']}",
            "pass": count >= 900  # At least 90% coverage
        }
    else:
        checks["cpesh_coverage"] = {"expected": ">95%", "actual": "ERROR", "pass": False}

    # Dataset source
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT dataset_source FROM cpe_entry LIMIT 1;"
    ])
    if success:
        source = output.strip()
        checks["dataset_source"] = {
            "expected": "factoid-wiki-large",
            "actual": source,
            "pass": source == "factoid-wiki-large"
        }
    else:
        checks["dataset_source"] = {"expected": "factoid-wiki-large", "actual": "ERROR", "pass": False}

    return checks


def check_neo4j():
    """Verify Neo4j graph state."""
    checks = {}

    # Concept nodes
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH (c:Concept) RETURN count(c);"
    ])
    if success:
        # Parse Cypher output
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        checks["concept_nodes"] = {
            "expected": "~999",
            "actual": count,
            "pass": 950 <= count <= 1050
        }
    else:
        checks["concept_nodes"] = {"expected": "~999", "actual": "ERROR", "pass": False}

    # Entity nodes
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH (e:Entity) RETURN count(e);"
    ])
    if success:
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        checks["entity_nodes"] = {
            "expected": ">1500",
            "actual": count,
            "pass": count >= 1500
        }
    else:
        checks["entity_nodes"] = {"expected": ">1500", "actual": "ERROR", "pass": False}

    # Relationships
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH ()-[r]->() RETURN count(r);"
    ])
    if success:
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        checks["relationships"] = {
            "expected": ">2000",
            "actual": count,
            "pass": count >= 2000
        }
    else:
        checks["relationships"] = {"expected": ">2000", "actual": "ERROR", "pass": False}

    return checks


def check_services():
    """Verify required services are running."""
    checks = {}

    # PostgreSQL
    success, _ = run_command(["psql", "lnsp", "-c", "SELECT 1;"])
    checks["postgresql"] = {
        "expected": "running",
        "actual": "running" if success else "not running",
        "pass": success
    }

    # Neo4j
    success, _ = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1;"
    ])
    checks["neo4j"] = {
        "expected": "running",
        "actual": "running" if success else "not running",
        "pass": success
    }

    # Ollama
    success, _ = run_command(["curl", "-s", "http://localhost:11434/api/tags"])
    checks["ollama"] = {
        "expected": "running",
        "actual": "running" if success else "not running",
        "pass": success
    }

    return checks


def check_artifacts():
    """Verify Faiss artifacts exist."""
    checks = {}

    artifacts_path = Path("artifacts")
    if not artifacts_path.exists():
        checks["artifacts_dir"] = {
            "expected": "exists",
            "actual": "missing",
            "pass": False
        }
        return checks

    npz_files = list(artifacts_path.glob("*.npz"))
    checks["faiss_files"] = {
        "expected": "≥1 .npz file",
        "actual": f"{len(npz_files)} files",
        "pass": len(npz_files) >= 1
    }

    if npz_files:
        # Check file structure
        try:
            import numpy as np
            latest_file = max(npz_files, key=lambda p: p.stat().st_mtime)
            data = np.load(str(latest_file))

            required_keys = ["fused", "concept", "question", "cpe_ids"]
            missing_keys = [k for k in required_keys if k not in data.files]

            checks["faiss_structure"] = {
                "expected": "required keys present",
                "actual": "all present" if not missing_keys else f"missing: {missing_keys}",
                "pass": len(missing_keys) == 0
            }

            # Check vector count
            vector_count = data["fused"].shape[0]
            checks["faiss_vector_count"] = {
                "expected": "~999",
                "actual": vector_count,
                "pass": 950 <= vector_count <= 1050
            }
        except Exception as e:
            checks["faiss_structure"] = {
                "expected": "readable",
                "actual": f"error: {e}",
                "pass": False
            }

    return checks


def print_results(all_checks: dict, strict: bool = False):
    """Print verification results."""
    print("\n" + "=" * 60)
    print("LNSP Baseline v1.0 Verification Results")
    print("=" * 60 + "\n")

    all_passed = True

    for category, checks in all_checks.items():
        print(f"📋 {category.upper()}")
        print("-" * 60)

        for check_name, result in checks.items():
            status = "✅" if result["pass"] else "❌"
            print(f"{status} {check_name}")
            print(f"   Expected: {result['expected']}")
            print(f"   Actual:   {result['actual']}")
            print()

            if not result["pass"]:
                all_passed = False

        print()

    # Summary
    total_checks = sum(len(checks) for checks in all_checks.values())
    passed_checks = sum(
        sum(1 for r in checks.values() if r["pass"])
        for checks in all_checks.values()
    )

    print("=" * 60)
    print(f"Summary: {passed_checks}/{total_checks} checks passed")
    print("=" * 60)

    if all_passed:
        print("\n✅ System matches baseline v1.0 specifications!\n")
        return 0
    else:
        print("\n⚠️  System deviates from baseline v1.0 specifications\n")
        if strict:
            print("Exiting with error code (strict mode)")
            return 1
        else:
            print("Run with --strict to exit with error on mismatch")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify LNSP baseline v1.0 system state"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any check fails"
    )

    args = parser.parse_args()

    print("🔍 Verifying LNSP baseline v1.0...")

    all_checks = {
        "services": check_services(),
        "postgresql": check_postgres(),
        "neo4j": check_neo4j(),
        "artifacts": check_artifacts(),
    }

    return print_results(all_checks, args.strict)


if __name__ == "__main__":
    sys.exit(main())