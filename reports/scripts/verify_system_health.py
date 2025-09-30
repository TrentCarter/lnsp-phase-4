#!/usr/bin/env python3
"""
LNSP System Health Verification Script

Verifies system health regardless of data size.
Adapts checks to current database state instead of hardcoded baseline.

Usage:
    python reports/scripts/verify_system_health.py
    python reports/scripts/verify_system_health.py --min-cpesh-pct 90  # Require 90% CPESH coverage
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


def check_postgres(min_cpesh_pct: float = 90.0):
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
            "expected": ">0",
            "actual": f"{count:,}",
            "pass": count > 0
        }
        total_entries = count
    else:
        checks["cpe_entries"] = {"expected": ">0", "actual": "ERROR", "pass": False}
        total_entries = 0

    # Vectors count (should match entries)
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*) FROM cpe_vectors;"
    ])
    if success:
        vector_count = int(output.strip())
        checks["vectors"] = {
            "expected": f"~{total_entries:,}",
            "actual": f"{vector_count:,}",
            "pass": abs(vector_count - total_entries) <= max(10, total_entries * 0.01)  # Within 1% or 10
        }
    else:
        checks["vectors"] = {"expected": f"~{total_entries:,}", "actual": "ERROR", "pass": False}

    # CPESH quality
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*) FROM cpe_entry WHERE jsonb_array_length(soft_negatives) > 0;"
    ])
    if success:
        cpesh_count = int(output.strip())
        cpesh_pct = (cpesh_count / total_entries * 100) if total_entries > 0 else 0
        checks["cpesh_coverage"] = {
            "expected": f"≥{min_cpesh_pct}%",
            "actual": f"{cpesh_count:,}/{total_entries:,} ({cpesh_pct:.1f}%)",
            "pass": cpesh_pct >= min_cpesh_pct
        }
    else:
        checks["cpesh_coverage"] = {"expected": f"≥{min_cpesh_pct}%", "actual": "ERROR", "pass": False}

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

    # Check for duplicates
    success, output = run_command([
        "psql", "lnsp", "-t", "-c",
        "SELECT COUNT(*), COUNT(DISTINCT chunk_position->>'doc_id') FROM cpe_entry;"
    ])
    if success:
        parts = output.strip().split('|')
        total = int(parts[0].strip())
        unique = int(parts[1].strip())
        checks["no_duplicates"] = {
            "expected": "total = unique",
            "actual": f"total={total:,}, unique={unique:,}",
            "pass": total == unique
        }
    else:
        checks["no_duplicates"] = {"expected": "total = unique", "actual": "ERROR", "pass": False}

    return checks, total_entries


def check_neo4j(expected_concepts: int):
    """Verify Neo4j graph state."""
    checks = {}

    # Concept nodes (should roughly match CPE entries)
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH (c:Concept) RETURN count(c);"
    ])
    if success:
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        variance = max(50, expected_concepts * 0.05)  # 5% or 50
        checks["concept_nodes"] = {
            "expected": f"~{expected_concepts:,}",
            "actual": f"{count:,}",
            "pass": abs(count - expected_concepts) <= variance
        }
    else:
        checks["concept_nodes"] = {"expected": f"~{expected_concepts:,}", "actual": "ERROR", "pass": False}

    # Entity nodes (should be > concepts)
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH (e:Entity) RETURN count(e);"
    ])
    if success:
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        checks["entity_nodes"] = {
            "expected": f">{expected_concepts:,}",
            "actual": f"{count:,}",
            "pass": count >= expected_concepts
        }
    else:
        checks["entity_nodes"] = {"expected": f">{expected_concepts:,}", "actual": "ERROR", "pass": False}

    # Relationships (should be > concepts)
    success, output = run_command([
        "cypher-shell", "-u", "neo4j", "-p", "password",
        "MATCH ()-[r]->() RETURN count(r);"
    ])
    if success:
        lines = output.strip().split('\n')
        count = int(lines[-1].strip()) if lines else 0
        checks["relationships"] = {
            "expected": f">{expected_concepts:,}",
            "actual": f"{count:,}",
            "pass": count >= expected_concepts
        }
    else:
        checks["relationships"] = {"expected": f">{expected_concepts:,}", "actual": "ERROR", "pass": False}

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


def check_artifacts(expected_vectors: int):
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

            # Check vector count (should be close to expected)
            vector_count = data["fused"].shape[0]
            variance = max(50, expected_vectors * 0.1)  # 10% or 50
            checks["faiss_vector_count"] = {
                "expected": f"~{expected_vectors:,}",
                "actual": f"{vector_count:,}",
                "pass": abs(vector_count - expected_vectors) <= variance
            }

            # Check vector dimensions
            vector_dim = data["fused"].shape[1]
            checks["vector_dimensions"] = {
                "expected": "784D",
                "actual": f"{vector_dim}D",
                "pass": vector_dim == 784
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
    print("LNSP System Health Verification")
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
        print("\n✅ System is healthy!\n")
        return 0
    else:
        print("\n⚠️  Some checks failed\n")
        if strict:
            print("Exiting with error code (strict mode)")
            return 1
        else:
            print("Run with --strict to exit with error on failure")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify LNSP system health"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any check fails"
    )
    parser.add_argument(
        "--min-cpesh-pct",
        type=float,
        default=90.0,
        help="Minimum CPESH coverage percentage (default: 90.0)"
    )

    args = parser.parse_args()

    print("🔍 Verifying LNSP system health...")

    # Check services first
    services = check_services()
    if not all(c["pass"] for c in services.values()):
        print("\n❌ Services not running. Please start all services first.\n")
        return 1

    # Get database state
    postgres_checks, total_entries = check_postgres(args.min_cpesh_pct)

    all_checks = {
        "services": services,
        "postgresql": postgres_checks,
        "neo4j": check_neo4j(total_entries),
        "artifacts": check_artifacts(total_entries),
    }

    return print_results(all_checks, args.strict)


if __name__ == "__main__":
    sys.exit(main())