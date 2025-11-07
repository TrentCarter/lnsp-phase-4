"""
PLMS KPI Validators - Lane-Specific Quality Gates
Ship as: services/plms/kpi_validators.py

Validators return numeric/boolean values checked against thresholds in quality_slo_gates.
"""

import json
import subprocess
from typing import Dict, Any


def code_tests(pytest_path: str) -> float:
    """
    Run pytest on artifact, return pass rate in [0,1].

    Args:
        pytest_path: Path to test file/directory

    Returns:
        pass_rate = passed_tests / total_tests
    """
    try:
        result = subprocess.run(
            ["pytest", pytest_path, "--maxfail=1", "--disable-warnings", "--json-report", "--json-report-file=/tmp/pytest_report.json"],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Read JSON report
        with open("/tmp/pytest_report.json", "r") as f:
            report = json.load(f)

        passed = report["summary"].get("passed", 0)
        total = report["summary"].get("total", 1)
        return passed / max(1, total)

    except Exception as e:
        print(f"Error running pytest: {e}")
        return 0.0


def linter_pass(repo_root: str) -> bool:
    """
    Run ruff linter, return True if no errors.

    Args:
        repo_root: Path to repository root

    Returns:
        True if linter passes (0 errors), False otherwise
    """
    try:
        proc = subprocess.run(
            ["ruff", "check", repo_root, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=60
        )

        findings = json.loads(proc.stdout)
        return len(findings) == 0

    except Exception as e:
        print(f"Error running linter: {e}")
        return False


def schema_diff(expected_schema: Dict[str, Any], table_fq: str) -> int:
    """
    Compare expected vs actual table schema, return diff count.

    Args:
        expected_schema: Dict with column names and types
        table_fq: Fully qualified table name (e.g., "public.users")

    Returns:
        Number of differences (0 = exact match)
    """
    try:
        actual_schema = get_table_schema(table_fq)  # Implement DB query
        return len(deep_diff(expected_schema, actual_schema))
    except Exception as e:
        print(f"Error checking schema: {e}")
        return 999  # High diff count indicates failure


def row_count_delta(expected_rows: int, table_fq: str) -> float:
    """
    Compare expected vs actual row count, return fractional delta.

    Args:
        expected_rows: Expected number of rows
        table_fq: Fully qualified table name

    Returns:
        abs(actual - expected) / expected
    """
    try:
        actual = get_row_count(table_fq)  # Implement DB query
        return abs(actual - expected_rows) / max(1, expected_rows)
    except Exception as e:
        print(f"Error checking row count: {e}")
        return 1.0  # 100% error


def graph_edge_count_delta(expected_edges: int, neo4j_query: str) -> float:
    """
    Compare expected vs actual edge count in Neo4j, return fractional delta.

    Args:
        expected_edges: Expected number of edges
        neo4j_query: Cypher query to count edges (e.g., "MATCH ()-[r]->() RETURN count(r)")

    Returns:
        abs(actual - expected) / expected
    """
    try:
        actual = run_cypher_count(neo4j_query)  # Implement Neo4j query
        return abs(actual - expected_edges) / max(1, expected_edges)
    except Exception as e:
        print(f"Error checking graph edges: {e}")
        return 1.0


def bleu_score(generated: str, reference: str) -> float:
    """
    Compute BLEU score (MT evaluation metric) for narrative quality.

    Args:
        generated: Generated text
        reference: Reference (ground truth) text

    Returns:
        BLEU score in [0,1]
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([reference.split()], generated.split()))
    except ImportError:
        print("Warning: nltk not installed, BLEU score unavailable")
        return 0.0
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return 0.0


def readability(text: str) -> float:
    """
    Compute Flesch-Kincaid grade level for readability.

    Args:
        text: Input text

    Returns:
        Grade level (lower = easier to read)
    """
    try:
        import textstat
        return textstat.flesch_kincaid_grade(text)
    except ImportError:
        print("Warning: textstat not installed, readability unavailable")
        return 12.0  # Assume college-level if unavailable
    except Exception as e:
        print(f"Error computing readability: {e}")
        return 12.0


# --- Helper functions (stubs - implement actual DB/graph queries) ---

def get_table_schema(table_fq: str) -> Dict[str, str]:
    """
    Query PostgreSQL for table schema.

    Returns:
        Dict mapping column names to types
    """
    # Stub: implement actual query
    # Example: SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users'
    return {
        "id": "integer",
        "name": "text",
        "created_at": "timestamp"
    }


def get_row_count(table_fq: str) -> int:
    """
    Query PostgreSQL for row count.

    Returns:
        Number of rows
    """
    # Stub: implement actual query
    # Example: SELECT COUNT(*) FROM users
    return 1000


def run_cypher_count(query: str) -> int:
    """
    Run Cypher query against Neo4j, return count.

    Args:
        query: Cypher query (should return a single count)

    Returns:
        Count result
    """
    # Stub: implement actual Neo4j query
    # Example: from neo4j import GraphDatabase; driver.session().run(query)
    return 500


def deep_diff(a: Dict, b: Dict) -> list:
    """
    Compute deep diff between two dictionaries.

    Returns:
        List of differences
    """
    diffs = []
    all_keys = set(a.keys()) | set(b.keys())

    for key in all_keys:
        if key not in a:
            diffs.append(f"Missing in A: {key}")
        elif key not in b:
            diffs.append(f"Missing in B: {key}")
        elif a[key] != b[key]:
            diffs.append(f"Value mismatch for {key}: {a[key]} vs {b[key]}")

    return diffs


# --- KPI Validator Registry ---

KPI_VALIDATORS = {
    "test_pass_rate": code_tests,
    "linter_pass": linter_pass,
    "schema_diff": schema_diff,
    "row_count_delta": row_count_delta,
    "graph_edge_count_delta": graph_edge_count_delta,
    "bleu_score": bleu_score,
    "readability": readability
}


def validate_kpi(kpi_name: str, threshold: Any, operator: str, actual_value: Any) -> bool:
    """
    Check if actual value meets KPI threshold.

    Args:
        kpi_name: Name of KPI (for error messages)
        threshold: Expected value
        operator: Comparison operator (">=", "<=", "==", "<", ">", "!=")
        actual_value: Measured value

    Returns:
        True if KPI passes, False otherwise
    """
    try:
        if operator == ">=":
            return actual_value >= threshold
        elif operator == "<=":
            return actual_value <= threshold
        elif operator == "==":
            return actual_value == threshold
        elif operator == ">":
            return actual_value > threshold
        elif operator == "<":
            return actual_value < threshold
        elif operator == "!=":
            return actual_value != threshold
        else:
            print(f"Unknown operator: {operator}")
            return False
    except Exception as e:
        print(f"Error validating KPI {kpi_name}: {e}")
        return False
