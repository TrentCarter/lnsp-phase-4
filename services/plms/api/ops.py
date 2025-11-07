"""
PLMS Ops API Endpoints

Provides operational health checks and status for HMI monitoring.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import glob
import json
import os
import sqlite3
from typing import Dict, Any, List

router = APIRouter(prefix="/api/ops", tags=["ops"])


def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health."""
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url, socket_timeout=2)
        r.ping()
        return {"status": "healthy", "url": redis_url}
    except ImportError:
        return {"status": "unavailable", "error": "redis-py not installed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_db_health(db_path: str) -> Dict[str, Any]:
    """Check SQLite database health."""
    try:
        if not os.path.exists(db_path):
            return {"status": "missing", "path": db_path}

        conn = sqlite3.connect(db_path, timeout=2)
        cursor = conn.cursor()

        # Simple query to verify read access
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        # Check if project_runs table exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='table' AND name='project_runs'
        """)
        has_project_runs = cursor.fetchone()[0] > 0

        conn.close()

        return {
            "status": "healthy",
            "path": db_path,
            "tables": table_count,
            "schema_valid": has_project_runs
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_filesystem_health(project_root: str) -> Dict[str, Any]:
    """Check filesystem permissions for passport writes."""
    try:
        artifacts_dir = os.path.join(project_root, "artifacts", "project-test")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Test write permissions
        test_file = os.path.join(artifacts_dir, "healthcheck.tmp")
        with open(test_file, 'w') as f:
            f.write("healthcheck")

        # Test read permissions
        with open(test_file, 'r') as f:
            content = f.read()

        # Cleanup
        os.remove(test_file)

        if content != "healthcheck":
            raise ValueError("Read/write mismatch")

        return {"status": "healthy", "path": artifacts_dir}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/healthz")
def healthz() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.

    Returns overall system health:
    - Redis connectivity
    - Database accessibility
    - Filesystem permissions

    Example:
        {
            "status": "healthy",
            "components": {
                "redis": {"status": "healthy", ...},
                "database": {"status": "healthy", ...},
                "filesystem": {"status": "healthy", ...}
            },
            "degraded": false
        }
    """
    project_root = os.environ.get("PROJECT_ROOT", os.path.abspath("."))
    db_path = os.environ.get("DB_PATH", os.path.join(project_root, "artifacts", "registry", "registry.db"))

    components = {
        "redis": check_redis_health(),
        "database": check_db_health(db_path),
        "filesystem": check_filesystem_health(project_root)
    }

    # Overall status
    unhealthy = [k for k, v in components.items() if v.get("status") != "healthy"]
    overall_status = "unhealthy" if unhealthy else "healthy"
    degraded = len(unhealthy) > 0

    return {
        "status": overall_status,
        "components": components,
        "degraded": degraded,
        "unhealthy_components": unhealthy
    }


@router.get("/invariants/latest")
def latest_invariants() -> Dict[str, Any]:
    """
    Get latest invariants check result.

    Returns:
        - status: "ok" | "fail" | "unknown"
        - log_path: Path to full log file
        - tail: Last 200 lines of log
        - timestamp: When check ran
        - violations: List of violations (if available)

    Example:
        {
            "status": "ok",
            "log_path": "/logs/plms/invariants_2025-11-06T07:00:03Z.log",
            "tail": "...",
            "timestamp": "2025-11-06T07:00:03Z",
            "violations": []
        }
    """
    project_root = os.environ.get("PROJECT_ROOT", os.path.abspath("."))
    log_dir = os.path.join(project_root, "logs", "plms")

    # Find latest log file
    log_pattern = os.path.join(log_dir, "invariants_*.log")
    log_files = sorted(glob.glob(log_pattern), reverse=True)

    if not log_files:
        return {
            "status": "unknown",
            "error": "No invariants check logs found",
            "log_dir": log_dir
        }

    latest_log = log_files[0]

    try:
        with open(latest_log, 'r') as f:
            content = f.read()

        # Extract last 200 lines
        lines = content.splitlines()
        tail = "\n".join(lines[-200:])

        # Determine status from content
        if "✓ All invariants passed" in content or "Total violations: 0" in content:
            status = "ok"
        elif "⚠️" in content or "🔴" in content or "Total violations:" in content:
            status = "fail"
        else:
            status = "unknown"

        # Extract timestamp from filename
        filename = os.path.basename(latest_log)
        timestamp = filename.replace("invariants_", "").replace(".log", "")

        # Try to parse violations (if any)
        violations = []
        if status == "fail":
            for line in lines:
                if line.strip().startswith("⚠️") or line.strip().startswith("🔴"):
                    violations.append(line.strip())

        return {
            "status": status,
            "log_path": latest_log,
            "tail": tail,
            "timestamp": timestamp,
            "violations": violations[:20]  # Limit to first 20
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "log_path": latest_log
        }


@router.get("/invariants/history")
def invariants_history(limit: int = 30) -> Dict[str, Any]:
    """
    Get invariants check history (for Grafana/HMI charts).

    Query Params:
        limit: Number of recent checks to return (default 30)

    Returns:
        List of recent check results with status and timestamp
    """
    project_root = os.environ.get("PROJECT_ROOT", os.path.abspath("."))
    log_dir = os.path.join(project_root, "logs", "plms")

    log_pattern = os.path.join(log_dir, "invariants_*.log")
    log_files = sorted(glob.glob(log_pattern), reverse=True)[:limit]

    history = []

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Extract filename timestamp
            filename = os.path.basename(log_file)
            timestamp = filename.replace("invariants_", "").replace(".log", "")

            # Determine status
            if "✓ All invariants passed" in content or "Total violations: 0" in content:
                status = "ok"
                violations = 0
            elif "Total violations:" in content:
                # Try to parse violation count
                for line in content.splitlines():
                    if "Total violations:" in line:
                        try:
                            violations = int(line.split(":")[-1].strip())
                        except ValueError:
                            violations = -1
                        break
                status = "fail"
            else:
                status = "unknown"
                violations = -1

            history.append({
                "timestamp": timestamp,
                "status": status,
                "violations": violations,
                "log_path": log_file
            })

        except Exception as e:
            # Skip failed reads
            continue

    return {
        "history": history,
        "count": len(history)
    }
