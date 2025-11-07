"""
PLMS Portfolio API Endpoints

Exposes portfolio-level metrics: queue status, lane concurrency, fair-share allocation.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from services.plms.portfolio_scheduler import get_scheduler, LANE_CONCURRENCY_LIMITS

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def get_current_user():
    """
    Stub: Extract user from JWT/session.
    Replace with actual auth middleware.
    """
    return {"username": "user@example.com", "scopes": ["portfolio.view"]}


def rbac(required_scope: str):
    """
    RBAC dependency: check if user has required scope.
    """
    def _rbac(user=Depends(get_current_user)):
        if required_scope not in user.get("scopes", []):
            raise HTTPException(status_code=403, detail=f"Missing scope: {required_scope}")
        return user
    return _rbac


@router.get("/status")
def portfolio_status(user=Depends(rbac("portfolio.view"))) -> Dict[str, Any]:
    """
    Get current portfolio scheduler status.

    Returns:
        - queue: Pending tasks per priority level
        - running_by_lane: Currently executing tasks per lane with utilization
        - caps: Lane concurrency limits
        - fairshare: Fair-share allocation metrics (future enhancement)

    Example:
        {
            "queue": {"priority_1": 5, "priority_2": 3},
            "running_by_lane": {
                "4200": {"running": 3, "limit": 4, "utilization": 0.75}
            },
            "total_queued": 8,
            "total_running": 3,
            "timestamp": "2025-11-06T12:34:56Z"
        }
    """
    scheduler = get_scheduler()
    status = scheduler.get_status()

    # Add lane limit reference for HMI
    status["lane_limits"] = LANE_CONCURRENCY_LIMITS

    # Future: Add fair-share allocation metrics
    # For now, return basic queue/running status
    return status


@router.get("/lanes")
def lane_capacities(user=Depends(rbac("portfolio.view"))) -> Dict[str, Any]:
    """
    Get configured lane concurrency limits.

    Returns:
        Map of lane_id -> max_concurrent_tasks
    """
    return {
        "lanes": LANE_CONCURRENCY_LIMITS,
        "description": {
            4200: "Code-API (lightweight, can parallelize)",
            4201: "Code-Test",
            4202: "Code-Docs",
            5100: "Data-Schema (expensive, heavy DB ops)",
            5101: "Data-Ingest",
            5102: "Data-Transform",
            6100: "Model-Train (GPU-bound)",
            6101: "Model-Eval"
        }
    }
