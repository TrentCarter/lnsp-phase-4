#!/usr/bin/env python3
"""
PAS Resource Manager — Port 6104
Manages system resource allocation: CPU, memory, GPU, ports.
Handles reservation requests, quota enforcement, and cleanup.
Part of Phase 1: Management Agents
"""
import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import contextmanager

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = Path("artifacts/resource_manager/resources.db")

# System capacity (adjust based on your machine)
# These are example values - will be auto-detected in production
DEFAULT_QUOTAS = {
    "cpu": 10.0,         # Total CPU cores available
    "mem_mb": 32768,     # Total memory in MB (32 GB)
    "gpu": 1,            # Total GPUs available
    "gpu_mem_mb": 8192,  # Total GPU memory in MB (8 GB)
    "ports": []          # Ports are checked dynamically
}

# ============================================================================
# Pydantic Models
# ============================================================================

class ResourceRequest(BaseModel):
    """Resource reservation request (resource_request.schema.json)"""
    job_id: str
    agent: str
    cpu: Optional[float] = None
    mem_mb: Optional[int] = None
    gpu: Optional[int] = None
    gpu_mem_mb: Optional[int] = None
    ports: Optional[List[int]] = None


class ResourceRelease(BaseModel):
    """Resource release request"""
    reservation_id: Optional[str] = None
    job_id: Optional[str] = None


class ResourceKill(BaseModel):
    """Force-kill a job and release resources"""
    job_id: str
    reason: str = "manual_kill"


class QuotaUpdate(BaseModel):
    """Update system quotas"""
    resource_type: str = Field(..., pattern="^(cpu|mem_mb|gpu|gpu_mem_mb)$")
    new_capacity: float


# ============================================================================
# Database Management
# ============================================================================

def init_db():
    """Initialize SQLite database with schema"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Quotas table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quotas (
            resource_type TEXT PRIMARY KEY,
            total_capacity REAL NOT NULL,
            allocated REAL DEFAULT 0,
            reserved REAL DEFAULT 0
        )
    """)

    # Reservations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            reservation_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            cpu REAL,
            mem_mb INTEGER,
            gpu INTEGER,
            gpu_mem_mb INTEGER,
            ports TEXT,  -- JSON array
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'released', 'expired', 'killed')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            released_at TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reservations_job_id ON reservations(job_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reservations_status ON reservations(status)")

    # Initialize default quotas if empty
    cursor.execute("SELECT COUNT(*) FROM quotas")
    if cursor.fetchone()[0] == 0:
        for resource_type, capacity in DEFAULT_QUOTAS.items():
            if resource_type != "ports":  # Ports handled separately
                cursor.execute("""
                    INSERT INTO quotas (resource_type, total_capacity, allocated, reserved)
                    VALUES (?, ?, 0, 0)
                """, (resource_type, capacity))

    conn.commit()
    conn.close()

    print(f"✓ Resource Manager database initialized at {DB_PATH}")


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PAS Resource Manager",
    description="System resource allocation and quota management",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Service information (root endpoint)"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get quota summary
        cursor.execute("SELECT resource_type, total_capacity, allocated, reserved FROM quotas")
        quotas = cursor.fetchall()

        quota_summary = {}
        for row in quotas:
            quota_summary[row["resource_type"]] = {
                "total": row["total_capacity"],
                "allocated": row["allocated"],
                "reserved": row["reserved"],
                "available": row["total_capacity"] - row["allocated"] - row["reserved"]
            }

        # Count active reservations
        cursor.execute("SELECT COUNT(*) as count FROM reservations WHERE status = 'active'")
        active_reservations = cursor.fetchone()["count"]

        return {
            "service": "PAS Resource Manager",
            "version": "1.0.0",
            "port": 6104,
            "status": "running",
            "quotas": quota_summary,
            "active_reservations": active_reservations,
            "endpoints": {
                "reserve": "POST /reserve",
                "release": "POST /release",
                "kill": "POST /kill",
                "quotas": "GET /quotas",
                "reservations": "GET /reservations",
                "update_quota": "POST /quotas/update"
            },
            "docs": "/docs"
        }


@app.post("/reserve")
async def reserve_resources(request: ResourceRequest):
    """
    Reserve system resources for a job

    Args:
        request: ResourceRequest with job_id, agent, and resource needs

    Returns:
        - reservation_id: UUID
        - granted: Dict of granted resources
        - status: 'granted' or 'denied'

    Raises:
        HTTPException: If quota exceeded or invalid request
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if enough resources available
        resources_to_check = {}
        if request.cpu is not None:
            resources_to_check["cpu"] = request.cpu
        if request.mem_mb is not None:
            resources_to_check["mem_mb"] = request.mem_mb
        if request.gpu is not None:
            resources_to_check["gpu"] = request.gpu
        if request.gpu_mem_mb is not None:
            resources_to_check["gpu_mem_mb"] = request.gpu_mem_mb

        # Validate against quotas
        for resource_type, requested_amount in resources_to_check.items():
            cursor.execute("""
                SELECT total_capacity, allocated, reserved
                FROM quotas
                WHERE resource_type = ?
            """, (resource_type,))

            row = cursor.fetchone()
            if not row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown resource type: {resource_type}"
                )

            total = row["total_capacity"]
            allocated = row["allocated"]
            reserved = row["reserved"]
            available = total - allocated - reserved

            if requested_amount > available:
                return {
                    "reservation_id": None,
                    "status": "denied",
                    "reason": f"Insufficient {resource_type}: requested {requested_amount}, available {available}",
                    "quotas": {
                        "total": total,
                        "allocated": allocated,
                        "reserved": reserved,
                        "available": available
                    }
                }

        # Grant reservation
        reservation_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO reservations (
                reservation_id, job_id, agent, cpu, mem_mb, gpu, gpu_mem_mb, ports, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """, (
            reservation_id, request.job_id, request.agent,
            request.cpu, request.mem_mb, request.gpu, request.gpu_mem_mb,
            json.dumps(request.ports) if request.ports else None
        ))

        # Update quotas (move from available to allocated)
        for resource_type, amount in resources_to_check.items():
            cursor.execute("""
                UPDATE quotas
                SET allocated = allocated + ?
                WHERE resource_type = ?
            """, (amount, resource_type))

        conn.commit()

        return {
            "reservation_id": reservation_id,
            "status": "granted",
            "granted": {
                "cpu": request.cpu,
                "mem_mb": request.mem_mb,
                "gpu": request.gpu,
                "gpu_mem_mb": request.gpu_mem_mb,
                "ports": request.ports
            },
            "ts": datetime.utcnow().isoformat()
        }


@app.post("/release")
async def release_resources(release: ResourceRelease):
    """
    Release resources from a reservation

    Args:
        release: ResourceRelease with reservation_id or job_id

    Returns:
        - success: bool
        - released: Dict of released resources
        - ts: Timestamp
    """
    if not release.reservation_id and not release.job_id:
        raise HTTPException(
            status_code=400,
            detail="Must provide either reservation_id or job_id"
        )

    with get_db() as conn:
        cursor = conn.cursor()

        # Find reservation
        if release.reservation_id:
            cursor.execute("""
                SELECT * FROM reservations
                WHERE reservation_id = ? AND status = 'active'
            """, (release.reservation_id,))
        else:
            cursor.execute("""
                SELECT * FROM reservations
                WHERE job_id = ? AND status = 'active'
            """, (release.job_id,))

        reservation = cursor.fetchone()

        if not reservation:
            raise HTTPException(
                status_code=404,
                detail="Active reservation not found"
            )

        # Release resources back to quota
        resources_to_release = {
            "cpu": reservation["cpu"],
            "mem_mb": reservation["mem_mb"],
            "gpu": reservation["gpu"],
            "gpu_mem_mb": reservation["gpu_mem_mb"]
        }

        for resource_type, amount in resources_to_release.items():
            if amount is not None:
                cursor.execute("""
                    UPDATE quotas
                    SET allocated = allocated - ?
                    WHERE resource_type = ?
                """, (amount, resource_type))

        # Mark reservation as released
        cursor.execute("""
            UPDATE reservations
            SET status = 'released', released_at = ?
            WHERE reservation_id = ?
        """, (datetime.utcnow().isoformat(), reservation["reservation_id"]))

        conn.commit()

        return {
            "success": True,
            "released": {k: v for k, v in resources_to_release.items() if v is not None},
            "ts": datetime.utcnow().isoformat()
        }


@app.post("/kill")
async def kill_job(kill: ResourceKill):
    """
    Force-kill a job and release its resources

    Args:
        kill: ResourceKill with job_id and reason

    Returns:
        - success: bool
        - killed_reservations: Number of reservations killed
        - ts: Timestamp
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Find all active reservations for this job
        cursor.execute("""
            SELECT * FROM reservations
            WHERE job_id = ? AND status = 'active'
        """, (kill.job_id,))

        reservations = cursor.fetchall()

        if not reservations:
            raise HTTPException(
                status_code=404,
                detail=f"No active reservations found for job {kill.job_id}"
            )

        # Release all resources
        for reservation in reservations:
            resources = {
                "cpu": reservation["cpu"],
                "mem_mb": reservation["mem_mb"],
                "gpu": reservation["gpu"],
                "gpu_mem_mb": reservation["gpu_mem_mb"]
            }

            for resource_type, amount in resources.items():
                if amount is not None:
                    cursor.execute("""
                        UPDATE quotas
                        SET allocated = allocated - ?
                        WHERE resource_type = ?
                    """, (amount, resource_type))

            # Mark as killed
            cursor.execute("""
                UPDATE reservations
                SET status = 'killed', released_at = ?
                WHERE reservation_id = ?
            """, (datetime.utcnow().isoformat(), reservation["reservation_id"]))

        conn.commit()

        print(f"⚠️  Killed job {kill.job_id}: {kill.reason}")

        return {
            "success": True,
            "killed_reservations": len(reservations),
            "reason": kill.reason,
            "ts": datetime.utcnow().isoformat()
        }


@app.get("/quotas")
async def get_quotas():
    """
    Get current system quotas and usage

    Returns:
        - quotas: Dict of resource quotas with usage stats
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM quotas")
        rows = cursor.fetchall()

        quotas = {}
        for row in rows:
            quotas[row["resource_type"]] = {
                "total_capacity": row["total_capacity"],
                "allocated": row["allocated"],
                "reserved": row["reserved"],
                "available": row["total_capacity"] - row["allocated"] - row["reserved"],
                "utilization": (row["allocated"] + row["reserved"]) / row["total_capacity"] if row["total_capacity"] > 0 else 0
            }

        return {"quotas": quotas}


@app.get("/reservations")
async def get_reservations(status: Optional[str] = None):
    """
    Get all reservations, optionally filtered by status

    Args:
        status: Filter by status (active, released, expired, killed)

    Returns:
        - reservations: List of reservations
    """
    with get_db() as conn:
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM reservations
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status,))
        else:
            cursor.execute("SELECT * FROM reservations ORDER BY created_at DESC")

        rows = cursor.fetchall()

        reservations = []
        for row in rows:
            reservations.append({
                "reservation_id": row["reservation_id"],
                "job_id": row["job_id"],
                "agent": row["agent"],
                "resources": {
                    "cpu": row["cpu"],
                    "mem_mb": row["mem_mb"],
                    "gpu": row["gpu"],
                    "gpu_mem_mb": row["gpu_mem_mb"],
                    "ports": json.loads(row["ports"]) if row["ports"] else None
                },
                "status": row["status"],
                "created_at": row["created_at"],
                "released_at": row["released_at"]
            })

        return {"reservations": reservations}


@app.post("/quotas/update")
async def update_quota(update: QuotaUpdate):
    """
    Update system quota capacity

    Args:
        update: QuotaUpdate with resource_type and new_capacity

    Returns:
        - success: bool
        - resource_type: Type of resource updated
        - old_capacity: Previous capacity
        - new_capacity: New capacity
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT total_capacity FROM quotas
            WHERE resource_type = ?
        """, (update.resource_type,))

        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Resource type {update.resource_type} not found"
            )

        old_capacity = row["total_capacity"]

        cursor.execute("""
            UPDATE quotas
            SET total_capacity = ?
            WHERE resource_type = ?
        """, (update.new_capacity, update.resource_type))

        conn.commit()

        return {
            "success": True,
            "resource_type": update.resource_type,
            "old_capacity": old_capacity,
            "new_capacity": update.new_capacity,
            "ts": datetime.utcnow().isoformat()
        }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6104)
