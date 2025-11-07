#!/usr/bin/env python3
"""
PAS Service Registry — Port 6121
Handles service registration, discovery, heartbeats, and TTL-based eviction.
Part of Phase 0: Core Infrastructure
"""
import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from contextlib import contextmanager

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = Path("artifacts/registry/registry.db")
DEFAULT_HEARTBEAT_INTERVAL_S = 60
DEFAULT_TTL_S = 90

# ============================================================================
# Pydantic Models (based on JSON schemas)
# ============================================================================

class ServiceRegistration(BaseModel):
    """Service registration request (service_registration.schema.json)"""
    service_id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern="^(model|tool|agent)$")
    role: str = Field(..., pattern="^(production|staging|canary|experimental)$")
    url: str
    caps: List[str] = Field(..., min_items=1)
    labels: Optional[Dict[str, Any]] = None
    ctx_limit: Optional[int] = None
    cost_hint: Optional[Dict[str, float]] = None
    heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S
    ttl_s: int = DEFAULT_TTL_S

    @validator('service_id', pre=True, always=True)
    def set_service_id(cls, v):
        return v or str(uuid.uuid4())


class ServiceHeartbeat(BaseModel):
    """Heartbeat update (heartbeat.schema.json)"""
    service_id: str
    status: str = Field("ok", pattern="^(ok|degraded|down|queued|running|blocked|waiting_approval|paused|error|done)$")
    p95_ms: Optional[float] = None
    queue_depth: Optional[int] = None
    load: Optional[float] = None
    message: Optional[str] = None


class ServicePromotion(BaseModel):
    """Service role promotion/demotion"""
    name: str
    from_role: str = Field(..., pattern="^(production|staging|canary|experimental)$")
    to_role: str = Field(..., pattern="^(production|staging|canary|experimental)$")


class ServiceDeregistration(BaseModel):
    """Service deregistration request"""
    service_id: str


# ============================================================================
# Database Management
# ============================================================================

def init_db():
    """Initialize SQLite database with schema"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS services (
            service_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('model', 'tool', 'agent')),
            role TEXT NOT NULL CHECK(role IN ('production', 'staging', 'canary', 'experimental')),
            url TEXT NOT NULL,
            caps TEXT NOT NULL,  -- JSON array
            labels TEXT,         -- JSON object
            ctx_limit INTEGER,
            cost_hint_usd_per_1k REAL,
            heartbeat_interval_s INTEGER DEFAULT 60,
            ttl_s INTEGER DEFAULT 90,
            status TEXT DEFAULT 'ok' CHECK(status IN ('ok', 'degraded', 'down', 'queued', 'running', 'blocked', 'waiting_approval', 'paused', 'error', 'done')),
            last_heartbeat_ts TEXT,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            p95_ms REAL,
            queue_depth INTEGER,
            load REAL
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_name_role ON services(name, role)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_type_role ON services(type, role)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_status ON services(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_last_heartbeat ON services(last_heartbeat_ts)")

    conn.commit()
    conn.close()

    print(f"✓ Registry database initialized at {DB_PATH}")


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
    title="PAS Service Registry",
    description="Service discovery, registration, and heartbeat monitoring",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.post("/register")
async def register_service(service: ServiceRegistration):
    """
    Register a new service (model, tool, or agent)

    Returns:
        - service_id (UUID)
        - registered_at (timestamp)
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Serialize JSON fields
        caps_json = json.dumps(service.caps)
        labels_json = json.dumps(service.labels) if service.labels else None
        cost_hint_usd = service.cost_hint.get("usd_per_1k") if service.cost_hint else None

        try:
            cursor.execute("""
                INSERT INTO services (
                    service_id, name, type, role, url, caps, labels,
                    ctx_limit, cost_hint_usd_per_1k, heartbeat_interval_s, ttl_s,
                    last_heartbeat_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                service.service_id, service.name, service.type, service.role,
                service.url, caps_json, labels_json, service.ctx_limit,
                cost_hint_usd, service.heartbeat_interval_s, service.ttl_s,
                datetime.utcnow().isoformat()
            ))

            conn.commit()

            # Fetch registered_at timestamp
            cursor.execute("SELECT registered_at FROM services WHERE service_id = ?", (service.service_id,))
            row = cursor.fetchone()

            return {
                "service_id": service.service_id,
                "registered_at": row["registered_at"]
            }

        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail=f"Service ID {service.service_id} already registered")


@app.put("/heartbeat")
async def update_heartbeat(heartbeat: ServiceHeartbeat):
    """
    Update service heartbeat

    Returns:
        - success: bool
        - ts: timestamp of update
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE services
            SET status = ?,
                last_heartbeat_ts = ?,
                p95_ms = ?,
                queue_depth = ?,
                load = ?
            WHERE service_id = ?
        """, (
            heartbeat.status,
            datetime.utcnow().isoformat(),
            heartbeat.p95_ms,
            heartbeat.queue_depth,
            heartbeat.load,
            heartbeat.service_id
        ))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Service {heartbeat.service_id} not found")

        conn.commit()

        return {
            "success": True,
            "ts": datetime.utcnow().isoformat()
        }


@app.get("/discover")
async def discover_services(
    type: Optional[str] = Query(None, pattern="^(model|tool|agent)$"),
    role: Optional[str] = Query(None, pattern="^(production|staging|canary|experimental)$"),
    cap: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    Discover services by filters

    Query parameters:
        - type: Service type (model, tool, agent)
        - role: Deployment role (production, staging, canary, experimental)
        - cap: Required capability (e.g., 'infer', 'classify')
        - name: Service name (exact match)
        - status: Service status (ok, degraded, down, etc.)

    Returns:
        - items: List of matching services with full metadata
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Build dynamic query
        query = "SELECT * FROM services WHERE 1=1"
        params = []

        if type:
            query += " AND type = ?"
            params.append(type)

        if role:
            query += " AND role = ?"
            params.append(role)

        if name:
            query += " AND name = ?"
            params.append(name)

        if status:
            query += " AND status = ?"
            params.append(status)

        if cap:
            # Search within JSON caps array
            query += " AND caps LIKE ?"
            params.append(f'%"{cap}"%')

        query += " ORDER BY name, role"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        items = []
        for row in rows:
            items.append({
                "service_id": row["service_id"],
                "name": row["name"],
                "type": row["type"],
                "role": row["role"],
                "url": row["url"],
                "caps": json.loads(row["caps"]),
                "labels": json.loads(row["labels"]) if row["labels"] else None,
                "ctx_limit": row["ctx_limit"],
                "cost_hint_usd_per_1k": row["cost_hint_usd_per_1k"],
                "status": row["status"],
                "last_heartbeat_ts": row["last_heartbeat_ts"],
                "registered_at": row["registered_at"],
                "p95_ms": row["p95_ms"],
                "queue_depth": row["queue_depth"],
                "load": row["load"]
            })

        return {"items": items}


@app.post("/promote")
async def promote_service(promotion: ServicePromotion):
    """
    Promote/demote service role

    Example: Move 'Transformer-Optimized' from 'staging' to 'production'

    Returns:
        - updated_instances: Number of services updated
        - ts: Timestamp of update
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE services
            SET role = ?
            WHERE name = ? AND role = ?
        """, (promotion.to_role, promotion.name, promotion.from_role))

        updated = cursor.rowcount
        conn.commit()

        if updated == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No services found with name='{promotion.name}' and role='{promotion.from_role}'"
            )

        return {
            "updated_instances": updated,
            "ts": datetime.utcnow().isoformat()
        }


@app.post("/deregister")
async def deregister_service(deregistration: ServiceDeregistration):
    """
    Deregister a service by ID

    Returns:
        - success: bool
        - ts: Timestamp of deregistration
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM services WHERE service_id = ?", (deregistration.service_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Service {deregistration.service_id} not found")

        conn.commit()

        return {
            "success": True,
            "ts": datetime.utcnow().isoformat()
        }


@app.get("/")
async def root():
    """
    Service information (root endpoint)
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM services")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as healthy FROM services WHERE status = 'ok'")
        healthy = cursor.fetchone()["healthy"]

        return {
            "service": "PAS Service Registry",
            "version": "1.0.0",
            "port": 6121,
            "status": "running",
            "registered_services": total,
            "healthy_services": healthy,
            "endpoints": {
                "health": "/health",
                "register": "POST /register",
                "heartbeat": "PUT /heartbeat",
                "discover": "GET /discover",
                "promote": "POST /promote",
                "deregister": "POST /deregister",
                "list_all": "GET /services",
                "get_service": "GET /services/{service_id}"
            },
            "docs": "/docs"
        }


@app.get("/health")
async def health():
    """
    Registry health check

    Returns:
        - status: 'ok'
        - registered: Total number of services
        - healthy: Number of services with status='ok'
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM services")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as healthy FROM services WHERE status = 'ok'")
        healthy = cursor.fetchone()["healthy"]

        return {
            "status": "ok",
            "registered": total,
            "healthy": healthy
        }


@app.get("/services/{service_id}")
async def get_service(service_id: str):
    """
    Get full details for a specific service

    Returns:
        - Full service record
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM services WHERE service_id = ?", (service_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")

        return {
            "service_id": row["service_id"],
            "name": row["name"],
            "type": row["type"],
            "role": row["role"],
            "url": row["url"],
            "caps": json.loads(row["caps"]),
            "labels": json.loads(row["labels"]) if row["labels"] else None,
            "ctx_limit": row["ctx_limit"],
            "cost_hint_usd_per_1k": row["cost_hint_usd_per_1k"],
            "heartbeat_interval_s": row["heartbeat_interval_s"],
            "ttl_s": row["ttl_s"],
            "status": row["status"],
            "last_heartbeat_ts": row["last_heartbeat_ts"],
            "registered_at": row["registered_at"],
            "p95_ms": row["p95_ms"],
            "queue_depth": row["queue_depth"],
            "load": row["load"]
        }


@app.get("/services")
async def list_all_services():
    """
    List all registered services (no filters)

    Returns:
        - items: List of all services
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM services ORDER BY name, role")
        rows = cursor.fetchall()

        items = []
        for row in rows:
            items.append({
                "service_id": row["service_id"],
                "name": row["name"],
                "type": row["type"],
                "role": row["role"],
                "url": row["url"],
                "caps": json.loads(row["caps"]),
                "labels": json.loads(row["labels"]) if row["labels"] else None,
                "status": row["status"],
                "last_heartbeat_ts": row["last_heartbeat_ts"]
            })

        return {"items": items}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6121)
