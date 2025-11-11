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

# Event counter (global) - tracks registrations, heartbeats, action logs
total_events = 0

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


class ActionLog(BaseModel):
    """Action log entry"""
    task_id: str
    parent_log_id: Optional[int] = None
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None
    action_type: str
    action_name: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    tier_from: Optional[int] = None
    tier_to: Optional[int] = None


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

    # Action logs table for tracking agent communication flows
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            parent_log_id INTEGER,
            timestamp TEXT NOT NULL,
            from_agent TEXT,
            to_agent TEXT,
            action_type TEXT NOT NULL,
            action_name TEXT,
            action_data TEXT,
            status TEXT,
            tier_from INTEGER,
            tier_to INTEGER,
            FOREIGN KEY (parent_log_id) REFERENCES action_logs(log_id)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_task_id ON action_logs(task_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_parent_log_id ON action_logs(parent_log_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_timestamp ON action_logs(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_from_agent ON action_logs(from_agent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_to_agent ON action_logs(to_agent)")

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
    global total_events

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

            # Increment event counter
            total_events += 1

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
    global total_events

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

        # Increment event counter
        total_events += 1

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
        - total_events: Total events (registrations + heartbeats + action logs)
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
            "healthy": healthy,
            "total_events": total_events
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
# Action Logs Endpoints
# ============================================================================

@app.post("/action_logs")
async def log_action(action_log: ActionLog):
    """
    Log an action/message in the agent communication flow

    Returns:
        - log_id: ID of the created log entry
        - timestamp: Timestamp of the log
    """
    global total_events

    with get_db() as conn:
        cursor = conn.cursor()

        timestamp = datetime.utcnow().isoformat()
        action_data_json = json.dumps(action_log.action_data) if action_log.action_data else None

        cursor.execute("""
            INSERT INTO action_logs (
                task_id, parent_log_id, timestamp,
                from_agent, to_agent,
                action_type, action_name, action_data,
                status, tier_from, tier_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            action_log.task_id, action_log.parent_log_id, timestamp,
            action_log.from_agent, action_log.to_agent,
            action_log.action_type, action_log.action_name, action_data_json,
            action_log.status, action_log.tier_from, action_log.tier_to
        ))

        log_id = cursor.lastrowid
        conn.commit()

        # Increment event counter
        total_events += 1

        return {
            "log_id": log_id,
            "timestamp": timestamp
        }


@app.get("/action_logs/tasks")
async def list_tasks():
    """
    List all tasks with their summary information

    Returns:
        - items: List of tasks with metadata (including human-readable task name)
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                task_id,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                COUNT(*) as action_count,
                GROUP_CONCAT(DISTINCT from_agent) as agents_involved
            FROM action_logs
            GROUP BY task_id
            ORDER BY start_time DESC
        """)

        rows = cursor.fetchall()

        items = []
        for row in rows:
            task_id = row["task_id"]

            # Fetch the task name from the first Gateway submission (Prime Directive)
            cursor.execute("""
                SELECT action_name
                FROM action_logs
                WHERE task_id = ? AND from_agent = 'Gateway' AND action_type = 'delegate'
                ORDER BY timestamp ASC
                LIMIT 1
            """, (task_id,))

            name_row = cursor.fetchone()
            task_name = name_row["action_name"] if name_row else task_id

            items.append({
                "task_id": task_id,
                "task_name": task_name,  # Human-readable name from Gateway submission
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "action_count": row["action_count"],
                "agents_involved": row["agents_involved"].split(",") if row["agents_involved"] else []
            })

        return {"items": items}


@app.get("/action_logs/task/{task_id}")
async def get_task_actions(task_id: str):
    """
    Get all actions for a specific task in hierarchical format

    Returns:
        - task_id: The task ID
        - actions: Hierarchical tree of actions
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Fetch all actions for this task
        cursor.execute("""
            SELECT
                log_id, parent_log_id, timestamp,
                from_agent, to_agent,
                action_type, action_name, action_data,
                status, tier_from, tier_to
            FROM action_logs
            WHERE task_id = ?
            ORDER BY timestamp ASC
        """, (task_id,))

        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Build flat list of actions
        actions = []
        action_map = {}

        for row in rows:
            action = {
                "log_id": row["log_id"],
                "parent_log_id": row["parent_log_id"],
                "timestamp": row["timestamp"],
                "from_agent": row["from_agent"],
                "to_agent": row["to_agent"],
                "action_type": row["action_type"],
                "action_name": row["action_name"],
                "action_data": json.loads(row["action_data"]) if row["action_data"] else None,
                "status": row["status"],
                "tier_from": row["tier_from"],
                "tier_to": row["tier_to"],
                "children": []
            }
            actions.append(action)
            action_map[action["log_id"]] = action

        # Build hierarchical structure
        root_actions = []
        for action in actions:
            if action["parent_log_id"] is None:
                root_actions.append(action)
            else:
                parent = action_map.get(action["parent_log_id"])
                if parent:
                    parent["children"].append(action)

        return {
            "task_id": task_id,
            "actions": root_actions
        }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6121)
