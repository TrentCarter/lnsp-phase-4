#!/usr/bin/env python3
"""
PAS Heartbeat Monitor — Port 6109
Monitors service heartbeats, detects missed beats, marks services down, and emits alerts.
Part of Phase 0: Core Infrastructure
"""
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import contextmanager

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = Path("artifacts/registry/registry.db")
EVENTS_DIR = Path("artifacts/hmi/events")
CHECK_INTERVAL_S = 30  # Check every 30 seconds
MAX_MISSES_BEFORE_DOWN = 2  # Mark 'down' after 2 missed heartbeats
MAX_MISSES_BEFORE_DEREGISTER = 3  # Deregister after 3 missed heartbeats

# Event counter (global)
total_events = 0

# ============================================================================
# Models
# ============================================================================

class HeartbeatAlert(BaseModel):
    """Alert for missed heartbeats (heartbeat_alert.schema.json)"""
    alert_type: str  # 'heartbeat_miss' | 'heartbeat_restored' | 'service_down' | 'service_deregistered'
    service_id: str
    service_name: str
    missed_beats: int
    last_seen: str  # ISO 8601
    action: str  # 'warning' | 'marked_down' | 'deregistered' | 'escalated'
    ts: str  # ISO 8601
    parent: str = None


class MonitorStats(BaseModel):
    """Heartbeat monitor statistics"""
    total_services: int
    healthy_services: int
    degraded_services: int
    down_services: int
    alerts_issued: int
    last_check: str


# ============================================================================
# Database Access
# ============================================================================

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
# Alert Emission
# ============================================================================

def emit_alert(alert: HeartbeatAlert):
    """
    Emit alert to HMI events directory (LDJSON format)

    Args:
        alert: HeartbeatAlert instance
    """
    global total_events

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Append to today's event log
    log_file = EVENTS_DIR / f"heartbeat_alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    with open(log_file, "a") as f:
        f.write(alert.json() + "\n")

    # Increment event counter
    total_events += 1

    # Also print to console for debugging
    print(f"⚠️  ALERT: {alert.alert_type} — {alert.service_name} (missed {alert.missed_beats} beats) → {alert.action}")


# ============================================================================
# Heartbeat Checking Logic
# ============================================================================

async def check_heartbeats():
    """
    Background task: Check all services for missed heartbeats

    Logic:
    - If last_heartbeat_ts > TTL → count as missed
    - After 2 misses → mark status='down', emit alert
    - After 3 misses → deregister service, emit alert
    """
    stats = {
        "total_checks": 0,
        "alerts_issued": 0
    }

    while True:
        await asyncio.sleep(CHECK_INTERVAL_S)

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                # Get all services with their TTL
                cursor.execute("""
                    SELECT service_id, name, type, role, url, labels,
                           heartbeat_interval_s, ttl_s, status,
                           last_heartbeat_ts, registered_at
                    FROM services
                    ORDER BY last_heartbeat_ts ASC
                """)

                services = cursor.fetchall()
                stats["total_checks"] += 1

                for service in services:
                    service_id = service["service_id"]
                    name = service["name"]
                    status = service["status"]
                    last_heartbeat_ts = service["last_heartbeat_ts"]
                    ttl_s = service["ttl_s"]
                    heartbeat_interval_s = service["heartbeat_interval_s"]

                    # Parse labels to find parent (if any)
                    labels = json.loads(service["labels"]) if service["labels"] else {}
                    parent = labels.get("parent")

                    # Calculate elapsed time since last heartbeat
                    if last_heartbeat_ts:
                        last_beat = datetime.fromisoformat(last_heartbeat_ts)
                        now = datetime.utcnow()
                        elapsed_s = (now - last_beat).total_seconds()

                        # Calculate missed beats
                        missed_beats = int(elapsed_s // heartbeat_interval_s)

                        # Action logic
                        if missed_beats >= MAX_MISSES_BEFORE_DEREGISTER:
                            # Deregister service
                            cursor.execute("DELETE FROM services WHERE service_id = ?", (service_id,))
                            conn.commit()

                            alert = HeartbeatAlert(
                                alert_type="service_deregistered",
                                service_id=service_id,
                                service_name=name,
                                missed_beats=missed_beats,
                                last_seen=last_heartbeat_ts,
                                action="deregistered",
                                ts=datetime.utcnow().isoformat(),
                                parent=parent
                            )
                            emit_alert(alert)
                            stats["alerts_issued"] += 1

                        elif missed_beats >= MAX_MISSES_BEFORE_DOWN and status != "down":
                            # Mark service as down
                            cursor.execute(
                                "UPDATE services SET status = 'down' WHERE service_id = ?",
                                (service_id,)
                            )
                            conn.commit()

                            alert = HeartbeatAlert(
                                alert_type="service_down",
                                service_id=service_id,
                                service_name=name,
                                missed_beats=missed_beats,
                                last_seen=last_heartbeat_ts,
                                action="marked_down",
                                ts=datetime.utcnow().isoformat(),
                                parent=parent
                            )
                            emit_alert(alert)
                            stats["alerts_issued"] += 1

                        elif missed_beats >= 1 and missed_beats < MAX_MISSES_BEFORE_DOWN:
                            # Warning (first missed beat)
                            alert = HeartbeatAlert(
                                alert_type="heartbeat_miss",
                                service_id=service_id,
                                service_name=name,
                                missed_beats=missed_beats,
                                last_seen=last_heartbeat_ts,
                                action="warning",
                                ts=datetime.utcnow().isoformat(),
                                parent=parent
                            )
                            emit_alert(alert)
                            stats["alerts_issued"] += 1

                        elif missed_beats == 0 and status == "down":
                            # Service recovered
                            cursor.execute(
                                "UPDATE services SET status = 'ok' WHERE service_id = ?",
                                (service_id,)
                            )
                            conn.commit()

                            alert = HeartbeatAlert(
                                alert_type="heartbeat_restored",
                                service_id=service_id,
                                service_name=name,
                                missed_beats=0,
                                last_seen=last_heartbeat_ts,
                                action="restored",
                                ts=datetime.utcnow().isoformat(),
                                parent=parent
                            )
                            emit_alert(alert)
                            stats["alerts_issued"] += 1

        except Exception as e:
            print(f"❌ Error in heartbeat check: {e}")

        # Print stats every 10 checks
        if stats["total_checks"] % 10 == 0:
            print(f"✓ Heartbeat Monitor: {stats['total_checks']} checks, {stats['alerts_issued']} alerts issued")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PAS Heartbeat Monitor",
    description="Monitors service heartbeats and emits alerts on misses",
    version="1.0.0"
)

# Background task handle
monitor_task = None


@app.on_event("startup")
async def startup_event():
    """Start background heartbeat checking task"""
    global monitor_task

    # Ensure event log directory exists
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start background task
    monitor_task = asyncio.create_task(check_heartbeats())
    print(f"✓ Heartbeat Monitor started (checking every {CHECK_INTERVAL_S}s)")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background task on shutdown"""
    global monitor_task
    if monitor_task:
        monitor_task.cancel()
        print("✓ Heartbeat Monitor stopped")


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

        cursor.execute("SELECT COUNT(*) as down FROM services WHERE status = 'down'")
        down = cursor.fetchone()["down"]

        return {
            "service": "PAS Heartbeat Monitor",
            "version": "1.0.0",
            "port": 6109,
            "status": "running",
            "check_interval_s": CHECK_INTERVAL_S,
            "monitoring": {
                "total_services": total,
                "healthy": healthy,
                "down": down
            },
            "thresholds": {
                "misses_before_down": MAX_MISSES_BEFORE_DOWN,
                "misses_before_deregister": MAX_MISSES_BEFORE_DEREGISTER
            },
            "endpoints": {
                "health": "/health",
                "stats": "/stats",
                "alerts": "/alerts"
            },
            "docs": "/docs"
        }


@app.get("/health")
async def health():
    """
    Heartbeat Monitor health check

    Returns:
        - status: 'ok'
        - check_interval_s: Interval between checks
        - max_misses_before_down: Threshold for marking service down
        - total_events: Total events (alerts) emitted
    """
    return {
        "status": "ok",
        "check_interval_s": CHECK_INTERVAL_S,
        "max_misses_before_down": MAX_MISSES_BEFORE_DOWN,
        "max_misses_before_deregister": MAX_MISSES_BEFORE_DEREGISTER,
        "total_events": total_events
    }


@app.get("/stats")
async def get_stats():
    """
    Get current heartbeat monitor statistics

    Returns:
        - total_services: Number of registered services
        - healthy_services: Services with status='ok'
        - degraded_services: Services with status='degraded'
        - down_services: Services with status='down'
        - last_check: Timestamp of last check
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM services")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as healthy FROM services WHERE status = 'ok'")
        healthy = cursor.fetchone()["healthy"]

        cursor.execute("SELECT COUNT(*) as degraded FROM services WHERE status = 'degraded'")
        degraded = cursor.fetchone()["degraded"]

        cursor.execute("SELECT COUNT(*) as down FROM services WHERE status = 'down'")
        down = cursor.fetchone()["down"]

        return {
            "total_services": total,
            "healthy_services": healthy,
            "degraded_services": degraded,
            "down_services": down,
            "last_check": datetime.utcnow().isoformat()
        }


@app.get("/alerts")
async def get_recent_alerts(limit: int = 50):
    """
    Get recent heartbeat alerts from event log

    Args:
        limit: Maximum number of alerts to return (default 50)

    Returns:
        - alerts: List of recent alerts
    """
    today_log = EVENTS_DIR / f"heartbeat_alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    if not today_log.exists():
        return {"alerts": []}

    with open(today_log, "r") as f:
        lines = f.readlines()

    # Return last N lines
    recent = lines[-limit:] if len(lines) > limit else lines

    alerts = [json.loads(line) for line in recent]

    return {"alerts": alerts}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6109)
