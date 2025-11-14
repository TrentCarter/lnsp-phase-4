#!/usr/bin/env python3
"""
Registry Heartbeat Utility

Provides automatic service registration and heartbeat transmission to the
Registry Service (port 6121) for services that need to be monitored by TRON.

Usage in FastAPI services:

    from services.common.registry_heartbeat import start_registry_heartbeat
    from fastapi import FastAPI

    app = FastAPI()

    @app.on_event("startup")
    async def startup_event():
        # Register and start sending heartbeats every 30s
        await start_registry_heartbeat(
            service_id="tron-heartbeat-monitor",
            name="TRON Heartbeat Monitor",
            type="agent",
            role="production",
            url="http://localhost:6109",
            caps=["health_monitoring", "timeout_detection", "service_alerts"],
            labels={"tier": "core", "category": "infrastructure", "port": 6109}
        )
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

REGISTRY_URL = "http://localhost:6121"
DEFAULT_HEARTBEAT_INTERVAL_S = 30
DEFAULT_TTL_S = 90

# Global task handle for heartbeat loop
_heartbeat_task: Optional[asyncio.Task] = None
_service_id: Optional[str] = None


async def register_service(
    service_id: str,
    name: str,
    type: str,  # "model" | "tool" | "agent"
    role: str,  # "production" | "staging" | "canary" | "experimental"
    url: str,
    caps: List[str],
    labels: Optional[Dict[str, Any]] = None,
    ctx_limit: Optional[int] = None,
    cost_hint: Optional[Dict[str, float]] = None,
    heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
    ttl_s: int = DEFAULT_TTL_S
) -> bool:
    """
    Register a service with the Registry Service

    Returns:
        True if registration succeeded, False otherwise
    """
    payload = {
        "service_id": service_id,
        "name": name,
        "type": type,
        "role": role,
        "url": url,
        "caps": caps,
        "labels": labels or {},
        "ctx_limit": ctx_limit,
        "cost_hint": cost_hint,
        "heartbeat_interval_s": heartbeat_interval_s,
        "ttl_s": ttl_s
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REGISTRY_URL}/register",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"✅ Registered {name} with Registry (ID: {result.get('service_id')})")
            return True
    except httpx.HTTPError as e:
        logger.error(f"❌ Failed to register {name} with Registry: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error registering {name}: {e}")
        return False


async def send_heartbeat(
    service_id: str,
    status: str = "ok",  # "ok" | "degraded" | "down" | "queued" | "running" | etc.
    p95_ms: Optional[float] = None,
    queue_depth: Optional[int] = None,
    load: Optional[float] = None,
    message: Optional[str] = None
) -> bool:
    """
    Send a heartbeat update to the Registry Service

    Returns:
        True if heartbeat succeeded, False otherwise
    """
    payload = {
        "service_id": service_id,
        "status": status,
        "p95_ms": p95_ms,
        "queue_depth": queue_depth,
        "load": load,
        "message": message
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{REGISTRY_URL}/heartbeat",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            logger.debug(f"💓 Heartbeat sent for {service_id}")
            return True
    except httpx.HTTPError as e:
        logger.warning(f"⚠️  Heartbeat failed for {service_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Unexpected error sending heartbeat for {service_id}: {e}")
        return False


async def heartbeat_loop(
    service_id: str,
    interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
    status: str = "ok"
):
    """
    Background task that sends periodic heartbeats to the Registry

    This runs indefinitely until the service shuts down.
    """
    logger.info(f"🔄 Starting heartbeat loop for {service_id} (every {interval_s}s)")

    while True:
        try:
            await asyncio.sleep(interval_s)
            success = await send_heartbeat(service_id, status=status)
            if not success:
                logger.warning(f"⚠️  Heartbeat failed for {service_id}, will retry in {interval_s}s")
        except asyncio.CancelledError:
            logger.info(f"🛑 Heartbeat loop cancelled for {service_id}")
            break
        except Exception as e:
            logger.error(f"❌ Error in heartbeat loop for {service_id}: {e}")
            # Continue despite errors


async def start_registry_heartbeat(
    service_id: str,
    name: str,
    type: str,
    role: str,
    url: str,
    caps: List[str],
    labels: Optional[Dict[str, Any]] = None,
    ctx_limit: Optional[int] = None,
    cost_hint: Optional[Dict[str, float]] = None,
    heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
    ttl_s: int = DEFAULT_TTL_S,
    status: str = "ok"
) -> bool:
    """
    Register service and start automatic heartbeat transmission

    This is the main entry point - call this from your FastAPI startup event.

    Returns:
        True if registration succeeded and heartbeat loop started, False otherwise
    """
    global _heartbeat_task, _service_id

    # Register the service first
    success = await register_service(
        service_id=service_id,
        name=name,
        type=type,
        role=role,
        url=url,
        caps=caps,
        labels=labels,
        ctx_limit=ctx_limit,
        cost_hint=cost_hint,
        heartbeat_interval_s=heartbeat_interval_s,
        ttl_s=ttl_s
    )

    if not success:
        logger.error(f"❌ Failed to register {name}, heartbeat loop will not start")
        return False

    # Start heartbeat loop
    _service_id = service_id
    _heartbeat_task = asyncio.create_task(
        heartbeat_loop(service_id, interval_s=heartbeat_interval_s, status=status)
    )

    logger.info(f"✅ Registry heartbeat started for {name}")
    return True


async def stop_registry_heartbeat():
    """
    Stop the heartbeat loop (call during shutdown)
    """
    global _heartbeat_task

    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("🛑 Registry heartbeat stopped")


# Optional: FastAPI lifespan context manager for FastAPI >= 0.93
try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI

    def create_registry_heartbeat_lifespan(
        service_id: str,
        name: str,
        type: str,
        role: str,
        url: str,
        caps: List[str],
        **kwargs
    ):
        """
        Create a FastAPI lifespan context manager with Registry heartbeat

        Usage:
            app = FastAPI(lifespan=create_registry_heartbeat_lifespan(
                service_id="tron",
                name="TRON Heartbeat Monitor",
                type="agent",
                role="production",
                url="http://localhost:6109",
                caps=["health_monitoring"]
            ))
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: register and start heartbeat
            await start_registry_heartbeat(
                service_id=service_id,
                name=name,
                type=type,
                role=role,
                url=url,
                caps=caps,
                **kwargs
            )
            yield
            # Shutdown: stop heartbeat
            await stop_registry_heartbeat()

        return lifespan

except ImportError:
    # FastAPI < 0.93, lifespan not available
    pass
