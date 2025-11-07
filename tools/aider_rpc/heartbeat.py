#!/usr/bin/env python3
"""
Registry Heartbeat & Health Reporting for Aider RPC

Manages service registration and periodic heartbeat to PAS Service Registry.

Based on PAS PRD Registry contract.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import httpx


@dataclass
class ServiceRegistration:
    """Service registration payload"""
    service_id: str
    name: str
    type: str  # agent|tool|model
    role: str  # execution|planning|review|etc
    url: str
    caps: list[str]  # Capabilities (git-edit, pytest-loop, etc.)
    labels: Dict[str, str]  # Arbitrary labels for routing
    ctx_limit: int  # Context window size
    cost_hint: Dict[str, float]  # Cost estimation (usd_per_1k)
    heartbeat_interval_s: int
    ttl_s: int  # Time-to-live (heartbeat_interval + buffer)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ServiceHeartbeat:
    """Heartbeat payload"""
    service_id: str
    status: str  # ok|degraded|error
    p95_ms: float  # P95 latency
    queue_depth: int  # Number of pending jobs
    load: float  # Load factor (0.0-1.0)
    ctx_used: Optional[int] = None  # Current context usage
    ctx_limit: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class RegistryClient:
    """
    Client for PAS Service Registry.

    Handles registration, heartbeat, and deregistration.
    """

    def __init__(
        self,
        registry_url: str,
        service_id: str,
        name: str,
        service_type: str,
        role: str,
        url: str,
        caps: list[str],
        ctx_limit: int = 131072,
        heartbeat_interval_s: int = 60,
    ):
        self.registry_url = registry_url.rstrip("/")
        self.service_id = service_id
        self.name = name
        self.service_type = service_type
        self.role = role
        self.url = url
        self.caps = caps
        self.ctx_limit = ctx_limit
        self.heartbeat_interval_s = heartbeat_interval_s

        self.registered = False
        self.heartbeat_task: Optional[asyncio.Task] = None

        # Metrics for heartbeat
        self.p95_latency = 0.0
        self.queue_depth = 0
        self.load = 0.0
        self.ctx_used = 0

        # Backoff state for retries
        self.failure_count = 0
        self.max_backoff_s = 60

    async def register(self):
        """Register service with Registry"""
        registration = ServiceRegistration(
            service_id=self.service_id,
            name=self.name,
            type=self.service_type,
            role=self.role,
            url=self.url,
            caps=self.caps,
            labels={"editor": "aider", "space": "n/a"},
            ctx_limit=self.ctx_limit,
            cost_hint={"usd_per_1k": 0.0},  # Local model, free
            heartbeat_interval_s=self.heartbeat_interval_s,
            ttl_s=self.heartbeat_interval_s + 30,
        )

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    f"{self.registry_url}/register",
                    json=registration.to_dict(),
                )
                response.raise_for_status()
                self.registered = True
                self.failure_count = 0  # Reset on success
                print(f"[Registry] Registered: {self.name} ({self.service_id})")
        except Exception as e:
            self.failure_count += 1
            print(f"[Registry] Registration failed (attempt {self.failure_count}): {e}")
            self.registered = False

    async def send_heartbeat(self):
        """Send heartbeat to Registry"""
        if not self.registered:
            print("[Registry] Not registered, skipping heartbeat")
            return

        heartbeat = ServiceHeartbeat(
            service_id=self.service_id,
            status="ok",  # TODO: Actual health check
            p95_ms=self.p95_latency,
            queue_depth=self.queue_depth,
            load=self.load,
            ctx_used=self.ctx_used,
            ctx_limit=self.ctx_limit,
        )

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.put(
                    f"{self.registry_url}/heartbeat",
                    json=heartbeat.to_dict(),
                )
                # Registry may return 404 if service expired, re-register
                if response.status_code == 404:
                    print("[Registry] Service expired, re-registering...")
                    await self.register()
                else:
                    response.raise_for_status()
                    self.failure_count = 0  # Reset on success
        except Exception as e:
            self.failure_count += 1
            print(f"[Registry] Heartbeat failed (attempt {self.failure_count}): {e}")

    async def start_heartbeat_loop(self):
        """Start periodic heartbeat loop with exponential backoff on failures"""
        await self.register()

        while True:
            # Exponential backoff: 1x, 2x, 4x, 8x, up to max_backoff_s
            if self.failure_count > 0:
                backoff_s = min(
                    self.heartbeat_interval_s * (2 ** (self.failure_count - 1)),
                    self.max_backoff_s
                )
                print(f"[Registry] Backing off for {backoff_s}s (failure count: {self.failure_count})")
                await asyncio.sleep(backoff_s)
            else:
                await asyncio.sleep(self.heartbeat_interval_s)

            await self.send_heartbeat()

    async def deregister(self):
        """Deregister service from Registry"""
        if not self.registered:
            return

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.delete(
                    f"{self.registry_url}/deregister/{self.service_id}"
                )
                response.raise_for_status()
                print(f"[Registry] Deregistered: {self.name} ({self.service_id})")
        except Exception as e:
            print(f"[Registry] Deregistration failed: {e}")
        finally:
            self.registered = False

    def update_metrics(self, p95_ms: float, queue_depth: int, load: float, ctx_used: int = 0):
        """Update metrics for next heartbeat"""
        self.p95_latency = p95_ms
        self.queue_depth = queue_depth
        self.load = load
        self.ctx_used = ctx_used


if __name__ == "__main__":
    # Self-test (requires Registry running on 6121)
    import sys

    async def test_heartbeat():
        registry_url = "http://127.0.0.1:6121"

        # Check if Registry is available
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                await client.get(f"{registry_url}/health")
        except Exception:
            print(f"✗ Registry not available at {registry_url}")
            print("  (This is OK for unit testing - start Registry to test heartbeat)")
            return

        client = RegistryClient(
            registry_url=registry_url,
            service_id="test-aider-001",
            name="Aider-LCO-Test",
            service_type="agent",
            role="execution",
            url="http://127.0.0.1:6150",
            caps=["git-edit", "pytest-loop"],
            heartbeat_interval_s=5,  # Short interval for testing
        )

        print("Heartbeat Self-Test:")
        print(f"  Registry: {registry_url}")
        print(f"  Service: {client.name} ({client.service_id})")

        # Register
        await client.register()
        if not client.registered:
            print("✗ Registration failed")
            return

        # Send a few heartbeats
        for i in range(3):
            await asyncio.sleep(client.heartbeat_interval_s)
            client.update_metrics(
                p95_ms=800.0 + i * 50,
                queue_depth=i,
                load=0.1 * i,
            )
            await client.send_heartbeat()
            print(f"✓ Heartbeat #{i+1} sent")

        # Deregister
        await client.deregister()
        print("✓ Test complete")

    asyncio.run(test_heartbeat())
