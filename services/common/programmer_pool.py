#!/usr/bin/env python3
"""
Programmer Pool - Load Balancing for Programmer Services

Responsibilities:
- Discover available Programmer services
- Load balance task assignments across Programmers
- Track Programmer state (idle, busy, failed)
- Provide routing logic for Managers
- Support runtime LLM selection (LLM-agnostic Programmers)

Load Balancing Strategy:
1. Round-robin among idle Programmers
2. Fallback to least-busy if all occupied
3. Remove failed Programmers from pool
4. Re-add Programmers when they recover

Integration:
- Managers call ProgrammerPool.assign_task() to get Programmer endpoint
- Managers send HTTP POST to Programmer's /execute endpoint
- Programmer Pool tracks state based on /health and /status checks
"""
import httpx
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict


class ProgrammerState(str, Enum):
    """Programmer service states"""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ProgrammerInfo:
    """Programmer service information"""
    agent_id: str
    port: int
    endpoint: str
    state: ProgrammerState = ProgrammerState.UNKNOWN
    current_tasks: int = 0
    max_tasks: int = 1  # Programmers are single-threaded
    total_tasks_completed: int = 0
    total_failures: int = 0
    last_health_check: float = 0.0
    last_assigned: float = 0.0
    avg_task_duration_s: float = 0.0

    def is_available(self) -> bool:
        """Check if Programmer can accept new tasks"""
        return (
            self.state in [ProgrammerState.IDLE, ProgrammerState.BUSY]
            and self.current_tasks < self.max_tasks
        )


class ProgrammerPool:
    """
    Programmer Pool with Load Balancing

    Singleton pattern for global access across Manager services.
    """

    _instance: Optional["ProgrammerPool"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize Programmer Pool (singleton)"""
        if self._initialized:
            return

        self.programmers: Dict[str, ProgrammerInfo] = {}
        self.round_robin_index: int = 0
        self.health_check_interval_s: float = 30.0
        self.last_discovery: float = 0.0
        self.discovery_interval_s: float = 60.0

        # HTTP client for health checks and task submission
        self.client = httpx.AsyncClient(timeout=5.0)

        self._initialized = True

    def register_programmer(
        self,
        agent_id: str,
        port: int,
        endpoint: Optional[str] = None
    ):
        """
        Register a Programmer service with the pool.

        Args:
            agent_id: Programmer agent ID (e.g., "Prog-001")
            port: Service port (e.g., 6151)
            endpoint: Optional custom endpoint (default: http://localhost:{port})
        """
        if endpoint is None:
            endpoint = f"http://localhost:{port}"

        self.programmers[agent_id] = ProgrammerInfo(
            agent_id=agent_id,
            port=port,
            endpoint=endpoint,
            state=ProgrammerState.UNKNOWN
        )

    def discover_programmers(self, port_range: range = range(6151, 6161)):
        """
        Auto-discover Programmer services by checking port range.

        Args:
            port_range: Port range to scan (default: 6151-6160)
        """
        for port in port_range:
            endpoint = f"http://localhost:{port}"
            try:
                response = httpx.get(f"{endpoint}/health", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    agent_id = data.get("agent", f"Prog-{port-6150:03d}")

                    if agent_id not in self.programmers:
                        self.programmers[agent_id] = ProgrammerInfo(
                            agent_id=agent_id,
                            port=port,
                            endpoint=endpoint,
                            state=ProgrammerState.IDLE,
                            last_health_check=time.time()
                        )
            except Exception:
                # Port not responding, skip
                continue

        self.last_discovery = time.time()

    async def health_check_all(self):
        """
        Perform health checks on all registered Programmers.

        Updates state based on /health endpoint response.
        """
        now = time.time()

        for agent_id, info in self.programmers.items():
            # Skip if recently checked
            if now - info.last_health_check < self.health_check_interval_s:
                continue

            try:
                response = await self.client.get(f"{info.endpoint}/health")
                if response.status_code == 200:
                    data = response.json()

                    # Update state based on health response
                    if data.get("status") == "ok":
                        # Check if busy by querying current tasks
                        # For now, assume IDLE if health check passes
                        if info.current_tasks == 0:
                            info.state = ProgrammerState.IDLE
                        else:
                            info.state = ProgrammerState.BUSY

                        info.last_health_check = now
                    else:
                        info.state = ProgrammerState.FAILED
                else:
                    info.state = ProgrammerState.FAILED

            except Exception:
                # Health check failed
                info.state = ProgrammerState.FAILED

    def get_available_programmers(self) -> List[ProgrammerInfo]:
        """
        Get list of available Programmers (can accept tasks).

        Returns:
            List of available ProgrammerInfo objects
        """
        return [
            info for info in self.programmers.values()
            if info.is_available()
        ]

    def assign_task(
        self,
        task_description: str = "",
        preferred_llm: Optional[str] = None
    ) -> Optional[ProgrammerInfo]:
        """
        Assign task to next available Programmer using round-robin.

        Args:
            task_description: Task description (for logging)
            preferred_llm: Optional preferred LLM (ignored for now, all Programmers are LLM-agnostic)

        Returns:
            ProgrammerInfo if available, None if all busy/failed
        """
        available = self.get_available_programmers()

        if not available:
            return None

        # Round-robin selection
        selected = available[self.round_robin_index % len(available)]
        self.round_robin_index += 1

        # Mark as busy
        selected.current_tasks += 1
        selected.state = ProgrammerState.BUSY
        selected.last_assigned = time.time()

        return selected

    def release_task(self, agent_id: str, success: bool = True):
        """
        Release task from Programmer (mark as idle).

        Args:
            agent_id: Programmer agent ID
            success: Whether task completed successfully
        """
        if agent_id not in self.programmers:
            return

        info = self.programmers[agent_id]
        info.current_tasks = max(0, info.current_tasks - 1)

        if success:
            info.total_tasks_completed += 1
        else:
            info.total_failures += 1

        # Update state
        if info.current_tasks == 0:
            info.state = ProgrammerState.IDLE
        else:
            info.state = ProgrammerState.BUSY

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool metrics
        """
        total = len(self.programmers)
        idle = sum(1 for p in self.programmers.values() if p.state == ProgrammerState.IDLE)
        busy = sum(1 for p in self.programmers.values() if p.state == ProgrammerState.BUSY)
        failed = sum(1 for p in self.programmers.values() if p.state == ProgrammerState.FAILED)
        total_tasks = sum(p.total_tasks_completed for p in self.programmers.values())
        total_failures = sum(p.total_failures for p in self.programmers.values())

        return {
            "total_programmers": total,
            "idle": idle,
            "busy": busy,
            "failed": failed,
            "available": idle + busy,  # Both can accept tasks if under max_tasks
            "total_tasks_completed": total_tasks,
            "total_failures": total_failures,
            "success_rate": (total_tasks / (total_tasks + total_failures))
                if (total_tasks + total_failures) > 0 else 0.0
        }

    def get_programmer(self, agent_id: str) -> Optional[ProgrammerInfo]:
        """Get Programmer info by agent ID"""
        return self.programmers.get(agent_id)

    def list_programmers(self) -> List[Dict[str, Any]]:
        """
        List all Programmers with their status.

        Returns:
            List of Programmer info dictionaries
        """
        return [
            {
                "agent_id": info.agent_id,
                "port": info.port,
                "endpoint": info.endpoint,
                "state": info.state,
                "current_tasks": info.current_tasks,
                "total_completed": info.total_tasks_completed,
                "total_failures": info.total_failures
            }
            for info in self.programmers.values()
        ]


# Global instance getter
_pool_instance: Optional[ProgrammerPool] = None


def get_programmer_pool() -> ProgrammerPool:
    """
    Get global ProgrammerPool instance (singleton).

    Returns:
        ProgrammerPool instance
    """
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = ProgrammerPool()
    return _pool_instance


# Auto-discovery on import (if this is the main process)
def init_pool():
    """Initialize pool with auto-discovery"""
    pool = get_programmer_pool()
    pool.discover_programmers()
    return pool


if __name__ == "__main__":
    # Self-test: discover and list Programmers
    print("Programmer Pool Self-Test\n")

    pool = init_pool()

    print(f"Discovered Programmers: {len(pool.programmers)}")
    for info in pool.list_programmers():
        print(f"  {info['agent_id']:<15} Port {info['port']}  State: {info['state']}")

    print(f"\nPool Stats:")
    stats = pool.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test assignment
    print(f"\nAssignment Test:")
    for i in range(5):
        programmer = pool.assign_task(f"Test task {i+1}")
        if programmer:
            print(f"  Task {i+1} → {programmer.agent_id} (Port {programmer.port})")
            pool.release_task(programmer.agent_id, success=True)
        else:
            print(f"  Task {i+1} → No available Programmer")

    print(f"\nPool Stats After Assignment:")
    stats = pool.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
