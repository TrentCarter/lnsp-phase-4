#!/usr/bin/env python3
"""
Manager Pool - Singleton pool for Manager allocation

Manages lifecycle of all Manager agents:
- Creation (via ManagerFactory)
- Allocation (assign to Directors)
- Pooling (reuse idle Managers)
- Termination (cleanup)

Thread-safe implementation using locks.
"""
import threading
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field


class ManagerState(Enum):
    """Manager lifecycle states"""
    CREATED = "created"  # Just created, not yet assigned
    IDLE = "idle"  # Available for work
    BUSY = "busy"  # Currently executing a job card
    FAILED = "failed"  # Failed execution, needs recovery
    TERMINATED = "terminated"  # Shut down, no longer available


@dataclass
class ManagerInfo:
    """Manager metadata"""
    manager_id: str
    lane: str  # Code, Models, Data, DevSecOps, Docs
    state: ManagerState
    created_at: float
    last_allocated_at: Optional[float] = None
    current_job_card_id: Optional[str] = None
    endpoint: Optional[str] = None  # RPC endpoint if Manager has HTTP server
    llm_model: Optional[str] = None
    parent_director: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ManagerPool:
    """
    Singleton Manager pool

    Maintains pool of Managers across all lanes.
    Thread-safe operations.

    Usage:
        pool = get_manager_pool()
        manager_id = pool.allocate_manager(lane="Code", director="Dir-Code")
        pool.mark_busy(manager_id, job_card_id="jc-123")
        # ... Manager executes job ...
        pool.mark_idle(manager_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize pool (only once)"""
        if self._initialized:
            return

        self._managers: Dict[str, ManagerInfo] = {}
        self._ops_lock = threading.Lock()
        self._initialized = True

    def register_manager(
        self,
        manager_id: str,
        lane: str,
        llm_model: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new Manager in the pool

        Args:
            manager_id: Unique Manager ID (e.g., "Mgr-Code-01")
            lane: Lane name (Code, Models, Data, DevSecOps, Docs)
            llm_model: LLM model used by Manager
            endpoint: RPC endpoint (if Manager has HTTP server)
            metadata: Additional metadata

        Returns:
            True if registered, False if already exists
        """
        with self._ops_lock:
            if manager_id in self._managers:
                return False

            self._managers[manager_id] = ManagerInfo(
                manager_id=manager_id,
                lane=lane,
                state=ManagerState.IDLE,
                created_at=time.time(),
                endpoint=endpoint,
                llm_model=llm_model,
                metadata=metadata or {}
            )
            return True

    def allocate_manager(
        self,
        lane: str,
        director: str,
        job_card_id: str,
        prefer_idle: bool = True
    ) -> Optional[str]:
        """
        Allocate a Manager for a job card

        Tries to reuse idle Manager first, creates new if none available.

        Args:
            lane: Lane name (Code, Models, Data, DevSecOps, Docs)
            director: Director requesting allocation (e.g., "Dir-Code")
            job_card_id: Job card ID being assigned
            prefer_idle: Prefer idle Managers over creating new ones

        Returns:
            manager_id if allocated, None if allocation failed
        """
        with self._ops_lock:
            # Try to find idle Manager in same lane
            if prefer_idle:
                for manager_id, info in self._managers.items():
                    if info.lane == lane and info.state == ManagerState.IDLE:
                        # Reuse idle Manager
                        info.state = ManagerState.BUSY
                        info.last_allocated_at = time.time()
                        info.current_job_card_id = job_card_id
                        info.parent_director = director
                        return manager_id

            # No idle Manager available - would create new one here
            # For now, return None (creation logic in ManagerFactory)
            return None

    def mark_busy(self, manager_id: str, job_card_id: str) -> bool:
        """
        Mark Manager as busy

        Args:
            manager_id: Manager ID
            job_card_id: Job card being executed

        Returns:
            True if marked busy, False if Manager not found
        """
        with self._ops_lock:
            if manager_id not in self._managers:
                return False

            info = self._managers[manager_id]
            info.state = ManagerState.BUSY
            info.current_job_card_id = job_card_id
            info.last_allocated_at = time.time()
            return True

    def mark_idle(self, manager_id: str) -> bool:
        """
        Mark Manager as idle (available for reuse)

        Args:
            manager_id: Manager ID

        Returns:
            True if marked idle, False if Manager not found
        """
        with self._ops_lock:
            if manager_id not in self._managers:
                return False

            info = self._managers[manager_id]
            info.state = ManagerState.IDLE
            info.current_job_card_id = None
            return True

    def mark_failed(self, manager_id: str) -> bool:
        """
        Mark Manager as failed

        Args:
            manager_id: Manager ID

        Returns:
            True if marked failed, False if Manager not found
        """
        with self._ops_lock:
            if manager_id not in self._managers:
                return False

            info = self._managers[manager_id]
            info.state = ManagerState.FAILED
            return True

    def terminate_manager(self, manager_id: str) -> bool:
        """
        Terminate Manager (remove from pool)

        Args:
            manager_id: Manager ID

        Returns:
            True if terminated, False if Manager not found
        """
        with self._ops_lock:
            if manager_id not in self._managers:
                return False

            info = self._managers[manager_id]
            info.state = ManagerState.TERMINATED
            # Keep in dict for historical tracking, but mark as terminated
            return True

    def get_manager_info(self, manager_id: str) -> Optional[ManagerInfo]:
        """
        Get Manager metadata

        Args:
            manager_id: Manager ID

        Returns:
            ManagerInfo if found, None otherwise
        """
        with self._ops_lock:
            return self._managers.get(manager_id)

    def get_managers_by_lane(self, lane: str, state: Optional[ManagerState] = None) -> List[ManagerInfo]:
        """
        Get all Managers for a lane

        Args:
            lane: Lane name
            state: Optional state filter

        Returns:
            List of ManagerInfo
        """
        with self._ops_lock:
            managers = [
                info for info in self._managers.values()
                if info.lane == lane
            ]

            if state:
                managers = [m for m in managers if m.state == state]

            return managers

    def get_idle_count(self, lane: Optional[str] = None) -> int:
        """
        Get count of idle Managers

        Args:
            lane: Optional lane filter

        Returns:
            Count of idle Managers
        """
        with self._ops_lock:
            managers = self._managers.values()

            if lane:
                managers = [m for m in managers if m.lane == lane]

            return sum(1 for m in managers if m.state == ManagerState.IDLE)

    def get_busy_count(self, lane: Optional[str] = None) -> int:
        """
        Get count of busy Managers

        Args:
            lane: Optional lane filter

        Returns:
            Count of busy Managers
        """
        with self._ops_lock:
            managers = self._managers.values()

            if lane:
                managers = [m for m in managers if m.lane == lane]

            return sum(1 for m in managers if m.state == ManagerState.BUSY)

    def get_all_managers(self) -> Dict[str, ManagerInfo]:
        """
        Get all Managers

        Returns:
            Dict mapping manager_id to ManagerInfo
        """
        with self._ops_lock:
            return self._managers.copy()


# Singleton accessor
_pool_instance = None


def get_manager_pool() -> ManagerPool:
    """Get singleton Manager pool instance"""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = ManagerPool()
    return _pool_instance
