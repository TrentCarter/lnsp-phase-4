#!/usr/bin/env python3
"""
Manager Factory - Creates Manager instances

Creates Managers dynamically based on:
- Lane (Code, Models, Data, DevSecOps, Docs)
- LLM configuration (model, provider)
- Resource allocation
- Job card requirements

Integrates with:
- Manager Pool (registration)
- Heartbeat Monitor (agent registration)
- Aider RPC (for code execution)
"""
import os
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

from services.common.manager_pool.manager_pool import get_manager_pool, ManagerState
from services.common.heartbeat import get_monitor, AgentState


class ManagerFactory:
    """
    Factory for creating Manager instances

    Managers are lightweight coordinators that manage 1-5 Programmers.
    They don't have their own HTTP servers - they communicate via:
    - File-based job queues (job card JSONL files)
    - Heartbeat system (status updates)
    - Aider RPC (for actual code execution by Programmers)

    Usage:
        factory = ManagerFactory()
        manager_id = factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-123",
            llm_model="qwen2.5-coder:7b"
        )
    """

    def __init__(self):
        """Initialize factory"""
        self.pool = get_manager_pool()
        self.heartbeat_monitor = get_monitor()
        self._manager_counter = {}  # Track IDs per lane

    def create_manager(
        self,
        lane: str,
        director: str,
        job_card_id: str,
        llm_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new Manager instance

        Args:
            lane: Lane name (Code, Models, Data, DevSecOps, Docs)
            director: Parent Director (e.g., "Dir-Code")
            job_card_id: Initial job card ID
            llm_model: LLM model to use (default: get from config)
            metadata: Additional metadata

        Returns:
            manager_id: Unique Manager ID (e.g., "Mgr-Code-01")
        """
        # Generate Manager ID
        manager_id = self._generate_manager_id(lane)

        # Get LLM configuration
        if llm_model is None:
            llm_model = self._get_default_llm(lane)

        # Register with Manager Pool
        self.pool.register_manager(
            manager_id=manager_id,
            lane=lane,
            llm_model=llm_model,
            endpoint=None,  # Managers are file-based, no HTTP endpoint
            metadata=metadata or {}
        )

        # Register with Heartbeat Monitor
        self.heartbeat_monitor.register_agent(
            agent=manager_id,
            parent=director,
            llm_model=llm_model,
            role="manager",
            tier="executor"
        )

        # Mark as busy (assigned to job card)
        self.pool.mark_busy(manager_id, job_card_id)

        # Create Manager working directory
        self._create_manager_workspace(manager_id, job_card_id)

        return manager_id

    def _generate_manager_id(self, lane: str) -> str:
        """
        Generate unique Manager ID

        Format: Mgr-{Lane}-{Counter:02d}
        Example: Mgr-Code-01, Mgr-Code-02, ...

        Args:
            lane: Lane name

        Returns:
            Unique Manager ID
        """
        if lane not in self._manager_counter:
            self._manager_counter[lane] = 0

        self._manager_counter[lane] += 1
        counter = self._manager_counter[lane]

        return f"Mgr-{lane}-{counter:02d}"

    def _get_default_llm(self, lane: str) -> str:
        """
        Get default LLM for lane

        Args:
            lane: Lane name

        Returns:
            LLM model string (e.g., "qwen2.5-coder:7b")
        """
        # Default LLMs per lane
        defaults = {
            "Code": "qwen2.5-coder:7b",
            "Models": "deepseek-r1:7b-q4_k_m",
            "Data": "gemini/gemini-2.5-flash",
            "DevSecOps": "gemini/gemini-2.5-flash",
            "Docs": "anthropic/claude-sonnet-4-5"
        }

        # Check environment overrides
        env_key = f"MANAGER_{lane.upper()}_LLM"
        return os.getenv(env_key, defaults.get(lane, "qwen2.5-coder:7b"))

    def _create_manager_workspace(self, manager_id: str, job_card_id: str):
        """
        Create Manager working directory

        Directory structure:
        artifacts/managers/{manager_id}/
        ├── job_cards/
        │   └── {job_card_id}.json
        ├── artifacts/
        │   ├── diffs/
        │   └── test_results/
        └── logs/
            └── manager.log

        Args:
            manager_id: Manager ID
            job_card_id: Initial job card ID
        """
        base_dir = Path(f"artifacts/managers/{manager_id}")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (base_dir / "job_cards").mkdir(exist_ok=True)
        (base_dir / "artifacts" / "diffs").mkdir(parents=True, exist_ok=True)
        (base_dir / "artifacts" / "test_results").mkdir(parents=True, exist_ok=True)
        (base_dir / "logs").mkdir(exist_ok=True)

    def allocate_manager(
        self,
        lane: str,
        director: str,
        job_card_id: str,
        prefer_idle: bool = True
    ) -> str:
        """
        Allocate Manager (reuse idle or create new)

        Args:
            lane: Lane name
            director: Parent Director
            job_card_id: Job card ID
            prefer_idle: Prefer reusing idle Managers

        Returns:
            manager_id: Allocated Manager ID
        """
        # Try to allocate from pool first
        if prefer_idle:
            manager_id = self.pool.allocate_manager(
                lane=lane,
                director=director,
                job_card_id=job_card_id,
                prefer_idle=True
            )

            if manager_id:
                # Reusing idle Manager
                return manager_id

        # No idle Manager available - create new one
        return self.create_manager(
            lane=lane,
            director=director,
            job_card_id=job_card_id
        )

    def release_manager(self, manager_id: str):
        """
        Release Manager (mark as idle for reuse)

        Args:
            manager_id: Manager ID
        """
        self.pool.mark_idle(manager_id)

        # Send idle heartbeat
        self.heartbeat_monitor.heartbeat(
            agent=manager_id,
            state=AgentState.IDLE,
            message="Idle, waiting for job card"
        )

    def terminate_manager(self, manager_id: str):
        """
        Terminate Manager (remove from pool)

        Args:
            manager_id: Manager ID
        """
        self.pool.terminate_manager(manager_id)

        # Final heartbeat
        self.heartbeat_monitor.heartbeat(
            agent=manager_id,
            state=AgentState.TERMINATED,
            message="Terminated"
        )


# Singleton accessor
_factory_instance = None


def get_manager_factory() -> ManagerFactory:
    """Get singleton Manager factory instance"""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = ManagerFactory()
    return _factory_instance
