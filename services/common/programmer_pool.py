#!/usr/bin/env python3
"""
Programmer Pool Load Balancer

Manages 10 programmers with:
- Load balancing (least_loaded, round_robin, capability_match)
- Health tracking
- Capability-based routing
- Cost optimization

Used by Manager-Code to dispatch tasks to the programmer pool.
"""
import asyncio
import httpx
import yaml
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path


class ProgrammerPool:
    """
    Manages pool of 10 programmers with load balancing and failover.
    """

    def __init__(self, config_path: str = "configs/pas/programmer_pool.yaml"):
        """Initialize programmer pool from config"""
        # Resolve config path relative to project root
        if not Path(config_path).is_absolute():
            config_path = Path.cwd() / config_path
            
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.programmers = self.config["programmers"]
        self.load_balancing = self.config.get("load_balancing", {})
        self.failover = self.config.get("failover", {})

        # Track programmer state
        self.health_cache: Dict[str, Dict[str, Any]] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.queue_depth: Dict[str, int] = {p["id"]: 0 for p in self.programmers}

        # Round-robin counter
        self.round_robin_index = 0

    def _get_programmer_by_id(self, prog_id: str) -> Optional[Dict[str, Any]]:
        """Get programmer config by ID"""
        for prog in self.programmers:
            if prog["id"] == prog_id:
                return prog
        return None

    def _get_programmer_url(self, prog_id: str) -> str:
        """Get programmer RPC URL"""
        prog = self._get_programmer_by_id(prog_id)
        if not prog:
            raise ValueError(f"Programmer {prog_id} not found")
        port = prog["port"]
        return f"http://127.0.0.1:{port}"

    async def _check_programmer_health(self, prog_id: str) -> Dict[str, Any]:
        """Check programmer health via /health endpoint"""
        url = self._get_programmer_url(prog_id)

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                health = response.json()

                # Cache health status
                self.health_cache[prog_id] = health
                self.last_health_check[prog_id] = datetime.now()

                return health
        except Exception as e:
            # Mark as unhealthy
            self.health_cache[prog_id] = {"status": "error", "error": str(e)}
            self.last_health_check[prog_id] = datetime.now()
            return {"status": "error", "error": str(e)}

    async def _get_cached_health(self, prog_id: str) -> Dict[str, Any]:
        """Get cached health or refresh if stale"""
        cache_ttl = self.load_balancing.get("health_check_interval_s", 30)

        # Check if cache is stale
        last_check = self.last_health_check.get(prog_id)
        if not last_check or (datetime.now() - last_check) > timedelta(seconds=cache_ttl):
            # Refresh cache
            return await self._check_programmer_health(prog_id)

        # Return cached
        return self.health_cache.get(prog_id, {"status": "unknown"})

    async def get_available_programmers(
        self,
        capabilities: Optional[List[str]] = None,
        prefer_free: bool = True
    ) -> List[str]:
        """
        Get list of available programmer IDs matching criteria.

        Args:
            capabilities: Required capabilities (e.g., ["fast", "free"])
            prefer_free: Prefer free models over paid

        Returns:
            List of programmer IDs sorted by preference
        """
        available = []

        # Filter by capabilities
        if capabilities:
            # Get all programmers matching ALL capabilities
            capability_routing = self.config.get("capability_routing", {})
            candidates = None
            for cap in capabilities:
                cap_progs = set(capability_routing.get(cap, []))
                if candidates is None:
                    candidates = cap_progs
                else:
                    candidates = candidates.intersection(cap_progs)

            if not candidates:
                # No programmers match all capabilities - fall back to any programmer
                candidates = {p["id"] for p in self.programmers}
        else:
            # No capability filter - all programmers
            candidates = {p["id"] for p in self.programmers}

        # Check health for candidates
        for prog_id in candidates:
            health = await self._get_cached_health(prog_id)
            if health.get("status") == "ok":
                available.append(prog_id)

        # Sort by preference
        if prefer_free and self.load_balancing.get("prefer_free_models", True):
            # Prefer free local models
            free_progs = set(self.config.get("capability_routing", {}).get("free", []))
            available.sort(key=lambda p: 0 if p in free_progs else 1)

        return available

    async def select_programmer(
        self,
        capabilities: Optional[List[str]] = None,
        prefer_free: bool = True
    ) -> Optional[str]:
        """
        Select best programmer for task using load balancing strategy.

        Args:
            capabilities: Required capabilities
            prefer_free: Prefer free models over paid

        Returns:
            Programmer ID or None if none available
        """
        available = await self.get_available_programmers(capabilities, prefer_free)

        if not available:
            return None

        strategy = self.load_balancing.get("strategy", "least_loaded")

        if strategy == "least_loaded":
            # Select programmer with lowest queue depth
            available.sort(key=lambda p: self.queue_depth.get(p, 0))
            return available[0]

        elif strategy == "round_robin":
            # Round-robin selection
            selected = available[self.round_robin_index % len(available)]
            self.round_robin_index += 1
            return selected

        elif strategy == "capability_match":
            # Already filtered by capabilities - just take first
            return available[0]

        else:
            # Default: first available
            return available[0]

    async def dispatch_task(
        self,
        task_description: str,
        files: List[str],
        run_id: str,
        capabilities: Optional[List[str]] = None,
        prefer_free: bool = True
    ) -> Dict[str, Any]:
        """
        Dispatch task to best available programmer.

        Args:
            task_description: Task description for Aider
            files: Files to edit
            run_id: Run ID for tracking
            capabilities: Required capabilities
            prefer_free: Prefer free models

        Returns:
            Result dict with status, programmer_id, and response

        Raises:
            Exception if no programmers available or task fails
        """
        # Select programmer
        prog_id = await self.select_programmer(capabilities, prefer_free)

        if not prog_id:
            raise Exception("No programmers available matching criteria")

        # Increment queue depth
        self.queue_depth[prog_id] = self.queue_depth.get(prog_id, 0) + 1

        try:
            # Dispatch to programmer
            url = self._get_programmer_url(prog_id)

            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(
                    f"{url}/aider/edit",
                    json={
                        "message": task_description,
                        "files": files,
                        "dry_run": False,
                        "run_id": run_id
                    }
                )
                response.raise_for_status()
                result = response.json()

            return {
                "status": "ok" if result.get("ok") else "error",
                "programmer_id": prog_id,
                "programmer_url": url,
                "result": result
            }

        finally:
            # Decrement queue depth
            self.queue_depth[prog_id] = max(0, self.queue_depth.get(prog_id, 0) - 1)

    async def get_pool_status(self) -> Dict[str, Any]:
        """
        Get status of entire programmer pool.

        Returns:
            Dict with health, queue depths, and availability
        """
        # Refresh all health checks
        health_tasks = [self._check_programmer_health(p["id"]) for p in self.programmers]
        await asyncio.gather(*health_tasks, return_exceptions=True)

        # Build status
        programmers_status = []
        for prog in self.programmers:
            prog_id = prog["id"]
            health = self.health_cache.get(prog_id, {})

            programmers_status.append({
                "id": prog_id,
                "port": prog["port"],
                "primary_llm": prog["primary_llm"],
                "backup_llm": prog["backup_llm"],
                "capabilities": prog.get("capabilities", []),
                "health": health.get("status", "unknown"),
                "using_backup": health.get("llm", {}).get("using_backup", False),
                "current_llm": health.get("llm", {}).get("current", "unknown"),
                "queue_depth": self.queue_depth.get(prog_id, 0)
            })

        # Count availability
        available = sum(1 for p in programmers_status if p["health"] == "ok")
        using_backup = sum(1 for p in programmers_status if p["using_backup"])

        return {
            "pool_size": len(self.programmers),
            "available": available,
            "unavailable": len(self.programmers) - available,
            "using_backup": using_backup,
            "programmers": programmers_status,
            "load_balancing": self.load_balancing,
            "total_queue_depth": sum(self.queue_depth.values())
        }


# Global pool instance (singleton)
_pool_instance = None


def get_programmer_pool() -> ProgrammerPool:
    """Get global programmer pool instance (singleton)"""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = ProgrammerPool()
    return _pool_instance
