#!/usr/bin/env python3
"""
Heartbeat & Monitoring System for PAS Multi-Tier Architecture

Responsibilities:
- Track agent heartbeats (60s intervals)
- Detect missed heartbeats (2-miss rule)
- Escalate failures to parent agents
- Aggregate status across agent hierarchy
- Health dashboard data

Usage:
    from services.common.heartbeat import HeartbeatMonitor

    monitor = HeartbeatMonitor()
    monitor.register_agent("Architect", parent="PAS Root")
    monitor.heartbeat("Architect", state="planning", message="Decomposing Prime Directive")

    # Check health
    health = monitor.get_health("Architect")
    if not health["healthy"]:
        print(f"Agent unhealthy: {health['reason']}")
"""
import time
import threading
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path


class AgentState(str, Enum):
    """Standard agent states"""
    IDLE = "idle"
    PLANNING = "planning"
    DELEGATING = "delegating"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    VALIDATING = "validating"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    HIBERNATED = "hibernated"


@dataclass
class HeartbeatRecord:
    """Single heartbeat record from an agent"""
    agent: str
    run_id: Optional[str]
    timestamp: float
    state: AgentState
    message: str
    llm_model: Optional[str] = None
    parent_agent: Optional[str] = None
    children_agents: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHealth:
    """Agent health status"""
    agent: str
    healthy: bool
    last_heartbeat: float
    missed_count: int
    state: AgentState
    message: str
    reason: Optional[str] = None  # Why unhealthy (if applicable)


class HeartbeatMonitor:
    """
    Global heartbeat monitor for all PAS agents

    Features:
    - 60s heartbeat interval tracking
    - 2-miss escalation (agent considered failed after 2 missed heartbeats)
    - Parent-child hierarchy tracking
    - Status aggregation
    - Thread-safe operations
    """

    # Singleton instance
    _instance: Optional['HeartbeatMonitor'] = None
    _lock = threading.Lock()

    # Constants
    HEARTBEAT_INTERVAL_S = 60  # Expected heartbeat interval
    MISS_THRESHOLD = 2  # Number of missed heartbeats before escalation
    MISS_TIMEOUT_S = HEARTBEAT_INTERVAL_S * MISS_THRESHOLD + 30  # 150s grace period

    def __new__(cls):
        """Singleton pattern - only one monitor instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize heartbeat monitor (only once)"""
        if self._initialized:
            return

        self._initialized = True
        self._heartbeats: Dict[str, HeartbeatRecord] = {}  # agent -> latest heartbeat
        self._registry: Dict[str, Dict[str, Any]] = {}  # agent -> metadata
        self._lock = threading.RLock()  # Reentrant lock for nested access

        # Start background checker thread
        self._checker_thread = threading.Thread(target=self._check_health_loop, daemon=True)
        self._checker_thread.start()

    def register_agent(
        self,
        agent: str,
        parent: Optional[str] = None,
        llm_model: Optional[str] = None,
        role: Optional[str] = None,
        tier: Optional[str] = None
    ) -> None:
        """
        Register an agent in the monitoring system

        Args:
            agent: Agent ID (e.g., "Architect", "Dir-Code", "Mgr-Code-01")
            parent: Parent agent ID (e.g., "PAS Root")
            llm_model: LLM model used by this agent
            role: Agent role (architect, director, manager, programmer)
            tier: Agent tier (coordinator, executor)
        """
        with self._lock:
            self._registry[agent] = {
                "parent": parent,
                "llm_model": llm_model,
                "role": role,
                "tier": tier,
                "registered_at": time.time()
            }

    def heartbeat(
        self,
        agent: str,
        run_id: Optional[str] = None,
        state: AgentState = AgentState.IDLE,
        message: str = "",
        llm_model: Optional[str] = None,
        parent_agent: Optional[str] = None,
        children_agents: Optional[List[str]] = None,
        progress: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a heartbeat from an agent

        Args:
            agent: Agent ID
            run_id: Current run ID (if active)
            state: Current agent state
            message: Status message
            llm_model: LLM model being used
            parent_agent: Parent agent ID
            children_agents: List of child agent IDs
            progress: Progress (0.0 to 1.0)
            metadata: Additional metadata
        """
        with self._lock:
            record = HeartbeatRecord(
                agent=agent,
                run_id=run_id,
                timestamp=time.time(),
                state=state,
                message=message,
                llm_model=llm_model,
                parent_agent=parent_agent,
                children_agents=children_agents or [],
                progress=progress,
                metadata=metadata or {}
            )
            self._heartbeats[agent] = record

    def get_health(self, agent: str) -> AgentHealth:
        """
        Get health status for an agent

        Returns:
            AgentHealth object with healthy flag and details
        """
        with self._lock:
            if agent not in self._heartbeats:
                return AgentHealth(
                    agent=agent,
                    healthy=False,
                    last_heartbeat=0,
                    missed_count=0,
                    state=AgentState.IDLE,
                    message="No heartbeat recorded",
                    reason="Agent not initialized"
                )

            record = self._heartbeats[agent]
            now = time.time()
            time_since_last = now - record.timestamp
            missed_count = int(time_since_last / self.HEARTBEAT_INTERVAL_S)

            # Healthy if last heartbeat within timeout
            healthy = time_since_last < self.MISS_TIMEOUT_S
            reason = None if healthy else f"No heartbeat for {time_since_last:.0f}s ({missed_count} missed)"

            return AgentHealth(
                agent=agent,
                healthy=healthy,
                last_heartbeat=record.timestamp,
                missed_count=missed_count,
                state=record.state,
                message=record.message,
                reason=reason
            )

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """Get health status for all registered agents"""
        with self._lock:
            return {agent: self.get_health(agent) for agent in self._heartbeats.keys()}

    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agent IDs"""
        with self._lock:
            return [
                agent
                for agent, health in self.get_all_health().items()
                if not health.healthy
            ]

    def get_hierarchy(self, root: str = "PAS Root") -> Dict[str, Any]:
        """
        Get agent hierarchy tree starting from root

        Returns:
            Nested dict structure showing parent-child relationships
        """
        with self._lock:
            def build_tree(agent: str) -> Dict[str, Any]:
                children = [
                    hb.agent
                    for hb in self._heartbeats.values()
                    if hb.parent_agent == agent
                ]

                health = self.get_health(agent)

                return {
                    "agent": agent,
                    "healthy": health.healthy,
                    "state": health.state.value,
                    "message": health.message,
                    "children": [build_tree(child) for child in children]
                }

            return build_tree(root)

    def get_status_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get executive status summary

        Args:
            run_id: Filter by run ID (if specified)

        Returns:
            Summary dict with overall health, agent counts, progress
        """
        with self._lock:
            # Filter by run_id if specified
            if run_id:
                relevant_agents = [
                    agent
                    for agent, hb in self._heartbeats.items()
                    if hb.run_id == run_id
                ]
            else:
                relevant_agents = list(self._heartbeats.keys())

            all_health = self.get_all_health()

            healthy_count = sum(1 for a in relevant_agents if all_health[a].healthy)
            unhealthy_count = len(relevant_agents) - healthy_count

            # Aggregate progress
            total_progress = sum(
                self._heartbeats[a].progress
                for a in relevant_agents
                if a in self._heartbeats
            )
            avg_progress = total_progress / len(relevant_agents) if relevant_agents else 0.0

            # Count by state
            state_counts = {}
            for agent in relevant_agents:
                state = self._heartbeats[agent].state.value
                state_counts[state] = state_counts.get(state, 0) + 1

            return {
                "total_agents": len(relevant_agents),
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "overall_healthy": unhealthy_count == 0,
                "avg_progress": round(avg_progress, 2),
                "state_counts": state_counts,
                "timestamp": time.time()
            }

    def save_snapshot(self, path: Path) -> None:
        """Save current heartbeat state to JSON file"""
        with self._lock:
            snapshot = {
                "timestamp": time.time(),
                "heartbeats": {
                    agent: asdict(record)
                    for agent, record in self._heartbeats.items()
                },
                "registry": self._registry
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2))

    def load_snapshot(self, path: Path) -> None:
        """Load heartbeat state from JSON file"""
        with self._lock:
            snapshot = json.loads(path.read_text())

            self._heartbeats = {
                agent: HeartbeatRecord(**data)
                for agent, data in snapshot["heartbeats"].items()
            }
            self._registry = snapshot["registry"]

    def _check_health_loop(self) -> None:
        """Background thread to check agent health periodically"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds

                unhealthy = self.get_unhealthy_agents()
                if unhealthy:
                    # Log unhealthy agents (could escalate here)
                    # For now, just track - Directors will handle escalation
                    pass

            except Exception:
                # Silently continue (monitoring thread should never crash)
                pass


# Global singleton instance
_monitor = HeartbeatMonitor()


def get_monitor() -> HeartbeatMonitor:
    """Get global heartbeat monitor instance"""
    return _monitor
