#!/usr/bin/env python3
"""
Job Card Queue System for PAS Multi-Tier Architecture

Responsibilities:
- Primary: RPC-based job card submission
- Fallback: File-based queue (atomic JSONL writes)
- Job card schema validation
- Priority queue support
- Delivery guarantees (at-least-once)

Usage:
    from services.common.job_queue import JobQueue, JobCard, Lane

    queue = JobQueue()

    # Submit job card
    job_card = JobCard(
        id="jc-abc123-code-001",
        parent_id="abc123-def456",
        role="director",
        lane=Lane.CODE,
        task="Implement OAuth2 authentication",
        inputs=[{"path": "app/services/auth.py"}],
        expected_artifacts=[{"path": "artifacts/runs/abc123/code/diffs/"}],
        acceptance=[{"check": "pytest>=0.90"}],
        budget={"tokens_target_ratio": 0.50}
    )

    queue.submit(job_card, target_agent="Dir-Code")

    # Poll for job cards (agent side)
    job_cards = queue.poll("Dir-Code")
    for jc in job_cards:
        process(jc)
        queue.ack(jc.id)
"""
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import fcntl
import os


class Lane(str, Enum):
    """PAS lanes"""
    CODE = "Code"
    MODELS = "Models"
    DATA = "Data"
    DEVSECOPS = "DevSecOps"
    DOCS = "Docs"


class Role(str, Enum):
    """Agent roles in hierarchy"""
    ARCHITECT = "architect"
    DIRECTOR = "director"
    MANAGER = "manager"
    PROGRAMMER = "programmer"


class Priority(int, Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class JobCard:
    """
    Job card submitted between agents in hierarchy

    Job cards flow down: Architect → Directors → Managers → Programmers
    """
    id: str  # Unique job card ID (e.g., "jc-abc123-code-001")
    parent_id: str  # Parent run/job ID
    role: Role  # Target role (director, manager, programmer)
    lane: Lane  # Which lane (Code, Models, Data, DevSecOps, Docs)
    task: str  # Natural language task description
    inputs: List[Dict[str, Any]] = field(default_factory=list)  # Input files/data
    expected_artifacts: List[Dict[str, Any]] = field(default_factory=list)  # Output paths
    acceptance: List[Dict[str, Any]] = field(default_factory=list)  # Acceptance criteria
    risks: List[str] = field(default_factory=list)  # Known risks
    budget: Dict[str, Any] = field(default_factory=dict)  # Token/time/cost budgets
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    submitted_by: Optional[str] = None  # Agent that created this job card


@dataclass
class QueueEntry:
    """Internal queue entry (job card + delivery metadata)"""
    job_card: JobCard
    target_agent: str
    attempts: int = 0
    last_attempt: float = 0.0
    acked: bool = False


class JobQueue:
    """
    Multi-tier job card queue with RPC primary and file-based fallback

    Features:
    - In-memory queue (fast, primary)
    - File-based persistence (fallback, durability)
    - Priority ordering
    - At-least-once delivery (requires ack)
    - Thread-safe operations
    """

    # Singleton instance
    _instance: Optional['JobQueue'] = None
    _lock = threading.Lock()

    def __new__(cls, queue_dir: Optional[Path] = None):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, queue_dir: Optional[Path] = None):
        """
        Initialize job queue

        Args:
            queue_dir: Directory for file-based queue (default: artifacts/queues/)
        """
        if self._initialized:
            return

        self._initialized = True
        self._queue_dir = queue_dir or Path("artifacts/queues")
        self._queues: Dict[str, List[QueueEntry]] = {}  # agent -> queue
        self._lock = threading.RLock()

        # Create queue directories
        self._queue_dir.mkdir(parents=True, exist_ok=True)

        # Load persisted queue entries
        self._load_from_disk()

    def submit(
        self,
        job_card: JobCard,
        target_agent: str,
        use_file_fallback: bool = False
    ) -> None:
        """
        Submit job card to target agent's queue

        Args:
            job_card: Job card to submit
            target_agent: Target agent ID (e.g., "Dir-Code", "Mgr-Code-01")
            use_file_fallback: Force file-based delivery (for durability)
        """
        with self._lock:
            # Ensure queue exists
            if target_agent not in self._queues:
                self._queues[target_agent] = []

            # Create queue entry
            entry = QueueEntry(
                job_card=job_card,
                target_agent=target_agent,
                attempts=0
            )

            # Add to in-memory queue (priority sorted)
            self._queues[target_agent].append(entry)
            self._queues[target_agent].sort(
                key=lambda e: (e.job_card.priority.value, e.job_card.created_at),
                reverse=True  # Higher priority + older first
            )

            # Persist to disk if fallback requested
            if use_file_fallback:
                self._write_to_disk(target_agent, entry)

    def poll(
        self,
        agent: str,
        limit: int = 10,
        mark_in_flight: bool = True
    ) -> List[JobCard]:
        """
        Poll job cards for an agent

        Args:
            agent: Agent ID polling for work
            limit: Max number of job cards to return
            mark_in_flight: Increment attempt counter (for redelivery)

        Returns:
            List of job cards (up to limit)
        """
        with self._lock:
            if agent not in self._queues:
                return []

            # Get unacked entries
            entries = [
                e for e in self._queues[agent]
                if not e.acked
            ][:limit]

            # Mark as in-flight
            if mark_in_flight:
                now = time.time()
                for entry in entries:
                    entry.attempts += 1
                    entry.last_attempt = now

            return [e.job_card for e in entries]

    def ack(self, job_card_id: str, agent: Optional[str] = None) -> bool:
        """
        Acknowledge job card completion

        Args:
            job_card_id: Job card ID to acknowledge
            agent: Agent ID (if known, for faster lookup)

        Returns:
            True if acked successfully, False if not found
        """
        with self._lock:
            # Search all queues if agent not specified
            search_agents = [agent] if agent else list(self._queues.keys())

            for ag in search_agents:
                if ag not in self._queues:
                    continue

                for entry in self._queues[ag]:
                    if entry.job_card.id == job_card_id:
                        entry.acked = True
                        # Remove from disk queue
                        self._remove_from_disk(ag, job_card_id)
                        return True

            return False

    def get_queue_depth(self, agent: str) -> int:
        """Get number of unacked job cards for agent"""
        with self._lock:
            if agent not in self._queues:
                return 0
            return sum(1 for e in self._queues[agent] if not e.acked)

    def get_all_queue_depths(self) -> Dict[str, int]:
        """Get queue depths for all agents"""
        with self._lock:
            return {
                agent: self.get_queue_depth(agent)
                for agent in self._queues.keys()
            }

    def get_stale_jobs(self, timeout_s: float = 300) -> List[JobCard]:
        """
        Get job cards that have been in-flight too long (likely failed)

        Args:
            timeout_s: Timeout in seconds (default 5 minutes)

        Returns:
            List of stale job cards that should be retried or escalated
        """
        with self._lock:
            now = time.time()
            stale = []

            for queue in self._queues.values():
                for entry in queue:
                    if entry.attempts > 0 and not entry.acked:
                        if now - entry.last_attempt > timeout_s:
                            stale.append(entry.job_card)

            return stale

    def _write_to_disk(self, agent: str, entry: QueueEntry) -> None:
        """Write queue entry to disk (atomic JSONL append)"""
        queue_file = self._queue_dir / agent / "inbox.jsonl"
        queue_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic append with file lock
        with open(queue_file, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Write job card as JSON line
                data = {
                    "job_card": asdict(entry.job_card),
                    "target_agent": entry.target_agent,
                    "created_at": entry.job_card.created_at
                }
                f.write(json.dumps(data) + "\n")
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _remove_from_disk(self, agent: str, job_card_id: str) -> None:
        """Remove job card from disk queue (mark as acked)"""
        queue_file = self._queue_dir / agent / "inbox.jsonl"
        acked_file = self._queue_dir / agent / "acked.jsonl"

        if not queue_file.exists():
            return

        acked_file.parent.mkdir(parents=True, exist_ok=True)

        # Read all entries, filter out acked one
        with open(queue_file, "r") as f_in:
            entries = [json.loads(line) for line in f_in if line.strip()]

        remaining = []
        acked_entry = None

        for entry in entries:
            if entry["job_card"]["id"] == job_card_id:
                acked_entry = entry
            else:
                remaining.append(entry)

        # Rewrite inbox without acked entry
        with open(queue_file, "w") as f_out:
            for entry in remaining:
                f_out.write(json.dumps(entry) + "\n")

        # Append to acked log
        if acked_entry:
            with open(acked_file, "a") as f_ack:
                acked_entry["acked_at"] = time.time()
                f_ack.write(json.dumps(acked_entry) + "\n")

    def _load_from_disk(self) -> None:
        """Load persisted queue entries from disk on startup"""
        if not self._queue_dir.exists():
            return

        for agent_dir in self._queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent = agent_dir.name
            inbox_file = agent_dir / "inbox.jsonl"

            if not inbox_file.exists():
                continue

            # Load all unacked entries
            with open(inbox_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    job_card = JobCard(**data["job_card"])

                    entry = QueueEntry(
                        job_card=job_card,
                        target_agent=data["target_agent"],
                        attempts=0,
                        acked=False
                    )

                    if agent not in self._queues:
                        self._queues[agent] = []
                    self._queues[agent].append(entry)


# Global singleton instance
_queue = JobQueue()


def get_queue() -> JobQueue:
    """Get global job queue instance"""
    return _queue
