"""
PLMS Portfolio Fair-Share Scheduler

Prevents single ingestion/generation job from hogging capacity.

Key design:
- Round-robin by project_priority
- Enforce concurrency ceilings per lane (Data-Schema ≤2, Code-API ≤4)
- Expose /portfolio/status endpoint
"""

from typing import List, Dict, Any
from collections import defaultdict, deque
from datetime import datetime


# Lane concurrency limits (tune based on resource profiles)
LANE_CONCURRENCY_LIMITS = {
    4200: 4,  # Code-API (lightweight, can parallelize)
    4201: 4,  # Code-Test
    4202: 3,  # Code-Docs
    5100: 2,  # Data-Schema (expensive, heavy DB ops)
    5101: 3,  # Data-Ingest
    5102: 2,  # Data-Transform
    6100: 2,  # Model-Train (GPU-bound)
    6101: 3,  # Model-Eval
}


class PortfolioScheduler:
    """
    Fair-share scheduler with per-lane concurrency caps.

    Maintains:
    - Priority-ordered queue of pending tasks
    - Currently executing tasks per lane
    - Round-robin cursor for fair scheduling
    """

    def __init__(self):
        """Initialize scheduler state."""
        # Priority queues: priority -> deque of tasks
        self.queues: Dict[int, deque] = defaultdict(deque)

        # Currently executing tasks per lane
        self.running_by_lane: Dict[int, List[Dict]] = defaultdict(list)

        # Round-robin cursor
        self.current_priority = 1

    def submit(self, task: Dict[str, Any]):
        """
        Submit task to scheduler.

        Args:
            task: {
                "task_id": int,
                "project_id": int,
                "priority": int (1-5, higher = more important),
                "lane_id": int,
                "estimated_duration_ms": int
            }
        """
        priority = task.get("priority", 3)  # Default: medium priority
        self.queues[priority].append(task)

    def get_next_task(self) -> Dict[str, Any] | None:
        """
        Get next task to execute (round-robin by priority, respects lane caps).

        Returns:
            Task dict or None if no tasks available / all lanes at capacity
        """
        # Try up to 10 priority levels
        for _ in range(10):
            # Round-robin: cycle through priorities
            if self.current_priority not in self.queues or not self.queues[self.current_priority]:
                # Move to next priority
                self.current_priority = (self.current_priority % 5) + 1
                continue

            # Try to pop task from current priority queue
            task = self.queues[self.current_priority].popleft()
            lane_id = task["lane_id"]

            # Check lane concurrency limit
            limit = LANE_CONCURRENCY_LIMITS.get(lane_id, 3)  # Default: 3
            currently_running = len(self.running_by_lane[lane_id])

            if currently_running < limit:
                # Lane has capacity, dispatch task
                self.running_by_lane[lane_id].append(task)
                return task
            else:
                # Lane at capacity, re-queue task and try next priority
                self.queues[self.current_priority].append(task)
                self.current_priority = (self.current_priority % 5) + 1

        # No tasks available or all lanes at capacity
        return None

    def mark_completed(self, task_id: int, lane_id: int):
        """
        Mark task as completed, freeing up lane slot.

        Args:
            task_id: Task ID
            lane_id: Lane ID
        """
        self.running_by_lane[lane_id] = [
            t for t in self.running_by_lane[lane_id] if t["task_id"] != task_id
        ]

    def get_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status.

        Returns:
            {
                "queued": {"priority_1": 5, "priority_2": 3, ...},
                "running_by_lane": {
                    4200: {"running": 3, "limit": 4, "utilization": 0.75},
                    ...
                },
                "total_queued": int,
                "total_running": int
            }
        """
        queued = {
            f"priority_{p}": len(q) for p, q in self.queues.items()
        }

        running_by_lane = {}
        for lane_id, tasks in self.running_by_lane.items():
            limit = LANE_CONCURRENCY_LIMITS.get(lane_id, 3)
            running_by_lane[lane_id] = {
                "running": len(tasks),
                "limit": limit,
                "utilization": len(tasks) / max(1, limit),
                "tasks": [t["task_id"] for t in tasks]
            }

        total_queued = sum(len(q) for q in self.queues.values())
        total_running = sum(len(tasks) for tasks in self.running_by_lane.values())

        return {
            "queued": queued,
            "running_by_lane": running_by_lane,
            "total_queued": total_queued,
            "total_running": total_running,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global scheduler instance
_scheduler: PortfolioScheduler | None = None


def get_scheduler() -> PortfolioScheduler:
    """Get global portfolio scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PortfolioScheduler()
    return _scheduler
