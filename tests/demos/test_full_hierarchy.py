"""
Test Full Agent Hierarchy: 2 Directors, 4 Managers, 8 Programmers

This demo creates a realistic programming task that requires:
- 1 Architect (task decomposition)
- 2 Directors (director-code, director-data)
- 4 Managers (manager-code-impl, manager-code-api, manager-data-schema, manager-narrative)
- 8 Programmers (executor-llm × 4, executor-tool × 4)

Task: Build a REST API with database backend
├── Director-Code
│   ├── Manager-Code-API-Design
│   │   ├── Programmer-1: Design OpenAPI spec
│   │   └── Programmer-2: Generate API docs
│   └── Manager-Code-Impl
│       ├── Programmer-3: Implement endpoints
│       └── Programmer-4: Write unit tests
└── Director-Data
    ├── Manager-Data-Schema
    │   ├── Programmer-5: Design PostgreSQL schema
    │   └── Programmer-6: Create migrations
    └── Manager-Narrative
        ├── Programmer-7: Write README
        └── Programmer-8: Create deployment guide
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List

import requests

# PAS Stub endpoint
PAS_URL = "http://localhost:6200"


# ============================================================================
# Agent Definitions
# ============================================================================


class Agent:
    """Base agent class."""

    def __init__(self, agent_id: str, role: str, name: str):
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.tasks: List[str] = []
        self.children: List["Agent"] = []

    def add_child(self, child: "Agent"):
        """Add a child agent."""
        self.children.append(child)

    def __repr__(self):
        return f"{self.role}({self.name})"


class Architect(Agent):
    """Architect: Expands job card → task graph."""

    def __init__(self):
        super().__init__("architect-001", "Architect", "Chief Architect")

    def decompose_task(self, project_description: str) -> Dict:
        """
        Decompose project into task graph.

        Returns: {
            "directors": [...],
            "managers": [...],
            "programmers": [...],
            "dependencies": {...}
        }
        """
        return {
            "task_graph": {
                "task-1": {
                    "lane": "Code-API-Design",
                    "description": "Design OpenAPI spec",
                    "assigned_to": "programmer-1",
                    "deps": [],
                },
                "task-2": {
                    "lane": "Narrative",
                    "description": "Generate API docs from spec",
                    "assigned_to": "programmer-2",
                    "deps": ["task-1"],
                },
                "task-3": {
                    "lane": "Data-Schema",
                    "description": "Design PostgreSQL schema",
                    "assigned_to": "programmer-5",
                    "deps": [],
                },
                "task-4": {
                    "lane": "Data-Schema",
                    "description": "Create database migrations",
                    "assigned_to": "programmer-6",
                    "deps": ["task-3"],
                },
                "task-5": {
                    "lane": "Code-Impl",
                    "description": "Implement REST endpoints",
                    "assigned_to": "programmer-3",
                    "deps": ["task-1", "task-4"],
                },
                "task-6": {
                    "lane": "Code-Impl",
                    "description": "Write unit tests",
                    "assigned_to": "programmer-4",
                    "deps": ["task-5"],
                },
                "task-7": {
                    "lane": "Narrative",
                    "description": "Write README.md",
                    "assigned_to": "programmer-7",
                    "deps": ["task-2", "task-5"],
                },
                "task-8": {
                    "lane": "Narrative",
                    "description": "Create deployment guide",
                    "assigned_to": "programmer-8",
                    "deps": ["task-6", "task-7"],
                },
            },
            "directors": ["director-code", "director-data"],
            "managers": [
                "manager-code-api",
                "manager-code-impl",
                "manager-data-schema",
                "manager-narrative",
            ],
            "programmers": [
                "programmer-1",
                "programmer-2",
                "programmer-3",
                "programmer-4",
                "programmer-5",
                "programmer-6",
                "programmer-7",
                "programmer-8",
            ],
        }


class Director(Agent):
    """Director: Lane-specific orchestrator."""

    def __init__(self, director_id: str, name: str, lanes: List[str]):
        super().__init__(director_id, "Director", name)
        self.lanes = lanes

    def allocate_tasks(self, task_graph: Dict) -> Dict[str, List[str]]:
        """Allocate tasks to managers based on lanes."""
        allocation = {}
        for task_id, task in task_graph.items():
            if task["lane"] in self.lanes:
                # Find appropriate manager
                if "API" in task["lane"]:
                    manager = "manager-code-api"
                elif "Impl" in task["lane"]:
                    manager = "manager-code-impl"
                elif "Schema" in task["lane"]:
                    manager = "manager-data-schema"
                elif "Narrative" in task["lane"]:
                    manager = "manager-narrative"
                else:
                    manager = "manager-default"

                if manager not in allocation:
                    allocation[manager] = []
                allocation[manager].append(task_id)

        return allocation


class Manager(Agent):
    """Manager: Step-by-step execution, retries, heartbeats."""

    def __init__(self, manager_id: str, name: str):
        super().__init__(manager_id, "Manager", name)

    async def execute_tasks(self, tasks: List[str], task_graph: Dict, run_id: str):
        """Execute assigned tasks in topological order."""
        for task_id in tasks:
            task = task_graph[task_id]
            print(
                f"  [{self.name}] Executing {task_id}: {task['description']} (Lane: {task['lane']})"
            )

            # Submit job card to PAS
            response = requests.post(
                f"{PAS_URL}/pas/v1/jobcards",
                json={
                    "project_id": 1,
                    "run_id": run_id,
                    "lane": task["lane"],
                    "priority": 0.5,
                    "deps": task["deps"],
                    "payload": {"description": task["description"]},
                    "budget_usd": 1.0,
                },
                headers={"Idempotency-Key": f"{run_id}-{task_id}"},
            )
            print(f"    → Task submitted: {response.json()}")


class Programmer(Agent):
    """Programmer: Does the actual work (executor)."""

    def __init__(self, programmer_id: str, name: str, executor_type: str):
        super().__init__(programmer_id, "Programmer", name)
        self.executor_type = executor_type  # "llm" or "tool"

    def execute(self, task_description: str) -> Dict:
        """Execute a single task."""
        print(
            f"    [{self.name}] ({self.executor_type}) Working on: {task_description}"
        )
        time.sleep(0.5)  # Simulate work
        return {"status": "success", "output": f"Completed: {task_description}"}


# ============================================================================
# Test Orchestration
# ============================================================================


async def run_full_hierarchy_test():
    """
    Run complete test: Architect → Directors → Managers → Programmers.
    """
    print("\n" + "=" * 80)
    print("FULL HIERARCHY TEST: 2 Directors, 4 Managers, 8 Programmers")
    print("=" * 80 + "\n")

    # ========================================================================
    # STEP 1: Initialize Agents
    # ========================================================================

    print("STEP 1: Initializing Agents...")

    # Architect
    architect = Architect()
    print(f"  ✓ {architect}")

    # Directors
    director_code = Director("director-code", "Director of Code", ["Code-API-Design", "Code-Impl"])
    director_data = Director("director-data", "Director of Data", ["Data-Schema", "Narrative"])
    print(f"  ✓ {director_code} (Lanes: {director_code.lanes})")
    print(f"  ✓ {director_data} (Lanes: {director_data.lanes})")

    # Managers
    manager_code_api = Manager("manager-code-api", "Code API Manager")
    manager_code_impl = Manager("manager-code-impl", "Code Impl Manager")
    manager_data_schema = Manager("manager-data-schema", "Data Schema Manager")
    manager_narrative = Manager("manager-narrative", "Narrative Manager")

    managers = {
        "manager-code-api": manager_code_api,
        "manager-code-impl": manager_code_impl,
        "manager-data-schema": manager_data_schema,
        "manager-narrative": manager_narrative,
    }
    for m in managers.values():
        print(f"  ✓ {m}")

    # Programmers
    programmers = {
        "programmer-1": Programmer("programmer-1", "OpenAPI Designer", "llm"),
        "programmer-2": Programmer("programmer-2", "Doc Generator", "tool"),
        "programmer-3": Programmer("programmer-3", "Backend Developer", "llm"),
        "programmer-4": Programmer("programmer-4", "Test Writer", "tool"),
        "programmer-5": Programmer("programmer-5", "Schema Designer", "llm"),
        "programmer-6": Programmer("programmer-6", "Migration Builder", "tool"),
        "programmer-7": Programmer("programmer-7", "README Writer", "llm"),
        "programmer-8": Programmer("programmer-8", "Deployment Guide Writer", "tool"),
    }
    for p in programmers.values():
        print(f"  ✓ {p} (Type: {p.executor_type})")

    print()

    # ========================================================================
    # STEP 2: Architect Decomposes Task
    # ========================================================================

    print("STEP 2: Architect Decomposing Task...")
    project_description = "Build a REST API with PostgreSQL backend for user management"
    task_plan = architect.decompose_task(project_description)

    print(f"  ✓ Task graph created: {len(task_plan['task_graph'])} tasks")
    print(f"  ✓ Directors: {task_plan['directors']}")
    print(f"  ✓ Managers: {task_plan['managers']}")
    print(f"  ✓ Programmers: {task_plan['programmers']}")
    print()

    # ========================================================================
    # STEP 3: Directors Allocate Tasks to Managers
    # ========================================================================

    print("STEP 3: Directors Allocating Tasks to Managers...")

    allocation_code = director_code.allocate_tasks(task_plan["task_graph"])
    allocation_data = director_data.allocate_tasks(task_plan["task_graph"])

    print(f"  [{director_code.name}] Allocated tasks:")
    for manager, tasks in allocation_code.items():
        print(f"    → {manager}: {tasks}")

    print(f"  [{director_data.name}] Allocated tasks:")
    for manager, tasks in allocation_data.items():
        print(f"    → {manager}: {tasks}")

    print()

    # ========================================================================
    # STEP 4: Start Run in PAS
    # ========================================================================

    print("STEP 4: Starting PAS Run...")
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    response = requests.post(
        f"{PAS_URL}/pas/v1/runs/start",
        json={
            "project_id": 1,
            "run_id": run_id,
            "run_kind": "baseline",
            "rehearsal_pct": 0.0,
        },
    )
    print(f"  ✓ Run started: {response.json()}")
    print()

    # ========================================================================
    # STEP 5: Managers Execute Tasks
    # ========================================================================

    print("STEP 5: Managers Executing Tasks...")

    # Combine allocations
    all_allocations = {**allocation_code, **allocation_data}

    for manager_id, task_ids in all_allocations.items():
        manager = managers[manager_id]
        print(f"\n[{manager.name}] Starting execution...")
        await manager.execute_tasks(task_ids, task_plan["task_graph"], run_id)

    print()

    # ========================================================================
    # STEP 6: Wait for PAS Execution (synthetic)
    # ========================================================================

    print("STEP 6: Waiting for PAS execution to complete...")
    print("  (PAS stub is executing tasks in background...)")
    await asyncio.sleep(10)  # Give PAS time to process

    # ========================================================================
    # STEP 7: Check Run Status
    # ========================================================================

    print("\nSTEP 7: Checking Run Status...")
    response = requests.get(f"{PAS_URL}/pas/v1/runs/status", params={"run_id": run_id})
    status = response.json()

    print(f"\n  Run Status: {status['status']}")
    print(f"  Tasks Total: {status['tasks_total']}")
    print(f"  Tasks Completed: {status['tasks_completed']}")
    print(f"  Tasks Failed: {status['tasks_failed']}")
    print(f"  Total Spend: ${status['spend_usd']:.2f}")
    print(f"  Energy Used: {status['spend_energy_kwh']:.3f} kWh")
    print(f"  Runway: {status['runway_minutes']} minutes")

    if status["kpi_violations"]:
        print(f"\n  ⚠️  KPI Violations: {len(status['kpi_violations'])}")
        for violation in status["kpi_violations"][:3]:  # Show first 3
            print(f"    - {violation['kpi_name']}: {violation['actual']} (expected: {violation['expected']})")
    else:
        print("\n  ✓ All KPIs passed!")

    print("\n  Task Details:")
    for task in status["tasks"]:
        emoji = "✓" if task["status"] == "succeeded" else "✗" if task["status"] == "failed" else "⏳"
        print(f"    {emoji} {task['task_id']} ({task['lane']}): {task['status']}")

    print()

    # ========================================================================
    # STEP 8: Visualize Hierarchy
    # ========================================================================

    print("STEP 8: Agent Hierarchy Visualization")
    print("\n" + "=" * 80)
    print("AGENT HIERARCHY TREE")
    print("=" * 80)
    print()
    print("Architect (Chief Architect)")
    print("├── Director-Code (Director of Code)")
    print("│   ├── Manager-Code-API-Design (Code API Manager)")
    print("│   │   ├── Programmer-1 (OpenAPI Designer) [llm]")
    print("│   │   └── Programmer-2 (Doc Generator) [tool]")
    print("│   └── Manager-Code-Impl (Code Impl Manager)")
    print("│       ├── Programmer-3 (Backend Developer) [llm]")
    print("│       └── Programmer-4 (Test Writer) [tool]")
    print("└── Director-Data (Director of Data)")
    print("    ├── Manager-Data-Schema (Data Schema Manager)")
    print("    │   ├── Programmer-5 (Schema Designer) [llm]")
    print("    │   └── Programmer-6 (Migration Builder) [tool]")
    print("    └── Manager-Narrative (Narrative Manager)")
    print("        ├── Programmer-7 (README Writer) [llm]")
    print("        └── Programmer-8 (Deployment Guide Writer) [tool]")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"  Total Agents: 15")
    print(f"    - 1 Architect")
    print(f"    - 2 Directors")
    print(f"    - 4 Managers")
    print(f"    - 8 Programmers (4 LLM + 4 Tool)")
    print()
    print(f"  Tasks Executed: {status['tasks_total']}")
    print(f"  Success Rate: {(status['tasks_completed'] - status['tasks_failed']) / status['tasks_total'] * 100:.1f}%")
    print(f"  Total Cost: ${status['spend_usd']:.2f}")
    print(f"  Total Energy: {status['spend_energy_kwh']:.3f} kWh")
    print()
    print("  ✅ Full hierarchy test complete!")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Check if PAS stub is running
    try:
        response = requests.get(f"{PAS_URL}/health", timeout=2)
        print(f"✓ PAS stub is running: {response.json()}")
    except requests.exceptions.RequestException:
        print(f"✗ ERROR: PAS stub not running on {PAS_URL}")
        print("  Start it with: ./.venv/bin/python services/pas/stub/app.py")
        exit(1)

    # Run test
    asyncio.run(run_full_hierarchy_test())
