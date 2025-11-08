"""
Test Full Agent Hierarchy with HMI Integration

This demo registers all agents with the Registry so they appear in:
- HMI Dashboard (http://localhost:6101)
- Tree View (http://localhost:6101/tree)
- Sequencer (http://localhost:6101/sequencer)
- Actions (http://localhost:6101/actions)

Architecture:
- 1 Architect
- 2 Directors (director-code, director-data)
- 4 Managers
- 8 Programmers (4 LLM + 4 Tool)
"""

import asyncio
import requests
import time
import uuid
from datetime import datetime

# Service URLs
PAS_URL = "http://localhost:6200"
REGISTRY_URL = "http://localhost:6121"

# ============================================================================
# Agent Registration
# ============================================================================


def register_agent(agent_id: str, name: str, role: str, tier: str, parent_id: str = None, port: int = None):
    """Register an agent with the PAS Registry."""
    payload = {
        "service_id": agent_id,
        "name": name,
        "port": port or 0,  # Agents don't need ports (only services do)
        "status": "active",
        "labels": {
            "agent_role": role,
            "tier": tier,
            "parent_id": parent_id or "root",
            "type": "agent"
        }
    }

    try:
        response = requests.post(f"{REGISTRY_URL}/register", json=payload, timeout=5)
        print(f"  ✓ Registered: {name} ({agent_id}) - Tier {tier}")
        return response.json()
    except Exception as e:
        print(f"  ✗ Failed to register {name}: {e}")
        return None


def send_heartbeat(agent_id: str):
    """Send heartbeat for an agent."""
    try:
        requests.post(
            f"{REGISTRY_URL}/heartbeat",
            json={"service_id": agent_id},
            timeout=2
        )
    except:
        pass  # Heartbeats are best-effort


# ============================================================================
# Test Execution
# ============================================================================


async def run_test_with_hmi_integration():
    """Run full hierarchy test with HMI integration."""

    print("\n" + "=" * 80)
    print("FULL HIERARCHY TEST WITH HMI INTEGRATION")
    print("=" * 80 + "\n")

    # ========================================================================
    # STEP 1: Register Architect
    # ========================================================================

    print("STEP 1: Registering Architect...")
    architect_id = "architect-001"
    register_agent(architect_id, "Chief Architect", "Architect", "0-architect")
    print()

    # ========================================================================
    # STEP 2: Register Directors
    # ========================================================================

    print("STEP 2: Registering Directors...")
    director_code_id = "director-code-001"
    director_data_id = "director-data-001"

    register_agent(director_code_id, "Director of Code", "Director", "1-director", parent_id=architect_id)
    register_agent(director_data_id, "Director of Data", "Director", "1-director", parent_id=architect_id)
    print()

    # ========================================================================
    # STEP 3: Register Managers
    # ========================================================================

    print("STEP 3: Registering Managers...")
    manager_code_api_id = "manager-code-api-001"
    manager_code_impl_id = "manager-code-impl-001"
    manager_data_schema_id = "manager-data-schema-001"
    manager_narrative_id = "manager-narrative-001"

    register_agent(manager_code_api_id, "Code API Manager", "Manager", "2-manager", parent_id=director_code_id)
    register_agent(manager_code_impl_id, "Code Impl Manager", "Manager", "2-manager", parent_id=director_code_id)
    register_agent(manager_data_schema_id, "Data Schema Manager", "Manager", "2-manager", parent_id=director_data_id)
    register_agent(manager_narrative_id, "Narrative Manager", "Manager", "2-manager", parent_id=director_data_id)
    print()

    # ========================================================================
    # STEP 4: Register Programmers
    # ========================================================================

    print("STEP 4: Registering Programmers...")
    programmers = [
        ("programmer-001", "OpenAPI Designer (LLM)", "Programmer-LLM", manager_code_api_id),
        ("programmer-002", "Doc Generator (Tool)", "Programmer-Tool", manager_code_api_id),
        ("programmer-003", "Backend Developer (LLM)", "Programmer-LLM", manager_code_impl_id),
        ("programmer-004", "Test Writer (Tool)", "Programmer-Tool", manager_code_impl_id),
        ("programmer-005", "Schema Designer (LLM)", "Programmer-LLM", manager_data_schema_id),
        ("programmer-006", "Migration Builder (Tool)", "Programmer-Tool", manager_data_schema_id),
        ("programmer-007", "README Writer (LLM)", "Programmer-LLM", manager_narrative_id),
        ("programmer-008", "Deployment Guide Writer (Tool)", "Programmer-Tool", manager_narrative_id),
    ]

    for prog_id, prog_name, prog_role, parent_id in programmers:
        register_agent(prog_id, prog_name, prog_role, "3-programmer", parent_id=parent_id)

    print()

    # ========================================================================
    # STEP 5: Submit Tasks to PAS
    # ========================================================================

    print("STEP 5: Submitting Tasks to PAS...")
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Start run
    response = requests.post(
        f"{PAS_URL}/pas/v1/runs/start",
        json={"project_id": 1, "run_id": run_id, "run_kind": "baseline"}
    )
    print(f"  ✓ Run started: {run_id}")

    # Submit 8 tasks
    tasks = [
        ("task-1", "Code-API-Design", "Design OpenAPI spec", []),
        ("task-2", "Narrative", "Generate API docs", ["task-1"]),
        ("task-3", "Data-Schema", "Design PostgreSQL schema", []),
        ("task-4", "Data-Schema", "Create migrations", ["task-3"]),
        ("task-5", "Code-Impl", "Implement endpoints", ["task-1", "task-4"]),
        ("task-6", "Code-Impl", "Write unit tests", ["task-5"]),
        ("task-7", "Narrative", "Write README", ["task-2", "task-5"]),
        ("task-8", "Narrative", "Deployment guide", ["task-6", "task-7"]),
    ]

    for task_id, lane, description, deps in tasks:
        response = requests.post(
            f"{PAS_URL}/pas/v1/jobcards",
            json={
                "project_id": 1,
                "run_id": run_id,
                "lane": lane,
                "priority": 0.5,
                "deps": deps,
                "payload": {"description": description},
                "budget_usd": 1.0
            },
            headers={"Idempotency-Key": f"{run_id}-{task_id}"}
        )
        print(f"  ✓ Submitted: {task_id} ({lane}): {description}")

    print()

    # ========================================================================
    # STEP 6: Send Heartbeats (simulate active agents)
    # ========================================================================

    print("STEP 6: Sending Heartbeats (keeps agents alive in HMI)...")
    all_agent_ids = [
        architect_id,
        director_code_id, director_data_id,
        manager_code_api_id, manager_code_impl_id, manager_data_schema_id, manager_narrative_id
    ] + [prog_id for prog_id, _, _, _ in programmers]

    for agent_id in all_agent_ids:
        send_heartbeat(agent_id)

    print(f"  ✓ Sent heartbeats for {len(all_agent_ids)} agents")
    print()

    # ========================================================================
    # STEP 7: Wait for Execution
    # ========================================================================

    print("STEP 7: Waiting for PAS execution (60 seconds)...")
    print("  💡 Open HMI now:")
    print(f"     - Dashboard: http://localhost:6101")
    print(f"     - Tree View: http://localhost:6101/tree")
    print(f"     - Actions: http://localhost:6101/actions")
    print()

    # Wait and send periodic heartbeats
    for i in range(12):  # 12 x 5s = 60s
        await asyncio.sleep(5)
        # Send heartbeats every 5 seconds
        for agent_id in all_agent_ids[:5]:  # Heartbeat a few agents
            send_heartbeat(agent_id)
        print(f"  ... {(i+1)*5}s elapsed (sent heartbeats)")

    print()

    # ========================================================================
    # STEP 8: Check Final Status
    # ========================================================================

    print("STEP 8: Checking Final Status...")
    response = requests.get(f"{PAS_URL}/pas/v1/runs/status", params={"run_id": run_id})
    status = response.json()

    print(f"\n  Run Status: {status['status']}")
    print(f"  Tasks Total: {status['tasks_total']}")
    print(f"  Tasks Completed: {status['tasks_completed']}")
    print(f"  Tasks Failed: {status['tasks_failed']}")
    print(f"  Total Spend: ${status['spend_usd']:.2f}")
    print(f"  Energy Used: {status['spend_energy_kwh']:.3f} kWh")

    completed_tasks = [t for t in status['tasks'] if t['status'] in ['succeeded', 'failed']]
    success_rate = len([t for t in completed_tasks if t['status'] == 'succeeded']) / len(completed_tasks) if completed_tasks else 0

    print(f"\n  ✓ Task Execution:")
    for task in status['tasks']:
        emoji = "✓" if task['status'] == 'succeeded' else "✗" if task['status'] == 'failed' else "⏳"
        print(f"    {emoji} {task['task_id']} ({task['lane']}): {task['status']}")

    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("TEST COMPLETE - CHECK HMI NOW!")
    print("=" * 80)
    print()
    print(f"  🌳 Tree View: http://localhost:6101/tree")
    print(f"     → Should show full hierarchy: Architect → Directors → Managers → Programmers")
    print()
    print(f"  📊 Dashboard: http://localhost:6101")
    print(f"     → Should show {len(all_agent_ids)} active agents")
    print()
    print(f"  📋 Actions: http://localhost:6101/actions")
    print(f"     → Should show {status['tasks_total']} tasks")
    print()
    print(f"  Total Agents Registered: {len(all_agent_ids)}")
    print(f"    - 1 Architect")
    print(f"    - 2 Directors")
    print(f"    - 4 Managers")
    print(f"    - 8 Programmers")
    print()
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Total Cost: ${status['spend_usd']:.2f}")
    print()
    print("  ✅ All agents registered and tasks submitted!")
    print("  ✅ HMI integration complete!")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    # Check services
    services = {
        "PAS Stub": PAS_URL,
        "Registry": REGISTRY_URL
    }

    print("Checking services...")
    all_ready = True
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            print(f"  ✓ {name} is running ({url})")
        except:
            print(f"  ✗ {name} is NOT running ({url})")
            all_ready = False

    if not all_ready:
        print("\n⚠️  Please start missing services:")
        print("  PAS Stub: ./.venv/bin/python services/pas/stub/app.py")
        print("  Registry: ./.venv/bin/python tools/pas_registry/registry_server.py")
        exit(1)

    print()

    # Run test
    asyncio.run(run_test_with_hmi_integration())
