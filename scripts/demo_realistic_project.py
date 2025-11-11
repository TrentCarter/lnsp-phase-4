#!/usr/bin/env python3
"""
Realistic PAS Demo: Add Authentication to E-Commerce Platform
Demonstrates the full hierarchy with proper comms logging
"""

import os
import sys
import time
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.common.comms_logger import get_logger

# Configuration
REGISTRY_DB = PROJECT_ROOT / "artifacts" / "registry" / "registry.db"
DEMO_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs"

# Initialize comms logger
logger = get_logger()

# Task definition
TASK_ID = f"demo-auth-{int(time.time())}"
RUN_ID = TASK_ID
PRIME_DIRECTIVE = "Add authentication system to e-commerce platform"

# Agent hierarchy
AGENTS = {
    "Gateway": {"tier": -1, "name": "Gateway"},
    "PAS Root": {"tier": 0, "name": "PAS Root"},
    "Architect": {"tier": 0, "name": "Architect"},
    "Dir-Code": {"tier": 1, "name": "Director of Code"},
    "Dir-DevSecOps": {"tier": 1, "name": "Director of DevSecOps"},
    "Mgr-Code-01": {"tier": 2, "name": "Code Manager 01"},
    "Mgr-Code-02": {"tier": 2, "name": "Code Manager 02"},
    "Prog-Qwen-001": {"tier": 3, "name": "Programmer Qwen 001"},
    "Prog-Qwen-002": {"tier": 3, "name": "Programmer Qwen 002"},
    "Prog-Qwen-003": {"tier": 3, "name": "Programmer Qwen 003"},
}

LLM_MODEL = "ollama/qwen2.5-coder:7b-instruct"


def register_agents():
    """Register all agents in the registry"""
    conn = sqlite3.connect(REGISTRY_DB)
    cursor = conn.cursor()

    # Services table should already exist, just insert/update agents
    for service_id, info in AGENTS.items():
        cursor.execute("""
            INSERT OR REPLACE INTO services
            (service_id, name, type, role, url, caps, labels, heartbeat_interval_s, ttl_s, last_heartbeat_ts, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            service_id,
            info["name"],
            "agent",
            "production",
            f"http://localhost:9999/{service_id}",
            json.dumps(["code", "delegate", "plan"]),
            json.dumps({"tier": str(info["tier"])}),
            60,
            90,
            datetime.now(timezone.utc).isoformat(),
            "ok"
        ))

    conn.commit()
    conn.close()
    print(f"✅ Registered {len(AGENTS)} agents")


def run_demo():
    """Execute the demo with realistic task flow"""
    print("\n" + "=" * 80)
    print(f"🎯 REALISTIC DEMO: {PRIME_DIRECTIVE}")
    print("=" * 80)
    print(f"Task ID: {TASK_ID}")
    print(f"Run ID: {RUN_ID}")
    print()

    # Register agents
    register_agents()

    # Create run directory
    run_dir = DEMO_OUTPUT_DIR / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log START separator
    logger.log_separator(label=TASK_ID, run_id=RUN_ID)

    # Phase 1: User → Gateway → PAS Root (Prime Directive Submission)
    print("\n📋 Phase 1: Prime Directive Submission")
    time.sleep(0.5)

    logger.log_cmd(
        from_agent="Gateway",
        to_agent="PAS Root",
        message=f"Submit Prime Directive: {PRIME_DIRECTIVE}",
        run_id=RUN_ID
    )
    time.sleep(0.3)

    logger.log_response(
        from_agent="PAS Root",
        to_agent="Gateway",
        message=f"Queued: {PRIME_DIRECTIVE}",
        run_id=RUN_ID,
        status="queued"
    )
    time.sleep(0.2)

    logger.log_response(
        from_agent="PAS Root",
        to_agent="Gateway",
        message=f"Started execution: {PRIME_DIRECTIVE}",
        run_id=RUN_ID,
        status="running"
    )
    time.sleep(0.5)

    # Phase 2: PAS Root → Architect (Delegation)
    print("📋 Phase 2: Architect Analysis")

    logger.log_cmd(
        from_agent="PAS Root",
        to_agent="Architect",
        message=f"Analyze and delegate Prime Directive: {PRIME_DIRECTIVE}",
        llm_model=LLM_MODEL,
        run_id=RUN_ID
    )
    time.sleep(0.5)

    # Architect heartbeats
    for i in range(3):
        logger.log_heartbeat(
            from_agent="Architect",
            to_agent="PAS Root",
            message=f"Analyzing project structure ({i+1}/3)",
            llm_model=LLM_MODEL,
            run_id=RUN_ID,
            status="running",
            progress=0.3 + (i * 0.2)
        )
        time.sleep(0.3)

    logger.log_response(
        from_agent="Architect",
        to_agent="PAS Root",
        message="Analysis complete: Breaking down into 8 tasks (5 new files, 3 modifications)",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="completed",
        metadata={"tasks": 8, "new_files": 5, "modified_files": 3}
    )
    time.sleep(0.5)

    # Phase 3: Architect → Directors (Delegation)
    print("📋 Phase 3: Director Delegation")

    # Code Director
    logger.log_cmd(
        from_agent="Architect",
        to_agent="Dir-Code",
        message=f"Delegate to Code Director: {PRIME_DIRECTIVE} (8 files)",
        llm_model=LLM_MODEL,
        run_id=RUN_ID
    )
    time.sleep(0.3)

    logger.log_response(
        from_agent="Dir-Code",
        to_agent="Architect",
        message="Received: Will delegate to Code Manager for implementation",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="running"
    )
    time.sleep(0.3)

    # Phase 4: Directors → Managers (Delegation)
    print("📋 Phase 4: Manager Assignment")

    logger.log_cmd(
        from_agent="Dir-Code",
        to_agent="Mgr-Code-01",
        message="Assign to Code Manager: Implement User Authentication (8 files)",
        llm_model=LLM_MODEL,
        run_id=RUN_ID
    )
    time.sleep(0.3)

    logger.log_response(
        from_agent="Mgr-Code-01",
        to_agent="Dir-Code",
        message="Accepted: Will execute via Programmer agent",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="running"
    )
    time.sleep(0.5)

    # Phase 5: Manager → Programmers (Task Execution)
    print("📋 Phase 5: Programmer Execution")

    tasks = [
        {
            "programmer": "Prog-Qwen-001",
            "files": ["models/user.py", "models/session.py"],
            "description": "Create auth models, routes, middleware, tests, docs"
        },
        {
            "programmer": "Prog-Qwen-002",
            "files": ["routes/auth.py", "middleware/jwt.py"],
            "description": "Implement JWT authentication and password hashing"
        },
        {
            "programmer": "Prog-Qwen-003",
            "files": ["tests/test_auth.py", "docs/auth_api.md"],
            "description": "Write comprehensive tests and API documentation"
        }
    ]

    for task in tasks:
        programmer = task["programmer"]

        # Manager → Programmer delegation
        logger.log_cmd(
            from_agent="Mgr-Code-01",
            to_agent=programmer,
            message=f"Execute Prime Directive: {task['description']}",
            llm_model=LLM_MODEL,
            run_id=RUN_ID
        )
        time.sleep(0.5)

        # Programmer execution with heartbeats
        total_files = len(task["files"])
        for idx, file in enumerate(task["files"]):
            progress = (idx + 1) / total_files

            logger.log_heartbeat(
                from_agent=programmer,
                to_agent="Mgr-Code-01",
                message=f"Generating {file} ({idx + 1}/{total_files})",
                llm_model=LLM_MODEL,
                run_id=RUN_ID,
                status="running",
                progress=progress,
                metadata={"current_file": file, "files_done": idx + 1, "total_files": total_files}
            )
            time.sleep(0.8)

        # Programmer completion
        logger.log_response(
            from_agent=programmer,
            to_agent="Mgr-Code-01",
            message=f"Completed: Generated {total_files} files successfully",
            llm_model=LLM_MODEL,
            run_id=RUN_ID,
            status="completed",
            metadata={"files_created": total_files, "files": task["files"]}
        )
        time.sleep(0.5)

    # Phase 6: Completion Reporting (Bottom-Up)
    print("📋 Phase 6: Completion Reporting")

    # Manager → Director
    logger.log_response(
        from_agent="Mgr-Code-01",
        to_agent="Dir-Code",
        message="All programmer tasks completed successfully",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="completed",
        metadata={"programmers_completed": 3, "total_files": 8}
    )
    time.sleep(0.5)

    # Director → Architect
    logger.log_response(
        from_agent="Dir-Code",
        to_agent="Architect",
        message="Code division completed: All 8 files generated",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="completed"
    )
    time.sleep(0.5)

    # Architect → PAS Root
    logger.log_response(
        from_agent="Architect",
        to_agent="PAS Root",
        message="Prime Directive execution complete",
        llm_model=LLM_MODEL,
        run_id=RUN_ID,
        status="completed",
        metadata={"total_tasks": 8, "success": True}
    )
    time.sleep(0.5)

    # PAS Root → Gateway (Final notification)
    logger.log_response(
        from_agent="PAS Root",
        to_agent="Gateway",
        message=f"Completed: {PRIME_DIRECTIVE}",
        run_id=RUN_ID,
        status="completed",
        metadata={"execution_time": 40, "agents_used": 7, "files_generated": 8}
    )

    # Log END separator
    logger.log_separator(label=TASK_ID, run_id=RUN_ID)

    # Summary
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Task ID: {TASK_ID}")
    print(f"Agents involved: {len(AGENTS)}")
    print(f"Files simulated: 8")
    print(f"Execution time: ~40 seconds")
    print(f"\n📊 View results:")
    print(f"   - Sequencer: http://localhost:6100/sequencer?task_id={TASK_ID}")
    print(f"   - Parse logs: ./tools/parse_comms_log.py --run-id {RUN_ID}")
    print()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
