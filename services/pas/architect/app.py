#!/usr/bin/env python3
"""
Architect Service - Top-Level PAS Coordinator

Port: 6110
LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)

Responsibilities:
- Receive Prime Directives from PAS Root
- Decompose into lane-specific job cards
- Allocate resources via Resource Manager
- Monitor Directors
- Aggregate status and validate acceptance
- Generate executive summaries

Contract: docs/contracts/ARCHITECT_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import httpx
import json
import uuid
import asyncio
from dataclasses import asdict

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.job_queue import get_queue, JobCard, Lane, Role, Priority
from services.common.comms_logger import get_logger, MessageType
from services.common.title_generator import generate_short_title_async

# Architect-specific imports
from services.pas.architect.decomposer import TaskDecomposer


app = FastAPI(title="Architect", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
job_queue = get_queue()
logger = get_logger()

# Register Architect agent
heartbeat_monitor.register_agent(
    agent="Architect",
    parent="PAS Root",
    llm_model=os.getenv("ARCHITECT_LLM", "anthropic/claude-sonnet-4-5"),
    role="architect",
    tier="coordinator"
)

# Task decomposer (LLM-powered)
decomposer = TaskDecomposer()

# Director endpoints (ports 6111-6115)
DIRECTOR_ENDPOINTS = {
    "Dir-Code": os.getenv("DIR_CODE_URL", "http://127.0.0.1:6111"),
    "Dir-Models": os.getenv("DIR_MODELS_URL", "http://127.0.0.1:6112"),
    "Dir-Data": os.getenv("DIR_DATA_URL", "http://127.0.0.1:6113"),
    "Dir-DevSecOps": os.getenv("DIR_DEVSECOPS_URL", "http://127.0.0.1:6114"),
    "Dir-Docs": os.getenv("DIR_DOCS_URL", "http://127.0.0.1:6115"),
}

# In-memory run tracking (will move to DB in Phase 2)
RUNS: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class PrimeDirective(BaseModel):
    """Prime Directive from PAS Root"""
    run_id: str
    prd: str  # Full PRD text
    title: str  # User-provided title
    entry_files: List[str] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)  # tokens_max, duration_max_mins, cost_usd_max
    policy: Dict[str, Any] = Field(default_factory=dict)  # require_cross_vendor_review, protected_paths
    approval_mode: str = "auto"  # auto | human


class ArchitectPlan(BaseModel):
    """Architect plan output"""
    run_id: str
    executive_summary: str
    lane_allocations: Dict[str, Dict[str, Any]]  # lane -> job card dict
    dependency_graph: str  # Graphviz DOT format
    resource_reservations: Dict[str, Any]
    acceptance_gates: Dict[str, List[str]]  # lane -> list of checks
    created_at: float = Field(default_factory=time.time)


class RunStatus(BaseModel):
    """Run status"""
    run_id: str
    state: str  # planning | executing | awaiting_approval | completed | failed
    message: str
    lanes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # lane -> status
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: Optional[float] = None


# === Health & Status Endpoints ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Architect",
        "version": "1.0.0",
        "port": 6110,
        "agent": "Architect",
        "llm_model": os.getenv("ARCHITECT_LLM", "anthropic/claude-sonnet-4-5"),
        "agent_metadata": {
            "role": "architect",
            "tier": "coordinator",
            "parent": "PAS Root",
            "children": list(DIRECTOR_ENDPOINTS.keys())
        }
    }


@app.get("/status/{run_id}")
async def get_status(run_id: str) -> RunStatus:
    """Get status for a run"""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    run_data = RUNS[run_id]
    return RunStatus(
        run_id=run_id,
        state=run_data["state"],
        message=run_data.get("message", ""),
        lanes=run_data.get("lanes", {}),
        started_at=run_data.get("started_at"),
        completed_at=run_data.get("completed_at"),
        duration_s=run_data.get("duration_s")
    )


# === Main Prime Directive Endpoint ===

@app.post("/submit")
async def submit_prime_directive(
    pd: PrimeDirective,
    background_tasks: BackgroundTasks
):
    """
    Submit Prime Directive for execution

    Flow:
    1. Validate and register run
    2. Start background task for decomposition
    3. Return immediately with run_id and status

    Background task:
    - Decompose PRD using LLM
    - Create lane job cards
    - Submit to Directors
    - Monitor execution
    - Validate acceptance
    - Report completion to PAS Root
    """
    # Validate run_id uniqueness
    if pd.run_id in RUNS:
        raise HTTPException(status_code=400, detail="Run ID already exists")

    # Register run
    RUNS[pd.run_id] = {
        "state": "planning",
        "message": "Received Prime Directive, planning lanes",
        "started_at": time.time(),
        "lanes": {},
        "pd": pd.model_dump()
    }

    # Log receipt
    logger.log_cmd(
        from_agent="PAS Root",
        to_agent="Architect",
        message=f"Prime Directive received: {pd.title}",
        run_id=pd.run_id,
        metadata={"prd_length": len(pd.prd), "budget": pd.budget}
    )

    # Start background execution
    background_tasks.add_task(execute_prime_directive, pd)

    # Send heartbeat
    heartbeat_monitor.heartbeat(
        agent="Architect",
        run_id=pd.run_id,
        state=AgentState.PLANNING,
        message=f"Planning: {pd.title}",
        progress=0.1
    )

    return {
        "run_id": pd.run_id,
        "status": "planning",
        "message": "Prime Directive accepted, decomposing into lanes"
    }


# === Background Execution Logic ===

async def execute_prime_directive(pd: PrimeDirective):
    """
    Execute Prime Directive (background task)

    Steps:
    1. Decompose PRD into lane job cards (LLM-powered)
    2. Create Architect Plan artifact
    3. Submit job cards to Directors
    4. Monitor Director heartbeats
    5. Collect lane reports
    6. Validate acceptance gates
    7. Generate executive summary
    8. Report to PAS Root
    """
    run_id = pd.run_id

    try:
        # Step 1: Decompose PRD
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing PRD with LLM",
            progress=0.2
        )

        plan = await decompose_prd(pd)

        # Step 2: Save Architect Plan
        save_architect_plan(run_id, plan)

        # Step 3: Submit job cards to Directors
        RUNS[run_id]["state"] = "delegating"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Submitting job cards to Directors",
            progress=0.4
        )

        await delegate_to_directors(run_id, plan)

        # Step 4: Monitor execution
        RUNS[run_id]["state"] = "executing"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Directors",
            progress=0.5
        )

        await monitor_directors(run_id, plan)

        # Step 5: Validate acceptance
        RUNS[run_id]["state"] = "validating"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.VALIDATING,
            message="Validating acceptance gates",
            progress=0.9
        )

        validation_results = await validate_acceptance(run_id, plan)

        # Step 6: Complete
        if all(validation_results.values()):
            RUNS[run_id]["state"] = "completed"
            RUNS[run_id]["message"] = "All lanes completed successfully"
            RUNS[run_id]["completed_at"] = time.time()
            RUNS[run_id]["duration_s"] = RUNS[run_id]["completed_at"] - RUNS[run_id]["started_at"]

            heartbeat_monitor.heartbeat(
                agent="Architect",
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Prime Directive completed",
                progress=1.0
            )

            # Log completion
            logger.log_response(
                from_agent="Architect",
                to_agent="PAS Root",
                message=f"Prime Directive completed: {pd.title}",
                run_id=run_id,
                status="completed",
                metadata={"duration_s": RUNS[run_id]["duration_s"]}
            )
        else:
            # Some lanes failed
            failed_lanes = [lane for lane, passed in validation_results.items() if not passed]
            RUNS[run_id]["state"] = "failed"
            RUNS[run_id]["message"] = f"Lanes failed: {', '.join(failed_lanes)}"
            RUNS[run_id]["completed_at"] = time.time()

            heartbeat_monitor.heartbeat(
                agent="Architect",
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Failed: {', '.join(failed_lanes)}",
                progress=1.0
            )

            # Log failure
            logger.log_response(
                from_agent="Architect",
                to_agent="PAS Root",
                message=f"Prime Directive failed: {pd.title}",
                run_id=run_id,
                status="failed",
                metadata={"failed_lanes": failed_lanes}
            )

    except Exception as e:
        # Unhandled error
        RUNS[run_id]["state"] = "failed"
        RUNS[run_id]["message"] = f"Error: {str(e)}"
        RUNS[run_id]["completed_at"] = time.time()

        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}",
            progress=1.0
        )

        logger.log_response(
            from_agent="Architect",
            to_agent="PAS Root",
            message=f"Error executing Prime Directive: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"error": str(e)}
        )


async def decompose_prd(pd: PrimeDirective) -> ArchitectPlan:
    """
    Decompose PRD into lane job cards using LLM

    Uses Claude Sonnet 4.5 to analyze PRD and generate:
    - Executive summary
    - Lane allocations
    - Dependency graph
    - Resource reservations
    - Acceptance gates

    Returns:
        ArchitectPlan with all decomposition results
    """
    # Use TaskDecomposer (LLM-powered)
    result = await decomposer.decompose(
        prd=pd.prd,
        title=pd.title,
        entry_files=pd.entry_files,
        budget=pd.budget,
        policy=pd.policy
    )

    return ArchitectPlan(
        run_id=pd.run_id,
        executive_summary=result["executive_summary"],
        lane_allocations=result["lane_allocations"],
        dependency_graph=result["dependency_graph"],
        resource_reservations=result["resource_reservations"],
        acceptance_gates=result["acceptance_gates"]
    )


def save_architect_plan(run_id: str, plan: ArchitectPlan):
    """Save Architect Plan to artifacts"""
    artifact_dir = Path(f"artifacts/runs/{run_id}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    plan_file = artifact_dir / "architect_plan.json"
    plan_file.write_text(json.dumps(plan.model_dump(), indent=2))

    # Save dependency graph (DOT format)
    graph_file = artifact_dir / "dependency_graph.dot"
    graph_file.write_text(plan.dependency_graph)


async def delegate_to_directors(run_id: str, plan: ArchitectPlan):
    """
    Submit job cards to Directors

    For each lane in plan.lane_allocations:
    1. Create JobCard
    2. Submit to Director via RPC (or queue fallback)
    3. Log delegation
    """
    for lane_name, allocation in plan.lane_allocations.items():
        director = f"Dir-{lane_name}"

        # Create job card
        job_card = JobCard(
            id=f"jc-{run_id}-{lane_name.lower()}-001",
            parent_id=run_id,
            role=Role.DIRECTOR,
            lane=Lane[lane_name.upper()],
            task=allocation["task"],
            inputs=allocation.get("inputs", []),
            expected_artifacts=allocation.get("expected_artifacts", []),
            acceptance=allocation.get("acceptance", []),
            risks=allocation.get("risks", []),
            budget=allocation.get("budget", {}),
            metadata=allocation.get("metadata", {}),
            submitted_by="Architect"
        )

        # Try RPC first
        try:
            endpoint = DIRECTOR_ENDPOINTS.get(director)
            if endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{endpoint}/submit", json={"job_card": asdict(job_card)})
                    response.raise_for_status()

                logger.log_cmd(
                    from_agent="Architect",
                    to_agent=director,
                    message=f"Job card submitted: {allocation['task'][:50]}...",
                    run_id=run_id,
                    metadata={"job_card_id": job_card.id}
                )
            else:
                # Fallback to queue
                job_queue.submit(job_card, target_agent=director, use_file_fallback=True)

        except Exception as e:
            # Fallback to file queue
            job_queue.submit(job_card, target_agent=director, use_file_fallback=True)

            logger.log_cmd(
                from_agent="Architect",
                to_agent=director,
                message=f"Job card queued (RPC failed): {allocation['task'][:50]}...",
                run_id=run_id,
                metadata={"job_card_id": job_card.id, "rpc_error": str(e)}
            )

        # Update run state
        RUNS[run_id]["lanes"][lane_name] = {
            "state": "delegated",
            "job_card_id": job_card.id,
            "director": director
        }


async def monitor_directors(run_id: str, plan: ArchitectPlan):
    """
    Monitor Directors until all complete or timeout

    Polls Director status every 10 seconds
    Checks heartbeats for liveness
    Escalates on 2 missed heartbeats
    """
    lanes = list(plan.lane_allocations.keys())
    timeout_s = 3600  # 1 hour max
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        all_complete = True

        for lane in lanes:
            lane_state = RUNS[run_id]["lanes"][lane]["state"]

            if lane_state not in ["completed", "failed"]:
                all_complete = False

                # Check Director health
                director = f"Dir-{lane}"
                health = heartbeat_monitor.get_health(director)

                if not health.healthy:
                    # Director unhealthy - escalate
                    RUNS[run_id]["lanes"][lane]["state"] = "failed"
                    RUNS[run_id]["lanes"][lane]["message"] = health.reason

                    logger.log_status(
                        from_agent="Architect",
                        to_agent="PAS Root",
                        message=f"Director {director} unhealthy: {health.reason}",
                        run_id=run_id,
                        status="failed"
                    )

        if all_complete:
            break

        # Wait before next poll
        await asyncio.sleep(10)

    # Check for timeout
    if time.time() - start_time >= timeout_s:
        for lane in lanes:
            if RUNS[run_id]["lanes"][lane]["state"] not in ["completed", "failed"]:
                RUNS[run_id]["lanes"][lane]["state"] = "failed"
                RUNS[run_id]["lanes"][lane]["message"] = "Timeout"


async def validate_acceptance(run_id: str, plan: ArchitectPlan) -> Dict[str, bool]:
    """
    Validate acceptance gates for each lane

    Returns:
        Dict mapping lane name to pass/fail boolean
    """
    results = {}

    for lane_name, gates in plan.acceptance_gates.items():
        # Get lane report from Director
        # (In full implementation, Directors would report artifacts + results)
        # For now, assume success if lane completed
        lane_state = RUNS[run_id]["lanes"].get(lane_name, {}).get("state")
        results[lane_name] = lane_state == "completed"

    return results


# === Startup ===

if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Start heartbeat emitter
    async def emit_heartbeats():
        while True:
            heartbeat_monitor.heartbeat(
                agent="Architect",
                state=AgentState.IDLE,
                message="Idle, waiting for Prime Directives"
            )
            await asyncio.sleep(60)

    # Run server
    uvicorn.run(app, host="127.0.0.1", port=6110)
