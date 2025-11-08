#!/usr/bin/env python3
"""
VP (VP Agent) - Local Code Operator CLI

Terminal client for LCO (Local Code Operator) that integrates with:
- PLMS API (port 6100) for planning/estimation
- PAS Stub (port 6200) for execution

Usage:
  vp new --name <project>
  vp plan [--from-prd docs/PRD.md]
  vp estimate
  vp simulate --rehearsal 0.01
  vp start
  vp status
  vp logs [--tail 20]

Version: 0.1.0 (MVP for Phase 3)
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import requests

# ============================================================================
# Configuration
# ============================================================================

PLMS_API = os.getenv("PLMS_API_BASE_URL", "http://localhost:6100")
PAS_API = os.getenv("PAS_API_BASE_URL", "http://localhost:6200")
VP_STATE_DIR = Path.home() / ".vp"
VP_STATE_FILE = VP_STATE_DIR / "state.json"

# ============================================================================
# State Management (Local File)
# ============================================================================


def load_state() -> dict:
    """Load VP state from local file."""
    if not VP_STATE_FILE.exists():
        return {}
    with open(VP_STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict):
    """Save VP state to local file."""
    VP_STATE_DIR.mkdir(exist_ok=True)
    with open(VP_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_current_project() -> Optional[dict]:
    """Get current project from state."""
    state = load_state()
    project_id = state.get("current_project_id")
    if not project_id:
        return None
    return state.get("projects", {}).get(str(project_id))


def set_current_project(project_id: int, run_id: str):
    """Set current project in state."""
    state = load_state()
    state["current_project_id"] = project_id
    if "projects" not in state:
        state["projects"] = {}
    state["projects"][str(project_id)] = {
        "project_id": project_id,
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_state(state)


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """VP Agent - Local Code Operator"""
    pass


@cli.command()
@click.option("--name", required=True, help="Project name")
def new(name: str):
    """Initialize a new project."""
    click.echo(f"Creating project: {name}")

    # Generate IDs
    project_id = abs(hash(name)) % 10000  # Simple ID generation for MVP
    run_id = f"r-{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}"

    # TODO: Create project in PLMS (when PLMS projects endpoint is ready)
    # For now, just store locally
    set_current_project(project_id, run_id)

    click.echo(f"✓ Project created")
    click.echo(f"  Project ID: {project_id}")
    click.echo(f"  Run ID: {run_id}")
    click.echo()
    click.echo("Next steps:")
    click.echo("  vp plan      # Generate execution plan")
    click.echo("  vp estimate  # Get cost/time estimates")
    click.echo("  vp start     # Start execution")


@cli.command()
@click.option("--from-prd", type=click.Path(exists=True), help="Path to PRD file")
def plan(from_prd: Optional[str]):
    """Generate execution plan."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    click.echo(f"Planning project {project['project_id']}...")

    if from_prd:
        click.echo(f"  Reading PRD from: {from_prd}")
        with open(from_prd) as f:
            prd_content = f.read()
        click.echo(f"  PRD size: {len(prd_content)} characters")

    # TODO: Call PLMS clarify endpoint (when available)
    click.echo("✓ Plan generated (stub)")
    click.echo()
    click.echo("Estimated tasks:")
    click.echo("  1. Code-Impl: Implement feature X")
    click.echo("  2. Data-Schema: Update schema for Y")
    click.echo("  3. Vector-Ops: Refresh code index")
    click.echo()
    click.echo("Next: vp estimate")


@cli.command()
def estimate():
    """Get cost/time estimates."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    click.echo(f"Getting estimates for project {project['project_id']}...")

    # TODO: Call PLMS metrics endpoint when integrated
    # For MVP: Show synthetic estimates
    click.echo()
    click.echo("📊 Estimates (90% confidence intervals):")
    click.echo()
    click.echo("  Tokens:   15,000 (13,200 - 16,800)")
    click.echo("  Duration: 25 min (20 - 30 min)")
    click.echo("  Cost:     $2.50 ($2.20 - $2.80)")
    click.echo("  Energy:   0.35 kWh (0.30 - 0.40 kWh)")
    click.echo()
    click.echo("Next: vp start")


@cli.command()
@click.option("--rehearsal", type=float, default=0.0, help="Rehearsal percentage (0.0-0.05)")
def simulate(rehearsal: float):
    """Simulate rehearsal run."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    if not (0.0 <= rehearsal <= 0.05):
        click.echo("❌ Rehearsal must be between 0.0 and 0.05", err=True)
        sys.exit(1)

    click.echo(f"Simulating rehearsal ({rehearsal*100:.1f}%) for run {project['run_id']}...")

    try:
        resp = requests.post(
            f"{PAS_API}/pas/v1/runs/simulate",
            json={
                "run_id": project["run_id"],
                "rehearsal_pct": rehearsal,
                "stratified": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        click.echo()
        click.echo("✓ Rehearsal simulation complete")
        click.echo()
        click.echo(f"  Strata coverage:   {data['strata_coverage']*100:.0f}%")
        click.echo(f"  Rehearsal tokens:  {data['rehearsal_tokens']:,}")
        click.echo(f"  Projected tokens:  {data['projected_tokens']:,}")
        click.echo(f"  Confidence band:   {data['ci_lower']:,} - {data['ci_upper']:,}")
        click.echo(f"  Risk score:        {data['risk_score']:.2f}")
        click.echo()
        click.echo("Next: vp start")

    except requests.exceptions.ConnectionError:
        click.echo(f"❌ PAS stub not running on {PAS_API}", err=True)
        click.echo("   Start with: make run-pas-stub", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def start():
    """Start execution."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    click.echo(f"Starting run {project['run_id']}...")

    try:
        # Start run via PAS
        resp = requests.post(
            f"{PAS_API}/pas/v1/runs/start",
            json={
                "project_id": project["project_id"],
                "run_id": project["run_id"],
                "run_kind": "baseline",
                "rehearsal_pct": 0.0,
                "budget_caps": {
                    "budget_usd": 50.0,
                    "energy_kwh": 2.0,
                },
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        click.echo(f"✓ Run started: {data['status']}")

        # Submit sample job cards (for demo)
        click.echo()
        click.echo("Submitting sample tasks...")

        tasks = [
            {
                "lane": "Code-Impl",
                "payload": {
                    "goal": "Implement feature",
                    "tests": ["tests/test_feature.py"],
                },
            },
            {
                "lane": "Vector-Ops",
                "payload": {
                    "operation": "refresh",
                    "scope": "src/",
                },
            },
        ]

        for task in tasks:
            resp = requests.post(
                f"{PAS_API}/pas/v1/jobcards",
                headers={"Idempotency-Key": str(uuid.uuid4())},
                json={
                    "project_id": project["project_id"],
                    "run_id": project["run_id"],
                    "lane": task["lane"],
                    "priority": 0.5,
                    "deps": [],
                    "payload": task["payload"],
                    "budget_usd": 1.0,
                    "ci_width_hint": 0.3,
                },
                timeout=10,
            )
            resp.raise_for_status()
            task_data = resp.json()
            click.echo(f"  ✓ {task['lane']}: {task_data['task_id']}")

        click.echo()
        click.echo("Next: vp status (monitor progress)")

    except requests.exceptions.ConnectionError:
        click.echo(f"❌ PAS stub not running on {PAS_API}", err=True)
        click.echo("   Start with: make run-pas-stub", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Get current project status."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    try:
        resp = requests.get(
            f"{PAS_API}/pas/v1/runs/status",
            params={"run_id": project["run_id"]},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        click.echo(f"Run: {data['run_id']}")
        click.echo(f"Status: {data['status']}")
        click.echo()
        click.echo("📊 Progress:")
        click.echo(f"  Tasks:       {data['tasks_completed']}/{data['tasks_total']} completed")
        click.echo(f"  Failed:      {data['tasks_failed']}")
        click.echo()
        click.echo("💰 Spend:")
        click.echo(f"  Cost:        ${data['spend_usd']:.2f}")
        click.echo(f"  Energy:      {data['spend_energy_kwh']:.3f} kWh")
        click.echo()
        click.echo(f"⏰ Runway:     {data['runway_minutes']} minutes")
        click.echo()

        if data["kpi_violations"]:
            click.echo("⚠️  KPI Violations:")
            for kpi in data["kpi_violations"]:
                click.echo(f"  - {kpi['kpi_name']}: {kpi['actual']} (expected: {kpi['expected']})")
            click.echo()

        if data["tasks"]:
            click.echo("Tasks:")
            for task in data["tasks"]:
                status_icon = "✓" if task["status"] == "succeeded" else "⏳" if task["status"] == "running" else "❌"
                click.echo(f"  {status_icon} {task['lane']}: {task['status']}")

    except requests.exceptions.ConnectionError:
        click.echo(f"❌ PAS stub not running on {PAS_API}", err=True)
        click.echo("   Start with: make run-pas-stub", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--tail", type=int, default=20, help="Number of lines to show")
def logs(tail: int):
    """Show run logs."""
    project = get_current_project()
    if not project:
        click.echo("❌ No active project. Run: vp new --name <project>", err=True)
        sys.exit(1)

    # TODO: Implement log streaming from PAS
    click.echo(f"Showing last {tail} log lines for run {project['run_id']}:")
    click.echo("(Log streaming not yet implemented in stub)")


if __name__ == "__main__":
    cli()
