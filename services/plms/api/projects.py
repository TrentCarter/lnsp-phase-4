"""
PLMS Project API Endpoints - Tier 1 Enhanced
Includes: Idempotency, RBAC hooks, Rehearsal mode, Multi-run support
"""

from fastapi import APIRouter, HTTPException, Header, Depends, Body, Query, Request
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib
import json
import uuid
import os

# Import Redis-backed idempotency cache
from services.plms.idempotency import get_cache, InMemoryIdempotencyCache
from services.plms.stratified_sampler import stratified_sample, validate_strata_coverage

router = APIRouter(prefix="/api/projects", tags=["projects"])

# --- Idempotency Cache Initialization ---

# Use Redis if REDIS_URL env var set, otherwise fallback to in-memory (dev only)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS = os.environ.get("PLMS_USE_REDIS", "true").lower() == "true"

try:
    if USE_REDIS:
        idempotency_cache = get_cache(REDIS_URL)
        print(f"✓ Redis idempotency cache initialized: {REDIS_URL}")
    else:
        idempotency_cache = InMemoryIdempotencyCache()
        print("⚠️  Using in-memory idempotency cache (NOT production-safe!)")
except Exception as e:
    print(f"⚠️  Redis unavailable ({e}), falling back to in-memory cache")
    idempotency_cache = InMemoryIdempotencyCache()

# --- Dependencies (stubs - wire to actual auth/DB) ---

def get_current_user():
    """
    Stub: Extract user from JWT/session.
    Replace with actual auth middleware.
    """
    return {"username": "user@example.com", "scopes": ["projects.view", "projects.start", "projects.simulate"]}

def rbac(required_scope: str):
    """
    RBAC dependency: check if user has required scope.
    """
    def _rbac(user=Depends(get_current_user)):
        if required_scope not in user.get("scopes", []):
            raise HTTPException(status_code=403, detail=f"Missing scope: {required_scope}")
        return user
    return _rbac

def idempotency_guard(key: str = Header(..., alias="Idempotency-Key")):
    """
    Idempotency guard: require Idempotency-Key header for POST /start.
    Now backed by Redis with 24h TTL.
    """
    if not key or len(key) < 8:
        raise HTTPException(status_code=400, detail="Invalid or missing Idempotency-Key header")
    return key

# Stub helpers (wire to actual DB/PAS)
def provider_router_snapshot() -> Dict:
    """Snapshot current provider matrix (models + versions)."""
    return {
        "openai/gpt-4": "gpt-4-0613",
        "local/llama-8b": "llama-3.1-8b-v1.2"
    }

def capability_catalog_snapshot() -> Dict:
    """Snapshot provider capability graph."""
    return {"capabilities": ["text-generation", "embeddings", "code-completion"]}

def prd_sha(project_id: int) -> str:
    """Compute SHA256 of PRD file."""
    # Stub: read PRD from disk, compute hash
    return hashlib.sha256(b"stub-prd-content").hexdigest()[:16]

def env_fingerprint() -> Dict:
    """Capture environment fingerprint for deterministic replay."""
    return {
        "git_commit": "abc123",
        "python_version": "3.11.5",
        "os": "Darwin"
    }

def pas_submit_jobcard(project_id: int, run_kind: str, provider_matrix: Dict) -> str:
    """
    Submit job card to PAS Architect.
    Returns: run_id (UUID)
    """
    # Stub: wire to actual PAS submission endpoint
    return str(uuid.uuid4())

def persist_artifact(project_id: int, run_id: str, filename: str, content: Any) -> str:
    """Save artifact to disk/S3, return path."""
    path = f"artifacts/project-{project_id}/{filename}"
    # Stub: actual file write
    return path

def db_project_runs_insert(**kwargs):
    """Insert row into project_runs table."""
    # Stub: wire to actual DB
    pass

def load_task_estimates(project_id: int) -> List[Dict]:
    """Load task tree from database."""
    # Stub
    return [{"task_id": 1, "name": "Design API", "complexity": "simple"}]

def stratified_sample(plan: List[Dict], pct: float) -> List[Dict]:
    """Sample tasks stratified by complexity."""
    # Stub: implement stratified sampling
    return plan[:max(1, int(len(plan) * pct))]

def pas_execute_canary(project_id: int, canary_tasks: List[Dict], write_sandbox: bool) -> Dict:
    """Execute canary tasks via PAS."""
    # Stub: wire to PAS execution with write-suppression flag
    return {
        "tokens": 190,
        "duration_ms": 21000,
        "cost_usd": 0.011,
        "echo_cos_mean": 0.84
    }

def extrapolate_full(plan: List[Dict], rehearsal_actual: Dict) -> Dict:
    """Extrapolate canary results to full project with CIs."""
    n_total = len(plan)
    n_sample = len(stratified_sample(plan, 0.01))
    scale_factor = n_total / max(1, n_sample)

    tokens_mean = rehearsal_actual["tokens"] * scale_factor
    tokens_stddev = tokens_mean * 0.1  # Assume 10% stddev (tune from data)

    return {
        "tokens": int(tokens_mean),
        "duration_ms": int(rehearsal_actual["duration_ms"] * scale_factor),
        "cost_usd": round(rehearsal_actual["cost_usd"] * scale_factor, 2),
        "echo_cos_mean": rehearsal_actual["echo_cos_mean"],
        "ci_90": {
            "tokens": [int(tokens_mean - 1.645 * tokens_stddev), int(tokens_mean + 1.645 * tokens_stddev)],
            "cost_usd": [round((tokens_mean - 1.645 * tokens_stddev) * 0.00006, 2),
                        round((tokens_mean + 1.645 * tokens_stddev) * 0.00006, 2)]
        }
    }

def compute_risk_factors(plan: List[Dict], rehearsal_actual: Dict, extrapolated: Dict) -> List[Dict]:
    """Identify risk factors from canary execution."""
    # Stub: detect high variance, KPI violations, etc.
    return [
        {"lane": 4202, "issue": "high_variance", "samples": 3}
    ]

def load_estimates_with_ci(project_id: int) -> Dict:
    """Load estimates with credible intervals."""
    # Stub
    return {
        "tokens": {"mean": 19000, "ci_90": [17100, 20900]},
        "duration_ms": {"mean": 2100000, "ci_90": [1890000, 2310000]},
        "cost_usd": {"mean": 1.14, "ci_90": [1.03, 1.25]}
    }

def load_estimates(project_id: int) -> Dict:
    """Load point estimates (no CIs)."""
    return {"tokens": 19000, "duration_minutes": 35, "cost_usd": 1.14}

def load_actuals(project_id: int) -> Dict:
    """Load actual metrics from completed run."""
    return {"tokens": 17500, "duration_ms": 1920000, "cost_usd": 1.05}

def assess_accuracy(est: Dict, actual: Dict, with_ci: bool) -> Dict:
    """Compute accuracy metrics."""
    if with_ci:
        return {
            "tokens_in_ci": True,
            "duration_ms_in_ci": True,
            "cost_usd_in_ci": True,
            "token_error_pct": 7.9,
            "time_error_pct": 8.6,
            "cost_error_pct": 7.9
        }
    else:
        return {
            "token_error_pct": 7.9,
            "time_error_pct": 8.6,
            "cost_error_pct": 7.9
        }

def lane_error_breakdown(project_id: int) -> Dict:
    """Per-lane calibration error breakdown."""
    return {
        "lane_specific_errors": [
            {"lane_id": 4201, "lane_name": "Code-API-Design", "mae_tokens": 150, "mae_pct": 5.3},
            {"lane_id": 4202, "lane_name": "Code-API-Impl", "mae_tokens": 620, "mae_pct": 7.8}
        ]
    }

def db_list_lane_overrides(project_id: int) -> List[Dict]:
    """List all lane overrides for a project (active learning feedback)."""
    # Stub: query lane_overrides table
    return [
        {
            "task_id": 2,
            "task_description": "Implement JWT authentication middleware",
            "predicted_lane": 4202,
            "corrected_lane": 5301,
            "corrected_by": "user@example.com",
            "corrected_at": "2025-11-07T14:32:00Z"
        }
    ]

# --- API Endpoints ---

@router.post("/{project_id}/start")
def start_project(
    project_id: int,
    payload: Dict[str, Any] = Body(...),
    idem_key: str = Depends(idempotency_guard),
    user=Depends(rbac("projects.start"))
):
    """
    Start execution with Redis-backed idempotency + replay passport.

    Headers:
        Idempotency-Key: UUID (required for safe retries)
        Returns Idempotent-Replay: true if served from cache

    Body:
        run_kind: "baseline" | "rehearsal" | "replay" | "hotfix"
        write_sandbox: bool (explicitly recorded)
    """
    # Check Redis cache first
    cached_response = idempotency_cache.check_and_store(
        method="POST",
        path=f"/api/projects/{project_id}/start",
        body=payload,
        user_id=user["username"]
    )

    if cached_response:
        # Return cached response with replay header
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=cached_response,
            headers={"Idempotent-Replay": "true"}
        )

    run_kind = payload.get("run_kind", "baseline")
    if run_kind not in {"baseline", "rehearsal", "replay", "hotfix"}:
        raise HTTPException(400, "invalid run_kind")

    # Snapshot providers/capabilities for deterministic replay
    provider_matrix = provider_router_snapshot()
    capability_snapshot = capability_catalog_snapshot()

    run_id = pas_submit_jobcard(project_id, run_kind, provider_matrix)

    passport = {
        "project_id": project_id,
        "run_id": run_id,
        "run_kind": run_kind,
        "provider_matrix": provider_matrix,
        "capabilities": capability_snapshot,
        "prd_sha": prd_sha(project_id),
        "env": env_fingerprint(),
        "started_at": datetime.utcnow().isoformat() + "Z"
    }
    passport_path = persist_artifact(project_id, run_id, "replay_passport.json", passport)

    # Persist run row with snapshot + critical flags
    db_project_runs_insert(
        project_id=project_id,
        run_id=run_id,
        run_kind=run_kind,
        provider_matrix_json=json.dumps(provider_matrix),
        capability_snapshot=json.dumps(capability_snapshot),
        write_sandbox=payload.get("write_sandbox", False),  # Explicitly record
        validation_pass=None  # Will be set after validation phase
    )

    resp = {
        "status": "executing",
        "run_id": run_id,
        "provider_matrix": provider_matrix,
        "replay_passport_path": passport_path
    }

    # Store in Redis cache (24h TTL)
    idempotency_cache.check_and_store(
        method="POST",
        path=f"/api/projects/{project_id}/start",
        body=payload,
        user_id=user["username"],
        response=resp
    )

    # Return fresh response with replay=false header
    from fastapi.responses import JSONResponse
    return JSONResponse(content=resp, headers={"Idempotent-Replay": "false"})


@router.post("/{project_id}/simulate")
def simulate_project(
    project_id: int,
    rehearsal_pct: float = Query(0.01, ge=0.001, le=0.2),
    write_sandbox: bool = Query(False),
    user=Depends(rbac("projects.simulate"))
):
    """
    Simulate with stratified rehearsal (guaranteed strata coverage).

    Query Params:
        rehearsal_pct: Requested fraction (default 0.01 = 1%)
                      Auto-bumps if needed to cover all strata (cap 5%)
        write_sandbox: Suppress DB/artifact writes (default False)

    Returns:
        strata_coverage: 1.0 means all strata sampled
        auto_bumped: true if rehearsal_pct was increased
    """
    plan = load_task_estimates(project_id)

    # Use stratified sampler with coverage guarantee
    canary_tasks, sampling_metrics = stratified_sample(
        plan,
        rehearsal_pct=rehearsal_pct,
        min_per_stratum=1,
        max_pct=0.05
    )

    # Validate coverage and collect warnings
    warnings = validate_strata_coverage(sampling_metrics)

    # Execute canary via PAS with write-suppression if requested
    rehearsal_actual = pas_execute_canary(project_id, canary_tasks, write_sandbox)

    # Extrapolate with uncertainty (seeded by priors, widened for small n)
    extrapolated = extrapolate_full(plan, rehearsal_actual)

    # Compute risk factors (wide CI, KPI early failures, lane variance)
    risks = compute_risk_factors(plan, rehearsal_actual, extrapolated)

    return {
        "simulation_results": {
            "rehearsal_actual": rehearsal_actual,
            "extrapolated_full": extrapolated,
            "risk_factors": risks
        },
        "sampling_metrics": {
            "n_total": sampling_metrics["n_total"],
            "n_sample": sampling_metrics["n_sample"],
            "actual_pct": sampling_metrics["actual_pct"],
            "requested_pct": sampling_metrics["requested_pct"],
            "strata_coverage": sampling_metrics["strata_coverage"],
            "auto_bumped": sampling_metrics["auto_bumped"],
            "n_strata": sampling_metrics["n_strata"]
        },
        "warnings": warnings
    }


@router.get("/{project_id}/metrics")
def project_metrics(
    project_id: int,
    with_ci: bool = Query(False),
    user=Depends(rbac("projects.view"))
):
    """
    Get metrics with optional credible intervals (Bayesian calibration).

    Query Params:
        with_ci: Include 90% credible intervals (default False)
    """
    est = load_estimates_with_ci(project_id) if with_ci else load_estimates(project_id)
    actual = load_actuals(project_id)
    accuracy = assess_accuracy(est, actual, with_ci=with_ci)
    calibration = lane_error_breakdown(project_id) if with_ci else {}

    return {
        "estimated": est,
        "actual": actual,
        "accuracy": accuracy,
        "calibration": calibration
    }


@router.get("/{project_id}/lane-overrides")
def lane_overrides(project_id: int, user=Depends(rbac("projects.view"))):
    """
    Get lane overrides for active learning feedback.

    Returns list of human corrections to TMD lane predictions.
    """
    return db_list_lane_overrides(project_id)


# --- Example: Additional endpoints (stubs) ---

@router.get("/{project_id}/budget-runway")
def budget_runway(project_id: int, user=Depends(rbac("projects.view"))):
    """Budget runway gauge (time to depletion)."""
    return {
        "budget": {"usd_max": 2.00, "usd_spent": 1.10, "burn_per_min": 0.083},
        "runway": {"minutes_to_depletion": 12.0, "projected_overrun_usd": 0.32},
        "status": "warning"
    }


@router.get("/{project_id}/risk-heatmap")
def risk_heatmap(project_id: int, user=Depends(rbac("projects.view"))):
    """Lane × phase risk heatmap."""
    return {
        "lanes": [
            {
                "lane_id": 4200,
                "name": "Code-API",
                "estimation_risk": "low",
                "execution_risk": "medium",
                "validation_risk": "low",
                "signals": {"mae_pct": 0.12, "ci_width_pct": 0.28}
            },
            {
                "lane_id": 5100,
                "name": "Data-Schema",
                "estimation_risk": "high",
                "execution_risk": "medium",
                "validation_risk": "high",
                "signals": {"mae_pct": 0.37, "ci_width_pct": 0.55}
            }
        ],
        "legend": {
            "low":    {"mae_pct_lt": 0.15, "ci_width_lt": 0.30},
            "medium": {"mae_pct_le": 0.30, "ci_width_le": 0.50},
            "high":   {"else": True}
        }
    }
