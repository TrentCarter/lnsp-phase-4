#!/usr/bin/env python3
"""
PAS Token Governor — Port 6105
Tracks context usage per agent, enforces token budgets.
Triggers Save-State → Clear → Resume on context breaches.
Part of Phase 1: Management Agents
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import contextmanager

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = Path("artifacts/token_governor/tokens.db")
SUMMARIES_DIR = Path("docs/runs")

DEFAULT_TARGET_RATIO = 0.50  # Target 50% context usage
DEFAULT_HARD_MAX_RATIO = 0.75  # Hard max 75% context usage

# ============================================================================
# Pydantic Models
# ============================================================================

class ContextTrack(BaseModel):
    """Track context usage for an agent"""
    agent: str
    run_id: Optional[str] = None
    ctx_used: int = Field(..., ge=0)
    ctx_limit: int = Field(..., gt=0)
    target_ratio: Optional[float] = Field(DEFAULT_TARGET_RATIO, ge=0, le=1)
    hard_max_ratio: Optional[float] = Field(DEFAULT_HARD_MAX_RATIO, ge=0, le=1)


class SummarizeRequest(BaseModel):
    """Request to summarize agent context"""
    agent: str
    run_id: str
    trigger_reason: str = Field(..., pattern="^(hard_max_breach|manual|soft_timeout)$")
    content_to_summarize: Optional[str] = None


# ============================================================================
# Database Management
# ============================================================================

def init_db():
    """Initialize SQLite database with schema"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Agent contexts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_contexts (
            agent TEXT PRIMARY KEY,
            run_id TEXT,
            ctx_used INTEGER DEFAULT 0,
            ctx_limit INTEGER NOT NULL,
            target_ratio REAL DEFAULT 0.50,
            hard_max_ratio REAL DEFAULT 0.75,
            status TEXT DEFAULT 'ok' CHECK(status IN ('ok', 'warning', 'breach')),
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Summarizations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summarizations (
            summary_id TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            run_id TEXT NOT NULL,
            trigger_reason TEXT NOT NULL,
            ctx_before INTEGER,
            ctx_after INTEGER,
            summary_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summarizations_agent ON summarizations(agent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summarizations_run_id ON summarizations(run_id)")

    conn.commit()
    conn.close()

    print(f"✓ Token Governor database initialized at {DB_PATH}")


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# Summarization Logic
# ============================================================================

def generate_summary(agent: str, run_id: str, content: Optional[str] = None) -> str:
    """
    Generate context summary (stub for now, will use local LLM in future)

    Args:
        agent: Agent name
        run_id: Run identifier
        content: Content to summarize (optional)

    Returns:
        Path to summary file
    """
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = SUMMARIES_DIR / f"{run_id}_summary.md"

    # Generate stub summary (replace with LLM call in production)
    summary_content = f"""# Context Summary for {agent}

**Run ID:** {run_id}
**Generated:** {datetime.utcnow().isoformat()}
**Trigger:** Hard max context breach

## Summary

Context window exceeded threshold. Agent state has been summarized and cleared.

## Next Steps

- Resume operation with cleared context
- Monitor context usage closely
- Consider breaking task into smaller chunks

---

*This is a stub summary. In production, this will use local LLM (llama-3.1-8b) to generate intelligent summaries.*
"""

    with open(summary_path, "w") as f:
        f.write(summary_content)

    print(f"✓ Generated summary: {summary_path}")

    return str(summary_path)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PAS Token Governor",
    description="Context budget enforcement and summarization",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Service information (root endpoint)"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Count agents by status
        cursor.execute("SELECT status, COUNT(*) as count FROM agent_contexts GROUP BY status")
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Count total summarizations
        cursor.execute("SELECT COUNT(*) as count FROM summarizations")
        total_summaries = cursor.fetchone()["count"]

        return {
            "service": "PAS Token Governor",
            "version": "1.0.0",
            "port": 6105,
            "status": "running",
            "defaults": {
                "target_ratio": DEFAULT_TARGET_RATIO,
                "hard_max_ratio": DEFAULT_HARD_MAX_RATIO
            },
            "tracking": {
                "total_agents": sum(status_counts.values()),
                "ok": status_counts.get("ok", 0),
                "warning": status_counts.get("warning", 0),
                "breach": status_counts.get("breach", 0)
            },
            "summarizations": {
                "total": total_summaries
            },
            "endpoints": {
                "track": "POST /track",
                "status": "GET /status",
                "summarize": "POST /summarize",
                "summaries": "GET /summaries",
                "clear": "POST /clear"
            },
            "docs": "/docs"
        }


@app.post("/track")
async def track_context(track: ContextTrack):
    """
    Track or update context usage for an agent

    Args:
        track: ContextTrack with agent, ctx_used, ctx_limit

    Returns:
        - agent: Agent name
        - ctx_used: Current context usage
        - ctx_limit: Context limit
        - ctx_ratio: Usage ratio (0-1)
        - status: 'ok' | 'warning' | 'breach'
        - action: Recommended action (if any)
    """
    ctx_ratio = track.ctx_used / track.ctx_limit

    # Determine status
    if ctx_ratio >= track.hard_max_ratio:
        status = "breach"
        action = "save_state_clear_resume"
    elif ctx_ratio >= track.target_ratio:
        status = "warning"
        action = "monitor_closely"
    else:
        status = "ok"
        action = None

    with get_db() as conn:
        cursor = conn.cursor()

        # Upsert agent context
        cursor.execute("""
            INSERT INTO agent_contexts (
                agent, run_id, ctx_used, ctx_limit, target_ratio, hard_max_ratio, status, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent) DO UPDATE SET
                run_id = excluded.run_id,
                ctx_used = excluded.ctx_used,
                ctx_limit = excluded.ctx_limit,
                target_ratio = excluded.target_ratio,
                hard_max_ratio = excluded.hard_max_ratio,
                status = excluded.status,
                last_updated = excluded.last_updated
        """, (
            track.agent, track.run_id, track.ctx_used, track.ctx_limit,
            track.target_ratio, track.hard_max_ratio, status,
            datetime.utcnow().isoformat()
        ))

        conn.commit()

    response = {
        "agent": track.agent,
        "ctx_used": track.ctx_used,
        "ctx_limit": track.ctx_limit,
        "ctx_ratio": round(ctx_ratio, 3),
        "status": status,
        "action": action,
        "ts": datetime.utcnow().isoformat()
    }

    # Print warning if breach
    if status == "breach":
        print(f"⚠️  CONTEXT BREACH: {track.agent} ({track.ctx_used}/{track.ctx_limit} = {ctx_ratio:.1%})")

    return response


@app.get("/status")
async def get_status(agent: Optional[str] = None):
    """
    Get context status for one or all agents

    Args:
        agent: Agent name (optional, returns all if not specified)

    Returns:
        - agents: List of agent context statuses
    """
    with get_db() as conn:
        cursor = conn.cursor()

        if agent:
            cursor.execute("SELECT * FROM agent_contexts WHERE agent = ?", (agent,))
        else:
            cursor.execute("SELECT * FROM agent_contexts ORDER BY last_updated DESC")

        rows = cursor.fetchall()

        agents = []
        for row in rows:
            ctx_ratio = row["ctx_used"] / row["ctx_limit"] if row["ctx_limit"] > 0 else 0
            agents.append({
                "agent": row["agent"],
                "run_id": row["run_id"],
                "ctx_used": row["ctx_used"],
                "ctx_limit": row["ctx_limit"],
                "ctx_ratio": round(ctx_ratio, 3),
                "target_ratio": row["target_ratio"],
                "hard_max_ratio": row["hard_max_ratio"],
                "status": row["status"],
                "last_updated": row["last_updated"]
            })

        return {"agents": agents}


@app.post("/summarize")
async def summarize_context(request: SummarizeRequest):
    """
    Trigger Save-State → Clear → Resume workflow

    Args:
        request: SummarizeRequest with agent, run_id, trigger_reason

    Returns:
        - summary_id: UUID
        - agent: Agent name
        - summary_path: Path to summary file
        - ctx_before: Context usage before clearing
        - ctx_after: Context usage after clearing (0)
        - ts: Timestamp
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Get current context
        cursor.execute("SELECT * FROM agent_contexts WHERE agent = ?", (request.agent,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {request.agent} not found in context tracking"
            )

        ctx_before = row["ctx_used"]

        # Generate summary
        summary_path = generate_summary(
            request.agent,
            request.run_id,
            request.content_to_summarize
        )

        # Record summarization
        summary_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO summarizations (
                summary_id, agent, run_id, trigger_reason, ctx_before, ctx_after, summary_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_id, request.agent, request.run_id, request.trigger_reason,
            ctx_before, 0, summary_path
        ))

        # Clear agent context
        cursor.execute("""
            UPDATE agent_contexts
            SET ctx_used = 0, status = 'ok', last_updated = ?
            WHERE agent = ?
        """, (datetime.utcnow().isoformat(), request.agent))

        conn.commit()

    print(f"✓ Summarized context for {request.agent}: {ctx_before} → 0 tokens")

    return {
        "summary_id": summary_id,
        "agent": request.agent,
        "summary_path": summary_path,
        "ctx_before": ctx_before,
        "ctx_after": 0,
        "ts": datetime.utcnow().isoformat()
    }


@app.get("/summaries")
async def get_summaries(agent: Optional[str] = None, limit: int = 50):
    """
    Get summarization history

    Args:
        agent: Filter by agent name (optional)
        limit: Maximum number of results

    Returns:
        - summaries: List of summarizations
    """
    with get_db() as conn:
        cursor = conn.cursor()

        if agent:
            cursor.execute("""
                SELECT * FROM summarizations
                WHERE agent = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (agent, limit))
        else:
            cursor.execute("""
                SELECT * FROM summarizations
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()

        summaries = []
        for row in rows:
            summaries.append({
                "summary_id": row["summary_id"],
                "agent": row["agent"],
                "run_id": row["run_id"],
                "trigger_reason": row["trigger_reason"],
                "ctx_before": row["ctx_before"],
                "ctx_after": row["ctx_after"],
                "summary_path": row["summary_path"],
                "created_at": row["created_at"]
            })

        return {"summaries": summaries}


@app.post("/clear")
async def clear_agent_context(agent: str):
    """
    Manually clear an agent's context (sets ctx_used to 0)

    Args:
        agent: Agent name

    Returns:
        - success: bool
        - agent: Agent name
        - ctx_before: Context usage before clearing
        - ts: Timestamp
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT ctx_used FROM agent_contexts WHERE agent = ?", (agent,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent} not found"
            )

        ctx_before = row["ctx_used"]

        cursor.execute("""
            UPDATE agent_contexts
            SET ctx_used = 0, status = 'ok', last_updated = ?
            WHERE agent = ?
        """, (datetime.utcnow().isoformat(), agent))

        conn.commit()

    return {
        "success": True,
        "agent": agent,
        "ctx_before": ctx_before,
        "ts": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6105)
