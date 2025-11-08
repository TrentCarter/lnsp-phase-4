
#!/usr/bin/env python3
import os, json, uuid, time, pathlib, urllib.request
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field

PORT = int(os.getenv("PAS_CLAUDE_PORT", "6151"))
SERVICE_ID = os.getenv("PAS_SERVICE_ID", str(uuid.uuid4()))
REGISTRY_URL = os.getenv("PAS_REGISTRY_URL", "").rstrip("/")
COST_DIR = pathlib.Path(os.getenv("PAS_COST_DIR", "artifacts/costs")); COST_DIR.mkdir(parents=True, exist_ok=True)
ACT_DIR = pathlib.Path(os.getenv("PAS_ACTION_DIR", "artifacts/actions")); ACT_DIR.mkdir(parents=True, exist_ok=True)

class Target(BaseModel):
    name: str = "Claude-LCO"
    type: str = "host"
    role: str = "execution"

class Payload(BaseModel):
    message: str
    files: List[str] = []
    dry_run: bool = False

class Policy(BaseModel):
    timeout_s: int = 120
    require_caps: List[str] = Field(default_factory=lambda: ["git-edit"])

class InvokeRequest(BaseModel):
    target: Target = Target()
    payload: Payload
    policy: Policy = Policy()
    run_id: Optional[str] = ""

app = FastAPI(title="Claude-LCO RPC", version="0.1.0")

def _write_json_atomic(path: pathlib.Path, data: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def _post_json(url: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

@app.get("/health")
def health():
    return {"status": "ok", "service_id": SERVICE_ID, "vendor": "claude", "port": PORT}

@app.get("/describe")
def describe():
    return {
        "name": "Claude-LCO",
        "service_id": SERVICE_ID,
        "vendor": "claude",
        "caps": ["git-edit", "files", "diff"],
        "ctx_limit": 200000,
        "cost_tier": "high",
        "role": "execution",
        "url": f"http://127.0.0.1:{PORT}",
    }

def log_action(run_id: str, payload: Payload, status: str, timings_ms: Dict[str, float], cost: float = 0.0):
    action = {
        "run_id": run_id,
        "vendor": "claude",
        "service": "Claude-LCO",
        "service_id": SERVICE_ID,
        "message": payload.message,
        "files": payload.files,
        "status": status,
        "timings_ms": timings_ms,
        "cost_usd": cost,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(ACT_DIR / f"{run_id}.json", action)
    if REGISTRY_URL:
        try:
            _post_json(f"{REGISTRY_URL}/actions", action)
        except Exception:
            pass

@app.post("/invoke")
def invoke(req: InvokeRequest, bg: BackgroundTasks):
    run_id = req.run_id or f"claude-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    # TODO: integrate real Claude Code edit here (Anthropic API or CLI), respecting allowlist/redaction
    time.sleep(0.05)  # simulate small latency
    total = (time.time() - t0) * 1000.0
    receipt = {
        "run_id": run_id,
        "resolved": {
            "name": "Claude-LCO",
            "vendor": "claude",
            "role": req.target.role,
            "url": f"http://127.0.0.1:{PORT}",
            "service_id": SERVICE_ID,
        },
        "status": "ok",
        "timings_ms": {"total": round(total, 2)},
        "cost_estimate": {"usd": 0.0},
        "kpis": {"files_changed": 0},
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(COST_DIR / f"{run_id}.json", receipt)
    bg.add_task(log_action, run_id, req.payload, receipt["status"], receipt["timings_ms"], 0.0)
    return {"routing_receipt": receipt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tools.claude_rpc.server:app", host="0.0.0.0", port=PORT, reload=False)
