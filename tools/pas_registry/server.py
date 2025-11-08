
#!/usr/bin/env python3
import os, json, time, uuid, pathlib
from typing import Dict, Any
from fastapi import FastAPI, Request

PORT = int(os.getenv("PAS_REGISTRY_PORT", "6121"))
DATA_DIR = pathlib.Path(os.getenv("PAS_REGISTRY_DIR", "artifacts/registry"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SERVICES_PATH = DATA_DIR / "services.json"
ACTIONS_DIR = pathlib.Path("artifacts/actions"); ACTIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PAS Registry", version="0.1.0")

def _load_services() -> Dict[str, Any]:
    if SERVICES_PATH.exists():
        return json.loads(SERVICES_PATH.read_text())
    return {"services": {}}

def _save_services(obj: Dict[str, Any]):
    SERVICES_PATH.write_text(json.dumps(obj, indent=2))

@app.get("/health")
def health():
    return {"status": "ok", "port": PORT}

@app.post("/register")
async def register(req: Request):
    body = await req.json()
    data = _load_services()
    sid = body.get("service_id") or str(uuid.uuid4())
    body["service_id"] = sid
    body["last_seen"] = time.time()
    data["services"][sid] = body
    _save_services(data)
    return {"ok": True, "service_id": sid}

@app.put("/heartbeat")
async def heartbeat(req: Request):
    body = await req.json()
    sid = body.get("service_id")
    data = _load_services()
    if sid in data["services"]:
        data["services"][sid]["last_seen"] = time.time()
        _save_services(data)
        return {"ok": True}
    return {"ok": False, "error": "unknown service_id"}

@app.get("/services")
def services():
    return _load_services()

@app.post("/actions")
async def actions_post(req: Request):
    body = await req.json()
    rid = body.get("run_id") or f"action-{int(time.time()*1000)}"
    (ACTIONS_DIR / f"{rid}.json").write_text(json.dumps(body, indent=2))
    return {"ok": True, "run_id": rid}

@app.get("/actions")
def actions_get():
    items = []
    for p in ACTIONS_DIR.glob("*.json"):
        try:
            items.append(json.loads(p.read_text()))
        except Exception:
            continue
    items.sort(key=lambda x: x.get("ts",""), reverse=True)
    return {"items": items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tools.pas_registry.server:app", host="0.0.0.0", port=PORT, reload=False)
