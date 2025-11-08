
#!/usr/bin/env python3
import os, json, urllib.request
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException

PORT = int(os.getenv("PAS_GATEWAY_PORT", "6120"))
REGISTRY_URL = os.getenv("PAS_REGISTRY_URL", "http://127.0.0.1:6121").rstrip("/")

app = FastAPI(title="PAS Gateway", version="0.1.0")

def _get_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))

def _post_json(url: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

def _pick_service(services: Dict[str, Any], vendor: str = "", caps: List[str] = None) -> Dict[str, Any]:
    caps = caps or []
    candidates = []
    for s in services.get("services", {}).values():
        if vendor and s.get("vendor") != vendor:
            continue
        scaps = set(s.get("caps", []))
        if caps and not set(caps).issubset(scaps):
            continue
        candidates.append(s)
    for s in candidates:
        if s.get("role") == "execution":
            return s
    return candidates[0] if candidates else {}

@app.get("/health")
def health():
    return {"status": "ok", "port": PORT, "registry": REGISTRY_URL}

@app.post("/run/by-vendor/{vendor}")
async def run_by_vendor(vendor: str, req: Request):
    body = await req.json()
    services = _get_json(f"{REGISTRY_URL}/services")
    caps = body.get("policy", {}).get("require_caps", [])
    svc = _pick_service(services, vendor=vendor, caps=caps)
    if not svc:
        raise HTTPException(status_code=404, detail="No matching service")
    return _post_json(svc["url"] + "/invoke", body)

@app.post("/run/by-cap")
async def run_by_cap(req: Request):
    body = await req.json()
    caps = body.get("policy", {}).get("require_caps", [])
    services = _get_json(f"{REGISTRY_URL}/services")
    svc = _pick_service(services, vendor="", caps=caps)
    if not svc:
        raise HTTPException(status_code=404, detail="No matching service")
    return _post_json(svc["url"] + "/invoke", body)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tools.pas_gateway.server:app", host="0.0.0.0", port=PORT, reload=False)
