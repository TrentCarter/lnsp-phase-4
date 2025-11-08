
#!/usr/bin/env python3
import os, json, pathlib, html
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

PORT = int(os.getenv("PAS_HMI_PORT", "6101"))
ACTIONS_DIR = pathlib.Path("artifacts/actions"); ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
COST_DIR = pathlib.Path("artifacts/costs"); COST_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PAS HMI", version="0.1.0")

def _load_actions() -> List[Dict[str, Any]]:
    items = []
    for p in ACTIONS_DIR.glob("*.json"):
        try:
            items.append(json.loads(p.read_text()))
        except Exception:
            continue
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items

@app.get("/health")
def health():
    return {"status": "ok", "port": PORT}

@app.get("/api/actions")
def api_actions():
    return {"items": _load_actions()}

@app.get("/api/receipt/{run_id}")
def api_receipt(run_id: str):
    p = COST_DIR / f"{run_id}.json"
    if p.exists():
        return JSONResponse(json.loads(p.read_text()))
    return JSONResponse({"error": "not found"}, status_code=404)

VENDOR_COLOR = {"aider":"#2f80ed","claude":"#8a2be2","gemini":"#1a73e8","codex":"#10b981"}

@app.get("/actions", response_class=HTMLResponse)
def actions_view():
    items = _load_actions()
    rows = []
    for it in items:
        vendor = (it.get("vendor") or "").lower()
        color = VENDOR_COLOR.get(vendor, "#888")
        badge = f'<span style="background:{color};color:#fff;padding:2px 6px;border-radius:6px;font-size:12px">{html.escape(vendor or "n/a")}</span>'
        rid = html.escape(it.get("run_id",""))
        status = html.escape(it.get("status",""))
        msg = html.escape(it.get("message",""))
        lat = it.get("timings_ms",{}).get("total","")
        cost = it.get("cost_usd","")
        link = f'/api/receipt/{rid}' if rid else '#'
        rows.append(f"<tr><td>{badge}</td><td>{rid}</td><td>{status}</td><td>{lat}</td><td>{cost}</td><td>{msg}</td><td><a href='{link}' target='_blank'>receipt</a></td></tr>")
    html_doc = f"""
<!doctype html><html><head><meta charset="utf-8"><title>PAS Actions</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #fafafa; }}
</style>
</head><body>
<h2>Actions</h2>
<table>
<thead><tr><th>Vendor</th><th>Run ID</th><th>Status</th><th>Latency (ms)</th><th>Cost (USD)</th><th>Message</th><th>Receipt</th></tr></thead>
<tbody>
{''.join(rows) if rows else "<tr><td colspan='7'>No actions yet.</td></tr>"}
</tbody></table>
</body></html>
"""
    return HTMLResponse(html_doc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tools.hmi.server:app", host="0.0.0.0", port=PORT, reload=False)
