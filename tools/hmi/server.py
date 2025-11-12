
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

# Settings persistence
SETTINGS_FILE = pathlib.Path("artifacts/pas_settings.json")
DEFAULT_SETTINGS = {
    "hhmrs": {
        "heartbeat_interval_s": 30,
        "timeout_threshold_s": 60,
        "max_restarts": 3,
        "max_llm_retries": 3,
        "enable_auto_restart": True,
        "enable_llm_switching": True
    },
    "tron_chime": {
        "enabled": True,
        "sound": "ping",  # ping, bell, chime, alert, alarm
        "volume": 50,  # 0-100
        "chime_on_timeout": True,
        "chime_on_restart": False,
        "chime_on_escalation": True,
        "chime_on_permanent_failure": True
    },
    "hmi_display": {
        "show_tron_status_bar": True,
        "auto_refresh_interval_s": 5,
        "theme": "light",  # light, dark, auto
        "show_agent_tree": True,
        "show_metrics_panel": True
    },
    "notifications": {
        "email_enabled": False,
        "email_address": "",
        "slack_enabled": False,
        "slack_webhook_url": "",
        "notify_on_permanent_failure": True,
        "notify_on_completion": False
    }
}

def _load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def _save_settings(settings: Dict[str, Any]) -> bool:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
        return True
    except Exception:
        return False

@app.get("/api/settings")
def api_get_settings():
    return _load_settings()

@app.post("/api/settings")
def api_save_settings(settings: Dict[str, Any]):
    if _save_settings(settings):
        return {"status": "ok", "message": "Settings saved successfully"}
    return JSONResponse({"status": "error", "message": "Failed to save settings"}, status_code=500)

@app.post("/api/settings/reset")
def api_reset_settings():
    if _save_settings(DEFAULT_SETTINGS):
        return {"status": "ok", "message": "Settings reset to defaults"}
    return JSONResponse({"status": "error", "message": "Failed to reset settings"}, status_code=500)

VENDOR_COLOR = {"aider":"#2f80ed","claude":"#8a2be2","gemini":"#1a73e8","codex":"#10b981"}

@app.get("/settings", response_class=HTMLResponse)
def settings_view():
    html_doc = """
<!doctype html><html><head><meta charset="utf-8"><title>PAS Settings</title>
<style>
:root {
    --primary: #2f80ed;
    --success: #27ae60;
    --warning: #f39c12;
    --danger: #e74c3c;
    --dark: #2c3e50;
    --light: #ecf0f1;
    --border: #ddd;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
}
.container {
    max-width: 900px;
    margin: 0 auto;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 30px;
}
h1 {
    margin: 0 0 10px 0;
    color: var(--dark);
}
.subtitle {
    color: #666;
    margin-bottom: 30px;
}
nav {
    margin-bottom: 20px;
}
nav a {
    text-decoration: none;
    color: var(--primary);
    margin-right: 15px;
}
nav a:hover {
    text-decoration: underline;
}
.section {
    margin-bottom: 30px;
    padding: 20px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #fafafa;
}
.section h2 {
    margin-top: 0;
    color: var(--dark);
    font-size: 18px;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 8px;
    margin-bottom: 20px;
}
.form-group {
    margin-bottom: 18px;
}
.form-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 6px;
    color: var(--dark);
}
.form-group .help-text {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}
input[type="number"], input[type="text"], input[type="email"], select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 14px;
    box-sizing: border-box;
}
input[type="range"] {
    width: 100%;
}
input[type="checkbox"] {
    margin-right: 8px;
    width: 18px;
    height: 18px;
    cursor: pointer;
}
.checkbox-label {
    display: flex;
    align-items: center;
    cursor: pointer;
}
.range-value {
    display: inline-block;
    margin-left: 10px;
    font-weight: 600;
    color: var(--primary);
}
.btn-group {
    display: flex;
    gap: 10px;
    margin-top: 30px;
}
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s;
}
.btn-primary {
    background: var(--primary);
    color: white;
}
.btn-primary:hover {
    background: #1a66cc;
}
.btn-secondary {
    background: #95a5a6;
    color: white;
}
.btn-secondary:hover {
    background: #7f8c8d;
}
.btn-danger {
    background: var(--danger);
    color: white;
}
.btn-danger:hover {
    background: #c0392b;
}
.alert {
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 20px;
    display: none;
}
.alert-success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.alert-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
</head><body>
<div class="container">
    <nav>
        <a href="/actions">← Back to Actions</a>
    </nav>

    <h1>PAS Settings</h1>
    <p class="subtitle">Configure HHMRS, TRON notifications, and HMI display preferences</p>

    <div id="alert" class="alert"></div>

    <form id="settings-form">
        <!-- HHMRS Settings -->
        <div class="section">
            <h2>⚡ HHMRS (Health Monitoring & Retry System)</h2>

            <div class="form-group">
                <label>Heartbeat Interval (seconds)</label>
                <input type="number" id="heartbeat_interval_s" min="10" max="120" value="30">
                <div class="help-text">How often agents send heartbeats (default: 30s)</div>
            </div>

            <div class="form-group">
                <label>Timeout Threshold (seconds)</label>
                <input type="number" id="timeout_threshold_s" min="30" max="300" value="60">
                <div class="help-text">TRON detects timeout after this duration (default: 60s = 2 missed heartbeats)</div>
            </div>

            <div class="form-group">
                <label>Max Restarts (Level 1)</label>
                <input type="number" id="max_restarts" min="0" max="10" value="3">
                <div class="help-text">How many times to restart agent with same config before escalating (default: 3)</div>
            </div>

            <div class="form-group">
                <label>Max LLM Retries (Level 2)</label>
                <input type="number" id="max_llm_retries" min="0" max="10" value="3">
                <div class="help-text">How many times to switch LLM before permanent failure (default: 3)</div>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="enable_auto_restart" checked>
                    Enable automatic restarts
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="enable_llm_switching" checked>
                    Enable LLM switching on failure
                </label>
            </div>
        </div>

        <!-- TRON Chime Settings -->
        <div class="section">
            <h2>🔔 TRON Chime Notifications</h2>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="chime_enabled" checked>
                    Enable chime notifications
                </label>
            </div>

            <div class="form-group">
                <label>Chime Sound</label>
                <select id="chime_sound">
                    <option value="ping">Ping (soft)</option>
                    <option value="bell">Bell (medium)</option>
                    <option value="chime">Chime (pleasant)</option>
                    <option value="alert">Alert (attention)</option>
                    <option value="alarm">Alarm (urgent)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Chime Volume: <span id="volume_display" class="range-value">50%</span></label>
                <input type="range" id="chime_volume" min="0" max="100" value="50" oninput="document.getElementById('volume_display').textContent = this.value + '%'">
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="chime_on_timeout" checked>
                    Chime on timeout detection
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="chime_on_restart">
                    Chime on agent restart
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="chime_on_escalation" checked>
                    Chime on escalation to parent
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="chime_on_permanent_failure" checked>
                    Chime on permanent failure
                </label>
            </div>
        </div>

        <!-- HMI Display Settings -->
        <div class="section">
            <h2>🖥️ HMI Display</h2>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="show_tron_status_bar" checked>
                    Show TRON status bar
                </label>
                <div class="help-text">Display thin TRON ORANGE alert bar at top when interventions occur</div>
            </div>

            <div class="form-group">
                <label>Auto-refresh Interval (seconds)</label>
                <input type="number" id="auto_refresh_interval_s" min="1" max="60" value="5">
                <div class="help-text">How often to refresh HMI dashboard (default: 5s)</div>
            </div>

            <div class="form-group">
                <label>Theme</label>
                <select id="theme">
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="auto">Auto (system)</option>
                </select>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="show_agent_tree" checked>
                    Show agent hierarchy tree
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="show_metrics_panel" checked>
                    Show metrics panel
                </label>
            </div>
        </div>

        <!-- Notification Settings -->
        <div class="section">
            <h2>📧 External Notifications</h2>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="email_enabled">
                    Enable email notifications
                </label>
            </div>

            <div class="form-group">
                <label>Email Address</label>
                <input type="email" id="email_address" placeholder="you@example.com">
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="slack_enabled">
                    Enable Slack notifications
                </label>
            </div>

            <div class="form-group">
                <label>Slack Webhook URL</label>
                <input type="text" id="slack_webhook_url" placeholder="https://hooks.slack.com/services/...">
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="notify_on_permanent_failure" checked>
                    Notify on permanent failures
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="notify_on_completion">
                    Notify on run completion
                </label>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="btn-group">
            <button type="submit" class="btn btn-primary">💾 Save Settings</button>
            <button type="button" class="btn btn-secondary" onclick="loadSettings()">🔄 Reload</button>
            <button type="button" class="btn btn-danger" onclick="resetSettings()">⚠️ Reset to Defaults</button>
        </div>
    </form>
</div>

<script>
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();

        // HHMRS
        document.getElementById('heartbeat_interval_s').value = data.hhmrs.heartbeat_interval_s;
        document.getElementById('timeout_threshold_s').value = data.hhmrs.timeout_threshold_s;
        document.getElementById('max_restarts').value = data.hhmrs.max_restarts;
        document.getElementById('max_llm_retries').value = data.hhmrs.max_llm_retries;
        document.getElementById('enable_auto_restart').checked = data.hhmrs.enable_auto_restart;
        document.getElementById('enable_llm_switching').checked = data.hhmrs.enable_llm_switching;

        // TRON Chime
        document.getElementById('chime_enabled').checked = data.tron_chime.enabled;
        document.getElementById('chime_sound').value = data.tron_chime.sound;
        document.getElementById('chime_volume').value = data.tron_chime.volume;
        document.getElementById('volume_display').textContent = data.tron_chime.volume + '%';
        document.getElementById('chime_on_timeout').checked = data.tron_chime.chime_on_timeout;
        document.getElementById('chime_on_restart').checked = data.tron_chime.chime_on_restart;
        document.getElementById('chime_on_escalation').checked = data.tron_chime.chime_on_escalation;
        document.getElementById('chime_on_permanent_failure').checked = data.tron_chime.chime_on_permanent_failure;

        // HMI Display
        document.getElementById('show_tron_status_bar').checked = data.hmi_display.show_tron_status_bar;
        document.getElementById('auto_refresh_interval_s').value = data.hmi_display.auto_refresh_interval_s;
        document.getElementById('theme').value = data.hmi_display.theme;
        document.getElementById('show_agent_tree').checked = data.hmi_display.show_agent_tree;
        document.getElementById('show_metrics_panel').checked = data.hmi_display.show_metrics_panel;

        // Notifications
        document.getElementById('email_enabled').checked = data.notifications.email_enabled;
        document.getElementById('email_address').value = data.notifications.email_address;
        document.getElementById('slack_enabled').checked = data.notifications.slack_enabled;
        document.getElementById('slack_webhook_url').value = data.notifications.slack_webhook_url;
        document.getElementById('notify_on_permanent_failure').checked = data.notifications.notify_on_permanent_failure;
        document.getElementById('notify_on_completion').checked = data.notifications.notify_on_completion;

        showAlert('Settings loaded successfully', 'success');
    } catch (error) {
        showAlert('Failed to load settings: ' + error.message, 'error');
    }
}

async function saveSettings(event) {
    event.preventDefault();

    const settings = {
        hhmrs: {
            heartbeat_interval_s: parseInt(document.getElementById('heartbeat_interval_s').value),
            timeout_threshold_s: parseInt(document.getElementById('timeout_threshold_s').value),
            max_restarts: parseInt(document.getElementById('max_restarts').value),
            max_llm_retries: parseInt(document.getElementById('max_llm_retries').value),
            enable_auto_restart: document.getElementById('enable_auto_restart').checked,
            enable_llm_switching: document.getElementById('enable_llm_switching').checked
        },
        tron_chime: {
            enabled: document.getElementById('chime_enabled').checked,
            sound: document.getElementById('chime_sound').value,
            volume: parseInt(document.getElementById('chime_volume').value),
            chime_on_timeout: document.getElementById('chime_on_timeout').checked,
            chime_on_restart: document.getElementById('chime_on_restart').checked,
            chime_on_escalation: document.getElementById('chime_on_escalation').checked,
            chime_on_permanent_failure: document.getElementById('chime_on_permanent_failure').checked
        },
        hmi_display: {
            show_tron_status_bar: document.getElementById('show_tron_status_bar').checked,
            auto_refresh_interval_s: parseInt(document.getElementById('auto_refresh_interval_s').value),
            theme: document.getElementById('theme').value,
            show_agent_tree: document.getElementById('show_agent_tree').checked,
            show_metrics_panel: document.getElementById('show_metrics_panel').checked
        },
        notifications: {
            email_enabled: document.getElementById('email_enabled').checked,
            email_address: document.getElementById('email_address').value,
            slack_enabled: document.getElementById('slack_enabled').checked,
            slack_webhook_url: document.getElementById('slack_webhook_url').value,
            notify_on_permanent_failure: document.getElementById('notify_on_permanent_failure').checked,
            notify_on_completion: document.getElementById('notify_on_completion').checked
        }
    };

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings)
        });
        const data = await response.json();

        if (data.status === 'ok') {
            showAlert('Settings saved successfully! Changes will take effect immediately.', 'success');
        } else {
            showAlert('Failed to save settings: ' + data.message, 'error');
        }
    } catch (error) {
        showAlert('Failed to save settings: ' + error.message, 'error');
    }
}

async function resetSettings() {
    if (!confirm('Are you sure you want to reset all settings to defaults? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/api/settings/reset', {method: 'POST'});
        const data = await response.json();

        if (data.status === 'ok') {
            showAlert('Settings reset to defaults', 'success');
            await loadSettings();
        } else {
            showAlert('Failed to reset settings: ' + data.message, 'error');
        }
    } catch (error) {
        showAlert('Failed to reset settings: ' + error.message, 'error');
    }
}

function showAlert(message, type) {
    const alert = document.getElementById('alert');
    alert.textContent = message;
    alert.className = 'alert alert-' + type;
    alert.style.display = 'block';
    setTimeout(() => {
        alert.style.display = 'none';
    }, 5000);
}

document.getElementById('settings-form').addEventListener('submit', saveSettings);
loadSettings();
</script>
</body></html>
"""
    return HTMLResponse(html_doc)

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
nav {{ margin-bottom: 20px; }}
nav a {{ text-decoration: none; color: #2f80ed; margin-right: 15px; }}
nav a:hover {{ text-decoration: underline; }}
</style>
</head><body>
<nav>
    <a href="/settings">⚙️ Settings</a>
</nav>
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
