#!/usr/bin/env python3
"""
Model Pool Manager — Port 8050
Dynamic LLM model lifecycle management with automatic load/unload.

Manages model services on ports 8051-8099 with:
- Automatic loading/unloading based on TTL
- Keep-alive tracking (default 15 min inactivity)
- Warmup models (qwen2.5-coder, llama3.1) stay loaded
- Service registration with PAS Registry
- OpenAI-compatible model endpoints

Author: Trent Carter
Date: 2025-11-11
"""

import asyncio
import httpx
import json
import psutil
import subprocess
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# Configuration
# ============================================================================

PORT = 8050
CONFIG_FILE = Path("configs/pas/model_pool_config.json")
REGISTRY_FILE = Path("configs/pas/model_pool_registry.json")
PAS_REGISTRY_URL = "http://localhost:6121"

# Default configuration
DEFAULT_CONFIG = {
    "default_ttl_minutes": 15,
    "min_ttl_minutes": 5,
    "max_ttl_minutes": 60,
    "check_interval_seconds": 30,
    "warmup_models": ["qwen2.5-coder:7b", "llama3.1:8b"],
    "port_range": {"start": 8051, "end": 8099},
    "max_concurrent_models": 5,
    "memory_limit_mb": 32000,
    "auto_unload": True,
    "graceful_shutdown_seconds": 30,
    "ollama_base_url": "http://localhost:11434"
}

# Default model registry
DEFAULT_REGISTRY = {
    "models": [
        {
            "model_id": "qwen2.5-coder:7b",
            "display_name": "Qwen 2.5 Coder 7B",
            "model_type": "code",
            "warmup": True,
            "ttl_minutes": 0,  # 0 = never unload
            "priority": "high",
            "tags": ["code", "python", "fast"]
        },
        {
            "model_id": "llama3.1:8b",
            "display_name": "Llama 3.1 8B",
            "model_type": "general",
            "warmup": True,
            "ttl_minutes": 0,
            "priority": "high",
            "tags": ["general", "reasoning"]
        },
        {
            "model_id": "deepseek-coder-v2:16b",
            "display_name": "DeepSeek Coder V2 16B",
            "model_type": "code",
            "warmup": False,
            "ttl_minutes": 15,
            "priority": "medium",
            "tags": ["code", "advanced", "large"]
        },
        {
            "model_id": "codellama:13b",
            "display_name": "Code Llama 13B",
            "model_type": "code",
            "warmup": False,
            "ttl_minutes": 15,
            "priority": "low",
            "tags": ["code", "fallback"]
        }
    ]
}


# ============================================================================
# Data Models
# ============================================================================

class ModelState(str, Enum):
    """Model lifecycle states"""
    COLD = "COLD"          # Not loaded, port available
    WARMING = "WARMING"    # Loading into Ollama
    HOT = "HOT"            # Loaded and ready
    COOLING = "COOLING"    # Grace period before unload
    UNLOADING = "UNLOADING"  # Actively removing


class ModelInfo(BaseModel):
    """Model metadata and state"""
    model_id: str
    display_name: str
    model_type: str
    warmup: bool
    ttl_minutes: int
    priority: str
    tags: List[str]

    # Runtime state
    state: ModelState = ModelState.COLD
    port: Optional[int] = None
    process: Optional[int] = None  # FastAPI subprocess PID
    last_request_time: Optional[datetime] = None
    load_time: Optional[datetime] = None
    request_count: int = 0
    memory_mb: Optional[int] = None


class ModelRegisterRequest(BaseModel):
    """Request to register a new model"""
    model_name: str
    display_name: Optional[str] = None
    model_type: str = "general"
    warmup: bool = False
    ttl_minutes: int = 15
    priority: str = "medium"
    tags: List[str] = Field(default_factory=list)


class ModelLoadRequest(BaseModel):
    """Request to force-load a model"""
    wait: bool = True  # Wait for model to be HOT before returning


class ModelUnloadRequest(BaseModel):
    """Request to force-unload a model"""
    force: bool = False  # Immediate unload vs graceful


class TTLExtendRequest(BaseModel):
    """Request to extend TTL"""
    minutes: int = Field(default=15, ge=1, le=60)


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration"""
    default_ttl_minutes: Optional[int] = None
    max_concurrent_models: Optional[int] = None
    check_interval_seconds: Optional[int] = None
    auto_unload: Optional[bool] = None


# ============================================================================
# Model Pool Manager
# ============================================================================

class ModelPoolManager:
    """Manages dynamic model loading/unloading with TTL tracking"""

    def __init__(self):
        self.config = self._load_config()
        self.registry = self._load_registry()
        self.models: Dict[str, ModelInfo] = {}
        self.port_pool = set(range(
            self.config["port_range"]["start"],
            self.config["port_range"]["end"] + 1
        ))
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self._ttl_task = None

        # Initialize models from registry
        for model_data in self.registry["models"]:
            model = ModelInfo(**model_data)
            self.models[model.model_id] = model

    def _load_config(self) -> dict:
        """Load configuration from file or use defaults"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
        else:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG

    def _load_registry(self) -> dict:
        """Load model registry from file or use defaults"""
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        else:
            REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REGISTRY_FILE, "w") as f:
                json.dump(DEFAULT_REGISTRY, f, indent=2)
            return DEFAULT_REGISTRY

    def _save_registry(self):
        """Persist registry to disk"""
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _allocate_port(self) -> int:
        """Allocate next available port"""
        if not self.port_pool:
            raise HTTPException(503, "No available ports")
        port = min(self.port_pool)
        self.port_pool.remove(port)
        return port

    def _release_port(self, port: int):
        """Release port back to pool"""
        self.port_pool.add(port)

    async def start_background_tasks(self):
        """Start TTL monitoring background task"""
        self._ttl_task = asyncio.create_task(self._ttl_monitor())

        # Load warmup models
        for model_id, model in self.models.items():
            if model.warmup:
                print(f"🔥 Loading warmup model: {model_id}")
                await self.load_model(model_id, wait=False)

    async def stop_background_tasks(self):
        """Stop background tasks and cleanup"""
        if self._ttl_task:
            self._ttl_task.cancel()
            try:
                await self._ttl_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        for model_id in list(self.models.keys()):
            if self.models[model_id].state not in [ModelState.COLD, ModelState.UNLOADING]:
                await self.unload_model(model_id, force=True)

    async def _ttl_monitor(self):
        """Background task to monitor TTL and unload idle models"""
        while True:
            try:
                await asyncio.sleep(self.config["check_interval_seconds"])

                if not self.config["auto_unload"]:
                    continue

                now = datetime.now()
                for model_id, model in self.models.items():
                    # Skip warmup models and non-HOT models
                    if model.warmup or model.state != ModelState.HOT:
                        continue

                    # Check if TTL expired
                    if model.last_request_time:
                        idle_time = (now - model.last_request_time).total_seconds() / 60
                        ttl = model.ttl_minutes or self.config["default_ttl_minutes"]

                        if idle_time >= ttl:
                            print(f"⏰ TTL expired for {model_id} ({idle_time:.1f} min idle)")
                            model.state = ModelState.COOLING

                            # Grace period before unload
                            await asyncio.sleep(120)  # 2 min grace

                            # Re-check state (might have received request during grace)
                            if model.state == ModelState.COOLING:
                                await self.unload_model(model_id)

            except Exception as e:
                print(f"❌ Error in TTL monitor: {e}")

    async def load_model(self, model_id: str, wait: bool = True) -> ModelInfo:
        """Load model into Ollama and start FastAPI service"""
        if model_id not in self.models:
            raise HTTPException(404, f"Model {model_id} not registered")

        model = self.models[model_id]

        # Check if already loaded
        if model.state in [ModelState.HOT, ModelState.WARMING]:
            if wait and model.state == ModelState.WARMING:
                await self._wait_for_hot(model_id)
            return model

        # Allocate port
        if not model.port:
            model.port = self._allocate_port()

        model.state = ModelState.WARMING
        model.load_time = datetime.now()

        print(f"🚀 Loading {model_id} on port {model.port}...")

        # Start FastAPI service for this model
        service_script = Path(__file__).parent / "model_service_template.py"
        process = subprocess.Popen([
            "./.venv/bin/python",
            str(service_script),
            "--model-id", model_id,
            "--port", str(model.port),
            "--ollama-url", self.config["ollama_base_url"]
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        model.process = process.pid

        # Wait for service to be ready
        await self._wait_for_service_ready(model.port)

        # Verify Ollama has model loaded
        await self._verify_ollama_model(model_id)

        model.state = ModelState.HOT
        print(f"✅ {model_id} ready on port {model.port}")

        # Register with PAS Registry
        await self._register_with_pas(model)

        return model

    async def unload_model(self, model_id: str, force: bool = False):
        """Unload model from Ollama and stop FastAPI service"""
        if model_id not in self.models:
            raise HTTPException(404, f"Model {model_id} not registered")

        model = self.models[model_id]

        if model.state == ModelState.COLD:
            return

        model.state = ModelState.UNLOADING

        print(f"🔄 Unloading {model_id}...")

        # Stop FastAPI service
        if model.process:
            try:
                process = psutil.Process(model.process)
                if not force:
                    process.terminate()
                    process.wait(timeout=self.config["graceful_shutdown_seconds"])
                else:
                    process.kill()
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass

        # Deregister from PAS Registry
        await self._deregister_from_pas(model)

        # Release port
        if model.port:
            self._release_port(model.port)
            model.port = None

        model.state = ModelState.COLD
        model.process = None
        model.memory_mb = None

        print(f"✅ {model_id} unloaded")

    async def extend_ttl(self, model_id: str, minutes: int):
        """Extend TTL for a model"""
        if model_id not in self.models:
            raise HTTPException(404, f"Model {model_id} not registered")

        model = self.models[model_id]
        model.last_request_time = datetime.now()
        print(f"⏱️  TTL extended for {model_id} (+{minutes} min)")

    async def _wait_for_service_ready(self, port: int, timeout: int = 120):
        """Wait for FastAPI service to be ready"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = await self.http_client.get(f"http://localhost:{port}/health")
                if resp.status_code == 200:
                    return
            except:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(f"Service on port {port} did not start in {timeout}s")

    async def _wait_for_hot(self, model_id: str, timeout: int = 120):
        """Wait for model to reach HOT state"""
        start = time.time()
        while time.time() - start < timeout:
            model = self.models[model_id]
            if model.state == ModelState.HOT:
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"Model {model_id} did not load in {timeout}s")

    async def _verify_ollama_model(self, model_id: str):
        """Verify Ollama has the model loaded"""
        try:
            resp = await self.http_client.get(f"{self.config['ollama_base_url']}/api/tags")
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model_id not in model_names:
                print(f"⚠️  Warning: {model_id} not found in Ollama, pulling...")
                # Trigger pull via Ollama API
                await self.http_client.post(
                    f"{self.config['ollama_base_url']}/api/pull",
                    json={"name": model_id}
                )
        except Exception as e:
            print(f"⚠️  Could not verify Ollama model: {e}")

    async def _register_with_pas(self, model: ModelInfo):
        """Register model service with PAS Registry"""
        try:
            await self.http_client.post(
                f"{PAS_REGISTRY_URL}/register",
                json={
                    "service_name": f"model_{model.model_id.replace(':', '_')}",
                    "service_type": "llm_model",
                    "host": "localhost",
                    "port": model.port,
                    "metadata": {
                        "model_id": model.model_id,
                        "model_type": model.model_type,
                        "tags": model.tags
                    }
                }
            )
        except Exception as e:
            print(f"⚠️  Could not register with PAS Registry: {e}")

    async def _deregister_from_pas(self, model: ModelInfo):
        """Deregister model service from PAS Registry"""
        try:
            service_name = f"model_{model.model_id.replace(':', '_')}"
            await self.http_client.delete(f"{PAS_REGISTRY_URL}/services/{service_name}")
        except Exception as e:
            print(f"⚠️  Could not deregister from PAS Registry: {e}")

    def get_status(self) -> dict:
        """Get overall pool status"""
        active_models = [m for m in self.models.values() if m.state in [ModelState.HOT, ModelState.WARMING]]
        total_memory = sum(m.memory_mb or 0 for m in active_models)

        return {
            "status": "healthy",
            "active_models": len(active_models),
            "total_models": len(self.models),
            "total_memory_mb": total_memory,
            "available_ports": len(self.port_pool),
            "config": self.config
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Model Pool Manager", version="1.0.0")
manager = ModelPoolManager()


@app.on_event("startup")
async def startup():
    """Initialize manager and start background tasks"""
    await manager.start_background_tasks()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await manager.stop_background_tasks()


# Health & Status
@app.get("/health")
async def health():
    """Health check endpoint"""
    return manager.get_status()


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    lines = []
    lines.append("# HELP model_pool_active_models Number of active models")
    lines.append("# TYPE model_pool_active_models gauge")
    active_count = len([m for m in manager.models.values() if m.state == ModelState.HOT])
    lines.append(f"model_pool_active_models {active_count}")

    lines.append("# HELP model_pool_memory_mb Total memory used by models")
    lines.append("# TYPE model_pool_memory_mb gauge")
    total_mem = sum(m.memory_mb or 0 for m in manager.models.values() if m.state == ModelState.HOT)
    lines.append(f"model_pool_memory_mb {total_mem}")

    lines.append("# HELP model_pool_requests_total Total requests per model")
    lines.append("# TYPE model_pool_requests_total counter")
    for model in manager.models.values():
        safe_name = model.model_id.replace(":", "_").replace(".", "_")
        lines.append(f'model_pool_requests_total{{model="{safe_name}"}} {model.request_count}')

    return "\n".join(lines)


# Model Registry
@app.get("/models")
async def list_models():
    """List all registered models with their states"""
    models_data = []
    for model in manager.models.values():
        uptime_minutes = 0
        if model.load_time and model.state == ModelState.HOT:
            uptime_minutes = (datetime.now() - model.load_time).total_seconds() / 60

        ttl_remaining_minutes = 0
        last_request_ago_seconds = 0
        if model.last_request_time and model.state == ModelState.HOT:
            idle_seconds = (datetime.now() - model.last_request_time).total_seconds()
            last_request_ago_seconds = int(idle_seconds)
            ttl = model.ttl_minutes or manager.config["default_ttl_minutes"]
            ttl_remaining_minutes = max(0, ttl - (idle_seconds / 60))

        models_data.append({
            "model_id": model.model_id,
            "display_name": model.display_name,
            "port": model.port,
            "state": model.state,
            "ttl_remaining_minutes": round(ttl_remaining_minutes, 1),
            "last_request_ago_seconds": last_request_ago_seconds,
            "request_count": model.request_count,
            "uptime_minutes": round(uptime_minutes, 1),
            "memory_mb": model.memory_mb,
            "warmup": model.warmup,
            "endpoint": f"http://localhost:{model.port}" if model.port else None
        })

    total_memory = sum(m.memory_mb or 0 for m in manager.models.values() if m.state == ModelState.HOT)

    return {
        "models": models_data,
        "total_memory_mb": total_memory,
        "available_ports": len(manager.port_pool)
    }


@app.post("/models/register")
async def register_model(req: ModelRegisterRequest):
    """Register a new model"""
    model_id = req.model_name

    if model_id in manager.models:
        raise HTTPException(400, f"Model {model_id} already registered")

    # Create model info
    model = ModelInfo(
        model_id=model_id,
        display_name=req.display_name or model_id,
        model_type=req.model_type,
        warmup=req.warmup,
        ttl_minutes=req.ttl_minutes,
        priority=req.priority,
        tags=req.tags
    )

    manager.models[model_id] = model

    # Update registry file
    manager.registry["models"].append({
        "model_id": model.model_id,
        "display_name": model.display_name,
        "model_type": model.model_type,
        "warmup": model.warmup,
        "ttl_minutes": model.ttl_minutes,
        "priority": model.priority,
        "tags": model.tags
    })
    manager._save_registry()

    return {
        "model_id": model_id,
        "state": model.state,
        "message": "Model registered successfully"
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from registry"""
    if model_id not in manager.models:
        raise HTTPException(404, f"Model {model_id} not found")

    model = manager.models[model_id]

    # Unload if active
    if model.state not in [ModelState.COLD, ModelState.UNLOADING]:
        await manager.unload_model(model_id, force=True)

    # Remove from registry
    del manager.models[model_id]
    manager.registry["models"] = [
        m for m in manager.registry["models"] if m["model_id"] != model_id
    ]
    manager._save_registry()

    return {
        "model_id": model_id,
        "message": "Model deleted successfully"
    }


# Model Lifecycle
@app.post("/models/{model_id}/load")
async def load_model(model_id: str, req: ModelLoadRequest = ModelLoadRequest()):
    """Load a model"""
    model = await manager.load_model(model_id, wait=req.wait)

    return {
        "model_id": model_id,
        "state": model.state,
        "port": model.port,
        "endpoint": f"http://localhost:{model.port}"
    }


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str, req: ModelUnloadRequest = ModelUnloadRequest()):
    """Unload a model"""
    await manager.unload_model(model_id, force=req.force)

    return {
        "model_id": model_id,
        "state": ModelState.COLD,
        "message": "Model unloaded successfully"
    }


@app.post("/models/{model_id}/extend-ttl")
async def extend_ttl(model_id: str, req: TTLExtendRequest):
    """Extend TTL for a model"""
    await manager.extend_ttl(model_id, req.minutes)

    model = manager.models[model_id]
    expires_at = None
    if model.last_request_time:
        ttl = model.ttl_minutes or manager.config["default_ttl_minutes"]
        expires_at = model.last_request_time + timedelta(minutes=ttl)

    return {
        "model_id": model_id,
        "ttl_remaining_minutes": req.minutes,
        "expires_at": expires_at.isoformat() if expires_at else None
    }


@app.get("/models/{model_id}/health")
async def model_health(model_id: str):
    """Check health of a specific model"""
    if model_id not in manager.models:
        raise HTTPException(404, f"Model {model_id} not found")

    model = manager.models[model_id]

    if model.state != ModelState.HOT or not model.port:
        return {
            "model_id": model_id,
            "state": model.state,
            "responsive": False
        }

    # Check if service is responsive
    try:
        start = time.time()
        resp = await manager.http_client.get(f"http://localhost:{model.port}/health")
        latency_ms = int((time.time() - start) * 1000)

        return {
            "model_id": model_id,
            "state": model.state,
            "endpoint": f"http://localhost:{model.port}",
            "responsive": resp.status_code == 200,
            "latency_ms": latency_ms,
            "last_error": None
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "state": model.state,
            "responsive": False,
            "last_error": str(e)
        }


# Configuration
@app.get("/config")
async def get_config():
    """Get current configuration"""
    return manager.config


@app.patch("/config")
async def update_config(req: ConfigUpdateRequest):
    """Update configuration"""
    if req.default_ttl_minutes is not None:
        manager.config["default_ttl_minutes"] = req.default_ttl_minutes
    if req.max_concurrent_models is not None:
        manager.config["max_concurrent_models"] = req.max_concurrent_models
    if req.check_interval_seconds is not None:
        manager.config["check_interval_seconds"] = req.check_interval_seconds
    if req.auto_unload is not None:
        manager.config["auto_unload"] = req.auto_unload

    # Save to file
    with open(CONFIG_FILE, "w") as f:
        json.dump(manager.config, f, indent=2)

    return {
        "status": "updated",
        "config": manager.config
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Starting Model Pool Manager on port {PORT}")
    print(f"📂 Config: {CONFIG_FILE}")
    print(f"📂 Registry: {REGISTRY_FILE}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
