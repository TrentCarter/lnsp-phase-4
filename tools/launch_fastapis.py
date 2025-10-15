#!/usr/bin/env python3
"""
Unified FastAPI launcher for LNSP services.

Starts multiple FastAPI apps (chunker, TMD router, GTR-T5, LVM, ingest, vec2text)
with a single command. Supports a global --reload switch that applies to ALL
launched services.

Usage examples:

  # Start all services without reload (recommended)
  ./venv/bin/python tools/launch_fastapis.py

  # Start all services with reload
  ./venv/bin/python tools/launch_fastapis.py --reload

  # Start a subset
  ./venv/bin/python tools/launch_fastapis.py --services chunker,tmd_router,ingest

  # Offset all default ports by +100 (e.g., for a second stack on same machine)
  ./venv/bin/python tools/launch_fastapis.py --port-offset 100

Notes:
- This script uses the current Python interpreter (sys.executable) to run
  `-m uvicorn ...`. To ensure correct env/deps, run it with your project venv:
  `./.venv/bin/python tools/launch_fastapis.py ...`
- Press Ctrl+C to stop. The launcher will gracefully terminate all child servers.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Service:
    name: str
    module_app: str  # e.g., "app.api.chunking:app"
    port: int
    host: str = "127.0.0.1"


# Default service registry (keep in sync with individual modules)
SERVICES: Dict[str, Service] = {
    "chunker": Service("chunker", "app.api.chunking:app", 8001),
    "tmd_router": Service("tmd_router", "app.api.tmd_router:app", 8002),
    "lvm": Service("lvm", "app.api.lvm_server:app", 8003),
    "ingest": Service("ingest", "app.api.ingest_chunks:app", 8004),
    "gtr_t5": Service("gtr_t5", "app.api.vec2text_embedding_server:app", 8767),
    "vec2text": Service("vec2text", "app.api.vec2text_server:app", 8766),
}


def build_uvicorn_cmd(py_exe: str, svc: Service, reload: bool, log_level: str) -> List[str]:
    cmd = [
        py_exe,
        "-m",
        "uvicorn",
        svc.module_app,
        "--host",
        svc.host,
        "--port",
        str(svc.port),
        "--log-level",
        log_level,
    ]
    if reload:
        cmd.append("--reload")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch multiple FastAPI services with a unified --reload flag")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn --reload for ALL services (high CPU on large repos)",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable uvicorn autoreload for ALL services (default)",
    )
    parser.set_defaults(reload=False)

    parser.add_argument(
        "--services",
        type=str,
        default=",".join(SERVICES.keys()),
        help=f"Comma-separated subset to launch. Available: {', '.join(SERVICES.keys())}",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (applies to all services)",
    )
    parser.add_argument(
        "--port-offset",
        type=int,
        default=0,
        help="Add an integer offset to each default port (e.g., +100)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level",
    )

    args = parser.parse_args()

    # Resolve Python interpreter (stick to current one)
    py_exe = sys.executable

    # Materialize service plan
    selected = [s.strip() for s in args.services.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SERVICES]
    if unknown:
        print(f"Unknown services: {', '.join(unknown)}")
        print(f"Available: {', '.join(SERVICES.keys())}")
        return 2

    # Apply host/port overrides
    launch_plan: List[Service] = []
    for name in selected:
        base = SERVICES[name]
        launch_plan.append(
            Service(
                name=base.name,
                module_app=base.module_app,
                port=base.port + args.port_offset,
                host=args.host,
            )
        )

    # Spawn processes
    procs: Dict[str, subprocess.Popen] = {}

    def terminate_all(sig: int = signal.SIGINT):
        if not procs:
            return
        print("\nStopping services...")
        # Send signal
        for name, p in procs.items():
            try:
                if p.poll() is None:
                    p.send_signal(sig)
            except Exception:
                pass
        # Wait a bit
        deadline = time.time() + 5
        while time.time() < deadline and any(p.poll() is None for p in procs.values()):
            time.sleep(0.1)
        # Force kill if needed
        for name, p in procs.items():
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    def handle_sigint(signum, frame):
        terminate_all(signal.SIGINT)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print("Launching services:\n")
    for svc in launch_plan:
        cmd = build_uvicorn_cmd(py_exe, svc, args.reload, args.log_level)
        env = os.environ.copy()
        # Keep environment as-is; user controls LNSP_* via env.
        print(f"  - {svc.name:10} http://{svc.host}:{svc.port}  ({svc.module_app})  reload={'on' if args.reload else 'off'}")
        try:
            p = subprocess.Popen(cmd, env=env)
            procs[svc.name] = p
        except FileNotFoundError:
            print(f"    ! Failed: uvicorn not found in interpreter {py_exe}. Ensure you run via your venv.")
            terminate_all()
            return 1
        except Exception as e:
            print(f"    ! Failed to start {svc.name}: {e}")
            terminate_all()
            return 1

    print("\nAll services launched. Press Ctrl+C to stop.")

    # Wait for any child to exit; if one dies, terminate others
    exit_code = 0
    try:
        while True:
            live = [(name, p) for name, p in procs.items() if p.poll() is None]
            if not live:
                break
            # If any process has exited, break and stop all
            dead = [(name, p) for name, p in procs.items() if p.poll() is not None]
            if dead:
                name, p = dead[0]
                print(f"\nService exited: {name} (code {p.returncode})")
                exit_code = p.returncode or exit_code
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        terminate_all()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
