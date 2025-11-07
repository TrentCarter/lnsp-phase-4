"""
Aider RPC Wrapper for PAS

This package provides a FastAPI-based RPC wrapper around the Aider coding agent,
enabling integration with the PAS (Project Agentic System) architecture.

Components:
- server.py: FastAPI service with /invoke, /health, /describe endpoints
- allowlist.py: Command sandboxing and file ACL enforcement
- redact.py: Secrets scrubbing for logs and diffs
- receipts.py: Cost and KPI tracking for Token Governor integration
- heartbeat.py: Registry registration and health reporting

Usage:
    export PAS_PORT=6150
    export PAS_REGISTRY_URL=http://127.0.0.1:6121
    python -m tools.aider_rpc.server
"""

__version__ = "0.1.0"
__all__ = ["server", "allowlist", "redact", "receipts", "heartbeat"]
