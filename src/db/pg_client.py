"""Compatibility helpers offering the legacy `get_pg_connection` entrypoint."""

from __future__ import annotations

from typing import Any

from src.db_postgres import connect

_connection: Any | None = None


def get_pg_connection() -> Any:
    """Return a cached psycopg2 connection using the shared project DSN."""
    global _connection

    if _connection is not None:
        # ``connection.closed`` is 0 when open; non-zero values indicate closed.
        if getattr(_connection, "closed", 1) == 0:  # pragma: no branch - simple check
            return _connection

    _connection = connect()
    return _connection
