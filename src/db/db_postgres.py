"""Compatibility shim exposing Postgres helpers under ``src.db``."""

from ..db_postgres import (  # noqa: F401
    PG_DSN,
    PostgresDB,
    connect,
    insert_entry,
    upsert_vectors,
)

__all__ = [
    "PG_DSN",
    "PostgresDB",
    "connect",
    "insert_entry",
    "upsert_vectors",
]
