#!/usr/bin/env python3
"""Rebuild the SQLite lookup index for CPESH Parquet segments."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pyarrow.parquet as pq

MANIFEST = Path("artifacts/cpesh_manifest.jsonl")
DB_PATH = Path("artifacts/cpesh_index.db")


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cpesh_index (
            cpe_id TEXT PRIMARY KEY,
            segment_id TEXT,
            segment_path TEXT NOT NULL,
            row_offset INTEGER NOT NULL,
            lane_index INTEGER,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cpesh_segment
        ON cpesh_index(segment_id, row_offset)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cpesh_lane
        ON cpesh_index(lane_index)
        """
    )
    conn.commit()


def iter_manifest() -> Iterator[dict]:
    if not MANIFEST.exists():
        return iter(())

    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_rows(entry: dict) -> Iterable[Tuple[str, str, str, int, int | None, str | None]]:
    path = entry.get("path")
    if not path:
        return []

    parquet_path = Path(path)
    if not parquet_path.exists():
        return []

    segment_id = entry.get("segment_id") or parquet_path.stem
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as exc:  # pragma: no cover - IO failure
        print(f"[cpesh-index] failed to open {parquet_path}: {exc}")
        return []

    offset = 0
    columns = ["cpe_id", "lane_index", "created_at"]

    for batch in parquet_file.iter_batches(columns=columns, batch_size=2048):
        data = batch.to_pydict()
        cpe_ids = data.get("cpe_id", [])
        lanes = data.get("lane_index", [])
        created = data.get("created_at", [])

        for idx, cpe_id in enumerate(cpe_ids):
            if not cpe_id:
                continue
            absolute = offset + idx
            lane_val = None
            if lanes:
                lane_val = lanes[idx]
            created_at = None
            if created:
                created_at = created[idx]
            yield (
                str(cpe_id),
                segment_id,
                str(parquet_path),
                int(absolute),
                int(lane_val) if isinstance(lane_val, (int, float)) else None,
                str(created_at) if created_at is not None else None,
            )
        offset += len(cpe_ids)


def refresh_index() -> None:
    if not MANIFEST.exists():
        print("[cpesh-index] manifest missing; nothing to index")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM cpesh_index")

        inserted = 0
        for entry in iter_manifest():
            rows = list(iter_rows(entry))
            if not rows:
                continue
            cur.executemany(
                """
                INSERT OR REPLACE INTO cpesh_index
                (cpe_id, segment_id, segment_path, row_offset, lane_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted += len(rows)
            conn.commit()

        print(f"[cpesh-index] indexed {inserted} records into {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    refresh_index()
