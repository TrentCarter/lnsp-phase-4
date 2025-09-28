"""Helper utilities for inspecting CPESH data payloads.

This module exposes `snapshot_cpesh_records` which surfaces merged chunk
metadata, CPESH fields, and human-readable TMD labels so we can manually
spot-check record completeness without relying on automated heuristics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Ensure `src/` imports resolve when invoked via pytest or ad-hoc scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tmd_extractor_v2 import extract_tmd_from_text  # noqa: E402


@dataclass
class ChunkRecord:
    chunk_id: str
    contents: str
    metadata: Dict[str, Any]


def _load_chunk_records(chunk_path: Path) -> Dict[str, ChunkRecord]:
    records: Dict[str, ChunkRecord] = {}
    if not chunk_path.exists():
        return records

    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            chunk_id = raw.get("id") or raw.get("chunk_id")
            if not chunk_id:
                continue
            records[chunk_id] = ChunkRecord(
                chunk_id=chunk_id,
                contents=raw.get("contents", ""),
                metadata=raw.get("metadata", {}),
            )
    return records


def _join_chunk_texts(source_ids: Iterable[str], chunk_map: Dict[str, ChunkRecord]) -> str:
    texts: List[str] = []
    for chunk_id in source_ids:
        record = chunk_map.get(chunk_id)
        if record and record.contents:
            texts.append(record.contents)
    return "\n\n".join(texts)


def snapshot_cpesh_records(
    limit: int = 5,
    *,
    cpesh_path: Path | str = "artifacts/cpesh_active_fixed.jsonl",
    chunk_path: Path | str = "artifacts/fw10k_chunks.jsonl",
) -> List[Dict[str, Any]]:
    """Return a structured snapshot of CPESH records for manual validation."""

    cpesh_file = Path(cpesh_path)
    chunk_file = Path(chunk_path)
    chunk_map = _load_chunk_records(chunk_file)

    results: List[Dict[str, Any]] = []
    if not cpesh_file.exists():
        return results

    with cpesh_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            cpesh_payload = record.get("cpesh", {})
            source_ids = record.get("source_ids", [])
            chunk_text = _join_chunk_texts(source_ids, chunk_map)

            tmd_snapshot = None
            if chunk_text:
                tmd_info = extract_tmd_from_text(chunk_text)
                tmd_snapshot = {
                    "code": f"{tmd_info['domain_code']}.{tmd_info['task_code']}.{tmd_info['modifier_code']}",
                    "labels": {
                        "domain": tmd_info.get("domain"),
                        "task": tmd_info.get("task"),
                        "modifier": tmd_info.get("modifier"),
                    },
                    "confidence": tmd_info.get("confidence"),
                }

            aggregated_metadata = [
                {
                    "chunk_id": chunk_id,
                    "metadata": chunk_map.get(chunk_id).metadata if chunk_id in chunk_map else {},
                }
                for chunk_id in source_ids
            ]

            results.append(
                {
                    "doc_id": record.get("doc_id"),
                    "chunk_id": record.get("chunk_id"),
                    "source_ids": source_ids,
                    "word_count": record.get("word_count"),
                    "chunk_count": record.get("chunk_count"),
                    "access_count": record.get("access_count"),
                    "cpesh": {
                        "concept": cpesh_payload.get("concept"),
                        "probe": cpesh_payload.get("probe"),
                        "expected": cpesh_payload.get("expected"),
                        "soft_negative": cpesh_payload.get("soft_negative"),
                        "hard_negative": cpesh_payload.get("hard_negative"),
                        "created_at": cpesh_payload.get("created_at"),
                        "last_accessed": cpesh_payload.get("last_accessed"),
                        "generation_method": cpesh_payload.get("generation_method"),
                    },
                    "chunk_text": chunk_text,
                    "tmd": tmd_snapshot,
                    "chunk_metadata": aggregated_metadata,
                }
            )

    return results


if __name__ == "__main__":
    snapshot = snapshot_cpesh_records(limit=3)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
