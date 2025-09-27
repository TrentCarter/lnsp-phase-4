#!/usr/bin/env python3
"""Sample CPESH records across storage tiers for quick inspection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

import pyarrow.dataset as ds

ACTIVE_DEFAULT = Path("artifacts/cpesh_active.jsonl")
MANIFEST_DEFAULT = Path("artifacts/cpesh_manifest.jsonl")


def sample_active(path: Path, limit: int) -> List[dict]:
    if limit <= 0 or not path.exists():
        return []

    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append(record)
            if len(samples) >= limit:
                break
    return samples


def sample_segments(manifest: Path, limit: int) -> List[dict]:
    if limit <= 0 or not manifest.exists():
        return []

    entries = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return []

    per_segment = max(1, limit // len(entries))
    chosen: List[dict] = []

    for entry in entries:
        if len(chosen) >= limit:
            break
        path = entry.get("path")
        if not path:
            continue
        dataset = ds.dataset(path, format="parquet")
        total = dataset.count()
        if total == 0:
            continue
        take = min(per_segment, total, limit - len(chosen))
        indices = sorted(random.sample(range(total), take)) if take < total else list(range(total))
        if not indices:
            continue
        table = dataset.take(indices)
        chosen.extend(table.to_pylist())

    return chosen[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample CPESH records across tiers")
    parser.add_argument("--k", type=int, default=100, help="Total samples to collect")
    parser.add_argument("--active-frac", type=int, default=30, help="Percentage of samples from the active tier")
    parser.add_argument("--active", type=Path, default=ACTIVE_DEFAULT, help="Path to active CPESH JSONL")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT, help="Path to CPESH manifest JSONL")
    args = parser.parse_args()

    k = max(0, args.k)
    active_take = min(k, max(0, int(k * args.active_frac / 100)))
    segment_take = k - active_take

    active_records = sample_active(args.active, active_take)
    segment_records = sample_segments(args.manifest, segment_take)

    report = {
        "requested": k,
        "active_samples": len(active_records),
        "segment_samples": len(segment_records),
        "total": len(active_records) + len(segment_records),
    }

    print(json.dumps(report))


if __name__ == "__main__":
    main()
