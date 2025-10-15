#!/usr/bin/env python3
"""Quick quality checks for serialized episodes."""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, median
from typing import List

import numpy as np


def load_meta(indir: str) -> List[dict]:
    path = os.path.join(indir, "meta.jsonl")
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def describe(values: List[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={mean(values):.2f} median={median(values):.2f} "
        f"min={min(values):.2f} max={max(values):.2f}"
    )


def describe_coherence(values: List[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={mean(values):.3f} median={median(values):.3f} "
        f"min={min(values):.3f} max={max(values):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize episodic packs")
    parser.add_argument("--indir", required=True, help="Episode directory")
    args = parser.parse_args()

    metas = load_meta(args.indir)
    if not metas:
        print("No metadata entries found.")
        return 1

    lengths = [float(m.get("concept_count", 0)) for m in metas if m.get("concept_count")]
    cohs = [float(m.get("mean_coherence", 0.0)) for m in metas if m.get("mean_coherence") is not None]
    mins = [float(m.get("min_coherence", 0.0)) for m in metas if m.get("min_coherence") is not None]

    print(f"Episodes: {len(metas)}")
    print(f"Lengths: {describe(lengths)}")
    print(f"Mean coherence: {describe_coherence(cohs)}")
    print(f"Min coherence: {describe_coherence(mins)}")

    globbed = glob.glob(os.path.join(args.indir, "*.npz"))
    if globbed:
        sample = np.load(globbed[0])
        print(f"Sample episode: {os.path.basename(globbed[0])} -> X{sample['X'].shape} y{sample['y'].shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
