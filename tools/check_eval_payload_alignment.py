#!/usr/bin/env python3
"""
Verify that evaluation truth keys align with payload metadata.

Reports coverage statistics and highlights missing (article_index, chunk_index)
pairs to catch dataset / payload mismatches.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Pair = Tuple[int, int]


def load_truth_pairs(eval_npz: Path) -> Sequence[Pair]:
    data = np.load(eval_npz, allow_pickle=True)
    if "truth_keys" not in data:
        raise KeyError(f"{eval_npz} missing 'truth_keys'")
    keys = data["truth_keys"]
    if keys.ndim != 2 or keys.shape[1] != 2:
        raise ValueError(f"'truth_keys' must have shape [N,2]; got {keys.shape}")
    return [tuple(map(int, pair)) for pair in keys]


def load_payload_pairs(payload_path: Path) -> Dict[Pair, int]:
    payload_obj = np.load(payload_path, allow_pickle=True)
    if isinstance(payload_obj, np.lib.npyio.NpzFile):
        mapping = {}
        for key in payload_obj.files:
            entry = payload_obj[key]
            if isinstance(entry, np.ndarray) and entry.dtype == object:
                entry = entry.item()
            if not isinstance(entry, dict) or "meta" not in entry:
                continue
            meta = entry["meta"]
            pair = (int(meta["article_index"]), int(meta["chunk_index"]))
            mapping[pair] = mapping.get(pair, 0) + 1
        return mapping

    if not hasattr(payload_obj, "item"):
        raise TypeError(f"{payload_path} must store a dict; got {type(payload_obj)}")

    mapping = {}
    data = payload_obj.item()
    for value in data.values():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        meta = value[1]
        if not isinstance(meta, dict):
            continue
        pair = (int(meta["article_index"]), int(meta["chunk_index"]))
        mapping[pair] = mapping.get(pair, 0) + 1
    return mapping


@dataclass
class AlignmentReport:
    eval_pairs: int
    eval_unique_pairs: int
    payload_pairs: int
    payload_unique_pairs: int
    missing_pairs: int
    coverage: float
    missing_examples: List[Pair]
    offset_coverage: float | None = None

    def to_dict(self) -> dict:
        out = {
            "eval_total": self.eval_pairs,
            "eval_unique": self.eval_unique_pairs,
            "payload_total": self.payload_pairs,
            "payload_unique": self.payload_unique_pairs,
            "missing": self.missing_pairs,
            "coverage": self.coverage,
            "missing_examples": self.missing_examples,
        }
        if self.offset_coverage is not None:
            out["offset_plus_one_coverage"] = self.offset_coverage
        return out


def build_report(eval_pairs: Sequence[Pair], payload_map: Dict[Pair, int]) -> AlignmentReport:
    payload_keys = set(payload_map.keys())
    missing = [pair for pair in eval_pairs if pair not in payload_keys]
    coverage = 1.0 - (len(missing) / len(eval_pairs)) if eval_pairs else 1.0

    # Check if an off-by-one article index shift would cover the missing keys
    if missing:
        shifted_hits = sum((pair[0] + 1, pair[1]) in payload_keys for pair in missing)
        offset_coverage = (len(eval_pairs) - len(missing) + shifted_hits) / len(eval_pairs)
    else:
        offset_coverage = None

    return AlignmentReport(
        eval_pairs=len(eval_pairs),
        eval_unique_pairs=len(set(eval_pairs)),
        payload_pairs=sum(payload_map.values()),
        payload_unique_pairs=len(payload_map),
        missing_pairs=len(missing),
        coverage=coverage,
        missing_examples=missing[:20],
        offset_coverage=offset_coverage,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True, help="Evaluation NPZ with truth_keys")
    ap.add_argument("--payload", type=Path, required=True, help="Payload .npy/.npz file")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    eval_pairs = load_truth_pairs(args.eval)
    payload_map = load_payload_pairs(args.payload)
    report = build_report(eval_pairs, payload_map)

    payload_articles = {pair[0] for pair in payload_map}
    payload_chunks = {pair[1] for pair in payload_map}

    summary = report.to_dict()
    summary["eval_article_range"] = [min(pair[0] for pair in eval_pairs), max(pair[0] for pair in eval_pairs)]
    summary["payload_article_range"] = [min(payload_articles), max(payload_articles)]
    summary["eval_chunk_range"] = [min(pair[1] for pair in eval_pairs), max(pair[1] for pair in eval_pairs)]
    summary["payload_chunk_range"] = [min(payload_chunks), max(payload_chunks)]

    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
