#!/usr/bin/env python3
"""
Remap evaluation truth keys to match payload metadata.

Filters eval data to only include (article_index, chunk_index) pairs that
exist in the payload, ensuring 100% coverage for validation.

Usage:
    python tools/remap_eval_to_payload.py \
        --eval artifacts/lvm/eval_v2_ready.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --out artifacts/lvm/eval_v2_ready_aligned.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np


Pair = Tuple[int, int]


def load_eval_data(eval_npz: Path) -> dict[str, np.ndarray]:
    """Load evaluation NPZ and return all arrays in a dict."""
    data = np.load(eval_npz, allow_pickle=True)

    required = ["truth_keys"]
    for key in required:
        if key not in data:
            raise KeyError(f"{eval_npz} missing required key: {key}")

    # Load all available keys
    result = {}
    for key in data.files:
        result[key] = data[key]

    print(f"Loaded eval data:")
    for key, arr in result.items():
        print(f"  {key}: {arr.shape}")

    return result


def load_payload_pairs(payload_path: Path) -> Set[Pair]:
    """Load payload and extract all (article_index, chunk_index) pairs."""
    payload_obj = np.load(payload_path, allow_pickle=True)

    if isinstance(payload_obj, np.lib.npyio.NpzFile):
        pairs = set()
        for key in payload_obj.files:
            entry = payload_obj[key]
            if isinstance(entry, np.ndarray) and entry.dtype == object:
                entry = entry.item()
            if not isinstance(entry, dict) or "meta" not in entry:
                continue
            meta = entry["meta"]
            pair = (int(meta["article_index"]), int(meta["chunk_index"]))
            pairs.add(pair)
        return pairs

    if not hasattr(payload_obj, "item"):
        raise TypeError(f"{payload_path} must store a dict; got {type(payload_obj)}")

    pairs = set()
    data = payload_obj.item()
    for value in data.values():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        meta = value[1]
        if not isinstance(meta, dict):
            continue
        pair = (int(meta["article_index"]), int(meta["chunk_index"]))
        pairs.add(pair)

    print(f"\nPayload contains {len(pairs)} unique (article, chunk) pairs")

    return pairs


def remap_eval_data(
    eval_data: dict[str, np.ndarray],
    payload_pairs: Set[Pair],
) -> tuple[dict[str, np.ndarray], dict]:
    """Filter eval data to only include pairs that exist in payload."""

    truth_keys = eval_data["truth_keys"]

    # Convert truth_keys to pairs
    eval_pairs = [tuple(map(int, pair)) for pair in truth_keys]

    # Find valid indices (pairs that exist in payload)
    valid_indices = []
    missing_pairs = []

    for i, pair in enumerate(eval_pairs):
        if pair in payload_pairs:
            valid_indices.append(i)
        else:
            missing_pairs.append(pair)

    valid_indices = np.array(valid_indices)

    # Filter all arrays in the eval data
    new_data = {}
    for key, arr in eval_data.items():
        new_data[key] = arr[valid_indices]

    # Build report
    report = {
        "original_samples": len(eval_pairs),
        "valid_samples": len(valid_indices),
        "dropped_samples": len(missing_pairs),
        "coverage_before": 0.0,
        "coverage_after": 1.0,
        "missing_examples": missing_pairs[:20],
    }

    if len(eval_pairs) > 0:
        report["coverage_before"] = len(valid_indices) / len(eval_pairs)

    print(f"\nRemapping results:")
    print(f"  Original samples: {report['original_samples']}")
    print(f"  Valid samples: {report['valid_samples']}")
    print(f"  Dropped samples: {report['dropped_samples']}")
    print(f"  Coverage before: {report['coverage_before']:.2%}")
    print(f"  Coverage after: {report['coverage_after']:.2%}")

    if missing_pairs:
        print(f"\nFirst {min(20, len(missing_pairs))} missing pairs:")
        for pair in missing_pairs[:20]:
            print(f"    {pair}")

    return new_data, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True, help="Evaluation NPZ to remap")
    ap.add_argument("--payload", type=Path, required=True, help="Payload .npy/.npz file")
    ap.add_argument("--out", type=Path, required=True, help="Output aligned NPZ path")
    ap.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    # Load data
    eval_data = load_eval_data(args.eval)
    payload_pairs = load_payload_pairs(args.payload)

    # Remap
    new_data, report = remap_eval_data(eval_data, payload_pairs)

    # Save aligned eval data
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **new_data)
    print(f"\nSaved aligned eval to: {args.out}")

    # Save report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Saved report to: {args.report}")

    # Verify 100% coverage
    print("\nVerifying coverage...")
    final_pairs = [tuple(map(int, pair)) for pair in new_data["truth_keys"]]
    all_in_payload = all(pair in payload_pairs for pair in final_pairs)

    if all_in_payload:
        print("SUCCESS: 100% coverage achieved")
        sys.exit(0)
    else:
        print("ERROR: Coverage is not 100%, something went wrong")
        sys.exit(1)


if __name__ == "__main__":
    main()
