#!/usr/bin/env python3
"""
Diagnostics for TMD alpha tuning.

Reads one or more TMD diagnostics JSONL files produced when running
RAG/vecrag_tmd_rerank.py with TMD_DIAG=1, and summarizes:
- Spearman correlation distribution between vec and TMD channels
- Percentage of queries whose top-k changed after re-ranking
- Normalization collapse frequency
- LLM zero-vector frequency (if present in footer)

Usage:
  python tools/tmd_alpha_diagnostics.py --files RAG/results/tmd_diag_*.jsonl

Optional:
  --hist N   Print N-bin histogram of spearman correlations
  --alpha    Group results by alpha (if mixed files)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_diag_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    # unique preserve order
    seen = set()
    uniq: List[str] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def summarize(files: List[str], bins: int = 10, group_by_alpha: bool = False) -> None:
    if not files:
        print("[diag] No files matched.")
        return

    # Aggregates
    bucket: Dict[str, Dict[str, List[float]]] = {}
    footer_aggs: Dict[str, Dict[str, float]] = {}

    def key_for(alpha: float | None) -> str:
        return f"alpha={alpha:.3f}" if alpha is not None and group_by_alpha else "all"

    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                except Exception:
                    continue

                alpha = j.get("alpha")
                k = key_for(alpha)

                if j.get("summary"):
                    ag = footer_aggs.setdefault(k, {
                        "collapse_count": 0.0,
                        "changed_queries": 0.0,
                        "llm_zero_tmd": 0.0,
                        "total_queries": 0.0,
                    })
                    ag["collapse_count"] += float(j.get("collapse_count", 0.0))
                    ag["changed_queries"] += float(j.get("changed_queries", 0.0))
                    ag["llm_zero_tmd"] += float(j.get("llm_zero_tmd", 0.0))
                    ag["total_queries"] += float(j.get("total_queries", 0.0))
                    continue

                # per-query
                b = bucket.setdefault(k, {
                    "spearman": [],
                    "changed_positions": [],
                    "vec_collapsed": [],
                })
                if "spearman_vec_tmd" in j:
                    b["spearman"].append(float(j["spearman_vec_tmd"]))
                if "changed_positions" in j:
                    b["changed_positions"].append(float(j["changed_positions"]))
                if "vec_collapsed" in j:
                    b["vec_collapsed"].append(1.0 if j["vec_collapsed"] else 0.0)

    def print_hist(values: List[float], bins: int) -> None:
        if not values:
            print("  (no data)")
            return
        mn, mx = min(values), max(values)
        if mx - mn < 1e-8:
            print(f"  all values = {mn:.4f}")
            return
        step = (mx - mn) / bins
        counts = [0] * bins
        for v in values:
            idx = min(bins - 1, int((v - mn) / step))
            counts[idx] += 1
        for i, c in enumerate(counts):
            lo = mn + i * step
            hi = lo + step
            print(f"  [{lo:+.2f},{hi:+.2f}): {c}")

    # Print summary
    for k in sorted(bucket.keys() | footer_aggs.keys()):
        spe = bucket.get(k, {}).get("spearman", [])
        chg = bucket.get(k, {}).get("changed_positions", [])
        col = bucket.get(k, {}).get("vec_collapsed", [])
        ft = footer_aggs.get(k, {})

        total = int(ft.get("total_queries", len(chg)))
        changed = int(ft.get("changed_queries", sum(1 for x in chg if x > 0)))
        collapse = int(ft.get("collapse_count", sum(col)))
        llm_zero = int(ft.get("llm_zero_tmd", 0))
        ratio = (changed / total) if total > 0 else 0.0

        print("=" * 72)
        print(f"Diagnostics Summary [{k}] from {len(files)} file(s)")
        print("-" * 72)
        print(f"Queries:	{total}")
        print(f"Changed@topk:	{changed} ({ratio*100:.1f}%)")
        print(f"Vec collapsed:	{collapse}")
        print(f"LLM zero TMD:	{llm_zero}")
        if spe:
            print(f"Spearman r:	mean={sum(spe)/len(spe):+.4f} min={min(spe):+.4f} max={max(spe):+.4f}")
            if bins > 0:
                print("Spearman histogram:")
                print_hist(spe, bins)
        else:
            print("Spearman r:	(no data)")

    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["RAG/results/tmd_diag_*.jsonl"], help="Glob(s) to diagnostics JSONL files")
    ap.add_argument("--hist", type=int, default=10, help="Histogram bins for Spearman (0 to disable)")
    ap.add_argument("--alpha", action="store_true", help="Group summaries by alpha value")
    args = ap.parse_args()

    files = load_diag_files(args.files)
    summarize(files, bins=args.hist, group_by_alpha=args.alpha)
