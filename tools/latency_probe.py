#!/usr/bin/env python3
"""
Latency probe for LNSP Retrieval API.
- Hits /search for each lane (L1_FACTOID, L2_GRAPH, L3_SYNTH)
- Records N iterations per lane
- Writes JSONL traces to eval/day6_latency_traces.jsonl
- Prints P50/P95 per lane and overall summary
- Also prints /admin/faiss payload for record

Usage:
  python tools/latency_probe.py --base http://127.0.0.1:8080 --iters 50 --out eval/day6_latency_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

LANES = ["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"]
QUERIES = {
    "L1_FACTOID": [
        "What is FAISS?",
        "What does IVF mean in FAISS?",
        "Define vector database"
    ],
    "L2_GRAPH": [
        "Explain IVF vs IVF_PQ",
        "How does an inverted file index work?",
        "What is product quantization?"
    ],
    "L3_SYNTH": [
        "How does a knowledge graph help RAG?",
        "Compare IVFFlat and IVFPQ for latency and recall",
        "When to use PQ versus Flat index?"
    ],
}


def _client():
    try:
        import httpx  # type: ignore
        return ("httpx", httpx)
    except Exception:
        import urllib.request
        return ("urllib", urllib.request)


def _post_json(base: str, path: str, payload: Dict[str, Any], client) -> Tuple[int, Dict[str, Any], float]:
    url = base.rstrip("/") + path
    t0 = time.perf_counter()
    if client[0] == "httpx":
        httpx = client[1]
        try:
            with httpx.Client(timeout=15.0) as s:
                r = s.post(url, json=payload)
                elapsed = (time.perf_counter() - t0) * 1000.0
                try:
                    data = r.json()
                except Exception:
                    data = {}
                return (r.status_code, data, elapsed)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return (599, {}, elapsed)
    else:
        urllib = client[1]
        try:
            req = urllib.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.urlopen(req, timeout=15) as r:  # type: ignore[attr-defined]
                elapsed = (time.perf_counter() - t0) * 1000.0
                try:
                    data = json.loads(r.read().decode("utf-8"))
                except Exception:
                    data = {}
                return (getattr(r, "status", 200), data, elapsed)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return (599, {}, elapsed)


def _get_json(base: str, path: str, client) -> Tuple[int, Dict[str, Any]]:
    url = base.rstrip("/") + path
    if client[0] == "httpx":
        httpx = client[1]
        try:
            with httpx.Client(timeout=10.0) as s:
                r = s.get(url)
                try:
                    data = r.json()
                except Exception:
                    data = {}
                return (r.status_code, data)
        except Exception:
            return (599, {})
    else:
        urllib = client[1]
        try:
            with urllib.urlopen(url, timeout=10) as r:  # type: ignore[attr-defined]
                try:
                    data = json.loads(r.read().decode("utf-8"))
                except Exception:
                    data = {}
                return (getattr(r, "status", 200), data)
        except Exception:
            return (599, {})


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=str, default="eval/day6_latency_traces.jsonl")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    client = _client()

    # Probe admin/faiss
    status_admin, admin_payload = _get_json(args.base, "/admin/faiss", client)
    print("/admin/faiss:", json.dumps(admin_payload, indent=2, sort_keys=True))

    all_latencies: Dict[str, List[float]] = {lane: [] for lane in LANES}
    total_latencies: List[float] = []

    with open(args.out, "w", encoding="utf-8") as fout:
        for lane in LANES:
            qs = QUERIES[lane]
            for i in range(args.iters):
                payload = {"q": qs[i % len(qs)], "lane": lane, "top_k": 5}
                code, data, elapsed = _post_json(args.base, "/search", payload, client)
                trace = {
                    "ts": time.time(),
                    "lane": lane,
                    "elapsed_ms": round(elapsed, 3),
                    "status": code,
                    "top_k": len(data.get("items", [])) if isinstance(data, dict) else 0,
                }
                fout.write(json.dumps(trace) + "\n")
                if code == 200:
                    all_latencies[lane].append(elapsed)
                    total_latencies.append(elapsed)

    # Compute stats
    print("\nLatency summary (ms):")
    for lane in LANES:
        p50 = percentile(all_latencies[lane], 50)
        p95 = percentile(all_latencies[lane], 95)
        print(f"- {lane}: P50={p50:.2f}, P95={p95:.2f}, n={len(all_latencies[lane])}")

    if total_latencies:
        print(f"\nOverall: P50={percentile(total_latencies,50):.2f}, P95={percentile(total_latencies,95):.2f}, n={len(total_latencies)}")

    print(f"\nWrote traces to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
