#!/usr/bin/env python3
"""
Pipeline Walk & Timing

- Reads a single-element JSON array containing one long text.
- Uses `src.chunker_v2.SemanticChunker` to produce chunks.
- Sends chunks to the ingestion API POST /ingest.
- Runs 5 iterations, sleeps 500ms between iterations, and reports the last run timings.

Usage:
  python tools/pipeline_walk.py --input data/samples/sample_prompts_1.json --api http://127.0.0.1:8004
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import requests

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.chunker_v2 import SemanticChunker  # type: ignore


def load_single_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], str):
        raise ValueError("Input must be a JSON array with a single string element containing the raw text.")
    return data[0]


def run_once(raw_text: str, api_base: str, dataset_source: str = "pipeline_walk") -> Dict[str, Any]:
    out: Dict[str, Any] = {"dataset_source": dataset_source}

    # 1) Chunking
    t0 = time.perf_counter()
    chunker = SemanticChunker(min_chunk_size=200)
    chunks = chunker.chunk(raw_text)
    t_chunk = (time.perf_counter() - t0) * 1000.0

    # Build ingest payload
    payload = {
        "dataset_source": dataset_source,
        "chunks": [
            {
                "text": c.text,
                "source_document": os.path.basename(dataset_source) or "pipeline_walk",
                "chunk_index": i,
                "metadata": {
                    "char_count": c.char_count,
                    "word_count": c.word_count,
                },
            }
            for i, c in enumerate(chunks)
        ],
    }

    # 2) Ingest
    t1 = time.perf_counter()
    resp = requests.post(f"{api_base}/ingest", json=payload, timeout=60)
    t_ingest = (time.perf_counter() - t1) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(f"Ingest failed: HTTP {resp.status_code} - {resp.text[:500]}")

    data = resp.json()
    out.update(
        {
            "n_chunks": len(payload["chunks"]),
            "chunk_ms": t_chunk,
            "ingest_ms": t_ingest,
            "total_ms": t_chunk + t_ingest,
            "ingest_response": {
                "total_chunks": data.get("total_chunks"),
                "successful": data.get("successful"),
                "failed": data.get("failed"),
                "batch_id": data.get("batch_id"),
                "processing_time_ms": data.get("processing_time_ms"),
            },
        }
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON array with a single long text")
    parser.add_argument("--api", default="http://127.0.0.1:8004", help="Ingestion API base URL")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default 5)")
    parser.add_argument("--sleep_ms", type=int, default=500, help="Sleep between runs in ms (default 500)")
    args = parser.parse_args()

    raw_text = load_single_text(args.input)

    results: List[Dict[str, Any]] = []
    for i in range(args.runs):
        res = run_once(raw_text, args.api, dataset_source=os.path.basename(args.input))
        results.append(res)
        print(f"Run {i+1}/{args.runs}: chunks={res['n_chunks']} chunk_ms={res['chunk_ms']:.1f} ingest_ms={res['ingest_ms']:.1f} total_ms={res['total_ms']:.1f}")
        if i < args.runs - 1:
            time.sleep(args.sleep_ms / 1000.0)

    last = results[-1]

    print("\n=== Pipeline Walk Summary (last run) ===")
    print(f"API: {args.api}")
    print(f"Input: {args.input}")
    print(f"Chunks: {last['n_chunks']}")
    print(f"Chunking: {last['chunk_ms']:.1f} ms")
    print(f"Ingest: {last['ingest_ms']:.1f} ms")
    print(f"Total: {last['total_ms']:.1f} ms")
    ir = last.get("ingest_response", {})
    print(
        f"IngestResponse: total={ir.get('total_chunks')} ok={ir.get('successful')} fail={ir.get('failed')} batch_id={ir.get('batch_id')} api_ms={ir.get('processing_time_ms')}"
    )


if __name__ == "__main__":
    main()
