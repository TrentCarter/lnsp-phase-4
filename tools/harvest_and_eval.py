#!/usr/bin/env python3
"""HandWiki + OSDev mini-harvester and evaluator for LNSP.

The script crawls a small set of seed pages (HandWiki + OSDev), extracts the
main article text, runs the text through two chunking profiles, ingests those
chunks, then evaluates quality metrics sourced from Postgres when available.

Environment variables:
    LNSP_CHUNK_API   Chunker endpoint (default: http://localhost:8001/chunk)
    LNSP_INGEST_API  Ingest endpoint (default: http://localhost:8004/ingest)

Optional Postgres configuration (defaults target a local dev install):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Example:
    python tools/harvest_and_eval.py \
        --dataset watercycle-mini \
        --seeds "https://handwiki.org/wiki/Earth:Water_cycle" \
                "https://wiki.osdev.org/Interrupts" \
        --max-pages 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
try:
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover - optional dependency
    tabulate = None  # type: ignore

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # pragma: no cover - dependency optional
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore


CHUNK_API = os.getenv("LNSP_CHUNK_API", "http://localhost:8001/chunk")
INGEST_API = os.getenv("LNSP_INGEST_API", "http://localhost:8004/ingest")

USER_AGENT = "lnsp-harvester/0.1 (+https://truesynthesis.ai)"
HEADERS = {"User-Agent": USER_AGENT}
ALLOWED_NETLOCS = {"handwiki.org", "wiki.osdev.org"}
MAIN_SELECTOR_HINTS = [
    "#content",
    "#bodyContent",
    "article",
    "#mw-content-text",
    ".mw-parser-output",
    "main",
]
CLEAN_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form"]

ECHO_GATE = float(os.getenv("LNSP_ECHO_GATE", "0.82"))
COS_DEDUPE_THRESHOLD = float(os.getenv("LNSP_COS_DEDUPE", "0.92"))


@dataclass
class Profile:
    name: str
    payload: Dict[str, object]


PROFILES: Sequence[Profile] = (
    Profile(
        name="simple++",
        payload={
            "mode": "simple",
            "max_chunk_size": 160,
            "min_chunk_size": 60,
        },
    ),
    Profile(
        name="semantic-75",
        payload={
            "mode": "semantic",
            "breakpoint_threshold": 75,
            "min_chunk_size": 100,
            "max_chunk_size": 320,
        },
    ),
)


def debug(msg: str) -> None:
    """Lightweight stderr logging."""
    sys.stderr.write(f"[harvest] {msg}\n")


def is_allowed(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc
    except Exception:
        return False
    return any(netloc.endswith(domain) for domain in ALLOWED_NETLOCS)


def fetch(url: str, timeout: int = 20) -> str:
    debug(f"fetching {url}")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def html_to_text(html: str) -> str:
    processed_html = html
    if Document is not None:
        try:
            doc = Document(html)
            processed_html = doc.summary(html_partial=True)
        except Exception:
            processed_html = html

    soup = BeautifulSoup(processed_html, "lxml")
    for tag_name in CLEAN_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    node = None
    for selector in MAIN_SELECTOR_HINTS:
        node = soup.select_one(selector)
        if node:
            break
    node = node or soup

    text = node.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, profile: Profile) -> List[Dict[str, object]]:
    payload = {"text": text}
    payload.update(profile.payload)
    debug(f"chunking text with profile {profile.name}")
    resp = requests.post(CHUNK_API, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    chunks = data.get("chunks", data)
    if not isinstance(chunks, list):
        raise ValueError("Unexpected chunk response payload")
    normalized = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            if chunk.get("text"):
                normalized.append(chunk)
        else:
            text_value = str(chunk)
            if text_value:
                normalized.append({"text": text_value})
    return normalized


def ingest_chunks(
    chunks: Sequence[Dict[str, object]] | Sequence[str],
    dataset_source: str,
    skip_cpesh: bool = False,
) -> Dict[str, object]:
    if not chunks:
        raise ValueError("No chunks to ingest")
    payload_chunks: List[Dict[str, object]] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            payload_chunks.append(chunk)
        else:
            payload_chunks.append({"text": str(chunk)})
    payload = {
        "chunks": payload_chunks,
        "dataset_source": dataset_source,
        "skip_cpesh": skip_cpesh,
    }
    debug(f"ingesting {len(chunks)} chunks as {dataset_source}")
    resp = requests.post(INGEST_API, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


class Metrics:
    def __init__(self, dataset_source: str) -> None:
        self.dataset_source = dataset_source
        self.pg_conn = self._connect_pg()

    def _connect_pg(self):  # pragma: no cover - integrates external service
        if psycopg2 is None:
            debug("psycopg2 not available; metrics fallback to NaN")
            return None
        try:
            conn = psycopg2.connect(
                host=os.getenv("PGHOST", "localhost"),
                port=int(os.getenv("PGPORT", "5432")),
                dbname=os.getenv("PGDATABASE", "lnsp"),
                user=os.getenv("PGUSER", "lnsp"),
                password=os.getenv("PGPASSWORD", "lnsp"),
            )
            conn.autocommit = True
            return conn
        except Exception as exc:  # pragma: no cover
            debug(f"unable to connect to Postgres: {exc}")
            return None

    # Postgres helpers --------------------------------------------------

    def _pg_fetch(self, query: str, params: Optional[dict] = None) -> List[dict]:
        if not self.pg_conn:
            return []
        try:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(query, params or {})
                return list(cur.fetchall())
        except Exception as exc:  # pragma: no cover - best effort fallback
            debug(f"postgres query failed: {exc}")
            return []

    # Metrics -----------------------------------------------------------

    def yield_per_1k(self) -> float:
        rows = self._pg_fetch(
            """
            SELECT SUM(char_length(source_text)) AS total_chars,
                   COUNT(*) AS total_chunks
            FROM cpe_entry
            WHERE dataset_source = %(ds)s
            """,
            {"ds": self.dataset_source},
        )
        if not rows or not rows[0]["total_chars"]:
            return float("nan")
        chars = float(rows[0]["total_chars"])
        chunks = float(rows[0]["total_chunks"])
        words = chars / 5.0
        return (chunks / max(words, 1.0)) * 1000.0

    def echo_pass_rate(self) -> float:
        rows = self._pg_fetch(
            """
            SELECT AVG((echo_score >= %(gate)s)::int) AS pass_rate
            FROM cpe_entry
            WHERE dataset_source = %(ds)s
              AND echo_score IS NOT NULL
            """,
            {"ds": self.dataset_source, "gate": ECHO_GATE},
        )
        if not rows or rows[0]["pass_rate"] is None:
            return float("nan")
        return float(rows[0]["pass_rate"]) * 100.0

    def vectors_by_lane(self) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        rows = self._pg_fetch(
            """
            SELECT e.cpe_id, e.lane_index, v.concept_vec
            FROM cpe_entry e
            JOIN cpe_vectors v USING(cpe_id)
            WHERE e.dataset_source = %(ds)s
            """,
            {"ds": self.dataset_source},
        )
        lanes: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for row in rows:
            try:
                lane = int(row["lane_index"])
            except Exception:
                debug("skipping vector row with invalid lane_index")
                continue
            raw_vec = row["concept_vec"]
            if isinstance(raw_vec, str):
                try:
                    raw_vec = json.loads(raw_vec)
                except Exception:
                    debug("skipping vector row with unparsable concept_vec")
                    continue
            if not isinstance(raw_vec, (list, tuple)):
                debug("skipping vector row without iterable concept_vec")
                continue
            vector = np.array(raw_vec, dtype=float)
            if vector.size == 0:
                continue
            lanes.setdefault(lane, []).append((str(row["cpe_id"]), vector))
        return lanes

    def unique_ratio(self) -> float:
        lanes = self.vectors_by_lane()
        total = 0
        unique = 0
        for _, items in lanes.items():
            if not items:
                continue
            vectors = np.vstack([vec for _, vec in items])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-8, None)
            sim_matrix = vectors @ vectors.T
            seen = set()
            for idx in range(len(items)):
                if idx in seen:
                    continue
                similar = np.where(sim_matrix[idx] >= COS_DEDUPE_THRESHOLD)[0]
                for j in similar:
                    seen.add(int(j))
                unique += 1
                total += len(similar)
        if total == 0:
            return float("nan")
        return (unique / total) * 100.0

    def retrieval_scores(self, k: int = 10) -> Tuple[float, float]:
        lanes = self.vectors_by_lane()
        mrrs: List[float] = []
        ndcgs: List[float] = []
        for items in lanes.values():
            if len(items) < 2:
                continue
            vectors = np.vstack([vec for _, vec in items])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-8, None)
            sim_matrix = vectors @ vectors.T
            for idx in range(len(items)):
                order = np.argsort(-sim_matrix[idx])
                order = [j for j in order if j != idx][:k]
                rel = [1.0 if sim_matrix[idx, j] >= 0.8 else 0.0 for j in order]
                rr = 0.0
                for rank, score in enumerate(rel, start=1):
                    if score:
                        rr = 1.0 / rank
                        break
                mrrs.append(rr)
                gains = [score / math.log2(i + 2) for i, score in enumerate(rel)]
                dcg = float(sum(gains))
                ideal_hits = int(sum(rel))
                ideal = float(sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)))
                ndcgs.append((dcg / ideal) if ideal > 0 else 0.0)
        if not mrrs:
            return (float("nan"), float("nan"))
        return (float(np.mean(mrrs)), float(np.mean(ndcgs)))


def run_profile(pages: Sequence[Tuple[str, str]], profile: Profile, dataset_prefix: str) -> Dict[str, object]:
    all_chunks: List[Dict[str, object]] = []
    source_words = 0
    for _, text in pages:
        source_words += len(text.split())
        chunks = chunk_text(text, profile)
        all_chunks.extend(chunks)
    dataset_source = f"{dataset_prefix}|{profile.name}"
    ingest_result = ingest_chunks(all_chunks, dataset_source, skip_cpesh=False)
    metrics = Metrics(dataset_source)
    yield_metric = metrics.yield_per_1k()
    unique_metric = metrics.unique_ratio()
    echo_pass = metrics.echo_pass_rate()
    mrr, ndcg = metrics.retrieval_scores()

    if math.isnan(yield_metric):
        if source_words:
            successful = ingest_result.get("successful")
            if successful is None:
                successful = len(all_chunks)
            yield_metric = float(successful) / (source_words / 1000.0)

    return {
        "profile": profile.name,
        "yield": yield_metric,
        "unique": unique_metric,
        "echo": echo_pass,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def format_metric(value: Optional[float], precision: int = 3, scale: float = 1.0) -> str:
    if isinstance(value, np.generic):  # convert numpy scalars
        value = float(value)
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    formatted = value * scale if isinstance(value, (int, float)) else value
    if isinstance(formatted, (int, float)):
        return f"{formatted:.{precision}f}"
    return str(formatted)


def collect_pages(seeds: Sequence[str], max_pages: int) -> List[Tuple[str, str]]:
    pages: List[Tuple[str, str]] = []
    for url in seeds:
        if len(pages) >= max_pages:
            break
        if not is_allowed(url):
            debug(f"skipping disallowed domain: {url}")
            continue
        try:
            html = fetch(url)
            text = html_to_text(html)
            if len(text) < 400:
                debug(f"skipping short page: {url}")
                continue
            pages.append((url, text))
        except Exception as exc:
            debug(f"failed to harvest {url}: {exc}")
    return pages


def render_table(results: Sequence[Dict[str, object]]) -> str:
    rows = []
    for record in results:
        rows.append(
            [
                record["profile"],
                format_metric(record["yield"], precision=1),
                format_metric(record["unique"], precision=1),
                format_metric(record["echo"], precision=1),
                format_metric(record["mrr"], precision=3),
                format_metric(record["ndcg"], precision=3),
            ]
        )
    headers = ["Profile", "Yield/1k", "Unique%", "EchoPass%", "MRR@10", "nDCG@10"]
    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")

    # Basic fallback so test still produces readable output without tabulate.
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def fmt_row(row_values: Sequence[object]) -> str:
        parts = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row_values)]
        return " | ".join(parts)

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HandWiki + OSDev mini harvester and evaluator")
    parser.add_argument("--dataset", required=True, help="Dataset source prefix (e.g., watercycle-mini)")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seed URLs to crawl")
    parser.add_argument("--max-pages", type=int, default=2, help="Maximum pages to pull (default: 2)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pages = collect_pages(args.seeds, args.max_pages)
    if not pages:
        debug("no pages harvested; aborting")
        print("No pages fetched; aborting.")
        return 1

    results: List[Dict[str, object]] = []
    for profile in PROFILES:
        try:
            results.append(run_profile(pages, profile, args.dataset))
        except requests.RequestException as exc:
            debug(f"API request failed for {profile.name}: {exc}")
            print(f"Profile {profile.name} failed: {exc}")
        except Exception as exc:  # pragma: no cover - debugging safety net
            debug(f"unhandled error for {profile.name}: {exc}")
            print(f"Profile {profile.name} failed: {exc}")

    if not results:
        print("No results to report.")
        return 2

    print(render_table(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
