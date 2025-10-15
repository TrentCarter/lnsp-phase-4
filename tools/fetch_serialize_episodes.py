#!/usr/bin/env python3
"""STEM + Software episodic fetcher and serializer.

Fetches ordered source material, chunks into tiny concepts (no minimum token
requirement), enforces cosine continuity when stitching, and saves episode packs
ready for next-vector training.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup

try:
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for episodic fetching") from exc

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency
    _ENC = None

USER_AGENT = "lnsp-episodic-fetcher/0.1"
HEADERS = {"User-Agent": USER_AGENT}
ALLOWED_HOSTS = {
    "openstax.org",
    "en.wikibooks.org",
    "simple.wikipedia.org",
    "en.wikipedia.org",
    "wiki.osdev.org",
    "developer.mozilla.org",
}
SECTION_HINTS = [
    "article",
    "main",
    "#content",
    "#mw-content-text",
    ".mw-parser-output",
]
DEFAULT_VECTOR_DIM = 768
TOK_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Source:
    name: str
    url: str
    group: str
    type: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_allowed(url: str) -> None:
    host = requests.utils.urlparse(url).netloc
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"Host not allowed: {host}")


def fetch_html(url: str, timeout: int = 30) -> str:
    ensure_allowed(url)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _strip_soup(soup: BeautifulSoup) -> None:
    for tag in ("script", "style", "nav", "header", "footer", "aside", "form"):
        for node in soup.find_all(tag):
            node.decompose()


def html_to_blocks(url: str, html: str) -> List[str]:
    processed = html
    if Document is not None:
        try:
            processed = Document(html).summary(html_partial=True)
        except Exception:
            processed = html
    soup = BeautifulSoup(processed, "lxml")
    _strip_soup(soup)
    root = None
    for selector in SECTION_HINTS:
        candidate = soup.select_one(selector)
        if candidate is not None:
            root = candidate
            break
    if root is None:
        root = soup
    blocks: List[str] = []
    for node in root.find_all(["h2", "h3", "h4", "p", "pre", "code", "li", "dd", "dt", "blockquote"]):
        text = node.get_text(" ").strip()
        if not text:
            continue
        if len(text) < 2:
            continue
        text = re.sub(r"\s+", " ", text)
        blocks.append(text)
    if not blocks:
        fallback = root.get_text(" ").strip()
        if fallback:
            blocks = [fallback]
    return blocks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def encode_tokens(text: str) -> List[int] | List[str]:
    if _ENC:
        return _ENC.encode(text)
    return text.split()


def decode_tokens(tokens: Sequence[int] | Sequence[str]) -> str:
    if _ENC:
        return _ENC.decode(tokens)  # type: ignore[arg-type]
    return " ".join(tokens)  # type: ignore[arg-type]


def tiny_chunker(
    text: str,
    min_tokens: int,
    max_tokens: int,
    hard_max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    tokens = encode_tokens(text)
    if not tokens:
        return []
    total = len(tokens)
    idx = 0
    chunks: List[str] = []
    while idx < total:
        span = min(idx + max_tokens, total)
        span = min(span, idx + hard_max_tokens)
        if min_tokens > 0 and (span - idx) < min_tokens and span < total:
            span = min(idx + min_tokens, total)
        segment = tokens[idx:span]
        if not segment:
            break
        chunks.append(decode_tokens(segment))
        if span == total:
            break
        if overlap_tokens > 0:
            idx = max(span - overlap_tokens, idx + 1)
        else:
            idx = span
    return [c.strip() for c in chunks if c.strip()]


def call_chunk_api(text: str, params: dict) -> List[str]:
    endpoint = os.getenv("LNSP_CHUNK_API")
    if not endpoint:
        return []
    payload = {"text": text}
    payload.update(params)
    resp = requests.post(endpoint, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    chunks = data.get("chunks", data)
    output: List[str] = []
    if isinstance(chunks, list):
        for item in chunks:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    output.append(str(text_value))
            elif isinstance(item, str):
                output.append(item)
    return output


def generate_concepts(
    blocks: Sequence[str],
    params: dict,
) -> List[str]:
    concepts: List[str] = []
    for block in blocks:
        remote = call_chunk_api(
            block,
            {
                "mode": "semantic",
                "max_chunk_tokens": params["max_tokens"],
                "min_chunk_tokens": params["min_tokens"],
                "hard_max_tokens": params["hard_max_tokens"],
                "overlap_tokens": params["overlap_tokens"],
                "preserve_order": True,
            },
        )
        if remote:
            concepts.extend(remote)
            continue
        concepts.extend(
            tiny_chunker(
                block,
                params["min_tokens"],
                params["max_tokens"],
                params["hard_max_tokens"],
                params["overlap_tokens"],
            )
        )
    return concepts


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _token_vector(token: str, dim: int = DEFAULT_VECTOR_DIM, seed: int = 0xC0FFEE) -> np.ndarray:
    """Deterministic lexical hash vector for a token."""
    digest = hashlib.blake2b((token.lower() + str(seed)).encode("utf-8"), digest_size=32).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def embed_batch(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DEFAULT_VECTOR_DIM), dtype=np.float32)
    endpoint = os.getenv("LNSP_VECTOR_API")
    if endpoint:
        resp = requests.post(endpoint, json={"texts": list(texts)}, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        vectors = payload.get("vectors") or payload.get("embeddings")
        if not isinstance(vectors, list):
            raise ValueError("Vector API response missing 'vectors' or 'embeddings'")
        arr = np.asarray(vectors, dtype=np.float32)
        return arr
    vectors = np.zeros((len(texts), DEFAULT_VECTOR_DIM), dtype=np.float32)
    for idx, text in enumerate(texts):
        tokens = TOK_RE.findall(text) or ["_empty"]
        acc = np.zeros(DEFAULT_VECTOR_DIM, dtype=np.float32)
        for token in tokens[:64]:
            acc += _token_vector(token)
        acc /= float(max(1, min(len(tokens), 64)))
        norm = float(np.linalg.norm(acc))
        vectors[idx] = acc / norm if norm > 1e-8 else acc
    return vectors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Episode building
# ---------------------------------------------------------------------------

def build_episodes(
    concepts: Sequence[str],
    vectors: np.ndarray,
    target_len: int,
    min_len: int,
    max_len: int,
    tau_local: float,
) -> Tuple[List[np.ndarray], List[dict]]:
    total_concepts = len(concepts)
    if vectors.size == 0 or total_concepts < 2:
        return [], []
    if total_concepts <= min_len:
        array = vectors[:total_concepts]
        meta = {
            "start_index": 0,
            "end_index": total_concepts - 1,
            "concept_count": total_concepts,
            "mean_coherence": 1.0,
            "min_coherence": 1.0,
            "span_index": 0,
            "slice_start": 0,
            "slice_end": total_concepts,
            "preview": list(concepts[:3]),
            "truncated": True,
        }
        return [array], [meta]
    episodes: List[List[int]] = []
    spans: List[Tuple[int, int]] = []
    start = 0
    coherence: List[float] = []
    for idx in range(1, len(concepts)):
        sim = cosine(vectors[idx - 1], vectors[idx])
        if sim < tau_local or (idx - start) >= max_len:
            span_len = idx - start
            if span_len >= min_len:
                episodes.append(list(range(start, idx)))
                spans.append((start, idx))
                coherence.append(sim)
            start = idx
    if len(concepts) - start >= min_len:
        episodes.append(list(range(start, len(concepts))))
        spans.append((start, len(concepts)))
    if not episodes:
        fallback_spans: List[Tuple[int, int]] = []
        window = max(min_len, min(target_len, len(concepts)))
        for start_idx in range(0, len(concepts), window):
            end_idx = min(start_idx + window, len(concepts))
            if end_idx - start_idx < 2:
                continue
            fallback_spans.append((start_idx, end_idx))
        if not fallback_spans:
            return [], []
        episodes = [list(range(s, e)) for s, e in fallback_spans]
        spans = fallback_spans
    packed: List[np.ndarray] = []
    metas: List[dict] = []
    for span_idx, (start_idx, end_idx) in enumerate(spans):
        length = end_idx - start_idx
        slice_start = start_idx
        while slice_start < end_idx:
            slice_end = min(slice_start + target_len, end_idx)
            if slice_end - slice_start < min_len and end_idx - start_idx >= min_len:
                break
            window = vectors[slice_start:slice_end]
            packed.append(window)
            window_texts = concepts[slice_start:slice_end]
            sims: List[float] = []
            for i in range(slice_start + 1, slice_end):
                sims.append(cosine(vectors[i - 1], vectors[i]))
            metas.append(
                {
                    "start_index": int(slice_start),
                    "end_index": int(slice_end - 1),
                    "concept_count": int(slice_end - slice_start),
                    "mean_coherence": float(np.mean(sims)) if sims else 1.0,
                    "min_coherence": float(np.min(sims)) if sims else 1.0,
                    "span_index": int(span_idx),
                    "slice_start": int(slice_start - start_idx),
                    "slice_end": int(slice_end - start_idx),
                    "preview": window_texts[:3],
                    "truncated": slice_end - slice_start < target_len,
                }
            )
            slice_start = slice_end
    return packed, metas


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def prepare_outdir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()
    return path


def save_episode(
    directory: pathlib.Path,
    source: Source,
    episode_index: int,
    array: np.ndarray,
    meta: dict,
) -> None:
    if array.shape[0] < 2:
        return
    eid = f"{source.name}-{episode_index:04d}"
    x = array[:-1]
    y = array[1:]
    np.savez_compressed(directory / f"{eid}.npz", X=x, y=y)
    record = {
        "episode_id": eid,
        "source": {
            "group": source.group,
            "name": source.name,
            "type": source.type,
            "url": source.url,
        },
        "vector_dim": int(array.shape[1]),
        **meta,
    }
    with open(directory / "meta.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, chunk, and serialize episodic packs")
    parser.add_argument("--config", required=True, help="YAML source configuration")
    parser.add_argument("--outdir", required=True, help="Episode output directory")
    parser.add_argument("--target-len", type=int, default=512, help="Desired episode length slice")
    parser.add_argument("--min-len", type=int, default=128, help="Minimum slice length")
    parser.add_argument("--max-len", type=int, default=1024, help="Maximum span length before splitting")
    parser.add_argument("--tau-local", type=float, default=0.5, help="Cosine gate between concepts")
    parser.add_argument("--min-toks", type=int, default=0, help="Minimum tokens per concept (0 = none)")
    parser.add_argument("--max-toks", type=int, default=32, help="Soft maximum tokens per concept")
    parser.add_argument("--hard-max", type=int, default=64, help="Hard maximum tokens per concept")
    parser.add_argument("--overlap-toks", type=int, default=4, help="Token overlap between concepts")
    return parser.parse_args(argv)


def iter_sources(config: Dict[str, object]) -> Iterable[Source]:
    for group_name, entries in config.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "name" not in entry or "url" not in entry:
                continue
            yield Source(
                name=str(entry["name"]),
                url=str(entry["url"]),
                group=str(group_name),
                type=str(entry.get("type", "unknown")),
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = prepare_outdir(pathlib.Path(args.outdir))

    config = load_yaml(args.config)
    chunk_params = {
        "min_tokens": max(args.min_toks, 0),
        "max_tokens": max(args.max_toks, 1),
        "hard_max_tokens": max(args.hard_max, 1),
        "overlap_tokens": max(args.overlap_toks, 0),
    }
    processed = 0
    for source in iter_sources(config):
        try:
            html = fetch_html(source.url)
            blocks = html_to_blocks(source.url, html)
            if not blocks:
                print(f"[warn] no blocks extracted for {source.url}")
                continue
            print(f"[info] {source.url}: blocks={len(blocks)}")
            concepts = generate_concepts(blocks, chunk_params)
            if not concepts:
                print(f"[warn] no concepts generated for {source.url}")
                continue
            print(f"[info] {source.url}: concepts={len(concepts)}")
            vectors = embed_batch(concepts)
            episodes, metas = build_episodes(
                concepts,
                vectors,
                target_len=args.target_len,
                min_len=args.min_len,
                max_len=args.max_len,
                tau_local=args.tau_local,
            )
            if not episodes:
                print(f"[warn] no episodes produced for {source.url}")
                continue
            for idx, (episode_array, meta) in enumerate(zip(episodes, metas)):
                save_episode(outdir, source, idx, episode_array, meta)
            kept = [m for m in metas if not m.get("truncated")] 
            print(
                f"[ok] {source.url} -> episodes={len(episodes)} "
                f"(full={len(kept)}, truncated={len(metas) - len(kept)})"
            )
            processed += 1
        except Exception as exc:
            print(f"[err] {source.url}: {exc}")
    if processed == 0:
        print("No sources processed successfully.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
