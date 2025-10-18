#!/usr/bin/env python3
"""Batch evaluation of semantic vector arithmetic on Wikipedia embeddings.

The script samples quadruples of consecutive vectors from the ordered Wikipedia
embedding set and evaluates the classic analogy operation

    v_{i+1} - v_i + v_{i+2} \approx v_{i+3}

For each sampled quadruple we measure:
  * Cosine similarity between the arithmetic result and the ground-truth vector.
  * Retrieval rank of the ground-truth vector among all vectors (higher is worse).
  * Top-1 nearest neighbour text and its similarity.
  * Optional decoding via the vec2text FastAPI server (port 8766) with
    text-level metrics: cosine in embedding space (via port 8767) and BLEU-2.

Results are saved as JSON for downstream analysis and a concise summary is
printed to stdout.

Usage example:

    python tools/semantic_vector_arithmetic_eval.py \
        --num-samples 200 --decode-samples 12 \
        --output artifacts/demos/semantic_vector_arithmetic_metrics.json

The script assumes the following services are running:
  * Vec2Text decoder FastAPI on http://localhost:8766 (for optional decoding).
  * Vec2Text-compatible encoder on http://localhost:8767 (for text metrics).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WIKI_PATH = ROOT / "artifacts" / "lvm" / "wikipedia_29322_ordered.npz"
DEFAULT_SAMPLES = 200
DEFAULT_DECODE = 12
VEC2TEXT_URL = "http://localhost:8766"
ENCODER_URL = "http://localhost:8767"

_local_orchestrator = None


@dataclass
class AnalogySample:
    """Container for analogy indices drawn from the dataset."""

    article_id: str
    idx_a: int
    idx_b: int
    idx_c: int
    idx_target: int
    seq_position: int


@dataclass
class AnalogyMetrics:
    """Evaluation metrics for a single analogy."""

    sample: AnalogySample
    cosine_to_target: float
    target_rank: int
    top1_index: int
    top1_cosine: float
    decoded_text: Optional[str]
    decoded_cosine: Optional[float]
    decoded_bleu2: Optional[float]


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load the ordered Wikipedia embedding dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    data = np.load(path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    concept_texts = data["concept_texts"]
    batch_ids = data["batch_ids"]
    seq_in_article = data["seq_in_article"]
    return {
        "vectors": vectors,
        "concept_texts": concept_texts,
        "batch_ids": batch_ids,
        "seq": seq_in_article,
    }


def build_article_index(batch_ids: Sequence[str], seq: Sequence[int]) -> Dict[str, List[Tuple[int, int]]]:
    """Group vector indices by article, sorted by sequence order."""

    mapping: Dict[str, List[Tuple[int, int]]] = {}
    for global_idx, (article_id, position) in enumerate(zip(batch_ids, seq)):
        mapping.setdefault(article_id, []).append((position, global_idx))

    for article_id, pairs in mapping.items():
        pairs.sort(key=lambda pair: pair[0])
        mapping[article_id] = pairs
    return mapping


def collect_analogies(mapping: Dict[str, List[Tuple[int, int]]]) -> List[AnalogySample]:
    """Collect all possible analogies from ordered article sequences."""

    analogies: List[AnalogySample] = []
    for article_id, pairs in mapping.items():
        indices = [idx for _, idx in pairs]
        positions = [pos for pos, _ in pairs]
        if len(indices) < 4:
            continue
        for start in range(len(indices) - 3):
            a = indices[start + 1]
            b = indices[start]
            c = indices[start + 2]
            target = indices[start + 3]
            seq_pos = positions[start + 3]
            analogies.append(
                AnalogySample(
                    article_id=article_id,
                    idx_a=a,
                    idx_b=b,
                    idx_c=c,
                    idx_target=target,
                    seq_position=seq_pos,
                )
            )
    return analogies


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def decode_vector(vec: np.ndarray, steps: int = 1) -> Optional[str]:
    payload = {
        "vectors": [vec.tolist()],
        "subscribers": "jxe,ielab",
        "steps": steps,
        "device": "cpu",
        "apply_adapter": True,
    }
    try:
        response = requests.post(f"{VEC2TEXT_URL}/decode", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        subscribers = data["results"][0]["subscribers"]
        # Prefer JXE output; IELab mirrors it in current config
        if "gtr → jxe" in subscribers:
            return subscribers["gtr → jxe"]["output"]
        if "jxe" in subscribers:  # legacy key fallback
            return subscribers["jxe"].get("output")
        # Fallback to any available decoder output
        first_decoder = next(iter(subscribers.values()), None)
        if isinstance(first_decoder, dict):
            return first_decoder.get("output")
        return None
    except Exception as exc:  # noqa: BLE001 - propagate as log entry
        print(f"[warn] Vec2Text decode failed: {exc}")
        return None


def encode_text(text: str, normalize: bool = True) -> Optional[np.ndarray]:
    payload = {"texts": [text], "normalize": normalize}
    try:
        response = requests.post(f"{ENCODER_URL}/embed", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        emb = np.array(data["embeddings"][0], dtype=np.float32)
        return emb
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Vec2Text encoder call failed: {exc}")

    # Fallback: instantiate in-process orchestrator once
    global _local_orchestrator
    try:
        from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
        if _local_orchestrator is None:
            _local_orchestrator = IsolatedVecTextVectOrchestrator()
        tensor = _local_orchestrator.encode_texts([text])
        if normalize:
            tensor = tensor / (tensor.norm(dim=1, keepdim=True) + 1e-8)
        return tensor.squeeze(0).cpu().numpy()
    except Exception as inner_exc:  # noqa: BLE001
        print(f"[warn] Local encoder fallback failed: {inner_exc}")
        return None


def bleu2(candidate: str, reference: str) -> float:
    """Compute a simple BLEU-2 score (1-gram & 2-gram, equal weights)."""

    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    def ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
        counts: Dict[Tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    precisions = []
    for n in (1, 2):
        cand_counts = ngram_counts(cand_tokens, n)
        if not cand_counts:
            precisions.append(0.0)
            continue
        ref_counts = ngram_counts(ref_tokens, n)
        overlap = 0
        total = sum(cand_counts.values())
        for ngram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))
        precisions.append(overlap / total if total else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / max(cand_len, 1)))
    return float(bp * math.exp(log_precision))


def evaluate_samples(
    samples: Sequence[AnalogySample],
    vectors: np.ndarray,
    normalized: np.ndarray,
    concept_texts: Sequence[str],
    decode_samples: int,
    decode_steps: int,
) -> List[AnalogyMetrics]:
    results: List[AnalogyMetrics] = []
    for idx, sample in enumerate(samples):
        vec_a = vectors[sample.idx_a]
        vec_b = vectors[sample.idx_b]
        vec_c = vectors[sample.idx_c]
        vec_target = vectors[sample.idx_target]

        result_vec = vec_a - vec_b + vec_c
        result_norm = result_vec / (np.linalg.norm(result_vec) + 1e-8)
        target_norm = normalized[sample.idx_target]

        cos_to_target = float(np.dot(result_norm, target_norm))

        sims = normalized @ result_norm
        sorted_idx = np.argsort(-sims)
        target_rank = int(np.where(sorted_idx == sample.idx_target)[0][0] + 1)
        top1_idx = int(sorted_idx[0])
        top1_cos = float(sims[top1_idx])

        decoded_text = None
        decoded_cosine = None
        decoded_bleu = None
        if idx < decode_samples:
            decoded_text = decode_vector(result_norm, steps=decode_steps)
            if decoded_text:
                decoded_vec = encode_text(decoded_text)
                if decoded_vec is not None:
                    decoded_cosine = cosine(decoded_vec, vec_target)
                decoded_bleu = bleu2(decoded_text, concept_texts[sample.idx_target])

        results.append(
            AnalogyMetrics(
                sample=sample,
                cosine_to_target=cos_to_target,
                target_rank=target_rank,
                top1_index=top1_idx,
                top1_cosine=top1_cos,
                decoded_text=decoded_text,
                decoded_cosine=decoded_cosine,
                decoded_bleu2=decoded_bleu,
            )
        )
    return results


def summarize(metrics: Sequence[AnalogyMetrics]) -> Dict[str, float]:
    cosines = [m.cosine_to_target for m in metrics]
    ranks = [m.target_rank for m in metrics]
    summary = {
        "num_samples": len(metrics),
        "cosine_mean": float(statistics.mean(cosines)),
        "cosine_median": float(statistics.median(cosines)),
        "cosine_p90": float(np.percentile(cosines, 90)),
        "cosine_ge_0.7": float(sum(c >= 0.7 for c in cosines) / len(cosines)),
        "cosine_ge_0.8": float(sum(c >= 0.8 for c in cosines) / len(cosines)),
        "rank_mean": float(statistics.mean(ranks)),
        "rank_median": float(statistics.median(ranks)),
        "rank_p90": float(np.percentile(ranks, 90)),
        "rank_top10_frac": float(sum(r <= 10 for r in ranks) / len(ranks)),
        "rank_top50_frac": float(sum(r <= 50 for r in ranks) / len(ranks)),
    }

    decoded_cosines = [m.decoded_cosine for m in metrics if m.decoded_cosine is not None]
    if decoded_cosines:
        summary.update(
            decoded_cosine_mean=float(statistics.mean(decoded_cosines)),
            decoded_cosine_median=float(statistics.median(decoded_cosines)),
        )
    decoded_bleus = [m.decoded_bleu2 for m in metrics if m.decoded_bleu2 is not None]
    if decoded_bleus:
        summary.update(
            decoded_bleu_mean=float(statistics.mean(decoded_bleus)),
            decoded_bleu_median=float(statistics.median(decoded_bleus)),
        )
    return summary


def save_results(
    path: Path,
    metrics: Sequence[AnalogyMetrics],
    summary: Dict[str, float],
    concept_texts: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "samples": [
            {
                "article_id": m.sample.article_id,
                "seq_position": int(m.sample.seq_position),
                "cosine_to_target": float(m.cosine_to_target),
                "target_rank": int(m.target_rank),
                "top1_index": int(m.top1_index),
                "top1_cosine": float(m.top1_cosine),
                "target_text": concept_texts[m.sample.idx_target],
                "top1_text": concept_texts[m.top1_index],
                "decoded_text": m.decoded_text,
                "decoded_cosine": None if m.decoded_cosine is None else float(m.decoded_cosine),
                "decoded_bleu2": None if m.decoded_bleu2 is None else float(m.decoded_bleu2),
            }
            for m in metrics
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic vector arithmetic at scale.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_WIKI_PATH,
        help="Path to wikipedia_*.npz dataset (default: artifacts/lvm/wikipedia_29322_ordered.npz)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of analogy samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=DEFAULT_DECODE,
        help="Number of samples to decode via vec2text for text-level metrics (default: 12)",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="Decoding steps to pass to vec2text (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "demos" / "semantic_vector_arithmetic_metrics.json",
        help="Output path for JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    vectors = data["vectors"]
    concept_texts = data["concept_texts"]
    mapping = build_article_index(data["batch_ids"], data["seq"])

    all_analogies = collect_analogies(mapping)
    if not all_analogies:
        raise RuntimeError("No analogies could be constructed from the dataset")

    sample_count = min(args.num_samples, len(all_analogies))
    samples = random.sample(all_analogies, sample_count)
    print(f"Evaluating {sample_count} analogies (out of {len(all_analogies)} available)...")

    normalized = normalize_vectors(vectors)
    metrics = evaluate_samples(
        samples,
        vectors,
        normalized,
        concept_texts,
        decode_samples=max(0, min(args.decode_samples, sample_count)),
        decode_steps=max(1, args.decode_steps),
    )

    summary = summarize(metrics)
    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    save_results(args.output, metrics, summary, concept_texts)
    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
