#!/usr/bin/env python3
"""Train per-lane calibrators for calibrated retrieval.

This script:
1. Extracts validation data from existing benchmark results
2. Trains isotonic/Platt calibrators per TMD lane (16 domains)
3. Tunes acceptance thresholds τ_lane for found@8 ≥ 0.85
4. Saves calibrators for inference use

Usage:
    python tools/train_calibrators.py \\
        --benchmark RAG/results/comprehensive_200.jsonl \\
        --npz artifacts/ontology_4k_full.npz \\
        --index artifacts/ontology_4k_ivf_flat_ip.index \\
        --output artifacts/calibrators/ \\
        --method isotonic \\
        --alpha 0.2
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend
from src.lvm.calibrated_retriever import CalibratedRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_training_data(
    benchmark_path: str,
    corpus_npz: str,
    faiss_db: FaissDB,
    embedding_backend: EmbeddingBackend,
    alpha: float = 0.2,
):
    """Extract (score, label, lane) tuples from benchmark results.

    For each query:
    - Retrieve top-K with FAISS (raw scores)
    - Label: 1 if retrieved ID matches target, 0 otherwise
    - Lane: TMD domain of the target concept
    """
    # Load corpus metadata
    npz = np.load(corpus_npz, allow_pickle=True)
    concept_texts = [str(x) for x in npz.get("concept_texts", [])]
    cpe_ids = [str(x) for x in npz.get("cpe_ids", [])]
    tmd_dense = np.asarray(npz.get("tmd_dense"), dtype=np.float32)

    # Create ID → index mapping
    id_to_idx = {cpe_id: idx for idx, cpe_id in enumerate(cpe_ids)}

    # Extract training samples per lane
    lane_data = defaultdict(lambda: {"scores": [], "labels": []})

    with open(benchmark_path, "r") as f:
        for line_num, line in enumerate(f):
            record = json.loads(line)

            query_text = record.get("query", "")
            target_id = str(record.get("target_id", ""))

            if not query_text or not target_id:
                continue

            # Get target concept's lane
            if target_id not in id_to_idx:
                logger.warning(f"Line {line_num}: target_id {target_id} not in corpus")
                continue

            target_idx = id_to_idx[target_id]
            if target_idx >= len(tmd_dense):
                logger.warning(f"Line {line_num}: target_idx {target_idx} out of bounds")
                continue

            lane = int(tmd_dense[target_idx][0])  # domain = first TMD component

            # Get target concept's TMD for fusion
            tmd_vec = tmd_dense[target_idx]

            # Encode query with α-weighted fusion
            gtr_vec = embedding_backend.encode([query_text])[0].astype(np.float32)
            gtr_norm = gtr_vec / np.linalg.norm(gtr_vec)
            tmd_norm = tmd_vec / (np.linalg.norm(tmd_vec) + 1e-9)
            fused_vec = np.concatenate([gtr_norm, alpha * tmd_norm])
            fused_vec = fused_vec / (np.linalg.norm(fused_vec) + 1e-9)

            # FAISS search
            fused_query = fused_vec.reshape(1, -1).astype(np.float32)
            scores, indices = faiss_db.search(fused_query, k=10)
            scores = scores[0]
            indices = indices[0]

            # Label each result: 1 if matches target, 0 otherwise
            for score, idx in zip(scores, indices):
                retrieved_id = cpe_ids[idx] if idx < len(cpe_ids) else str(idx)
                label = 1 if retrieved_id == target_id else 0

                lane_data[lane]["scores"].append(float(score))
                lane_data[lane]["labels"].append(label)

    # Convert to numpy arrays
    for lane in lane_data:
        lane_data[lane]["scores"] = np.array(lane_data[lane]["scores"])
        lane_data[lane]["labels"] = np.array(lane_data[lane]["labels"])

    logger.info(f"Extracted training data from {benchmark_path}:")
    for lane, data in sorted(lane_data.items()):
        n_samples = len(data["scores"])
        n_positive = np.sum(data["labels"])
        logger.info(f"  Lane {lane}: {n_samples} samples ({n_positive} positive, {n_samples - n_positive} negative)")

    return lane_data


def main():
    parser = argparse.ArgumentParser(description="Train per-lane calibrators")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--npz", required=True, help="Path to corpus NPZ")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--output", required=True, help="Output directory for calibrators")
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "platt"], help="Calibration method")
    parser.add_argument("--alpha", type=float, default=0.2, help="TMD fusion weight")
    parser.add_argument("--target-found", type=float, default=0.85, help="Target found@8 rate")
    args = parser.parse_args()

    # Load FAISS index
    logger.info(f"Loading FAISS index from {args.index}")
    faiss_db = FaissDB(index_path=args.index, meta_npz_path=args.npz)
    faiss_db.load()

    # Load embedding backend
    logger.info("Loading GTR-T5 embedding backend")
    embedding_backend = EmbeddingBackend()

    # Extract training data
    logger.info(f"Extracting training data from {args.benchmark}")
    lane_data = extract_training_data(
        benchmark_path=args.benchmark,
        corpus_npz=args.npz,
        faiss_db=faiss_db,
        embedding_backend=embedding_backend,
        alpha=args.alpha,
    )

    # Initialize calibrated retriever
    logger.info("Initializing calibrated retriever")
    retriever = CalibratedRetriever(
        faiss_db=faiss_db,
        embedding_backend=embedding_backend,
        npz_path=args.npz,
        alpha=args.alpha,
    )

    # Train calibrators per lane
    logger.info(f"Training {args.method} calibrators")
    for lane, data in sorted(lane_data.items()):
        if len(data["scores"]) < 10:
            logger.warning(f"Lane {lane}: insufficient data ({len(data['scores'])} samples) - skipping")
            continue

        retriever.train_calibrator(
            lane_id=lane,
            scores=data["scores"],
            labels=data["labels"],
            method=args.method,
        )

        retriever.tune_threshold(
            lane_id=lane,
            scores=data["scores"],
            labels=data["labels"],
            target_found_at_k=args.target_found,
            k=8,
        )

    # Save calibrators
    logger.info(f"Saving calibrators to {args.output}")
    retriever.save_calibrators(args.output)

    # Print summary
    logger.info("\n=== Calibration Summary ===")
    logger.info(f"Method: {args.method}")
    logger.info(f"Alpha (TMD weight): {args.alpha}")
    logger.info(f"Target found@8: {args.target_found}")
    logger.info(f"\nPer-lane thresholds (τ_lane):")
    for lane in sorted(retriever.tau_lanes.keys()):
        tau = retriever.tau_lanes[lane]
        calibrator_status = "✓ calibrated" if retriever.calibrators.get(lane) else "✗ uncalibrated"
        logger.info(f"  Lane {lane:2d}: τ={tau:.3f} ({calibrator_status})")

    logger.info(f"\n✓ Calibrators saved to {args.output}")


if __name__ == "__main__":
    main()
