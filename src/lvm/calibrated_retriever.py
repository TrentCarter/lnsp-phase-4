#!/usr/bin/env python3
"""Per-Lane Calibrated Retrieval for LVM Inference Pipeline.

This implements Stage 2 of the PRD: Calibrated vecRAG retrieval with:
- α-weighted fusion of GTR-T5 (768D) + TMD (16D) vectors
- Per-lane Platt/isotonic calibration (one calibrator per TMD domain)
- Dynamic threshold τ_lane for acceptance criteria

Usage:
    from src.lvm.calibrated_retriever import CalibratedRetriever

    retriever = CalibratedRetriever(faiss_index, npz_path="artifacts/ontology_4k_full.npz")
    retriever.load_calibrators("artifacts/calibrators/")  # Load pre-trained

    results = retriever.retrieve(
        concept_text="neural network",
        tmd_bits=(15, 14, 9),  # Tech/CodeGen/Technical
        tmd_dense=np.array([...16D...]),
        k=8
    )
"""
from __future__ import annotations
import json
import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available - calibration disabled")


@dataclass
class RetrievalCandidate:
    """Single retrieval candidate with calibrated probability."""
    index: int
    cpe_id: str
    concept_text: str
    raw_score: float
    calibrated_prob: float
    lane: int
    accepted: bool


@dataclass
class RetrievalResult:
    """Complete retrieval result for a concept."""
    concept_text: str
    tmd_bits: Tuple[int, int, int]
    tmd_lane: int
    candidates: List[RetrievalCandidate]
    accepted_candidates: List[RetrievalCandidate]
    FOUND: bool
    alpha_used: float


class CalibratedRetriever:
    """Per-lane calibrated retrieval with α-weighted fusion.

    Implements PRD Stage 2:
    - Fuses GTR-T5 (768D) + α*TMD (16D) → 784D query vector
    - Per-lane calibration: raw_score → P(match) per TMD domain
    - Acceptance threshold τ_lane (default 0.70, tunable per lane)
    """

    def __init__(
        self,
        faiss_db,
        embedding_backend,
        npz_path: str,
        alpha: float = 0.2,
        default_tau: float = 0.70,
    ):
        """Initialize calibrated retriever.

        Args:
            faiss_db: FaissDB instance with loaded index
            embedding_backend: EmbeddingBackend for GTR-T5 encoding
            npz_path: Path to NPZ with concept_texts, cpe_ids, tmd_dense
            alpha: Weight for TMD fusion (default 0.2)
            default_tau: Default acceptance threshold (default 0.70)
        """
        self.faiss_db = faiss_db
        self.embedding_backend = embedding_backend
        self.alpha = alpha
        self.default_tau = default_tau

        # Load metadata from NPZ
        self._load_metadata(npz_path)

        # Per-lane calibrators (16 TMD domains)
        # Keys: lane_id (0-15), Values: calibrator or None
        self.calibrators: Dict[int, Optional[object]] = {}

        # Per-lane thresholds (tuned for found@8 ≥ 0.85)
        # Keys: lane_id (0-15), Values: τ threshold
        self.tau_lanes: Dict[int, float] = {}

        # Initialize with defaults
        for lane_id in range(16):
            self.calibrators[lane_id] = None
            self.tau_lanes[lane_id] = default_tau

    def _load_metadata(self, npz_path: str):
        """Load concept metadata from NPZ file."""
        npz = np.load(npz_path, allow_pickle=True)

        # Concept texts (for result formatting)
        concept_texts_raw = npz.get("concept_texts")
        if concept_texts_raw is not None:
            self.concept_texts = [str(x) for x in concept_texts_raw]
        else:
            logger.warning(f"No concept_texts in {npz_path}")
            self.concept_texts = []

        # CPE IDs (for correlation)
        cpe_ids_raw = npz.get("cpe_ids")
        if cpe_ids_raw is not None:
            self.cpe_ids = [str(x) for x in cpe_ids_raw]
        else:
            logger.warning(f"No cpe_ids in {npz_path}")
            self.cpe_ids = []

        # TMD dense vectors (for lane detection)
        tmd_dense_raw = npz.get("tmd_dense")
        if tmd_dense_raw is not None:
            self.tmd_dense = np.asarray(tmd_dense_raw, dtype=np.float32)
        else:
            logger.warning(f"No tmd_dense in {npz_path}")
            self.tmd_dense = None

        logger.info(f"Loaded metadata: {len(self.concept_texts)} concepts, {len(self.cpe_ids)} IDs")

    def train_calibrator(
        self,
        lane_id: int,
        scores: np.ndarray,
        labels: np.ndarray,
        method: str = "isotonic",
    ):
        """Train calibrator for a specific lane.

        Args:
            lane_id: TMD domain (0-15)
            scores: Raw FAISS scores (cosine similarity)
            labels: Binary labels (1=match, 0=no match)
            method: "isotonic" or "platt" calibration
        """
        if not HAS_SKLEARN:
            logger.error("sklearn not available - cannot train calibrator")
            return

        if method == "isotonic":
            # Isotonic regression (non-parametric, monotonic)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(scores.reshape(-1), labels)
        elif method == "platt":
            # Platt scaling (logistic regression)
            base = DummyClassifier(strategy="prior")
            calibrator = CalibratedClassifierCV(base, cv=5, method="sigmoid")
            calibrator.fit(scores.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibrators[lane_id] = calibrator
        logger.info(f"Trained {method} calibrator for lane {lane_id}: {len(scores)} samples")

    def tune_threshold(
        self,
        lane_id: int,
        scores: np.ndarray,
        labels: np.ndarray,
        target_found_at_k: float = 0.85,
        k: int = 8,
    ):
        """Tune τ_lane threshold to achieve target found@k.

        Args:
            lane_id: TMD domain (0-15)
            scores: Raw FAISS scores
            labels: Binary labels
            target_found_at_k: Target fraction of queries with ≥1 accept (default 0.85)
            k: Top-K for found@k metric (default 8)
        """
        if self.calibrators[lane_id] is None:
            logger.warning(f"No calibrator for lane {lane_id} - using default τ={self.default_tau}")
            return

        # Predict calibrated probabilities
        calibrated_probs = self._predict_proba(lane_id, scores)

        # Grid search over thresholds
        best_tau = self.default_tau
        best_found_rate = 0.0

        for tau in np.arange(0.50, 0.90, 0.05):
            # Count queries with at least one accept
            found_count = 0
            for i in range(0, len(calibrated_probs), k):
                batch_probs = calibrated_probs[i:i+k]
                if np.any(batch_probs >= tau):
                    found_count += 1

            found_rate = found_count / (len(calibrated_probs) // k)

            if found_rate >= target_found_at_k and found_rate > best_found_rate:
                best_tau = tau
                best_found_rate = found_rate

        self.tau_lanes[lane_id] = best_tau
        logger.info(f"Tuned τ_lane[{lane_id}] = {best_tau:.3f} (found@{k} = {best_found_rate:.3f})")

    def _predict_proba(self, lane_id: int, scores: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities for a lane.

        Args:
            lane_id: TMD domain (0-15)
            scores: Raw FAISS scores

        Returns:
            Calibrated probabilities P(match|score)
        """
        calibrator = self.calibrators.get(lane_id)
        if calibrator is None:
            # No calibrator - use raw scores (clip to [0,1])
            return np.clip(scores, 0.0, 1.0)

        if isinstance(calibrator, IsotonicRegression):
            # Isotonic: predict directly on scores
            return calibrator.predict(scores.reshape(-1))
        else:
            # Platt: predict_proba returns (N,2), take positive class
            return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

    def retrieve(
        self,
        concept_text: str,
        tmd_bits: Tuple[int, int, int],
        tmd_dense: Optional[np.ndarray] = None,
        k: int = 8,
    ) -> RetrievalResult:
        """Calibrated retrieval for a single concept.

        Args:
            concept_text: Concept string (e.g., "neural network")
            tmd_bits: (domain, task, modifier) tuple
            tmd_dense: 16D TMD vector (optional, will use zeros if None)
            k: Top-K results to return

        Returns:
            RetrievalResult with candidates and acceptance status
        """
        # 1. Fuse GTR-T5 + α*TMD vectors
        gtr_vec = self.embedding_backend.encode([concept_text])[0].astype(np.float32)
        gtr_norm = gtr_vec / np.linalg.norm(gtr_vec)

        if tmd_dense is None:
            tmd_dense = np.zeros(16, dtype=np.float32)
        tmd_norm = tmd_dense / (np.linalg.norm(tmd_dense) + 1e-9)

        fused_vec = np.concatenate([gtr_norm, self.alpha * tmd_norm])
        fused_vec = fused_vec / (np.linalg.norm(fused_vec) + 1e-9)

        # 2. FAISS ANN search
        fused_query = fused_vec.reshape(1, -1).astype(np.float32)
        raw_scores, indices = self.faiss_db.search(fused_query, k)
        raw_scores = raw_scores[0]  # Shape: (k,)
        indices = indices[0]        # Shape: (k,)

        # 3. Per-lane calibration
        lane = tmd_bits[0]  # domain ∈ [0,15]
        calibrated_probs = self._predict_proba(lane, raw_scores)

        # 4. Build candidates
        tau = self.tau_lanes.get(lane, self.default_tau)
        candidates = []
        accepted_candidates = []

        for idx, raw_score, cal_prob in zip(indices, raw_scores, calibrated_probs):
            # Get metadata for this index
            cpe_id = self.cpe_ids[idx] if idx < len(self.cpe_ids) else str(idx)
            concept_text_result = self.concept_texts[idx] if idx < len(self.concept_texts) else ""

            accepted = cal_prob >= tau

            candidate = RetrievalCandidate(
                index=int(idx),
                cpe_id=cpe_id,
                concept_text=concept_text_result,
                raw_score=float(raw_score),
                calibrated_prob=float(cal_prob),
                lane=lane,
                accepted=accepted,
            )

            candidates.append(candidate)
            if accepted:
                accepted_candidates.append(candidate)

        # 5. Return result
        return RetrievalResult(
            concept_text=concept_text,
            tmd_bits=tmd_bits,
            tmd_lane=lane,
            candidates=candidates,
            accepted_candidates=accepted_candidates,
            FOUND=len(accepted_candidates) > 0,
            alpha_used=self.alpha,
        )

    def save_calibrators(self, output_dir: str):
        """Save trained calibrators to disk.

        Args:
            output_dir: Directory to save calibrator files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save calibrators
        for lane_id, calibrator in self.calibrators.items():
            if calibrator is not None:
                calibrator_file = output_path / f"calibrator_lane_{lane_id}.pkl"
                with open(calibrator_file, "wb") as f:
                    pickle.dump(calibrator, f)

        # Save thresholds and config
        config = {
            "tau_lanes": self.tau_lanes,
            "alpha": self.alpha,
            "default_tau": self.default_tau,
        }
        config_file = output_path / "calibration_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved calibrators to {output_dir}")

    def load_calibrators(self, input_dir: str):
        """Load trained calibrators from disk.

        Args:
            input_dir: Directory containing calibrator files
        """
        input_path = Path(input_dir)

        # Load calibrators
        for lane_id in range(16):
            calibrator_file = input_path / f"calibrator_lane_{lane_id}.pkl"
            if calibrator_file.exists():
                with open(calibrator_file, "rb") as f:
                    self.calibrators[lane_id] = pickle.load(f)

        # Load thresholds and config
        config_file = input_path / "calibration_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                # Convert string keys to int for tau_lanes
                self.tau_lanes = {int(k): v for k, v in config.get("tau_lanes", {}).items()}
                self.alpha = config.get("alpha", self.alpha)
                self.default_tau = config.get("default_tau", self.default_tau)

        logger.info(f"Loaded calibrators from {input_dir}")


def create_validation_dataset_from_benchmark(
    benchmark_jsonl: str,
    npz_path: str,
    output_path: str,
    n_samples: int = 200,
):
    """Create calibration training dataset from benchmark results.

    Extracts (score, label) pairs from vecRAG benchmark results where:
    - score: raw FAISS cosine similarity
    - label: 1 if top-1 result was correct, 0 otherwise

    Args:
        benchmark_jsonl: Path to benchmark results (e.g., RAG/results/*.jsonl)
        npz_path: Path to corpus NPZ (for lane detection)
        output_path: Path to save validation dataset
        n_samples: Number of samples to extract
    """
    import json

    # Load corpus metadata
    npz = np.load(npz_path, allow_pickle=True)
    tmd_dense = npz.get("tmd_dense")
    if tmd_dense is None:
        raise ValueError(f"No tmd_dense in {npz_path}")

    # Extract samples from benchmark
    validation_data = []

    with open(benchmark_jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break

            record = json.loads(line)

            # Extract vecRAG results
            vec_results = record.get("vec", {})
            if not vec_results:
                continue

            retrieved = vec_results.get("retrieved", [])
            scores = vec_results.get("scores", [])
            correct_id = record.get("target_id")  # Ground truth

            if not retrieved or not scores:
                continue

            # Label: 1 if top-1 matches target, 0 otherwise
            top1_id = retrieved[0] if retrieved else None
            label = 1 if str(top1_id) == str(correct_id) else 0

            # Get lane from target concept's TMD
            target_idx = int(correct_id) if str(correct_id).isdigit() else 0
            if target_idx < len(tmd_dense):
                lane = int(tmd_dense[target_idx][0])  # domain = first TMD component
            else:
                lane = 0

            # Record: (score, label, lane)
            validation_data.append({
                "score": float(scores[0]) if scores else 0.0,
                "label": label,
                "lane": lane,
                "query": record.get("query", ""),
                "target_id": correct_id,
            })

    # Save validation dataset
    with open(output_path, "w") as f:
        for item in validation_data:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Created validation dataset: {len(validation_data)} samples → {output_path}")
    return validation_data
