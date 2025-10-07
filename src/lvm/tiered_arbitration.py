"""
STAGE 4: Tiered Candidate Arbitration

4-tier ladder to minimize expensive vec2text calls:
  TIER 1: ANN within source lane (same domain)
  TIER 2: Graph expansion from top-2 seeds (1-hop neighbors)
  TIER 3: Cross-lane search (domain-compatible lanes)
  TIER 4: vec2text fallback (EXPENSIVE - log and minimize!)

Target: 70% ANN, 20% Graph, 7% Cross, <3% vec2text

See: docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md (lines 424-493)
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ResolutionPath(Enum):
    """Which tier resolved the candidate."""
    ANN = "ANN"           # Tier 1: Within-lane ANN
    GRAPH = "GRAPH"       # Tier 2: Graph expansion
    CROSS = "CROSS"       # Tier 3: Cross-lane search
    VEC2TEXT = "V2T"      # Tier 4: vec2text fallback


@dataclass
class Candidate:
    """Retrieval candidate."""
    id: str
    text: str
    score: float
    confidence: float
    lane: int  # TMD domain [0-15]


@dataclass
class ArbitrationResult:
    """Result from tiered arbitration."""
    candidates: List[Candidate]
    path: ResolutionPath
    confidence: float
    is_generated: bool = False
    latency_ms: float = 0.0


# Domain compatibility matrix (16x16)
# Compatible domains can be searched together (e.g., Tech → [Tech, AI, Science])
DOMAIN_COMPATIBILITY = {
    0: [0, 1, 2],      # Tech → Tech, AI, Science
    1: [1, 0, 2],      # AI → AI, Tech, Science
    2: [2, 0, 1],      # Science → Science, Tech, AI
    3: [3, 4],         # Medical → Medical, Health
    4: [4, 3],         # Health → Health, Medical
    5: [5, 6, 7],      # Business → Business, Finance, Legal
    6: [6, 5],         # Finance → Finance, Business
    7: [7, 5],         # Legal → Legal, Business
    8: [8, 9],         # Entertainment → Entertainment, Arts
    9: [9, 8],         # Arts → Arts, Entertainment
    10: [10, 11],      # Sports → Sports, Fitness
    11: [11, 10],      # Fitness → Fitness, Sports
    12: [12, 13, 14],  # Education → Education, Academic, Research
    13: [13, 12],      # Academic → Academic, Education
    14: [14, 12],      # Research → Research, Education
    15: [15],          # General → General only
}


class TieredArbitrator:
    """
    4-tier arbitration to resolve next concept from LVM prediction.

    Tries each tier in order until acceptance threshold met.
    Logs metrics for monitoring vec2text invocation rate.
    """

    def __init__(
        self,
        faiss_index,
        neo4j_driver,
        lane_calibrators: Dict[int, Any],
        vec2text_client,
        outbox_writer,
        tau_lane: float = 0.7,  # Acceptance threshold
        k: int = 8
    ):
        self.faiss = faiss_index
        self.neo4j = neo4j_driver
        self.calibrators = lane_calibrators
        self.vec2text = vec2text_client
        self.outbox = outbox_writer
        self.tau = tau_lane
        self.k = k

        # Metrics (in production: use Prometheus)
        self.metrics = {
            "tier1_resolves": 0,
            "tier2_resolves": 0,
            "tier3_resolves": 0,
            "tier4_resolves": 0,
            "total_calls": 0
        }

    def arbitrate(
        self,
        next_vec: np.ndarray,
        input_lane: int,
        s2_candidates: List[Candidate],
        mamba_confidence: float
    ) -> ArbitrationResult:
        """
        Run 4-tier arbitration to resolve next concept.

        Args:
            next_vec: 784D vector from LVM prediction
            input_lane: Domain of input concept [0-15]
            s2_candidates: Top-K candidates from Stage 2 retrieval
            mamba_confidence: Confidence from Mamba prediction

        Returns:
            ArbitrationResult with candidates and resolution path
        """
        self.metrics["total_calls"] += 1
        start_time = time.time()

        # ====================================================================
        # TIER 1: ANN within source lane
        # ====================================================================
        result = self._tier1_ann_lane(next_vec, input_lane)
        if result:
            self.metrics["tier1_resolves"] += 1
            result.latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Resolved via TIER 1 (ANN): {len(result.candidates)} candidates")
            return result

        # ====================================================================
        # TIER 2: Graph expansion from top-2 seeds
        # ====================================================================
        if len(s2_candidates) >= 2:
            result = self._tier2_graph_expand(next_vec, s2_candidates[:2], input_lane)
            if result:
                self.metrics["tier2_resolves"] += 1
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Resolved via TIER 2 (GRAPH): {len(result.candidates)} candidates")
                return result

        # ====================================================================
        # TIER 3: Cross-lane search (domain-compatible)
        # ====================================================================
        compatible_lanes = DOMAIN_COMPATIBILITY.get(input_lane, [input_lane])
        result = self._tier3_cross_lane(next_vec, compatible_lanes, input_lane)
        if result:
            self.metrics["tier3_resolves"] += 1
            result.latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Resolved via TIER 3 (CROSS): {len(result.candidates)} candidates")
            return result

        # ====================================================================
        # TIER 4: vec2text fallback (EXPENSIVE!)
        # ====================================================================
        logger.warning(f"vec2text fallback triggered! Lane={input_lane}, mamba_conf={mamba_confidence:.2f}")
        result = self._tier4_vec2text_fallback(next_vec, input_lane)
        self.metrics["tier4_resolves"] += 1
        result.latency_ms = (time.time() - start_time) * 1000

        return result

    def _tier1_ann_lane(self, vec: np.ndarray, lane: int) -> Optional[ArbitrationResult]:
        """TIER 1: ANN search within source lane."""
        # Search within lane (implementation depends on FAISS setup)
        # For now, assume FAISS has metadata filtering or separate lane indices
        results = self.faiss.search(vec.reshape(1, -1), k=self.k)
        distances, indices = results

        # Filter by lane (if FAISS supports metadata)
        # For demo: assume all results are same lane
        scores = 1.0 / (1.0 + distances[0])  # Convert distance to similarity

        # Calibrate probabilities
        calibrator = self.calibrators.get(lane)
        if not calibrator:
            return None

        probs = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]  # P(relevant)

        # Accept if any above threshold
        accepted_idx = np.where(probs >= self.tau)[0]
        if len(accepted_idx) == 0:
            return None

        candidates = [
            Candidate(
                id=str(indices[0][i]),
                text=f"concept_{indices[0][i]}",  # Lookup from DB in production
                score=float(scores[i]),
                confidence=float(probs[i]),
                lane=lane
            )
            for i in accepted_idx
        ]

        return ArbitrationResult(
            candidates=candidates,
            path=ResolutionPath.ANN,
            confidence=float(max(probs[accepted_idx]))
        )

    def _tier2_graph_expand(
        self,
        vec: np.ndarray,
        seeds: List[Candidate],
        lane: int
    ) -> Optional[ArbitrationResult]:
        """TIER 2: Graph expansion from top-2 seeds."""
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (seed:Concept)-[:BROADER|NARROWER|RELATED*1]-(neighbor:Concept)
                WHERE seed.id IN $seed_ids
                RETURN DISTINCT neighbor.id AS id, neighbor.text AS text
                LIMIT 100
            """, seed_ids=[s.id for s in seeds])

            neighbors = [{"id": r["id"], "text": r["text"]} for r in result]

        if not neighbors:
            return None

        # Get vectors for neighbors and compute similarity
        # (In production: batch fetch from FAISS)
        neighbor_vecs = np.random.randn(len(neighbors), 784)  # Placeholder
        scores = np.dot(neighbor_vecs, vec) / (
            np.linalg.norm(neighbor_vecs, axis=1) * np.linalg.norm(vec)
        )

        # Calibrate
        calibrator = self.calibrators.get(lane)
        if not calibrator:
            return None

        probs = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        # Accept if any above threshold
        accepted_idx = np.where(probs >= self.tau)[0]
        if len(accepted_idx) == 0:
            return None

        candidates = [
            Candidate(
                id=neighbors[i]["id"],
                text=neighbors[i]["text"],
                score=float(scores[i]),
                confidence=float(probs[i]),
                lane=lane
            )
            for i in accepted_idx
        ]

        return ArbitrationResult(
            candidates=candidates,
            path=ResolutionPath.GRAPH,
            confidence=float(max(probs[accepted_idx]))
        )

    def _tier3_cross_lane(
        self,
        vec: np.ndarray,
        compatible_lanes: List[int],
        primary_lane: int
    ) -> Optional[ArbitrationResult]:
        """TIER 3: Cross-lane search across domain-compatible lanes."""
        all_candidates = []

        for lane in compatible_lanes:
            results = self.faiss.search(vec.reshape(1, -1), k=self.k)
            distances, indices = results
            scores = 1.0 / (1.0 + distances[0])

            calibrator = self.calibrators.get(lane)
            if not calibrator:
                continue

            probs = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
            accepted_idx = np.where(probs >= self.tau)[0]

            for i in accepted_idx:
                all_candidates.append(
                    Candidate(
                        id=str(indices[0][i]),
                        text=f"concept_{indices[0][i]}_lane{lane}",
                        score=float(scores[i]),
                        confidence=float(probs[i]),
                        lane=lane
                    )
                )

        if not all_candidates:
            return None

        # Sort by confidence
        all_candidates.sort(key=lambda c: c.confidence, reverse=True)

        return ArbitrationResult(
            candidates=all_candidates[:self.k],
            path=ResolutionPath.CROSS,
            confidence=all_candidates[0].confidence
        )

    def _tier4_vec2text_fallback(
        self,
        vec: np.ndarray,
        lane: int
    ) -> ArbitrationResult:
        """TIER 4: vec2text inversion (EXPENSIVE - ~2000ms)."""
        # Invoke vec2text
        text = self.vec2text.invert(vec)

        # Create via outbox (staged write)
        provisional_id = self.outbox.create_concept_with_outbox(
            concept_text=text,
            tmd_bits=bytes([lane]),  # Placeholder
            tmd_dense=[0.0] * 16,
            vector_784d=vec.tolist(),
            parent_hint=None
        )

        return ArbitrationResult(
            candidates=[
                Candidate(
                    id=str(provisional_id),
                    text=text,
                    score=0.5,
                    confidence=0.5,
                    lane=lane
                )
            ],
            path=ResolutionPath.VEC2TEXT,
            confidence=0.5,
            is_generated=True
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current arbitration metrics."""
        total = self.metrics["total_calls"]
        if total == 0:
            return self.metrics

        return {
            **self.metrics,
            "tier1_pct": self.metrics["tier1_resolves"] / total * 100,
            "tier2_pct": self.metrics["tier2_resolves"] / total * 100,
            "tier3_pct": self.metrics["tier3_resolves"] / total * 100,
            "tier4_pct": self.metrics["tier4_resolves"] / total * 100,
        }


if __name__ == "__main__":
    print("Tiered Arbitration - 4-tier ladder")
    print("Target distribution: 70% ANN, 20% Graph, 7% Cross, <3% vec2text")
    print("\nSee docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md for details")
