"""
Two-stage retrieval: Paragraph ANN → Sentence Rerank

Stage 1: Retrieve top-K paragraphs using FAISS (current system)
Stage 2: Use last context sentence as query against sentence bank,
         boost paragraphs containing top-N matching sentences

Usage:
    from app.retriever.sent_para_rerank import SentenceParaReranker

    reranker = SentenceParaReranker(
        sent_bank_path="artifacts/arxiv_sentence_bank.npz",
        top_n_sents=5
    )

    # After paragraph retrieval
    para_scores = reranker.rerank_with_sentences(
        query_vec,          # Last context sentence vector [768]
        para_candidates,    # List of (para_id, cos_para, para_vec, ...)
        para_vecs           # Paragraph vectors [K, 768]
    )
"""
import numpy as np
from typing import List, Tuple, Dict, Optional

class SentenceParaReranker:
    """Two-stage retrieval: paragraph → sentence rerank."""

    def __init__(self, sent_bank_path: str, top_n_sents: int = 5):
        """
        Load sentence bank for reranking.

        Args:
            sent_bank_path: Path to sentence bank NPZ file
            top_n_sents: Number of top sentences to consider per paragraph
        """
        print(f"Loading sentence bank: {sent_bank_path}")
        data = np.load(sent_bank_path, allow_pickle=True)

        self.sent_vecs = data["sent_vecs"].astype(np.float32)  # [N, 768]
        self.sent_ids = data["sent_ids"]
        self.para_ids = data["para_ids"]
        self.article_ids = data["article_ids"]
        self.sent_idx = data["sent_idx"]  # Position within paragraph
        self.sent_texts = data.get("sent_texts", None)

        self.top_n_sents = top_n_sents

        # Build para_id → sentence indices mapping
        self.para_to_sents = {}
        for i, pid in enumerate(self.para_ids):
            if pid not in self.para_to_sents:
                self.para_to_sents[pid] = []
            self.para_to_sents[pid].append(i)

        print(f"  Loaded {len(self.sent_vecs)} sentences")
        print(f"  Covering {len(self.para_to_sents)} paragraphs")

    def fuse_scores(
        self,
        cos_para: float,
        cos_sent_hit: float,
        domain_prior: Optional[float] = None,
        is_prev: Optional[bool] = None,
        a: float = 0.75,
        b: float = 0.15,
        c: float = 0.10,
        d: float = 0.05
    ) -> float:
        """
        Fuse paragraph and sentence scores with optional priors.

        Args:
            cos_para: Paragraph-level cosine similarity
            cos_sent_hit: Max sentence cosine similarity within paragraph
            domain_prior: Domain/section prior (optional)
            is_prev: Whether paragraph is marked as "previous" (penalty)
            a, b, c, d: Fusion weights

        Returns:
            Fused score
        """
        score = a * cos_sent_hit + b * cos_para

        if domain_prior is not None:
            score += c * domain_prior

        if is_prev is not None and is_prev:
            score -= d

        return score

    def compute_sent_hit(
        self,
        query_vec: np.ndarray,
        para_id: int,
        top_k: int = 2
    ) -> Tuple[float, List[int]]:
        """
        Compute max (or top-K average) sentence cosine for a paragraph.

        Args:
            query_vec: Query vector [768]
            para_id: Paragraph ID to search
            top_k: Number of top sentences to average (1 = max)

        Returns:
            (cos_sent_hit, top_sentence_indices)
        """
        if para_id not in self.para_to_sents:
            return 0.0, []

        sent_indices = self.para_to_sents[para_id]
        sent_vecs = self.sent_vecs[sent_indices]

        # Compute cosine similarities
        cos_scores = sent_vecs @ query_vec  # [num_sents]

        # Get top-K
        top_k = min(top_k, len(cos_scores))
        top_indices = np.argsort(cos_scores)[-top_k:][::-1]

        cos_sent_hit = float(np.mean(cos_scores[top_indices]))
        global_sent_indices = [sent_indices[i] for i in top_indices]

        return cos_sent_hit, global_sent_indices

    def rerank_with_sentences(
        self,
        query_vec: np.ndarray,
        para_candidates: List[Tuple],
        para_vecs: np.ndarray,
        domain_priors: Optional[List[float]] = None,
        is_prev_flags: Optional[List[bool]] = None
    ) -> List[Tuple[int, float, float, float]]:
        """
        Rerank paragraph candidates using sentence-level hits.

        Args:
            query_vec: Last context sentence vector [768]
            para_candidates: List of (para_id, cos_para, ...)
            para_vecs: Paragraph vectors [K, 768]
            domain_priors: Optional domain/section priors
            is_prev_flags: Optional "is previous" flags

        Returns:
            List of (para_id, fused_score, cos_para, cos_sent_hit)
            sorted by fused_score descending
        """
        results = []

        for i, cand in enumerate(para_candidates):
            para_id = cand[0]
            cos_para = cand[1]

            # Compute sentence hit
            cos_sent_hit, top_sents = self.compute_sent_hit(query_vec, para_id, top_k=2)

            # Fuse scores
            domain_prior = domain_priors[i] if domain_priors else None
            is_prev = is_prev_flags[i] if is_prev_flags else None

            fused_score = self.fuse_scores(
                cos_para=cos_para,
                cos_sent_hit=cos_sent_hit,
                domain_prior=domain_prior,
                is_prev=is_prev
            )

            results.append((para_id, fused_score, cos_para, cos_sent_hit))

        # Sort by fused score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def expand_sentence_context(
        self,
        sent_idx: int,
        window: int = 1
    ) -> List[int]:
        """
        Expand sentence to include ±window neighboring sentences.

        Args:
            sent_idx: Global sentence index
            window: Number of neighbors on each side

        Returns:
            List of sentence indices in span
        """
        para_id = self.para_ids[sent_idx]
        sent_pos = self.sent_idx[sent_idx]

        # Get all sentences in this paragraph
        para_sents = self.para_to_sents.get(para_id, [])

        # Find neighbors
        span = []
        for s_idx in para_sents:
            s_pos = self.sent_idx[s_idx]
            if abs(s_pos - sent_pos) <= window:
                span.append(s_idx)

        return sorted(span)


def test_reranker():
    """Quick test of sentence-based reranking."""
    import os

    sent_bank = "artifacts/arxiv_sentence_bank.npz"
    if not os.path.exists(sent_bank):
        print(f"Sentence bank not found: {sent_bank}")
        return

    reranker = SentenceParaReranker(sent_bank, top_n_sents=5)

    # Dummy query and candidates
    query_vec = np.random.randn(768).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    para_candidates = [
        (0, 0.85),
        (1, 0.82),
        (2, 0.80),
    ]

    para_vecs = np.random.randn(3, 768).astype(np.float32)
    para_vecs /= np.linalg.norm(para_vecs, axis=1, keepdims=True)

    results = reranker.rerank_with_sentences(query_vec, para_candidates, para_vecs)

    print("\nReranked results:")
    for para_id, fused, cos_p, cos_s in results:
        print(f"  Para {para_id}: fused={fused:.3f}, cos_para={cos_p:.3f}, cos_sent={cos_s:.3f}")


if __name__ == "__main__":
    test_reranker()
