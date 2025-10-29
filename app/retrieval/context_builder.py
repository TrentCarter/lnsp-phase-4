# app/retrieval/context_builder.py
from __future__ import annotations
import os
import json
import faiss
import numpy as np
from typing import Optional, List, Tuple

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0.0] = 1.0
    return (v / n).astype(np.float32)

def _dedup_by_cosine(idxs: List[int], vecs: np.ndarray, thresh: float = 0.995) -> List[int]:
    out = []
    for i in idxs:
        if not out:
            out.append(i); continue
        v = vecs[i]
        keep = True
        for j in out:
            if float(np.dot(v, vecs[j])) >= thresh:
                keep = False; break
        if keep:
            out.append(i)
    return out

def _mmr_select(
    cand_idx: np.ndarray,  # shape [M]
    cand_vecs: np.ndarray, # shape [M, D] (normalized)
    qvec: np.ndarray,      # shape [D]     (normalized)
    k: int,
    lam: float = 0.7
) -> List[int]:
    """ Maximal Marginal Relevance selection to enforce diversity. """
    if len(cand_idx) == 0: return []
    selected: List[int] = []
    # precompute query sims
    q_sims = (cand_vecs @ qvec.astype(np.float32))
    remaining = set(range(len(cand_idx)))

    while len(selected) < min(k, len(cand_idx)):
        best_i, best_score = None, -1e9
        for r in remaining:
            # diversity term: max sim to anything already selected (or 0 if none)
            div = 0.0
            if selected:
                div = float(np.max(cand_vecs[r] @ cand_vecs[selected].T))
            score = lam * float(q_sims[r]) - (1.0 - lam) * div
            if score > best_score:
                best_score, best_i = score, r
        selected.append(best_i)
        remaining.remove(best_i)
    return [int(cand_idx[i]) for i in selected]

class RetrievalContextBuilder:
    """
    Builds a 5-vector context: [support_1, support_2, support_3, support_4, query_vector]
    Support selection:
      1) Try previous 4 chunks from the same doc as the top ANN 'anchor'
      2) Fill remainder with MMR-selected ANN neighbors (lane-filtered if available)
      3) Dedup by cosine, ensure total=4 supports
    """

    def __init__(
        self,
        index_path: str,
        vectors_path: str,
        meta_path: str,
        lane_path: Optional[str] = None,
        vector_dim: int = 768,
        nprobe: int = 16,
        max_candidates: int = 64,
    ):
        self.D = vector_dim
        self.index = faiss.read_index(index_path)
        try:
            # IVF/HNSW etc—set nprobe if supported
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = nprobe
        except Exception:
            pass

        # memory-map vectors for low-latency access
        # Expect float32 row-major [N, D], L2-normalized
        self.vecs = np.memmap(vectors_path, dtype="float32", mode="r")
        N = self.vecs.size // self.D
        self.vecs = self.vecs.reshape(N, self.D)

        # metadata (doc_id:int32, pos:int32); stored as .npz or .npy
        # If meta_path is .npz: expect keys 'doc_id', 'pos'
        # If .npy: expect structured array with fields 'doc_id','pos'
        if meta_path.endswith(".npz"):
            mz = np.load(meta_path, allow_pickle=False)
            self.doc_id = mz["doc_id"].astype(np.int32)
            self.pos    = mz["pos"].astype(np.int32)
        else:
            m = np.load(meta_path, allow_pickle=True)
            if isinstance(m, np.ndarray) and m.dtype.names and "doc_id" in m.dtype.names:
                self.doc_id = m["doc_id"].astype(np.int32)
                self.pos    = m["pos"].astype(np.int32)
            else:
                raise ValueError("Unsupported meta format. Use .npz with 'doc_id' and 'pos' arrays, or structured .npy.")

        self.lane = None
        if lane_path:
            # lane ids per row (int16 or int32)
            self.lane = np.load(lane_path).astype(np.int32)

        self.max_candidates = max_candidates

        # Safety checks
        assert self.vecs.shape[0] == self.doc_id.shape[0] == self.pos.shape[0], "Vector/meta length mismatch"
        assert self.vecs.shape[1] == self.D, "Vector dim mismatch"

    def _search(
        self,
        qvec: np.ndarray,
        k: int,
        lane_id: Optional[int] = None,
        oversample: int = 4
    ) -> np.ndarray:
        """Return candidate indices (np.ndarray of ints) filtered by lane (if provided)."""
        qvec = _normalize(qvec.astype(np.float32))
        want = min(self.max_candidates, max(k * oversample, k))
        # faiss expects shape [nq, D]
        D = qvec[None, :]
        sims, idxs = self.index.search(D, want)  # inner product preferred for cosine
        idxs = idxs[0]
        # Filter invalid (-1) and lane
        idxs = idxs[idxs >= 0]
        if lane_id is not None and self.lane is not None:
            lane_mask = (self.lane[idxs] == int(lane_id))
            idxs = idxs[lane_mask]
        return idxs[:want]

    def _same_doc_prev(self, anchor_idx: int, n: int) -> List[int]:
        """Collect up to n previous positions from the same doc as anchor."""
        did = int(self.doc_id[anchor_idx])
        p   = int(self.pos[anchor_idx])
        if p <= 0: return []
        # indices where doc_id==did and pos in [p-n, p-1]
        # Fast path: we assume 'pos' is dense per doc; we scan neighbors nearby.
        # You can replace with a doc->indices map for O(1).
        lo, hi = max(0, p - n), p - 1
        mask = (self.doc_id == did) & (self.pos >= lo) & (self.pos <= hi)
        prev = np.nonzero(mask)[0]
        # Sort by pos to preserve chronology
        order = np.argsort(self.pos[prev])
        return [int(i) for i in prev[order].tolist()][:n]

    def build_context(
        self,
        qvec: np.ndarray,
        lane_id: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Returns:
          ctx: np.ndarray shape [5, D] (supports x4 + qvec)
          support_indices: list of 4 global indices used
        """
        qvec = _normalize(qvec.astype(np.float32))

        # 1) ANN search to find anchor & pool
        cand = self._search(qvec, k=32, lane_id=lane_id, oversample=4)
        if cand.size == 0:
            # last-resort fallback: just clone small noise variants to avoid [v,v,v,v]
            noise = _normalize(qvec + 1e-3 * np.random.randn(4, self.D).astype(np.float32))
            ctx = np.vstack([noise, qvec[None, :]]).astype(np.float32)
            return ctx, []

        anchor = int(cand[0])

        # 2) Same-doc previous 4 if possible
        supports = self._same_doc_prev(anchor, n=4)

        # 3) Fill remaining with MMR-selected neighbors
        need = 4 - len(supports)
        if need > 0:
            # exclude already used & anchor
            used = set(supports + [anchor])
            pool = [int(i) for i in cand if int(i) not in used]
            if len(pool) > 0:
                pool = np.array(pool, dtype=np.int64)
                pool_vecs = self.vecs[pool]
                # Dedup near-duplicates first
                pool = np.array(_dedup_by_cosine(pool.tolist(), self.vecs, thresh=0.995), dtype=np.int64)
                pool_vecs = self.vecs[pool]
                mmr_sel = _mmr_select(pool, pool_vecs, qvec, k=need, lam=0.7)
                supports.extend(mmr_sel[:need])

        # 4) If still short (very rare), backfill with top remaining from cand
        if len(supports) < 4:
            for i in cand:
                ii = int(i)
                if ii not in supports and ii != anchor:
                    supports.append(ii)
                    if len(supports) == 4: break

        # 5) Finalize context matrix
        sup_vecs = self.vecs[supports] if len(supports) > 0 else np.zeros((0, self.D), dtype=np.float32)
        # Ensure normalized
        sup_vecs = _normalize(sup_vecs)
        ctx = np.vstack([sup_vecs, qvec[None, :]]).astype(np.float32)
        assert ctx.shape[0] == 5, f"Context shape invalid: {ctx.shape}"
        return ctx, supports
