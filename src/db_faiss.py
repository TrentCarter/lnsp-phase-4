import os
from typing import Optional, Sequence

import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from .utils.norms import l2_normalize
from .tmd_encoder import unpack_tmd


def _format_tmd_code(tmd_dense: np.ndarray, idx: int) -> str:
    """Format TMD code from dense vector index."""
    if tmd_dense is None or idx >= len(tmd_dense):
        return ""

    # Extract the first 3 values from TMD dense vector and convert to integers
    tmd_vec = tmd_dense[idx]
    if len(tmd_vec) < 3:
        return ""

    domain = int(tmd_vec[0])
    task = int(tmd_vec[1])
    modifier = int(tmd_vec[2])
    return f"{domain}.{task}.{modifier}"


def _default_nprobe() -> int:
    try:
        return max(1, int(os.getenv("FAISS_NPROBE", "16")))
    except Exception:
        return 16


def build_index(vectors: np.ndarray, nlist=64):
    if faiss is None:
        return None
    d = vectors.shape[1]
    vecs = l2_normalize(vectors.astype(np.float32))
    spec = f"IVF{max(8, int(nlist))},Flat"
    index = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT)
    # Train on a subset for speed; fall back to all if tiny
    N = vecs.shape[0]
    k = max(1, int(N * 0.05))
    ids = np.random.default_rng(0).choice(N, size=min(k, N), replace=False)
    index.train(vecs[ids])
    index.add(vecs)
    index.nprobe = _default_nprobe()
    return index


def search(index, queries: np.ndarray, topk=5):
    if index is None:
        return None
    q = l2_normalize(queries)
    scores, ids = index.search(q, topk)
    return scores, ids


class FaissDB:
    def __init__(self, index_path: str = 'artifacts/fw10k_ivf.index', meta_npz_path: str = None, nprobe: int = None, output_path: str = None, retriever_adapter=None):
        self.index_path = index_path
        # NPZ should carry doc_ids[] and other metadata needed by API
        if meta_npz_path is None:
            # Use fw10k_vectors.npz as default metadata file
            meta_npz_path = 'artifacts/fw10k_vectors.npz'
        self.meta_npz_path = meta_npz_path
        self.index = None
        self.doc_ids = None
        self.cpe_ids = None
        self.lane_indices = None
        self.concept_texts = None
        self.nprobe = nprobe or _default_nprobe()

        # Legacy attributes for backward compatibility
        self.output_path = output_path or meta_npz_path
        self.vectors_stored = []
        self.retriever_adapter = retriever_adapter

    def add_vector(self, cpe_record: dict) -> bool:
        if not cpe_record.get('fused_vec'):
            print(f"[FaissDB] No fused_vec in CPE record {cpe_record['cpe_id']}")
            return False
        fused_vector = np.array(cpe_record['fused_vec'], dtype=np.float32)
        concept_vector = np.array(cpe_record.get('concept_vec', []), dtype=np.float32)
        question_vector = np.array(cpe_record.get('question_vec', []), dtype=np.float32)
        chunk_position = cpe_record.get('chunk_position') or {}
        doc_id = chunk_position.get('doc_id', '')
        tmd_dense = cpe_record.get('tmd_dense')
        if tmd_dense is None:
            tmd_vector = np.zeros(16, dtype=np.float32)
        else:
            tmd_vector = np.array(tmd_dense, dtype=np.float32)

        self.vectors_stored.append({
            'cpe_id': cpe_record['cpe_id'],
            'doc_id': doc_id,
            'fused_vector': fused_vector,
            'concept_vector': concept_vector,
            'question_vector': question_vector,
            'lane_index': cpe_record.get('lane_index', 0),
            'concept_text': cpe_record.get('concept_text', ''),
            'tmd_dense': tmd_vector,
        })
        if self.retriever_adapter is not None:
            try:
                self.retriever_adapter.register_document(
                    fused_vector.tolist(),
                    {
                        'cpe_id': cpe_record['cpe_id'],
                        'doc_id': doc_id,
                        'lane_index': cpe_record.get('lane_index', 0),
                    },
                )
            except Exception as exc:
                print(f"[FaissDB] Retriever adapter error: {exc}")
        print(f"[FaissDB] Added vector for CPE {cpe_record['cpe_id']}")
        return True

    def save(self) -> bool:
        try:
            if not self.vectors_stored:
                print("[FaissDB] No vectors to save")
                return True
            fused_vectors = np.array([v['fused_vector'] for v in self.vectors_stored])
            concept_vectors = np.array([v['concept_vector'] for v in self.vectors_stored])
            question_vectors = np.array([v['question_vector'] for v in self.vectors_stored])
            cpe_ids = np.array([v['cpe_id'] for v in self.vectors_stored])
            lane_indices = np.array([v['lane_index'] for v in self.vectors_stored])
            concept_texts = np.array([v['concept_text'] for v in self.vectors_stored])
            doc_ids = np.array([v.get('doc_id', '') for v in self.vectors_stored])
            tmd_vectors = np.array([v['tmd_dense'] for v in self.vectors_stored])
            np.savez(
                self.output_path,
                fused=fused_vectors,
                concept=concept_vectors,
                question=question_vectors,
                vectors=fused_vectors,  # Keep 'vectors' for backward compatibility
                concept_vecs=concept_vectors,
                question_vecs=question_vectors,
                cpe_ids=cpe_ids,
                lane_indices=lane_indices,
                concept_texts=concept_texts,
                doc_ids=doc_ids,
                tmd_dense=tmd_vectors,
            )
            print(f"[FaissDB] Saved {len(self.vectors_stored)} vectors to {self.output_path}")
            return True
        except Exception as exc:
            print(f"[FaissDB] Error saving: {exc}")
            return False

    def load(self, index_path: str = None):
        """Load a pre-built FAISS index and associated metadata - pure load path only."""
        if index_path is None:
            index_path = self.index_path

        # 1) load prebuilt FAISS index only (no training/build here)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        index = faiss.read_index(str(index_path))
        self.index_path = index_path

        # 2) check if index is already ID-mapped
        is_id_mapped = isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2))
        print(f"[FaissDB] Index type: {type(index).__name__}, ID-mapped: {is_id_mapped}")

        # If not ID-mapped and empty, we could wrap, but prefer pre-built ID-mapped indices
        if not is_id_mapped:
            print(f"[FaissDB] WARNING: Index is not ID-mapped. Using positional IDs as fallback.")
            self._is_id_mapped = False
        else:
            self._is_id_mapped = True

        # 3) load npz metadata (doc_ids[] required)
        if not os.path.exists(self.meta_npz_path):
            raise FileNotFoundError(f"Missing FAISS npz meta: {self.meta_npz_path}")
        npz = np.load(self.meta_npz_path, allow_pickle=True)
        if "doc_ids" not in npz:
            raise ValueError(f"{self.meta_npz_path} missing 'doc_ids' array")

        doc_ids = npz["doc_ids"]
        # If index has no vectors, that's a build bug
        if index.ntotal == 0:
            raise RuntimeError("FAISS index has 0 vectors; rebuild artifacts before serving")

        # Load additional metadata
        self.cpe_ids = npz.get('cpe_ids', np.arange(len(doc_ids)))
        self.lane_indices = npz.get('lane_indices', np.zeros(len(doc_ids)))
        self.concept_texts = npz.get('concept_texts', [''] * len(doc_ids))
        self.tmd_dense = npz.get('tmd_dense', None)

        # nprobe tuning
        try:
            if hasattr(index, "nprobe"):
                index.nprobe = int(self.nprobe)
        except Exception:
            pass

        self.index = index
        self.doc_ids = doc_ids
        print(f"[FaissDB] Loaded index from {index_path} and metadata from {self.meta_npz_path}")
        return self

    @property
    def dim(self) -> int:
        return int(self.index.d) if self.index is not None else -1

    def search(self, qvecs: np.ndarray, topk: int):
        """Pure FAISS search interface for API layer."""
        if self.index is None:
            raise RuntimeError("FAISS not loaded")
        if qvecs.dtype != np.float32:
            qvecs = qvecs.astype(np.float32, copy=False)
        # shape check: must match 784 exactly
        if qvecs.shape[1] != self.dim:
            raise ValueError(f"Query dim {qvecs.shape[1]} != index dim {self.dim}")
        D, I = self.index.search(qvecs, topk)
        return D, I

    def search_legacy(
        self,
        query_vec: np.ndarray,
        topk: int = 5,
        use_lightrag: bool = False,
        *,
        nprobe: Optional[int] = None,
        boost_vectors: Optional[Sequence[np.ndarray]] = None,
    ):
        """Search for similar vectors."""
        if self.index is None:
            return []

        try:
            original_nprobe = None
            if nprobe is not None and hasattr(self.index, "nprobe"):
                original_nprobe = int(getattr(self.index, "nprobe", _default_nprobe()))
                self.index.nprobe = max(1, int(nprobe))

            query_normalized = l2_normalize(query_vec.reshape(1, -1))
            scores, indices = self.index.search(query_normalized, topk)

            # Return results in the format expected by the API
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0:
                    continue  # Skip invalid indices

                # For ID-mapped indices, idx is the ID we assigned (positional index)
                # For non-ID-mapped, idx is positional
                pos_idx = int(idx)  # Convert to int in case it's numpy type

                if pos_idx >= len(self.cpe_ids):
                    continue  # Out of bounds

                doc_id = ''
                if hasattr(self, 'doc_ids') and len(self.doc_ids) > pos_idx:
                    doc_id = str(self.doc_ids[pos_idx])

                tmd_code = ""
                if hasattr(self, 'tmd_dense') and self.tmd_dense is not None:
                    tmd_code = _format_tmd_code(self.tmd_dense, pos_idx)

                result = {
                    'cpe_id': str(self.cpe_ids[pos_idx]),
                    'score': float(score),
                    'rank': i + 1,
                    'lane_index': int(self.lane_indices[pos_idx]) if hasattr(self, 'lane_indices') else 0,
                    'retriever': 'faiss',
                    'tmd_code': tmd_code,
                    'metadata': {
                        'concept_text': str(self.concept_texts[pos_idx]) if hasattr(self, 'concept_texts') else ''
                    }
                }
                if doc_id:
                    result['doc_id'] = doc_id
                    result['metadata']['doc_id'] = doc_id

                results.append(result)
            return results
        except Exception as exc:
            print(f"[FaissDB] Search error: {exc}")
            return []
        finally:
            if nprobe is not None and hasattr(self.index, "nprobe") and original_nprobe is not None:
                try:
                    self.index.nprobe = int(original_nprobe)
                except Exception:
                    pass
