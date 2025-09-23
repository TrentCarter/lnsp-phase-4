import os
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from .utils.norms import l2_normalize


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
    def __init__(self, output_path: str = 'vectors.npz', retriever_adapter = None):
        self.output_path = output_path
        self.index = None
        self.vectors_stored = []
        self.retriever_adapter = retriever_adapter
        if faiss is not None:
            try:
                self.index = build_index(np.random.rand(0, 784))
                print(f"[FaissDB] Initialized")
            except Exception as exc:
                print(f"[FaissDB] Error initializing Faiss: {exc}")
                self.index = None
        else:
            print("[FaissDB] Faiss not available - using stubs")

    def add_vector(self, cpe_record: dict) -> bool:
        if not cpe_record.get('fused_vec'):
            print(f"[FaissDB] No fused_vec in CPE record {cpe_record['cpe_id']}")
            return False
        fused_vector = np.array(cpe_record['fused_vec'], dtype=np.float32)
        concept_vector = np.array(cpe_record.get('concept_vec', []), dtype=np.float32)
        question_vector = np.array(cpe_record.get('question_vec', []), dtype=np.float32)
        chunk_position = cpe_record.get('chunk_position') or {}
        doc_id = chunk_position.get('doc_id', '')

        self.vectors_stored.append({
            'cpe_id': cpe_record['cpe_id'],
            'doc_id': doc_id,
            'fused_vector': fused_vector,
            'concept_vector': concept_vector,
            'question_vector': question_vector,
            'lane_index': cpe_record.get('lane_index', 0),
            'concept_text': cpe_record.get('concept_text', ''),
        })
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
            np.savez(
                self.output_path,
                fused=fused_vectors,
                concept=concept_vectors,
                question=question_vectors,
                vectors=fused_vectors,  # Keep 'vectors' for backward compatibility
                cpe_ids=cpe_ids,
                lane_indices=lane_indices,
                concept_texts=concept_texts,
                doc_ids=doc_ids,
            )
            print(f"[FaissDB] Saved {len(self.vectors_stored)} vectors to {self.output_path}")
            return True
        except Exception as exc:
            print(f"[FaissDB] Error saving: {exc}")
            return False

    def load(self, npz_path: str) -> bool:
        """Load vectors from NPZ file and build index."""
        try:
            if not os.path.exists(npz_path):
                return False
            npz = np.load(npz_path)
            vectors = npz['vectors'].astype(np.float32)
            self.cpe_ids = npz['cpe_ids']
            self.lane_indices = npz.get('lane_indices', np.zeros(len(vectors)))
            self.concept_texts = npz.get('concept_texts', [''] * len(vectors))
            if 'doc_ids' in npz.files:
                self.doc_ids = npz['doc_ids']
            else:
                self.doc_ids = np.array([''] * len(vectors))

            # Build index with loaded vectors
            if len(vectors) > 0:
                if faiss is not None:
                    try:
                        # Use flat index for very small datasets
                        if len(vectors) < 32:
                            self.index = faiss.IndexFlatIP(vectors.shape[1])
                            # Flat index ignores nprobe; add normalized vectors directly
                            normalized_vecs = l2_normalize(vectors)
                            self.index.add(normalized_vecs)
                        else:
                            # Choose nlist based on corpus size; target 128 for ~10k
                            nlist = 128 if len(vectors) >= 8000 else min(32, max(8, len(vectors)//2))
                            self.index = build_index(vectors, nlist)
                        if self.index:
                            # If index was built inside build_index it already contains vectors
                            print(f"[FaissDB] Loaded and built index with {len(vectors)} vectors (nlist={getattr(self.index, 'nlist', 0)}, nprobe={getattr(self.index, 'nprobe', 'NA')})")
                            return True
                    except Exception as e:
                        print(f"[FaissDB] Error building index: {e}")
                        return False
            return False
        except Exception as exc:
            print(f"[FaissDB] Error loading {npz_path}: {exc}")
            return False

    def search(self, query_vec: np.ndarray, topk: int = 5, use_lightrag: bool = False):
        """Search for similar vectors."""
        if self.index is None:
            return []

        try:
            query_normalized = l2_normalize(query_vec.reshape(1, -1))
            scores, indices = self.index.search(query_normalized, topk)

            # Return results in the format expected by the API
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.cpe_ids):  # Valid index
                    doc_id = ''
                    if hasattr(self, 'doc_ids') and len(self.doc_ids) > idx:
                        doc_id = str(self.doc_ids[idx])

                    result = {
                        'cpe_id': str(self.cpe_ids[idx]),
                        'score': float(score),
                        'rank': i + 1,
                        'lane_index': int(self.lane_indices[idx]) if hasattr(self, 'lane_indices') else 0,
                        'retriever': 'faiss',
                        'metadata': {
                            'concept_text': str(self.concept_texts[idx]) if hasattr(self, 'concept_texts') else ''
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
