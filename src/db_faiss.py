import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from .utils.norms import l2_normalize


def build_index(vectors: np.ndarray, nlist=64):
    if faiss is None:
        return None
    d = vectors.shape[1]
    q = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    vecs = l2_normalize(vectors)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = 8
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
        vector = np.array(cpe_record['fused_vec'], dtype=np.float32)
        self.vectors_stored.append({
            'cpe_id': cpe_record['cpe_id'],
            'vector': vector,
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
            vectors = np.array([v['vector'] for v in self.vectors_stored])
            cpe_ids = np.array([v['cpe_id'] for v in self.vectors_stored])
            lane_indices = np.array([v['lane_index'] for v in self.vectors_stored])
            concept_texts = np.array([v['concept_text'] for v in self.vectors_stored])
            np.savez(self.output_path, vectors=vectors, cpe_ids=cpe_ids, lane_indices=lane_indices, concept_texts=concept_texts)
            print(f"[FaissDB] Saved {len(self.vectors_stored)} vectors to {self.output_path}")
            return True
        except Exception as exc:
            print(f"[FaissDB] Error saving: {exc}")
            return False
