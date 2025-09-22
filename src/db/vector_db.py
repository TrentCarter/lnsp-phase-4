from __future__ import annotations
from typing import Dict, Any, Optional
import os
import numpy as np
from uuid import UUID

try:
    from ..loaders.pg_writer import connect, upsert_cpe_vectors
    from ..loaders.faiss_writer import FaissShard
except ImportError:
    def upsert_cpe_vectors(conn, cpe_id, fused_vec, question_vec=None, concept_vec=None, tmd_dense=None, fused_norm=None):
        print(f"[VECTOR_DB STUB] Would upsert vectors for CPE: {cpe_id}")
    def connect():
        return None
    class FaissShard:
        def __init__(self, dim: int, nlist: int = 256):
            self.dim = dim
            self.index = None
        def build(self, vectors): return False
        def add(self, vectors): return False


class VectorDB:
    """Vector database writer for embeddings and Faiss indexing."""

    def __init__(self):
        self.use_pg = os.getenv("USE_POSTGRES", "false").lower() == "true"
        self.faiss_shards = {}  # lane_index -> FaissShard
        self.faiss_out_path = os.getenv("FAISS_OUT", "/tmp/factoid_vecs.npz")

    def insert_vector_entry(self, cpe_id: str, extraction: Dict[str, Any]):
        """Insert vector data into database and Faiss."""

        # Prepare vectors
        concept_vec = np.array(extraction["concept_vec"])
        question_vec = np.array(extraction["question_vec"])
        tmd_dense = np.array(extraction["tmd_dense"])

        # Fuse: TMD + Concept = 784D
        fused_vec = np.concatenate([tmd_dense, concept_vec])
        fused_norm = float(np.linalg.norm(fused_vec))

        # PG vector storage
        if self.use_pg:
            conn = connect()
            if conn:
                upsert_cpe_vectors(
                    conn, UUID(cpe_id), fused_vec, question_vec,
                    concept_vec, tmd_dense, fused_norm
                )
                conn.close()

        # Faiss indexing per lane
        lane_index = extraction["lane_index"]
        if lane_index not in self.faiss_shards:
            self.faiss_shards[lane_index] = FaissShard(dim=784, nlist=32)  # Small nlist for demo

        shard = self.faiss_shards[lane_index]
        if shard.index is None:
            # First vector in this lane - build index
            shard.build(fused_vec.reshape(1, -1))
        else:
            # Add to existing index
            shard.add(fused_vec.reshape(1, -1))

        print(f"[VECTOR_DB] CPE {cpe_id}: fused=784D norm={fused_norm:.3f} lane={lane_index}")

    def save_faiss_shards(self):
        """Save Faiss shards to disk."""
        if not self.faiss_shards:
            return

        # Simple NPZ save for demo
        shard_data = {}
        for lane_idx, shard in self.faiss_shards.items():
            if shard.index is not None:
                # In real implementation, you'd serialize the Faiss index properly
                shard_data[f"lane_{lane_idx}"] = {
                    "dim": shard.dim,
                    "nlist": shard.nlist,
                    "trained": shard.index.is_trained() if shard.index else False
                }

        np.savez(self.faiss_out_path, **shard_data)
        print(f"[VECTOR_DB] Saved {len(self.faiss_shards)} Faiss shards to {self.faiss_out_path}")
