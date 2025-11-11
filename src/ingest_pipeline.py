import faiss
import os

class IngestPipeline:
    def __init__(self, faiss_db):
        self.faiss_db = faiss_db

    def run(self):
        # Existing ingestion logic here...
        vectors = ...  # Build vectors in memory
        self.faiss_db.add(vectors)
        self.save_faiss_index()

    def save_faiss_index(self):
        index_path = 'artifacts/lnsp_index.index'
        faiss.write_index(self.faiss_db.index, index_path)
        print(f"FAISS index saved to {index_path}")

# Example usage
if __name__ == '__main__':
    from src.db import FaissDB
    faiss_db = FaissDB()
    pipeline = IngestPipeline(faiss_db)
    pipeline.run()
