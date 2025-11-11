import os
from unittest.mock import patch
from src.ingest_pipeline import IngestPipeline, FaissDB

def test_save_faiss_index():
    # Mock the FAISS index and database
    mock_index = faiss.IndexFlatL2(128)
    mock_db = FaissDB()
    mock_db.index = mock_index

    # Create an instance of IngestPipeline
    pipeline = IngestPipeline(mock_db)

    # Run the ingestion process
    with patch.object(FaissDB, 'add') as mock_add:
        pipeline.run()

    # Check if the FAISS index is saved
    assert os.path.exists('artifacts/lnsp_index.index')

def test_load_faiss_index():
    # Mock the FAISS index and database
    mock_index = faiss.IndexFlatL2(128)
    mock_db = FaissDB()
    mock_db.index = mock_index

    # Create an instance of IngestPipeline
    pipeline = IngestPipeline(mock_db)

    # Run the ingestion process
    with patch.object(FaissDB, 'add') as mock_add:
        pipeline.run()

    # Load the FAISS index
    loaded_index_path = 'artifacts/lnsp_index.index'
    loaded_index = faiss.read_index(loaded_index_path)
    assert isinstance(loaded_index, faiss.IndexFlatL2)

# Run tests
if __name__ == '__main__':
    test_save_faiss_index()
    test_load_faiss_index()
