import os
from src.db import FaissDB

def test_save_and_load():
    # Create a temporary FAISS index file
    index_path = 'artifacts/temp_index.index'
    faiss_db = FaissDB(index_path=index_path)

    # Add some vectors to the database
    vectors = [[1.0] * 128, [2.0] * 128]
    faiss_db.add(vectors)

    # Save the FAISS index
    faiss.write_index(faiss_db.index, index_path)

    # Load the FAISS index
    loaded_index = faiss.read_index(index_path)
    assert isinstance(loaded_index, faiss.IndexFlatL2)

    # Clean up
    os.remove(index_path)

# Run tests
if __name__ == '__main__':
    test_save_and_load()
