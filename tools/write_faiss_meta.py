import json
import os
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime, timezone

# Script to write Faiss metadata to JSON
def main():
    # Prefer environment overrides, then 10k artifacts, then 1k
    npz_candidates = [
        os.getenv('FAISS_NPZ_PATH'),
        'artifacts/fw10k_vectors.npz',
        'artifacts/fw1k_vectors.npz',
    ]
    index_candidates = [
        os.getenv('FAISS_INDEX_PATH'),
        'artifacts/fw10k_ivf.index',
        'artifacts/fw1k_ivf.index',
    ]

    npz_path = next((p for p in npz_candidates if p and Path(p).is_file()), None)
    if not npz_path:
        raise FileNotFoundError("No NPZ vectors file found (searched FAISS_NPZ_PATH, artifacts/fw10k_vectors.npz, artifacts/fw1k_vectors.npz)")

    index_path = next((p for p in index_candidates if p and Path(p).is_file()), None)
    
    # Load NPZ vectors
    vectors = np.load(npz_path)
    if 'vectors' not in vectors.files:
        raise KeyError(f"NPZ at {npz_path} missing 'vectors' array")
    num_vectors = int(vectors['vectors'].shape[0])
    dim = int(vectors['vectors'].shape[1]) if num_vectors > 0 else 0
    
    # Load Faiss index (if exists)
    if index_path and Path(index_path).is_file():
        index = faiss.read_index(index_path)
        index_type = index.__class__.__name__
        nlist = getattr(index, 'nlist', 0) if hasattr(index, 'nlist') else 0
    else:
        index_type = 'N/A'
        nlist = 0
    
    # Gather metadata
    meta = {
        'num_vectors': num_vectors,
        'index_type': index_type,
        'nlist': int(nlist),
        'dimension': dim,
        'npz_path': str(npz_path),
        'index_path': str(index_path) if index_path else None,
        'last_updated': datetime.now(timezone.utc).isoformat(),
    }
    
    # Write to JSON
    with open('artifacts/faiss_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print('faiss_meta.json:')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__' :
    main()
