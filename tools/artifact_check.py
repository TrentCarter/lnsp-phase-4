#!/usr/bin/env python3
"""Artifact validation script for LNSP components."""

import numpy as np
import json
import sys
from pathlib import Path

def check_artifacts():
    """Check that required artifacts exist and are valid."""
    artifacts_dir = Path('artifacts')
    if not artifacts_dir.exists():
        return {"artifacts_ok": False, "error": "artifacts directory not found"}

    # Check vector file
    vectors_ok = False
    vectors_file = artifacts_dir / 'fw1k_vectors.npz'
    if vectors_file.exists():
        try:
            data = np.load(vectors_file)
            vectors = data[list(data.keys())[0]]
            n_vectors, dim = vectors.shape
            vectors_ok = n_vectors > 0 and dim > 0
        except Exception as e:
            vectors_ok = False

    # Check index file
    index_ok = False
    index_files = ['fw1k_ivf.index', 'faiss_fw1k.ivf']
    for index_name in index_files:
        index_file = artifacts_dir / index_name
        if index_file.exists():
            index_ok = True
            break

    # Check metadata
    meta_ok = False
    meta_file = artifacts_dir / 'faiss_meta.json'
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            meta_ok = 'n_vectors' in meta and 'vector_dim' in meta
        except:
            meta_ok = False

    artifacts_ok = vectors_ok and index_ok and meta_ok

    result = {
        "artifacts_ok": artifacts_ok,
        "vectors_ok": vectors_ok,
        "index_ok": index_ok,
        "meta_ok": meta_ok,
    }

    return result

if __name__ == "__main__":
    result = check_artifacts()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["artifacts_ok"] else 1)
