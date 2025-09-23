#!/usr/bin/env python3
# Show Factoid entries by ID with Concept, TMD, CPE and truncated vectors.
# Reads local artifacts:
#  - artifacts/fw10k_chunks.jsonl (with {"id","concept","tmd":{...},"cpe":{...}})
#  - artifacts/fw10k_vectors.npz (emb: (N,768) or (N,784), ids: (N,))
# Usage:
#   python show_by_id.py 123 456 789

import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "artifacts/fw10k_chunks.jsonl"
VECT   = ROOT / "artifacts/fw10k_vectors.npz"

def load_chunks(path):
    idx = {}
    if not path.exists():
        print(f"[warn] {path} not found; returning empty index")
        return idx
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                rid = rec.get("id") or rec.get("doc_id") or rec.get("meta",{}).get("doc_id")
                if rid is not None:
                    idx[int(rid)] = rec
            except Exception:
                continue
    return idx

def load_vectors(path):
    if not path.exists():
        print(f"[warn] {path} not found; returning empty")
        return None, None
    npz = np.load(path)
    emb = npz["emb"]
    ids = npz["ids"].astype(int)
    return emb, ids

def trunc(arr, n=16):
    arr = np.asarray(arr).ravel()
    n = min(n, arr.size)
    head = ", ".join(f"{x:+.3f}" for x in arr[:n])
    return f"[{head}, …] (len={arr.size})"

def main(ids):
    chunks = load_chunks(CHUNKS)
    emb, all_ids = load_vectors(VECT)

    for s in ids:
        try:
            rid = int(s)
        except ValueError:
            print(f"\nID={s} (non-integer) — skipped")
            continue

        print(f"\n=== ID {rid} ===")
        rec = chunks.get(rid)
        if rec:
            concept = rec.get("concept") or rec.get("text") or "<unknown>"
            tmd = rec.get("tmd") or {}
            cpe = rec.get("cpe") or {}
            print(f"Concept: {concept}")
            print(f"TMD: {json.dumps(tmd, ensure_ascii=False)}")
            print(f"CPE: {json.dumps(cpe, ensure_ascii=False)}")
        else:
            print("No chunk metadata found.")

        if emb is not None and all_ids is not None:
            where = np.where(all_ids == rid)[0]
            if where.size:
                vec = emb[where[0]]
                if vec.shape[-1] in (768, 784):
                    # Show 4× 768D = 3072? The request says 'truncated to 4x of the 768D' – we interpret as 4×16=64 preview items.
                    # We'll print 64 elements (i.e., 4 blocks of 16 numbers).
                    print("768D/Fused Vector (truncated 64 els):", trunc(vec, 64))
                    # If we also store separate TMD16 or CPE vectors in rec, show them truncated:
                    tmd_vec = rec.get("tmd_vec")
                    cpe_vec = rec.get("cpe_vec")
                    if tmd_vec is not None:
                        print("TMD16 (truncated):", trunc(tmd_vec, 16))
                    if cpe_vec is not None:
                        print("CPE (truncated 16 els):", trunc(cpe_vec, 16))
                else:
                    print(f"Found vector with unexpected dim={vec.shape[-1]}")
            else:
                print("No vector found in NPZ for this ID.")
        else:
            print("Vectors NPZ not available.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_by_id.py <id1> <id2> ...")
        sys.exit(1)
    main(sys.argv[1:])
