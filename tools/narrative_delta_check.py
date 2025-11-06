#!/usr/bin/env python3
import argparse, glob, os, re, json, gzip
import numpy as np, requests, itertools, math, sys

ENCODER_URL = os.environ.get("ENCODER_URL", "http://localhost:8767/embed")

def read_texts_from_glob(pattern):
    paths = sorted(glob.glob(pattern, recursive=True))
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                yield p, f.read()
        except Exception as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)

def split_paragraphs(txt):
    # normalize and split on blank lines; drop very short or non-alpha blobs
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if len(p.strip()) > 0]
    paras = [p for p in paras if sum(ch.isalpha() for ch in p) >= 40]
    return paras

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk: return
        yield chunk

def encode_texts(texts, batch_size=64):
    vecs = []
    for batch in batched(texts, batch_size):
        resp = requests.post(ENCODER_URL, json={"texts": batch})
        resp.raise_for_status()
        V = np.array(resp.json()["embeddings"], dtype=np.float32)
        # L2 norm guard (some services already normalize)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        V = V / norms
        vecs.append(V)
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

def compute_delta(contexts, targets):
    """
    Δ_seq = cos(c_newest, next) - cos(c_newest, prev)
    contexts: (N,5,768), targets: (N,768)

    Where:
    - c_newest = contexts[:, 4] (most recent context)
    - next = targets (the chunk after c_newest)
    - prev = contexts[:, 3] (the chunk before c_newest)
    """
    c_new = contexts[:, -1, :]                       # (N,768) - most recent context
    cos_next = np.sum(c_new * targets, axis=1)       # (N,) - similarity to next
    cos_prev = np.sum(c_new * contexts[:, -2, :], axis=1)  # (N,) - similarity to previous
    delta = cos_next - cos_prev
    return delta, cos_next, cos_prev

def build_sequences(paras, ctx=5):
    seqs = []
    for i in range(len(paras) - ctx):
        C = paras[i:i+ctx]      # 5 context
        T = paras[i+ctx]        # next
        seqs.append((C, T))
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True,
                    help="Glob for narrative .txt files (e.g., data/narrative/**/*.txt)")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--out-npz", default="artifacts/lvm/narrative_probe.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)

    files = list(itertools.islice(read_texts_from_glob(args.input_glob), args.max_files))
    if not files:
        print("No files matched. Provide --input-glob to your local narrative texts.", file=sys.stderr)
        sys.exit(2)

    paras_all = []
    file_sources = []
    for p, txt in files:
        paras = split_paragraphs(txt)
        if len(paras) >= 12:
            paras_all.append(paras)
            file_sources.append(p)

    if not paras_all:
        print("No usable paragraphs found (need >=12 per file).", file=sys.stderr)
        sys.exit(3)

    # Build sequences across all files
    sequences = []
    source_idx = []
    for fi, paras in enumerate(paras_all):
        seqs = build_sequences(paras, ctx=5)
        for s in seqs:
            sequences.append(s)
            source_idx.append(fi)

    # Limit to ~2000 sequences for speed
    sequences = sequences[:2000]; source_idx = source_idx[:2000]

    print(f"Built {len(sequences)} sequences from {len(paras_all)} files", file=sys.stderr)

    # Flatten texts for a single encoding pass
    to_encode = []
    for C, T in sequences:
        to_encode.extend(C)
        to_encode.append(T)

    print(f"Encoding {len(to_encode)} text chunks...", file=sys.stderr)
    V = encode_texts(to_encode, batch_size=64)
    if V.shape[0] != len(to_encode):
        print("Encoder returned wrong count.", file=sys.stderr); sys.exit(4)

    D = V.shape[1]
    C_mat = np.zeros((len(sequences), 5, D), dtype=np.float32)
    T_mat = np.zeros((len(sequences), D), dtype=np.float32)

    k = 0
    for i, (C, T) in enumerate(sequences):
        C_mat[i] = V[k:k+5]; k += 5
        T_mat[i] = V[k];     k += 1

    delta, cos_next, cos_prev = compute_delta(C_mat, T_mat)

    # Summary
    m = float(np.mean(delta))
    p25, p50, p75 = [float(np.quantile(delta, q)) for q in (0.25, 0.50, 0.75)]

    result = {
        "N_sequences": int(len(sequences)),
        "N_files": len(paras_all),
        "mean_delta": m,
        "delta_quartiles": [p25, p50, p75],
        "mean_cos_next": float(np.mean(cos_next)),
        "mean_cos_prev": float(np.mean(cos_prev)),
        "decision": "RETRY_P8" if m >= 0.15 else ("BORDERLINE" if m >= 0.10 else "ABANDON_LVM")
    }

    print(json.dumps(result, indent=2))

    # Save small NPZ for possible P8 retry
    np.savez_compressed(args.out_npz,
        contexts=C_mat, targets=T_mat,
        delta=delta, cos_next=cos_next, cos_prev=cos_prev,
        source_file_idx=np.array(source_idx, dtype=np.int32))

    print(f"\n✅ Saved to {args.out_npz}", file=sys.stderr)

    # Decision output
    if m >= 0.15:
        print("\n🟢 DECISION: RETRY P8 on narrative data (Δ ≥ 0.15)", file=sys.stderr)
        print("Next: Split NPZ and run P8 pilot with same training script", file=sys.stderr)
    elif m >= 0.10:
        print("\n🟡 DECISION: BORDERLINE (0.10 ≤ Δ < 0.15)", file=sys.stderr)
        print("Suggest: Review quartiles and decide", file=sys.stderr)
    else:
        print("\n🔴 DECISION: ABANDON AR-LVM (Δ < 0.10)", file=sys.stderr)
        print("Next: Pivot to Option A (retrieval-only)", file=sys.stderr)

if __name__ == "__main__":
    main()
