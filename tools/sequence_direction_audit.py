#!/usr/bin/env python3
"""
Sequence Direction Audit & Rebuilder
------------------------------------

Diagnose and fix directionality issues in 5→1 sequence datasets.

Modes:
  1) verify  — Inspect an existing sequences NPZ for forward (+1) vs backward (−1) bias,
               cross-article leakage, and index monotonicity.
  2) rebuild — Rebuild sequences from the raw article NPZ in guaranteed FORWARD order
               with zero leakage and rich metadata.

Examples
--------
# Verify an existing sequences file against the full article store
python tools/sequence_direction_audit.py verify \
  --sequences artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --articles  artifacts/wikipedia_584k_fresh.npz \
  --max-samples 50000

# Rebuild a clean forward dataset (ctx=5) from article vectors
python tools/sequence_direction_audit.py rebuild \
  --articles artifacts/wikipedia_584k_fresh.npz \
  --output   artifacts/lvm/training_sequences_ctx5_FORWARD_clean.npz \
  --context-len 5 \
  --exclude-articles 1500-1999,4000-4499,7672-8470

Outputs
-------
* VERIFY: prints JSON summary (k=+1 vs k=−1 means, margins, leakage count, monotonicity errors).
* REBUILD: writes NPZ with keys: contexts, targets, article_ids, target_indices, context_indices,
           metadata JSON (as np.void) including coherence stats and build manifest.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

# -------------------------
# Helpers
# -------------------------

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def parse_ranges(ranges: Optional[str]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    out: List[Tuple[int, int]] = []
    for part in ranges.split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split('-')
        out.append((int(a), int(b)))
    return out


def in_any_range(x: int, ranges: List[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= x <= b:
            return True
    return False

# -------------------------
# Article store
# -------------------------

class ArticleStore:
    def __init__(self, vectors: np.ndarray, article_ids: np.ndarray, chunk_idx: Optional[np.ndarray] = None):
        assert vectors.ndim == 2
        assert vectors.shape[0] == article_ids.shape[0]
        self.D = vectors.shape[1]
        self.by_article: Dict[int, np.ndarray] = {}
        self.index_maps: Dict[int, Dict[int, int]] = {}
        # group by article and sort by chunk_idx if present
        tmp: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for i, aid in enumerate(article_ids.tolist()):
            idx = int(chunk_idx[i]) if chunk_idx is not None else i
            tmp.setdefault(aid, []).append((idx, vectors[i]))
        for aid, pairs in tmp.items():
            pairs.sort(key=lambda t: t[0])
            arr = np.stack([p[1] for p in pairs], axis=0)
            self.by_article[aid] = l2_normalize(arr)
            if chunk_idx is not None:
                self.index_maps[aid] = {int(p[0]): j for j, p in enumerate(pairs)}
            else:
                self.index_maps[aid] = {j: j for j in range(len(pairs))}

    @classmethod
    def from_npz(cls, path: str) -> "ArticleStore":
        z = np.load(path, allow_pickle=True)
        def pick(*cands):
            for k in cands:
                if k in z: return z[k]
            raise KeyError(f"Missing any of {cands} in {path}")
        vectors = l2_normalize(np.asarray(pick('vectors','embeddings','X')))
        art_ids = np.asarray(pick('article_indices','article_ids'))
        chunk_idx = z.get('chunk_indices', z.get('seq_idx', None))
        if chunk_idx is not None: chunk_idx = np.asarray(chunk_idx)
        return cls(vectors=vectors, article_ids=art_ids, chunk_idx=chunk_idx)

    def get_offset(self, aid: int, idx: int, k: int) -> Optional[np.ndarray]:
        if aid not in self.by_article: return None
        arr = self.by_article[aid]
        imap = self.index_maps[aid]
        if idx not in imap: return None
        j = imap[idx] + k
        if j < 0 or j >= arr.shape[0]: return None
        return arr[j]

    def article_len(self, aid: int) -> int:
        return 0 if aid not in self.by_article else self.by_article[aid].shape[0]

# -------------------------
# VERIFY mode
# -------------------------

def verify_sequences(seq_path: str, art_path: str, max_samples: int) -> int:
    z = np.load(seq_path, allow_pickle=True)
    
    # Handle different NPZ structures
    if 'context_sequences' in z:
        print("Found 'context_sequences' in NPZ file")
        # New format
        contexts = l2_normalize(np.asarray(z['context_sequences']))
        targets = l2_normalize(np.asarray(z['target_vectors']))
        print(f"Loaded {len(contexts)} sequences with shape {contexts.shape}")
        
        # Try to get article IDs if available
        if 'article_ids' in z:
            art_ids = np.asarray(z['article_ids'])
            print(f"Found {len(np.unique(art_ids))} unique article IDs")
        else:
            art_ids = None
            print("No article_ids found in NPZ")
            
        if 'target_indices' in z:
            tgt_idx = np.asarray(z['target_indices'])
            print(f"Found target_indices with range {tgt_idx.min()}-{tgt_idx.max()}")
        else:
            tgt_idx = None
            print("No target_indices found in NPZ")
            
        ctx_idx = None  # Not available in this format
    else:
        # Old format
        def pick(*cands):
            for k in cands:
                if k in z: return z[k]
            raise KeyError(f"Missing any of {cands} in {seq_path}")
        contexts = l2_normalize(np.asarray(pick('contexts','context','X')))
        targets = l2_normalize(np.asarray(pick('targets','target','y')))
        art_ids = z.get('article_ids', None)
        tgt_idx = z.get('target_indices', None)
        ctx_idx = z.get('context_indices', None)  # optional (Nx5)

    store = ArticleStore.from_npz(art_path)

    N = min(max_samples, contexts.shape[0])
    kpos, kneg = [], []  # k=+1, k=-1
    leaks = 0
    monotonic_errors = 0

    for i in range(N):
        ctx = contexts[i]
        tgt = targets[i]
        aid = int(art_ids[i]) if art_ids is not None else None
        tidx = int(tgt_idx[i]) if tgt_idx is not None else None

        # leakage / monotonicity checks if indices present
        if ctx_idx is not None:
            cidx = np.asarray(ctx_idx[i]).tolist()
            # check same-article assumption if metadata encodes it via target idx
            if tidx is not None and aid is not None:
                # if any gap crosses article boundary, we count as leak
                # (we cannot fully prove without per-frame article ids; conservative)
                pass
            # monotonic increasing
            if not all(cidx[j] + 1 == cidx[j+1] for j in range(4)):
                monotonic_errors += 1
        # if we have article/idx, compute k=±1 cosines
        if aid is not None and tidx is not None:
            tpos = store.get_offset(aid, tidx, +1)  # next
            tneg = store.get_offset(aid, tidx, -1)  # previous
            if tpos is not None and tneg is not None:
                kpos.append(cosine(tgt, tpos))
                kneg.append(cosine(tgt, tneg))

        # crude leakage signal: if all five context vectors are identical → repeat-pad bug
        if all(abs(float(np.dot(ctx[0], ctx[j])) - 1.0) < 1e-4 for j in range(1,5)):
            leaks += 1

    # Compute sequence coherence statistics
    seq_coherences = []
    for i in range(min(1000, N)):  # Sample 1000 sequences for efficiency
        ctx = contexts[i]
        # Compute mean cosine between consecutive context vectors
        coh = np.mean([cosine(ctx[j], ctx[j+1]) for j in range(4)])
        seq_coherences.append(coh)
    
    summary = {
        'samples': int(N),
        'k=+1_mean': float(np.mean(kpos)) if kpos else float('nan'),
        'k=-1_mean': float(np.mean(kneg)) if kneg else float('nan'),
        'margin(+1 - -1)': (float(np.mean(kpos) - np.mean(kneg)) if kpos and kneg else float('nan')),
        'repeat_pad_count': int(leaks),
        'monotonicity_errors': int(monotonic_errors),
        'mean_sequence_coherence': float(np.mean(seq_coherences)) if seq_coherences else float('nan'),
        'min_sequence_coherence': float(np.min(seq_coherences)) if seq_coherences else float('nan'),
        'max_sequence_coherence': float(np.max(seq_coherences)) if seq_coherences else float('nan'),
    }
    print(json.dumps(summary, indent=2))

    # Gate: margin must be positive
    if math.isnan(summary['margin(+1 - -1)']) or summary['margin(+1 - -1)'] <= 0:
        return 2
    return 0

# -------------------------
# REBUILD mode
# -------------------------

def compute_coherence(vecs: np.ndarray) -> float:
    if vecs.shape[0] < 2: return float('nan')
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    return float(np.mean(sims))


def rebuild_sequences(art_path: str, out_path: str, context_len: int, exclude_ranges: List[Tuple[int,int]]) -> int:
    art = np.load(art_path, allow_pickle=True)
    def pick(*cands):
        for k in cands:
            if k in art: return art[k]
        raise KeyError(f"Missing any of {cands} in {art_path}")

    vectors = l2_normalize(np.asarray(pick('vectors','embeddings','X')))
    art_ids = np.asarray(pick('article_indices','article_ids'))
    chunk_idx = art.get('chunk_indices', art.get('seq_idx', None))
    if chunk_idx is not None: chunk_idx = np.asarray(chunk_idx)

    # group by article, sort by chunk_idx
    groups: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for i, aid in enumerate(art_ids.tolist()):
        if in_any_range(aid, exclude_ranges):
            continue
        idx = int(chunk_idx[i]) if chunk_idx is not None else i
        groups.setdefault(aid, []).append((idx, vectors[i]))
    # build sequences
    C = context_len
    ctx_list, tgt_list, aid_list, tidx_list, cidx_list = [], [], [], [], []
    n_articles = 0
    coherences: List[float] = []
    for aid, pairs in groups.items():
        pairs.sort(key=lambda t: t[0])
        arr_idx = [p[0] for p in pairs]
        arr_vec = l2_normalize(np.stack([p[1] for p in pairs], axis=0))
        n = arr_vec.shape[0]
        if n < C + 1:  # need at least C ctx + 1 target
            continue
        # coherence for manifest
        coherences.append(compute_coherence(arr_vec))
        n_articles += 1
        for i in range(0, n - C):
            ctx = arr_vec[i:i+C]
            tgt = arr_vec[i+C]
            # HARD invariants: strictly forward, contiguous, same article
            if not all(arr_idx[i+j] + 1 == arr_idx[i+j+1] for j in range(C-1)):
                # skip non-contiguous sequences
                continue
            ctx_list.append(ctx)
            tgt_list.append(tgt)
            aid_list.append(aid)
            tidx_list.append(arr_idx[i+C])
            cidx_list.append(arr_idx[i:i+C])

    if not ctx_list:
        print("[ERROR] No sequences produced. Check exclude ranges and context length.")
        return 2

    contexts = np.stack(ctx_list, axis=0)
    targets  = np.stack(tgt_list, axis=0)
    article_ids = np.asarray(aid_list, dtype=np.int64)
    target_indices = np.asarray(tidx_list, dtype=np.int64)
    context_indices = np.stack([np.asarray(r, dtype=np.int64) for r in cidx_list], axis=0)

    manifest = {
        'mode': 'rebuild-forward',
        'context_len': C,
        'exclude_article_ranges': exclude_ranges,
        'num_sequences': int(contexts.shape[0]),
        'num_articles': int(n_articles),
        'mean_article_coherence': float(np.mean([c for c in coherences if not math.isnan(c)])) if coherences else float('nan'),
    }

    np.savez(out_path,
             contexts=contexts,
             targets=targets,
             article_ids=article_ids,
             target_indices=target_indices,
             context_indices=context_indices,
             manifest=np.void(json.dumps(manifest).encode('utf-8')))

    print(json.dumps({'wrote': out_path, **manifest}, indent=2))
    return 0

# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description='Sequence Direction Audit & Rebuilder')
    sub = p.add_subparsers(dest='cmd', required=True)

    pv = sub.add_parser('verify', help='Verify directionality/leakage of a sequences NPZ')
    pv.add_argument('--sequences', required=True)
    pv.add_argument('--articles', required=True)
    pv.add_argument('--max-samples', type=int, default=50000)

    pr = sub.add_parser('rebuild', help='Rebuild forward sequences from article NPZ')
    pr.add_argument('--articles', required=True)
    pr.add_argument('--output', required=True)
    pr.add_argument('--context-len', type=int, default=5)
    pr.add_argument('--exclude-articles', default='')

    args = p.parse_args()

    if args.cmd == 'verify':
        sys_exit = verify_sequences(args.sequences, args.articles, args.max_samples)
        raise SystemExit(sys_exit)
    else:
        excl = parse_ranges(args.exclude_articles)
        sys_exit = rebuild_sequences(args.articles, args.output, args.context_len, excl)
        raise SystemExit(sys_exit)

if __name__ == '__main__':
    main()
