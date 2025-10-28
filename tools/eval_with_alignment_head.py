#!/usr/bin/env python3
"""
Evaluate with Alignment Head
=============================

Tests retrieval performance with alignment head post-processing.

Pipeline:
  LVM prediction → Alignment head → Retrieval → Reranking
"""

import argparse
import json
import time
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from rerank_strategies_v2 import (
    dedup_candidates, mmr, rerank_with_sequence_bias, _l2norm
)


class AlignmentHead(nn.Module):
    """Tiny MLP with residual for vector alignment."""

    def __init__(self, dim=768, hidden_dim=256, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        residual = self.net(x)
        v_out = x + self.alpha * residual
        v_out = F.normalize(v_out, p=2, dim=-1)
        return v_out


class ShardAssistedRetrievalWithAlignment:
    """Retrieval with alignment head and shard assist."""

    def __init__(self, faiss_index, global_payload, article_shards, alignment_head):
        self.global_index = faiss_index
        self.global_payload = global_payload
        self.article_shards = article_shards
        self.alignment_head = alignment_head

    def search(self, query_vec, last_meta, K_global=50, K_local=20, K_union=60):
        """Search with alignment head and shard assist."""
        # Apply alignment head
        with torch.no_grad():
            q_tensor = torch.from_numpy(query_vec).float().unsqueeze(0)
            q_aligned = self.alignment_head(q_tensor).numpy()[0]

        # 1. Global IVF search
        q_norm = _l2norm(q_aligned.reshape(1, -1)).astype(np.float32)
        D_global, I_global = self.global_index.search(q_norm, K_global)

        global_cands = []
        for score, idx in zip(D_global[0], I_global[0]):
            if idx == -1:
                continue
            text, meta, vec = self.global_payload[int(idx)]
            global_cands.append((text, float(score), meta, vec))

        # 2. Local shard search (if article_index present)
        local_cands = []
        article_idx = last_meta.get("article_index")
        if article_idx is not None and article_idx in self.article_shards:
            shard = self.article_shards[article_idx]
            D_local, I_local = shard["index"].search(q_norm, min(K_local, shard["n_chunks"]))

            for score, local_idx in zip(D_local[0], I_local[0]):
                if local_idx == -1:
                    continue
                text, meta, vec, global_idx = shard["payload"][int(local_idx)]
                local_cands.append((text, float(score), meta, vec))

        # 3. Union + dedup
        all_cands = global_cands + local_cands
        seen = set()
        deduped = []
        for cand in all_cands:
            key = (int(cand[2]["article_index"]), int(cand[2]["chunk_index"]))
            if key not in seen:
                seen.add(key)
                deduped.append(cand)

        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped[:K_union]


def evaluate(retriever, dataset, args):
    """Evaluate with alignment head."""
    n = len(dataset)
    r1 = r5 = r10 = 0
    mrr = 0.0
    lat = []
    contain_20 = contain_50 = 0

    for ex in dataset:
        qv = ex["pred_vec"].astype(np.float32)
        last_meta = ex["last_meta"]
        truth_key = tuple(ex["truth_key"])

        t0 = time.perf_counter()

        # 1. Retrieval with alignment
        cands = retriever.search(
            qv, last_meta,
            K_global=args.K_global,
            K_local=args.K_local,
            K_union=args.K_union
        )

        # Track containment
        raw_keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands]
        if truth_key in raw_keys[:20]:
            contain_20 += 1
        if truth_key in raw_keys[:50]:
            contain_50 += 1

        # 2. Dedup
        cands = dedup_candidates(cands)

        # 3. MMR
        if args.do_mmr and len(cands) > args.top_final:
            vecs = np.stack([c[3] for c in cands], axis=0).astype(np.float32)
            sel = mmr(qv, vecs, lambda_=args.mmr_lambda, k=args.top_final)
            cands = [cands[i] for i in sel]
        else:
            cands = cands[:args.top_final]

        # 4. Sequence-bias reranking
        if args.use_seq_bias and cands:
            ranked = rerank_with_sequence_bias(
                candidates=cands,
                last_ctx_meta=last_meta,
                w_cos=1.0,
                w_same_article=args.w_same_article,
                w_next_gap=args.w_next_gap,
                tau=args.tau,
                directional_bonus=args.directional_bonus,
                pred_vec=qv,
                last_vec=None,
            )
            cands_sorted = [c for _, c in ranked]
        else:
            cands_sorted = cands

        keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands_sorted]

        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        # Check ground truth
        if truth_key in keys:
            idx = keys.index(truth_key)
            if idx == 0:
                r1 += 1
            if idx < 5:
                r5 += 1
            if idx < 10:
                r10 += 1
            mrr += 1.0 / (idx + 1)

    lat = np.array(lat)
    return {
        "N": n,
        "R@1": r1 / n,
        "R@5": r5 / n,
        "R@10": r10 / n,
        "MRR@10": mrr / n,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "Contain@20": contain_20 / n,
        "Contain@50": contain_50 / n,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate with alignment head")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True, help="Alignment head checkpoint")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--K_global", type=int, default=50)
    ap.add_argument("--K_local", type=int, default=20)
    ap.add_argument("--K_union", type=int, default=60)
    ap.add_argument("--do_mmr", action="store_true", default=True)
    ap.add_argument("--mmr_lambda", type=float, default=0.7)
    ap.add_argument("--top_final", type=int, default=10)
    ap.add_argument("--use_seq_bias", action="store_true", default=True)
    ap.add_argument("--w_same_article", type=float, default=0.05)
    ap.add_argument("--w_next_gap", type=float, default=0.12)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--directional_bonus", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=Path("artifacts/lvm/eval_with_alignment.json"))
    args = ap.parse_args()

    # Load alignment head
    print("Loading alignment head...")
    checkpoint = torch.load(args.alignment, map_location="cpu")
    config = checkpoint["config"]
    alignment_head = AlignmentHead(
        dim=config["dim"],
        hidden_dim=config["hidden_dim"],
        alpha=config["alpha"]
    )
    alignment_head.load_state_dict(checkpoint["model_state_dict"])
    alignment_head.eval()
    print(f"  Val cosine: {checkpoint['val_cosine']:.4f}\n")

    # Load resources
    print(f"Loading FAISS index (nprobe={args.nprobe})...")
    faiss_index = faiss.read_index(str(args.faiss))
    faiss_index.nprobe = args.nprobe

    print("Loading global payload...")
    global_payload = np.load(args.payload, allow_pickle=True).item()

    print("Loading article shards...")
    with open(args.shards, "rb") as f:
        article_shards = pickle.load(f)

    print("Loading dataset...")
    data = np.load(args.npz, allow_pickle=True)
    N = len(data["truth_keys"]) if args.limit is None else min(args.limit, len(data["truth_keys"]))
    dataset = [
        {
            "pred_vec": data["pred_vecs"][i],
            "last_meta": data["last_meta"].tolist()[i],
            "truth_key": tuple(data["truth_keys"].tolist()[i]),
        }
        for i in range(N)
    ]
    print(f"Loaded {N:,} samples\n")

    # Create retriever
    retriever = ShardAssistedRetrievalWithAlignment(
        faiss_index, global_payload, article_shards, alignment_head
    )

    # Run evaluation
    print("Running evaluation with alignment head...")
    results = evaluate(retriever, dataset, args)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print("\n" + "=" * 80)
    print("RESULTS WITH ALIGNMENT HEAD")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
