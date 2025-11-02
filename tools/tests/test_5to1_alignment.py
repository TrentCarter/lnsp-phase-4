#!/usr/bin/env python3
"""
5→1 Causal Alignment Test (5CAT)

Verifies that a 5-context → 1-next-vector LVM truly learned the +1 sequential mapping
and relies on order/contiguity — not just distributional matching.

Implements the test battery:
  A) Offset Alignment Sweep (+1 or bust)
  B) Retrieval Rank with Predicted Vector (within-article where possible)
  C) Context-Order Ablations (shuffle/reverse/leave-one-out/repeat-pad/cross-article)
  D) Multi-Step Horizon Consistency (rollout H steps)
  E) Coherence-Bin Stratification (Low vs Normal), using per-sample context coherence

CLI example:
  python tools/tests/test_5to1_alignment.py \
    --model artifacts/lvm/models/amn_clean_splits_*/best_model.pt \
    --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz \
    --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
    --articles-npz artifacts/wikipedia_584k_fresh.npz \
    --device mps --max-samples 10000

Exit code 0 on success; non-zero on gate failure.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

# Optional FAISS for global retrieval fallback
try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# -------------------------
# Utilities
# -------------------------

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


@dataclass
class SeqSample:
    ctx: np.ndarray  # (5, 768)
    tgt: np.ndarray  # (768,)
    article_id: Optional[int] = None
    target_idx: Optional[int] = None  # sequential index inside article (if provided)


@dataclass
class Dataset:
    name: str
    samples: List[SeqSample]


class ArticleStore:
    """Holds article vectors and allows (article_id, idx) → vector lookup.
       If chunk indices are missing, idx is treated as position order within that article group.
    """

    def __init__(self, vectors: np.ndarray, article_ids: np.ndarray, chunk_idx: Optional[np.ndarray] = None):
        assert vectors.ndim == 2, "vectors must be (N, D)"
        assert vectors.shape[0] == article_ids.shape[0]
        self.D = vectors.shape[1]
        self.by_article: Dict[int, np.ndarray] = {}
        self.index_maps: Dict[int, Dict[int, int]] = {}
        # group
        tmp: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for i, aid in enumerate(article_ids.tolist()):
            idx = int(chunk_idx[i]) if chunk_idx is not None else i  # temporary
            tmp.setdefault(aid, []).append((idx, vectors[i]))
        # sort by idx within each article and build arrays
        for aid, pairs in tmp.items():
            pairs.sort(key=lambda t: t[0])
            arr = np.stack([p[1] for p in pairs], axis=0)
            self.by_article[aid] = l2_normalize(arr)
            # map idx→row; if chunk_idx was None, we map sequential 0..n-1
            if chunk_idx is not None:
                self.index_maps[aid] = {int(p[0]): j for j, p in enumerate(pairs)}
            else:
                self.index_maps[aid] = {j: j for j in range(len(pairs))}

    def get(self, aid: int, idx: int) -> Optional[np.ndarray]:
        if aid not in self.by_article:
            return None
        imap = self.index_maps[aid]
        if idx not in imap:
            return None
        return self.by_article[aid][imap[idx]]

    def get_offset(self, aid: int, idx: int, k: int) -> Optional[np.ndarray]:
        if aid not in self.by_article:
            return None
        arr = self.by_article[aid]
        imap = self.index_maps[aid]
        if idx not in imap:
            return None
        j = imap[idx] + k
        if j < 0 or j >= arr.shape[0]:
            return None
        return arr[j]

    def article_vectors(self, aid: int) -> Optional[np.ndarray]:
        return self.by_article.get(aid, None)


# -------------------------
# Loading
# -------------------------

def load_sequences_npz(path: str, name: str) -> Dataset:
    """Load NPZ with contexts/targets and optional metadata.
    Expected keys (we try multiple fallbacks):
      contexts | context | X  → (N, 5, D)
      targets  | target  | y  → (N, D)
      article_ids        → (N,)
      target_indices     → (N,)  (index of the target inside its article)
    """
    arr = np.load(path, allow_pickle=True)
    keys = set(arr.keys())

    def pick(*candidates):
        for k in candidates:
            if k in arr:
                return arr[k]
        raise KeyError(f"None of keys {candidates} found in {path}. Available: {sorted(keys)}")

    contexts = pick("contexts", "context", "X")
    targets = pick("targets", "target", "y")
    article_ids = arr.get("article_ids", None)
    target_indices = arr.get("target_indices", None)

    contexts = np.asarray(contexts)
    targets = np.asarray(targets)
    assert contexts.ndim == 3 and contexts.shape[1] == 5, "contexts must be (N,5,D)"
    assert targets.ndim == 2 and targets.shape[0] == contexts.shape[0], "targets must be (N,D)"

    # L2 normalize defensively for eval
    contexts = l2_normalize(contexts, axis=-1)
    targets = l2_normalize(targets, axis=-1)

    n = contexts.shape[0]
    samples: List[SeqSample] = []
    for i in range(n):
        aid = int(article_ids[i]) if article_ids is not None else None
        tidx = int(target_indices[i]) if target_indices is not None else None
        samples.append(SeqSample(ctx=contexts[i], tgt=targets[i], article_id=aid, target_idx=tidx))

    return Dataset(name=name, samples=samples)


def load_articles_npz(path: str) -> ArticleStore:
    arr = np.load(path, allow_pickle=True)
    keys = set(arr.keys())

    def pick(*candidates):
        for k in candidates:
            if k in arr:
                return arr[k]
        raise KeyError(f"None of keys {candidates} found in {path}. Available: {sorted(keys)}")

    vectors = pick("vectors", "embeddings", "X")
    art_ids = pick("article_indices", "article_ids")
    chunk_idx = arr.get("chunk_indices", arr.get("seq_idx", None))

    vectors = l2_normalize(np.asarray(vectors))
    art_ids = np.asarray(art_ids)
    if chunk_idx is not None:
        chunk_idx = np.asarray(chunk_idx)

    return ArticleStore(vectors=vectors, article_ids=art_ids, chunk_idx=chunk_idx)


# -------------------------
# Model wrapper
# -------------------------

class ModelWrapper:
    """Thin wrapper for AMN/GRU/LSTM style models that take (B, 5, D) → (B, D)."""

    def __init__(self, model_path: str, device: str = "cpu", disable_pos: bool = False):
        if torch is None:
            raise RuntimeError("PyTorch is required to run this test.")
        self.device = torch.device(device)
        self.force_disable_pos = disable_pos

        # Load checkpoint (weights_only=False for compatibility with numpy scalars)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Check if it's a checkpoint dict or direct model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load model architecture
            model_type = checkpoint.get('model_type', 'amn').lower()

            # Import model classes from models.py (used by training)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from app.lvm.models import (
                AttentionMixtureNetwork,
                GRUStack,
                LSTMBaseline,
                TransformerVectorPredictor
            )

            # Detect input dimension from checkpoint (for positional encoding compatibility)
            input_dim = 768  # default
            if 'model_state_dict' in checkpoint:
                # Check input_proj.weight shape to detect if trained with positional encoding
                for key in checkpoint['model_state_dict'].keys():
                    if 'input_proj.weight' in key:
                        weight_shape = checkpoint['model_state_dict'][key].shape
                        input_dim = weight_shape[1]  # [out_dim, in_dim]
                        print(f"[INFO] Detected input_dim={input_dim} from checkpoint")
                        break

            # Instantiate correct architecture with detected input_dim
            # Output dimension is ALWAYS 768 (semantic vector dimension)
            if model_type == 'amn':
                self.model = AttentionMixtureNetwork(input_dim=input_dim, output_dim=768)
            elif model_type == 'gru':
                self.model = GRUStack(input_dim=input_dim, output_dim=768)
            elif model_type == 'lstm':
                self.model = LSTMBaseline(input_dim=input_dim, output_dim=768)
            elif model_type == 'transformer':
                self.model = TransformerVectorPredictor(input_dim=input_dim, output_dim=768)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Store whether positional encoding is needed
            self.use_positional = (input_dim == 769)
            self.pos_scale = 0.03  # Match training default

            # Override if --disable-pos flag is set
            if self.force_disable_pos:
                if self.use_positional:
                    print(f"[INFO] Positional encoding DISABLED by --disable-pos flag (model was trained with pos)")
                self.use_positional = False
            elif self.use_positional:
                print(f"[INFO] Positional encoding will be applied during inference (pos_scale={self.pos_scale})")

            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct model object
            self.model = checkpoint

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, ctx5: np.ndarray) -> np.ndarray:
        # ctx5: (5, D) or (B,5,D)
        x = ctx5
        if x.ndim == 2:
            x = x[None, ...]
        # Copy to ensure positive strides (PyTorch doesn't support negative strides)
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(x).to(self.device, dtype=torch.float32)

        # Apply positional encoding if model was trained with it
        if self.use_positional:
            B, T, D = t.shape
            pos = torch.linspace(0, 1, steps=T, device=self.device).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(B, T, 1) * self.pos_scale
            t = torch.cat([t, pos], dim=-1)  # 768 → 769

        out = self.model(t)
        out = out.detach().cpu().numpy()
        if out.ndim == 2:
            out = out[0]
        return l2_normalize(out)


# -------------------------
# Tests
# -------------------------

def sample_coherence(ctx5: np.ndarray) -> float:
    # mean cos of adjacent pairs in the context window (4 edges)
    return float(np.mean([cosine(ctx5[i], ctx5[i+1]) for i in range(4)]))


def offset_alignment_sweep(ds: Dataset, model: ModelWrapper, store: ArticleStore, max_samples: int) -> Dict[str, float]:
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    bucket: Dict[int, List[float]] = {k: [] for k in offsets}
    n = 0
    for s in ds.samples:
        if n >= max_samples:
            break
        if s.article_id is None or s.target_idx is None:
            continue  # need metadata
        vhat = model.predict(s.ctx)
        ok = True
        for k in offsets:
            tvec = store.get_offset(s.article_id, s.target_idx, k)
            if tvec is None:
                ok = False
                break
        if not ok:
            continue
        for k in offsets:
            tvec = store.get_offset(s.article_id, s.target_idx, k)
            bucket[k].append(cosine(vhat, tvec))
        n += 1
    means = {f"k={k}": (float(np.mean(v)) if v else float("nan")) for k, v in bucket.items()}
    # compute margin
    cpos = np.mean(bucket[1]) if bucket[1] else float("nan")
    other = [np.mean(bucket[k]) for k in offsets if k != 1 and bucket[k]]
    margin = float(cpos - max(other)) if other and not math.isnan(cpos) else float("nan")
    means["margin(+1)"] = margin
    means["samples"] = float(n)
    return means


def retrieval_rank_within_article(ds: Dataset, model: ModelWrapper, store: ArticleStore, max_samples: int) -> Dict[str, float]:
    r1 = []
    r5 = []
    rr = []  # reciprocal rank
    n = 0
    for s in ds.samples:
        if n >= max_samples:
            break
        if s.article_id is None or s.target_idx is None:
            continue
        article_vecs = store.article_vectors(s.article_id)
        if article_vecs is None or article_vecs.shape[0] < 6:
            continue
        vhat = model.predict(s.ctx)
        # cosine similarities to all vecs in this article
        sims = (article_vecs @ l2_normalize(vhat))  # since article_vecs are L2
        # true index inside article array
        true_vec = store.get(s.article_id, s.target_idx)
        if true_vec is None:
            continue
        true_sim = cosine(vhat, true_vec)
        # rank: number of sims strictly greater than true + 1
        rank = int(np.sum(sims > true_sim) + 1)
        r1.append(1.0 if rank == 1 else 0.0)
        r5.append(1.0 if rank <= 5 else 0.0)
        rr.append(1.0 / rank)
        n += 1
    return {
        "R@1": float(np.mean(r1)) if r1 else float("nan"),
        "R@5": float(np.mean(r5)) if r5 else float("nan"),
        "MRR": float(np.mean(rr)) if rr else float("nan"),
        "samples": float(n),
    }


def ablations(ds: Dataset, model: ModelWrapper, max_samples: int) -> Dict[str, float]:
    deltas = {"shuffled": [], "reverse": [], "loo_avg": [], "repeat_pad": [], "cross_article": []}
    n = 0
    rng = random.Random(17)
    for s in ds.samples:
        if n >= max_samples:
            break
        base_pred = model.predict(s.ctx)
        base_cos = cosine(base_pred, s.tgt)
        # shuffled
        perm = list(range(5))
        rng.shuffle(perm)
        ctx_shuf = s.ctx[perm]
        pred_shuf = model.predict(ctx_shuf)
        deltas["shuffled"].append(cosine(pred_shuf, s.tgt) - base_cos)
        # reverse
        pred_rev = model.predict(s.ctx[::-1])
        deltas["reverse"].append(cosine(pred_rev, s.tgt) - base_cos)
        # leave-one-out (average delta)
        loo = []
        for i in range(5):
            ctx_loo = np.concatenate([s.ctx[:i], s.ctx[i+1:]], axis=0)
            # pad by repeating last to keep size 5 for the model
            ctx_loo = np.concatenate([ctx_loo, s.ctx[-1:]], axis=0)
            pred_loo = model.predict(ctx_loo)
            loo.append(cosine(pred_loo, s.tgt) - base_cos)
        deltas["loo_avg"].append(float(np.mean(loo)))
        # repeat-pad (all same as last frame)
        ctx_rep = np.repeat(s.ctx[-1:,:], 5, axis=0)
        pred_rep = model.predict(ctx_rep)
        deltas["repeat_pad"].append(cosine(pred_rep, s.tgt) - base_cos)
        # cross-article (replace last frame with itself mean to break contiguity)
        # If we don't have another article handy, approximate by using the first frame
        ctx_cross = s.ctx.copy()
        ctx_cross[-1] = s.ctx[0]
        pred_cross = model.predict(ctx_cross)
        deltas["cross_article"].append(cosine(pred_cross, s.tgt) - base_cos)
        n += 1
    return {k: float(np.mean(v)) if v else float("nan") for k, v in deltas.items()} | {"samples": float(n)}


def rollout_consistency(ds: Dataset, model: ModelWrapper, store: ArticleStore, H: int, max_samples: int) -> Dict[str, float]:
    cos_h: List[float] = []
    n = 0
    for s in ds.samples:
        if n >= max_samples:
            break
        if s.article_id is None or s.target_idx is None:
            continue
        ctx = s.ctx.copy()
        ok = True
        future_vecs = []
        for h in range(1, H+1):
            tvec = store.get_offset(s.article_id, s.target_idx, h)
            if tvec is None:
                ok = False
                break
            future_vecs.append(tvec)
        if not ok:
            continue
        preds = []
        for h in range(1, H+1):
            vhat = model.predict(ctx)
            preds.append(vhat)
            ctx = np.concatenate([ctx[1:], vhat[None, :]], axis=0)  # slide
        ch = [cosine(preds[h-1], future_vecs[h-1]) for h in range(1, H+1)]
        cos_h.append(float(np.mean(ch)))
        n += 1
    return {
        f"avg_cos@H={H}": float(np.mean(cos_h)) if cos_h else float("nan"),
        "samples": float(n),
    }


def stratified_bins(ds: Dataset, model: ModelWrapper, max_samples: int) -> Dict[str, float]:
    low, normal = [], []
    n = 0
    for s in ds.samples:
        if n >= max_samples:
            break
        coh = sample_coherence(s.ctx)
        vhat = model.predict(s.ctx)
        c = cosine(vhat, s.tgt)
        if coh < 0.45:
            low.append(c)
        elif coh <= 0.55:
            normal.append(c)
        n += 1
    return {
        "low_mean": float(np.mean(low)) if low else float("nan"),
        "normal_mean": float(np.mean(normal)) if normal else float("nan"),
        "low_n": float(len(low)),
        "normal_n": float(len(normal)),
    }


# -------------------------
# Gating / Runner
# -------------------------

def run_suite(args) -> int:
    # Load
    print("[INFO] Loading datasets…")
    val = load_sequences_npz(args.val_npz, name="VAL")
    ood = load_sequences_npz(args.ood_npz, name="OOD")
    store = load_articles_npz(args.articles_npz)

    print("[INFO] Loading model…")
    model = ModelWrapper(args.model, device=args.device, disable_pos=args.disable_pos)

    maxN = args.max_samples

    results = {"VAL": {}, "OOD": {}}

    for ds, label in [(val, "VAL"), (ood, "OOD")]:
        print(f"[RUN] Offset alignment sweep — {label}")
        resA = offset_alignment_sweep(ds, model, store, maxN)
        for k, v in resA.items():
            results[label][f"A:{k}"] = v

        print(f"[RUN] Retrieval rank (within-article) — {label}")
        resB = retrieval_rank_within_article(ds, model, store, maxN)
        for k, v in resB.items():
            results[label][f"B:{k}"] = v

        print(f"[RUN] Ablations — {label}")
        resC = ablations(ds, model, maxN)
        for k, v in resC.items():
            results[label][f"C:{k}"] = v

        print(f"[RUN] Rollout H={args.horizon} — {label}")
        resD = rollout_consistency(ds, model, store, args.horizon, maxN)
        for k, v in resD.items():
            results[label][f"D:{k}"] = v

        print(f"[RUN] Stratified bins — {label}")
        resE = stratified_bins(ds, model, maxN)
        for k, v in resE.items():
            results[label][f"E:{k}"] = v

    # Print summary
    print("\n===== 5→1 Causal Alignment Test — Summary =====")
    print(json.dumps(results, indent=2))

    # Gates
    def gate_offset(label: str) -> bool:
        cpos = results[label].get("A:k=1", float("nan"))
        margin = results[label].get("A:margin(+1)", float("nan"))
        ok1 = (not math.isnan(cpos)) and (cpos >= (0.50 if label == "VAL" else 0.48))
        ok2 = (not math.isnan(margin)) and (margin >= (0.12 if label == "VAL" else 0.10))
        return ok1 and ok2

    def gate_retr(label: str) -> bool:
        R1 = results[label].get("B:R@1", 0.0)
        R5 = results[label].get("B:R@5", 0.0)
        MRR = results[label].get("B:MRR", 0.0)
        tR1 = 0.60 if label == "VAL" else 0.55
        tR5 = 0.95 if label == "VAL" else 0.92
        tMRR = 0.80 if label == "VAL" else 0.75
        return (R1 >= tR1) and (R5 >= tR5) and (MRR >= tMRR)

    def gate_ablate(label: str) -> bool:
        d = results[label]
        ok = True
        ok &= d.get("C:shuffled", 0.0) <= -0.15 + 1e-6
        ok &= d.get("C:reverse", 0.0) <= -0.10 + 1e-6
        ok &= d.get("C:loo_avg", 0.0) <= -0.08 + 1e-6
        ok &= d.get("C:repeat_pad", 0.0) <= 0.05  # should be near/under 0
        ok &= d.get("C:cross_article", 0.0) <= 0.05
        return ok

    def gate_rollout(label: str) -> bool:
        ac = results[label].get(f"D:avg_cos@H={args.horizon}", 0.0)
        thr = 0.45 if label == "VAL" else 0.42
        return ac >= thr

    def gate_bins_delta() -> bool:
        v = results["VAL"].get("E:normal_mean", float("nan"))
        o = results["OOD"].get("E:normal_mean", float("nan"))
        if math.isnan(v) or math.isnan(o):
            return False
        return abs(v - o) <= 0.05

    g1 = gate_offset("VAL") and gate_offset("OOD")
    g2 = gate_retr("VAL") and gate_retr("OOD")
    g3 = gate_ablate("VAL") and gate_ablate("OOD")
    g4 = gate_rollout("VAL") and gate_rollout("OOD")
    g5 = gate_bins_delta()

    all_ok = g1 and g2 and g3 and g4 and g5
    print("\n[RESULT] Gates:")
    print(json.dumps({
        "offset_sweep": g1,
        "retrieval_rank": g2,
        "ablations": g3,
        "rollout": g4,
        "bins_delta": g5,
    }, indent=2))

    if not all_ok:
        print("\n[FAIL] One or more gates failed. See summary above.")
        return 2
    print("\n[PASS] 5→1 causal alignment confirmed.")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="5→1 Causal Alignment Test (5CAT)")
    p.add_argument("--model", required=True, help="Path to torch model .pt")
    p.add_argument("--val-npz", required=True, help="Validation sequences NPZ (ctx=5)")
    p.add_argument("--ood-npz", required=True, help="OOD sequences NPZ (ctx=5)")
    p.add_argument("--articles-npz", required=True, help="Full article vectors NPZ for offset/retrieval")
    p.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    p.add_argument("--max-samples", type=int, default=10000, help="Max samples per split to evaluate")
    p.add_argument("--horizon", type=int, default=5, help="Rollout steps for test D")
    p.add_argument("--disable-pos", action="store_true", help="Force disable positional encoding at inference (for debugging)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_suite(parse_args()))
