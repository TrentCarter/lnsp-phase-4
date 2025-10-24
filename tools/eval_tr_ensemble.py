#!/usr/bin/env python3
"""
Evaluate Tiny Recursion (TR) ensembles vs direct LVM predictions for retrieval.

Scenarios:
  A) baseline_direct:      use direct LVM vectors
  B) single_tr:            run TR once
  C) ensemble_union_tr2:   run TR twice (diverse temps/seeds), union candidates, rerank
  D) ensemble_avg_tr2:     run TR twice, average vectors, single retrieval

Requirements
-----------
- FAISS index (cosine/IP ready)
- Payload mapping: id -> (text, meta, vec768)
- NPZ with at least:
    pred_vecs_direct:  [N,768]  (for A)
    last_meta:         [N] object array of {"article_index":int, "chunk_index":int}
    truth_keys:        [N,2] int (article_index, chunk_index)
  Optional fields:
    contexts:          [N] (only needed if you want to generate TR via callback)
    last_vecs:         [N,768] (enables directional bonus)
    tr_pred_vecs:      [N,768] (if present, used for single_tr without callback)

- If you want to generate TR vectors on the fly, pass --tr-module and --tr-fn.
  The function signature must be:
      def tiny_recursion_predict(contexts: List[Any], temp: float, seed: int, attempts: int) -> np.ndarray
  It must return an array of shape [len(contexts), 768] (L2-normalized preferred).

Outputs
-------
JSON with metrics:
  R@1,R@5,R@10,R@20, MRR@10, p50_ms, p95_ms, N, scenario

Example
-------
python tools/eval_tr_ensemble.py \
  --scenario ensemble_union_tr2 \
  --npz artifacts/lvm/wikipedia_ood_test_ctx5_v2.npz \
  --payload artifacts/wikipedia_500k_payload.npy \
  --faiss artifacts/faiss/wiki_500k.index \
  --K 30 --top-final 10 --mmr-lambda 0.7 \
  --w-same-article 0.05 --w-next-gap 0.12 --tau 3.0 \
  --directional-bonus 0.03 \
  --tr-module app.tiny_recursion --tr-fn predict \
  --tr-temp1 0.05 --tr-temp2 0.08 --tr-seed1 1337 --tr-seed2 4242 --tr-attempts 2 \
  --out artifacts/lvm/eval_tr_union.json
"""
from __future__ import annotations
import argparse, json, time, importlib, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Local utilities
from tools.rerank_strategies_v2 import (
    dedup_candidates,
    mmr,
    rerank_with_sequence_bias,
    _l2norm,
    cosine_softmax_weights,
)

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("FAISS not available: install faiss-cpu/faiss-gpu") from e

Candidate = Tuple[str, float, Dict[str, Any], np.ndarray]  # (text, cosine, meta, vec)


# ----------------------- Retrieval / Dataset shims -----------------------
class RetrievalShim:
    def __init__(self, faiss_index, id_to_payload: Dict[int, Tuple[str, Dict[str,Any], np.ndarray]]):
        self.index = faiss_index
        self.payload = id_to_payload

    def search(self, query_vec: np.ndarray, K: int) -> List[Candidate]:
        q = _l2norm(query_vec.reshape(1, -1)).astype(np.float32)
        D, I = self.index.search(q, K)
        out: List[Candidate] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            text, meta, vec = self.payload[int(idx)]
            vec_arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            vec_norm = _l2norm(vec_arr)[0]
            out.append((text, float(score), meta, vec_norm))
        return out


def load_payload(payload_path: Path) -> Dict[int, Tuple[str, Dict[str,Any], np.ndarray]]:
    obj = np.load(payload_path, allow_pickle=True).item()
    # Expect mapping: id -> (text:str, meta:dict, vec:np.ndarray[768])
    return obj


def load_npz(npz_path: Path) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    # Sanity
    assert "last_meta" in out and "truth_keys" in out, "NPZ must have last_meta and truth_keys"
    assert out["truth_keys"].ndim == 2 and out["truth_keys"].shape[1] == 2, "truth_keys must be [N,2]"
    return out


# ---------------------------- TR adapters -------------------------------

def run_tr_vectors(npz_data: Dict[str, Any], module_path: str|None, fn_name: str|None,
                   temp: float, seed: int, attempts: int) -> np.ndarray:
    if "tr_pred_vecs" in npz_data:
        return _l2norm(np.asarray(npz_data["tr_pred_vecs"], dtype=np.float32))
    if module_path and fn_name:
        if "contexts" not in npz_data:
            raise ValueError("NPZ missing 'contexts' needed for TR callback generation")
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
        vecs = fn(list(npz_data["contexts"]), temp=temp, seed=seed, attempts=attempts)
        vecs = np.asarray(vecs, dtype=np.float32)
        return _l2norm(vecs)
def cosine_softmax_weights(cosines: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Compute softmax weights from cosine similarities."""
    exp_cos = np.exp(cosines / temperature)
    return exp_cos / np.sum(exp_cos)

def _l2norm(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    return vec / (np.linalg.norm(vec) + 1e-8)


def evaluate(
    scenario: str,
    retriever: RetrievalShim,
    last_meta: List[Dict[str,Any]],
    truth_keys: np.ndarray,
    vecs_direct: np.ndarray|None,
    vecs_tr1: np.ndarray|None,
    vecs_tr2: np.ndarray|None,
    last_vecs: np.ndarray|None,
    K: int,
    top_final: int,
    mmr_lambda: float,
    use_seq_bias: bool,
    w_same_article: float,
    w_next_gap: float,
    tau: float,
    directional_bonus: float,
    fusion_temperature: float,
) -> Dict[str, Any]:
    """Original evaluate function for basic scenarios."""
    N = len(last_meta)
    r1 = r5 = r10 = r20 = 0
    mrr = 0.0
    lat: List[float] = []

    if vecs_direct is not None:
        vecs_direct = _l2norm(np.asarray(vecs_direct, dtype=np.float32))
    if vecs_tr1 is not None:
        vecs_tr1 = _l2norm(np.asarray(vecs_tr1, dtype=np.float32))
    if vecs_tr2 is not None:
        vecs_tr2 = _l2norm(np.asarray(vecs_tr2, dtype=np.float32))
    if last_vecs is not None:
        last_vecs = np.asarray(last_vecs, dtype=np.float32)

    def retrieve_for_vector(v: np.ndarray, lm: Dict[str, Any], last_vec: np.ndarray | None) -> List[Candidate]:
        v_norm = _l2norm(v.reshape(1, -1))[0].astype(np.float32)
        cands = retriever.search(v_norm, K)
        cands = dedup_candidates(cands)
        if len(cands) > top_final:
            vec_stack = np.stack([c[3] for c in cands], axis=0).astype(np.float32)
            sel = mmr(v_norm, vec_stack, lambda_=mmr_lambda, k=top_final)
            cands = [cands[i] for i in sel]
        else:
            cands = cands[:top_final]
        if use_seq_bias and cands:
            ranked = rerank_with_sequence_bias(
                candidates=cands,
                last_ctx_meta=lm,
                w_cos=1.0,
                w_same_article=w_same_article,
                w_next_gap=w_next_gap,
                tau=tau,
                directional_bonus=directional_bonus if (directional_bonus and last_vec is not None) else 0.0,
                pred_vec=v_norm,
                last_vec=last_vec,
            )
            cands = [c for _, c in ranked]
        return cands

    def rerank_union_candidates(
        candidates: List[Candidate],
        base_vec: np.ndarray,
        lm: Dict[str, Any],
        last_vec: np.ndarray | None,
    ) -> List[Candidate]:
        if not candidates:
            return []
        vec_stack = np.stack([c[3] for c in candidates], axis=0).astype(np.float32)
        if len(candidates) > top_final:
            sel = mmr(base_vec, vec_stack, lambda_=mmr_lambda, k=top_final)
            candidates = [candidates[idx] for idx in sel]
        else:
            candidates = candidates[:top_final]
        if use_seq_bias and candidates:
            ranked = rerank_with_sequence_bias(
                candidates=candidates,
                last_ctx_meta=lm,
                w_cos=1.0,
                w_same_article=w_same_article,
                w_next_gap=w_next_gap,
                tau=tau,
                directional_bonus=directional_bonus if (directional_bonus and last_vec is not None) else 0.0,
                pred_vec=base_vec,
                last_vec=last_vec,
            )
            candidates = [c for _, c in ranked]
        return candidates

    for i in range(N):
        lm = last_meta[i]
        tk = (int(truth_keys[i, 0]), int(truth_keys[i, 1]))
        last_vec = last_vecs[i] if last_vecs is not None else None

        t0 = time.perf_counter()

        if scenario == "baseline_direct":
            assert vecs_direct is not None, "pred_vecs_direct required for baseline_direct"
            v = vecs_direct[i]
            cands = retrieve_for_vector(v, lm, last_vec)

        elif scenario == "single_tr":
            assert vecs_tr1 is not None, "TR vectors required for single_tr"
            v = vecs_tr1[i]
            cands = retrieve_for_vector(v, lm, last_vec)

        elif scenario == "ensemble_avg_tr2":
            assert vecs_tr1 is not None and vecs_tr2 is not None, "Need two TR vectors"
            v = _l2norm((vecs_tr1[i] + vecs_tr2[i]).reshape(1, -1))[0].astype(np.float32)
            cands = retrieve_for_vector(v, lm, last_vec)

        elif scenario == "ensemble_union_tr2":
            assert vecs_tr1 is not None and vecs_tr2 is not None, "Need two TR vectors"
            v1 = vecs_tr1[i]
            v2 = vecs_tr2[i]

            c1 = retriever.search(v1, K)
            c2 = retriever.search(v2, K)
            merged = dedup_candidates(c1 + c2)

            fused_candidates: List[Candidate] = []
            if merged:
                cand_vecs = np.stack([c[3] for c in merged], axis=0).astype(np.float32)
                v1_norm = _l2norm(v1.reshape(1, -1))[0]
                v2_norm = _l2norm(v2.reshape(1, -1))[0]
                cos1 = cand_vecs @ v1_norm
                cos2 = cand_vecs @ v2_norm
                for idx, cand in enumerate(merged):
                    text, _, meta, vec = cand
                    local_cos = np.array([cos1[idx], cos2[idx]], dtype=np.float32)
                    weights = cosine_softmax_weights(local_cos, temperature=fusion_temperature)
                    fused_cos = float(weights[0] * cos1[idx] + weights[1] * cos2[idx])
                    fused_candidates.append((text, fused_cos, meta, vec))

            v_fused = _l2norm((v1 + v2).reshape(1, -1))[0].astype(np.float32)
            cands = rerank_union_candidates(fused_candidates, v_fused, lm, last_vec)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands]
        if tk in keys:
            idx = keys.index(tk)
            if idx == 0:
                r1 += 1
            if idx < 5:
                r5 += 1
            if idx < 10:
                r10 += 1
            if idx < 20:
                r20 += 1
            mrr += 1.0 / (idx + 1)

    lat = np.array(lat, dtype=np.float32)
    return {
        "scenario": scenario,
        "N": N,
        "R@1": r1 / N,
        "R@5": r5 / N,
        "R@10": r10 / N,
        "R@20": r20 / N,
        "MRR@10": mrr / N,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
    }
    """Evaluate with Best-of-N selection and adaptive K."""

    N = len(last_meta)
    r1=r5=r10=r20=0
    mrr=0.0
    lat=[]
    decisions=[]  # Track which method was chosen

    def retrieve_for_vector(v: np.ndarray, lm: Dict[str,Any], k: int) -> List[Candidate]:
        cands = retriever.search(v, k)
        cands = dedup_candidates(cands)
        if len(cands) > top_final:
            vec_stack = np.stack([c[3] for c in cands], axis=0).astype(np.float32)
            sel = mmr(v, vec_stack, lambda_=mmr_lambda, k=top_final)
            cands = [cands[i] for i in sel]
        if use_seq_bias and cands:
            ranked = rerank_with_sequence_bias(
                candidates=cands,
                last_ctx_meta=lm,
                w_cos=1.0,
                w_same_article=w_same_article,
                w_next_gap=w_next_gap,
                tau=tau,
                directional_bonus=directional_bonus if last_vecs is not None else 0.0,
                pred_vec=v,
                last_vec=last_vecs[i] if last_vecs is not None else None,
            )
            cands = [c for _, c in ranked]
        return cands

    def continuity_score(cands: List[Candidate], direct_cos: float, tr_cos: float,
                        seq_bonus: float, directional: float) -> float:
        """Compute continuity score for Best-of-N selection."""
        if not cands:
            return 0.0
        cosine_at_1 = cands[0][1] if cands else 0.0
        return 0.80 * cosine_at_1 + 0.15 * seq_bonus + 0.05 * directional

    for i in range(N):
        lm = last_meta[i]
        tk = (int(truth_keys[i,0]), int(truth_keys[i,1]))

        t0 = time.perf_counter()

        if scenario == "best_of_direct_tr":
            # Best-of-N between Direct and Single TR
            assert vecs_direct is not None and vecs_tr1 is not None

            # Get predictions
            direct_vec = vecs_direct[i]
            tr_vec = vecs_tr1[i]

            # Adaptive K based on confidence (placeholder - use simple heuristic)
            tr_conf = float(np.dot(_l2norm(tr_vec), _l2norm(vecs_direct[i])))  # Cosine as confidence proxy
            if adaptive_k:
                adaptive_k_val = int(10 + np.floor((0.72 - tr_conf) * 40))
                adaptive_k_val = max(10, min(50, adaptive_k_val))
            else:
                adaptive_k_val = K

            # Retrieve with both
            direct_cands = retrieve_for_vector(direct_vec, lm, adaptive_k_val)
            tr_cands = retrieve_for_vector(tr_vec, lm, adaptive_k_val)

            # Compute continuity scores
            direct_score = continuity_score(direct_cands, 1.0, 1.0, 1.0, 1.0)
            tr_score = continuity_score(tr_cands, 1.0, 1.0, 1.0, 1.0)

            # Choose best
            if direct_score >= tr_score:
                cands = direct_cands
                decisions.append("direct")
            else:
                cands = tr_cands
                decisions.append("tr")

        elif scenario == "confidence_gated_multi_tr":
            # Confidence-gated approach
            assert vecs_direct is not None and vecs_tr1 is not None

            direct_vec = vecs_direct[i]
            tr_vec = vecs_tr1[i]

            # Compute TR confidence (cosine between TR and Direct as proxy)
            tr_conf = float(np.dot(_l2norm(tr_vec), _l2norm(direct_vec)))

            # Adaptive K
            if adaptive_k:
                adaptive_k_val = int(10 + np.floor((0.72 - tr_conf) * 40))
                adaptive_k_val = max(10, min(50, adaptive_k_val))
            else:
                adaptive_k_val = K

            if tr_conf >= 0.75:
                # High confidence - use Direct
                cands = retrieve_for_vector(direct_vec, lm, adaptive_k_val)
                decisions.append("direct_high_conf")
            elif 0.60 <= tr_conf < 0.75:
                # Medium confidence - use Single TR
                cands = retrieve_for_vector(tr_vec, lm, adaptive_k_val)
                decisions.append("tr_medium_conf")
            else:
                # Low confidence - use Double TR union
                cands1 = retrieve_for_vector(tr_vec, lm, adaptive_k_val)
                # For simplicity, use TR candidates for low confidence
                cands = cands1
                decisions.append("double_tr_low_conf")

        elif scenario == "ensemble_best_of_2":
            # Best-of-2 TR runs
            assert vecs_tr1 is not None and vecs_tr2 is not None

            tr1_vec = vecs_tr1[i]
            tr2_vec = vecs_tr2[i]

            # Adaptive K based on average confidence
            conf1 = float(np.dot(_l2norm(tr1_vec), _l2norm(vecs_direct[i])))
            conf2 = float(np.dot(_l2norm(tr2_vec), _l2norm(vecs_direct[i])))
            avg_conf = (conf1 + conf2) / 2

            if adaptive_k:
                adaptive_k_val = int(10 + np.floor((0.72 - avg_conf) * 40))
                adaptive_k_val = max(10, min(50, adaptive_k_val))
            else:
                adaptive_k_val = K

            # Get candidates from both
            cands1 = retrieve_for_vector(tr1_vec, lm, adaptive_k_val)
            cands2 = retrieve_for_vector(tr2_vec, lm, adaptive_k_val)

            # Choose best based on continuity score
            score1 = continuity_score(cands1, conf1, conf1, 1.0, 1.0)
            score2 = continuity_score(cands2, conf2, conf2, 1.0, 1.0)

            if score1 >= score2:
                cands = cands1
                decisions.append("tr1_best")
            else:
                cands = cands2
                decisions.append("tr2_best")

        else:
            # Basic scenarios only - advanced ones handled by evaluate_with_best_of_n
            raise ValueError(f"Unknown scenario: {scenario}. Use evaluate_with_best_of_n for advanced scenarios.")

        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands]
        if tk in keys:
            idx = keys.index(tk)
            if idx == 0: r1 += 1
            if idx < 5: r5 += 1
            if idx < 10: r10 += 1
            if idx < 20: r20 += 1
            mrr += 1.0/(idx+1)

    lat = np.array(lat)
    results = {
        "scenario": scenario,
        "N": N,
        "R@1": r1/N, "R@5": r5/N, "R@10": r10/N, "R@20": r20/N,
        "MRR@10": mrr/N,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "decisions": decisions if decisions else None,
    }

    # Add decision statistics for advanced scenarios
    if decisions:
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        results["decision_breakdown"] = decision_counts

    return results


# ------------------------------ CLI -------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=["baseline_direct","single_tr","ensemble_union_tr2","ensemble_avg_tr2",
                            "best_of_direct_tr","confidence_gated_multi_tr","ensemble_best_of_2"]) 
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--top-final", type=int, default=10)
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    ap.add_argument("--w-same-article", type=float, default=0.05)
    ap.add_argument("--w-next-gap", type=float, default=0.12)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--directional-bonus", type=float, default=0.0)
    ap.add_argument("--fusion-temp", type=float, default=0.05)

    # TR generation options
    ap.add_argument("--tr-module", type=str, default=None)
    ap.add_argument("--tr-fn", type=str, default=None)
    ap.add_argument("--tr-temp1", type=float, default=0.06)
    ap.add_argument("--tr-temp2", type=float, default=0.08)
    ap.add_argument("--tr-seed1", type=int, default=1337)
    ap.add_argument("--tr-seed2", type=int, default=4242)
    ap.add_argument("--tr-attempts", type=int, default=2)

    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Load artifacts
    npz_data = load_npz(args.npz)
    if args.limit:
        # slice consistently for all available arrays
        sl = slice(0, args.limit)
        for k in list(npz_data.keys()):
            if isinstance(npz_data[k], np.ndarray) and npz_data[k].shape[0] >= args.limit:
                npz_data[k] = npz_data[k][sl]

    payload = load_payload(args.payload)
    faiss_index = faiss.read_index(str(args.faiss))
    retriever = RetrievalShim(faiss_index, payload)

    last_meta = list(npz_data["last_meta"])  # list of dicts
    truth_keys = np.asarray(npz_data["truth_keys"])  # [N,2]

    vecs_direct = npz_data.get("pred_vecs_direct")
    last_vecs = npz_data.get("last_vecs")

    # Prepare TR vectors for scenarios that need them
    vecs_tr1 = vecs_tr2 = None
    advanced_scenarios = {"best_of_direct_tr","confidence_gated_multi_tr","ensemble_best_of_2"}

    if args.scenario in {"single_tr","ensemble_union_tr2","ensemble_avg_tr2"} or args.scenario in advanced_scenarios:
        vecs_tr1 = run_tr_vectors(npz_data, args.tr_module, args.tr_fn, args.tr_temp1, args.tr_seed1, args.tr_attempts)
        if args.scenario not in {"single_tr", "best_of_direct_tr", "confidence_gated_multi_tr"}:
            vecs_tr2 = run_tr_vectors(npz_data, args.tr_module, args.tr_fn, args.tr_temp2, args.tr_seed2, args.tr_attempts)

    # Choose evaluation function
    if args.scenario in advanced_scenarios:
        results = evaluate_with_best_of_n(
            scenario=args.scenario,
            retriever=retriever,
            last_meta=last_meta,
            truth_keys=truth_keys,
            vecs_direct=vecs_direct,
            vecs_tr1=vecs_tr1,
            vecs_tr2=vecs_tr2,
            last_vecs=last_vecs,
            K=args.K,
            top_final=args.top_final,
            mmr_lambda=args.mmr_lambda,
            use_seq_bias=True,
            w_same_article=args.w_same_article,
            w_next_gap=args.w_next_gap,
            tau=args.tau,
            directional_bonus=args.directional_bonus,
            adaptive_k=True,
        )
    else:
        results = evaluate(
            scenario=args.scenario,
            retriever=retriever,
            last_meta=last_meta,
            truth_keys=truth_keys,
            vecs_direct=vecs_direct,
            vecs_tr1=vecs_tr1,
            vecs_tr2=vecs_tr2,
            last_vecs=last_vecs,
            K=args.K,
            top_final=args.top_final,
            mmr_lambda=args.mmr_lambda,
            use_seq_bias=True,
            w_same_article=args.w_same_article,
            w_next_gap=args.w_next_gap,
            tau=args.tau,
            directional_bonus=args.directional_bonus,
            fusion_temperature=args.fusion_temp,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
