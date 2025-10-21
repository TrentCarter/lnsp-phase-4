#!/usr/bin/env python3
"""
Hybrid Retrieval Evaluation - Quick Win Path

Strategy:
1. GTR-T5 (strong generic encoder) → FAISS top-K₀
2. Phase-3 LVM → FAISS top-K₀
3. Fuse with Reciprocal Rank Fusion (RRF)
4. Optional: LVM re-rank top-K₁
5. Optional: TMD re-rank final top-K

Expected: GTR-T5 gets targets into candidate pool, LVM/TMD refine ranking.
"""

import argparse
import numpy as np
import torch
import faiss
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import time

# Import GTR-T5 encoder
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator


def load_faiss_index(index_path: str, nprobe: int = 16) -> faiss.Index:
    """Load FAISS index with specified nprobe."""
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)

    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe
        print(f"  Set nprobe={nprobe}")

    print(f"  Index type: {type(index).__name__}")
    print(f"  Vectors: {index.ntotal:,}")

    return index


def load_vector_bank(npz_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load vector bank."""
    print(f"Loading vector bank: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']
    cpe_ids = data['cpe_ids']

    print(f"  Vectors: {vectors.shape}")
    print(f"  CPE IDs: {len(cpe_ids):,}")

    return vectors, cpe_ids


def load_lvm_model(model_path: str, device: str = 'mps'):
    """Load Phase-3 LVM model."""
    print(f"Loading Phase-3 LVM model: {model_path}")

    from app.lvm.memory_gru import MemoryAugmentedGRU

    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Phase-3 actual config (from checkpoint inspection)
        config = {
            'd_model': 768,
            'hidden_dim': 512,
            'memory_slots': 2048,
            'num_layers': 4,
            'use_memory_write': True
        }

    model = MemoryAugmentedGRU(
        d_model=config.get('d_model', config.get('input_dim', 768)),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 4),
        memory_slots=config.get('memory_slots', config.get('memory_dim', 2048)),
        use_memory_write=config.get('use_memory_write', True)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Model loaded")
    print(f"  Config: {config}")

    return model, config


def load_validation_data(npz_path: str) -> Dict[str, np.ndarray]:
    """Load validation data."""
    print(f"Loading validation data: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    context_seqs = data['context_sequences']
    target_vecs = data['target_vectors']
    target_indices = data['target_indices']

    print(f"  Sequences: {len(context_seqs):,}")
    print(f"  Context length: {context_seqs.shape[1]}")

    return {
        'context': context_seqs,
        'targets': target_vecs,
        'target_indices': target_indices
    }


def reciprocal_rank_fusion(
    lists: List[List[Tuple[int, float]]],
    K: int = 60,
    top_k: int = 1000
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples
        K: Constant for RRF formula (default: 60)
        top_k: Number of top results to return

    Returns:
        Fused ranked list of (id, rrf_score) tuples
    """
    scores = defaultdict(float)

    for ranked_list in lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (K + rank)

    # Sort by RRF score descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return fused[:top_k]


def get_gtr_t5_candidates(
    orchestrator: IsolatedVecTextVectOrchestrator,
    context_seq: np.ndarray,
    faiss_index: faiss.Index,
    k: int = 1000
) -> List[Tuple[int, float]]:
    """
    Get candidates using GTR-T5 dense encoder.

    Use the last context vector as the query (simple baseline).
    """
    # Use last vector in context as query
    query_vec = context_seq[-1:, :]  # (1, 768)

    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    # Search FAISS
    distances, indices = faiss_index.search(query_norm.astype(np.float32), k)

    # Convert to (id, score) list
    candidates = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    return candidates


def get_lvm_candidates(
    model: torch.nn.Module,
    context_seq: np.ndarray,
    faiss_index: faiss.Index,
    device: str,
    k: int = 1000
) -> List[Tuple[int, float]]:
    """
    Get candidates using Phase-3 LVM predictions.
    """
    # Convert to torch
    context_tensor = torch.from_numpy(context_seq).unsqueeze(0).to(device)  # (1, T, 768)

    # Get LVM prediction
    with torch.no_grad():
        pred_vec = model(context_tensor)  # (1, 768)

    pred_np = pred_vec.cpu().numpy()

    # Normalize
    pred_norm = pred_np / (np.linalg.norm(pred_np) + 1e-8)

    # Search FAISS
    distances, indices = faiss_index.search(pred_norm.astype(np.float32), k)

    # Convert to (id, score) list
    candidates = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    return candidates


def lvm_rerank(
    model: torch.nn.Module,
    context_seq: np.ndarray,
    candidates: List[int],
    bank_vectors: np.ndarray,
    device: str
) -> List[Tuple[int, float]]:
    """
    Re-rank candidates using LVM prediction similarity.
    """
    # Get LVM prediction
    context_tensor = torch.from_numpy(context_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_vec = model(context_tensor).cpu().numpy()[0]  # (768,)

    # Normalize
    pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)

    # Get candidate vectors
    cand_vecs = bank_vectors[candidates]  # (K, 768)
    cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)

    # Compute similarities
    sims = np.dot(cand_norm, pred_norm)  # (K,)

    # Sort by similarity descending
    sorted_indices = np.argsort(-sims)

    reranked = [(candidates[i], float(sims[i])) for i in sorted_indices]

    return reranked


def tmd_rerank(
    candidates: List[Tuple[int, float]],
    bank_vectors: np.ndarray,
    cpe_ids: List[str],
    query_lane: str,
    lambda_weight: float = 0.7
) -> List[Tuple[int, float]]:
    """
    Re-rank with TMD lane boosting.

    Args:
        candidates: List of (id, score) tuples
        bank_vectors: Full vector bank
        cpe_ids: CPE IDs (used to extract lane info)
        query_lane: Target TMD lane
        lambda_weight: Weight for original score vs TMD boost
    """
    # Simplified TMD: boost candidates from same lane
    # In practice, you'd extract lane from concept metadata

    # For now, use a heuristic or skip TMD if metadata unavailable
    # Just return candidates unchanged (TMD integration requires metadata)

    return candidates


def compute_metrics(
    candidates: List[int],
    target_idx: int,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute Hit@K and MRR metrics."""
    metrics = {}

    # Find rank of target (1-indexed)
    try:
        rank = candidates.index(target_idx) + 1
    except ValueError:
        rank = len(candidates) + 1  # Not found

    # Hit@K
    for k in k_values:
        metrics[f'hit@{k}'] = 1.0 if rank <= k else 0.0

    # MRR
    metrics['mrr'] = 1.0 / rank if rank <= len(candidates) else 0.0

    return metrics


def evaluate_hybrid(
    model: torch.nn.Module,
    orchestrator: IsolatedVecTextVectOrchestrator,
    faiss_index: faiss.Index,
    val_data: Dict[str, np.ndarray],
    bank_vectors: np.ndarray,
    cpe_ids: List[str],
    config: Dict,
    device: str = 'mps'
) -> Dict:
    """
    Evaluate hybrid retrieval pipeline.

    Pipeline:
    1. GTR-T5 → FAISS top-K₀
    2. LVM → FAISS top-K₀
    3. RRF fusion → top-K₀
    4. LVM re-rank → top-K₁
    5. TMD re-rank → top-K_final
    """
    K0_dense = config.get('K0_dense', 1000)
    K0_lvm = config.get('K0_lvm', 1000)
    K0_fused = config.get('K0_fused', 1000)
    K1 = config.get('K1', 50)
    K_final = config.get('K_final', 10)
    RRF_K = config.get('RRF_K', 60)
    use_lvm_rerank = config.get('use_lvm_rerank', True)
    use_tmd_rerank = config.get('use_tmd_rerank', False)

    context_seqs = val_data['context']
    target_indices = val_data['target_indices']

    print(f"\n{'='*60}")
    print(f"HYBRID EVALUATION CONFIG")
    print(f"{'='*60}")
    print(f"  K0_dense: {K0_dense}")
    print(f"  K0_lvm: {K0_lvm}")
    print(f"  K0_fused: {K0_fused}")
    print(f"  K1 (LVM re-rank): {K1}")
    print(f"  K_final (TMD): {K_final}")
    print(f"  RRF_K: {RRF_K}")
    print(f"  LVM re-rank: {use_lvm_rerank}")
    print(f"  TMD re-rank: {use_tmd_rerank}")
    print(f"  Validation samples: {len(context_seqs):,}")

    # Metrics accumulators
    stage_metrics = {
        'stage1_dense': defaultdict(list),
        'stage1_lvm': defaultdict(list),
        'stage2_fused': defaultdict(list),
        'stage3_lvm_rerank': defaultdict(list),
        'stage4_tmd': defaultdict(list)
    }

    latencies = {
        'dense_retrieval': [],
        'lvm_retrieval': [],
        'rrf_fusion': [],
        'lvm_rerank': [],
        'tmd_rerank': [],
        'total': []
    }

    for i, (context, target_idx) in enumerate(tqdm(zip(context_seqs, target_indices), total=len(context_seqs), desc="Evaluating")):
        t_start = time.time()

        # Stage 1a: GTR-T5 dense retrieval
        t0 = time.time()
        dense_cands = get_gtr_t5_candidates(orchestrator, context, faiss_index, k=K0_dense)
        latencies['dense_retrieval'].append(time.time() - t0)

        dense_ids = [cid for cid, _ in dense_cands]
        for k in [1, 5, 10, 100, 500, 1000]:
            if k <= len(dense_ids):
                stage_metrics['stage1_dense'][f'recall@{k}'].append(
                    1.0 if target_idx in dense_ids[:k] else 0.0
                )

        # Stage 1b: LVM retrieval
        t0 = time.time()
        lvm_cands = get_lvm_candidates(model, context, faiss_index, device, k=K0_lvm)
        latencies['lvm_retrieval'].append(time.time() - t0)

        lvm_ids = [cid for cid, _ in lvm_cands]
        for k in [1, 5, 10, 100, 500, 1000]:
            if k <= len(lvm_ids):
                stage_metrics['stage1_lvm'][f'recall@{k}'].append(
                    1.0 if target_idx in lvm_ids[:k] else 0.0
                )

        # Stage 2: RRF Fusion
        t0 = time.time()
        fused = reciprocal_rank_fusion([dense_cands, lvm_cands], K=RRF_K, top_k=K0_fused)
        latencies['rrf_fusion'].append(time.time() - t0)

        fused_ids = [cid for cid, _ in fused]
        for k in [1, 5, 10, 100, 500, 1000]:
            if k <= len(fused_ids):
                stage_metrics['stage2_fused'][f'recall@{k}'].append(
                    1.0 if target_idx in fused_ids[:k] else 0.0
                )

        # Stage 3: LVM Re-rank (optional)
        if use_lvm_rerank and len(fused_ids) > 0:
            t0 = time.time()
            reranked = lvm_rerank(model, context, fused_ids[:K0_fused], bank_vectors, device)
            latencies['lvm_rerank'].append(time.time() - t0)

            reranked_ids = [cid for cid, _ in reranked[:K1]]
        else:
            reranked_ids = fused_ids[:K1]
            latencies['lvm_rerank'].append(0.0)

        for k in [1, 5, 10]:
            if k <= len(reranked_ids):
                stage_metrics['stage3_lvm_rerank'][f'hit@{k}'].append(
                    1.0 if target_idx in reranked_ids[:k] else 0.0
                )

        # Stage 4: TMD Re-rank (optional)
        if use_tmd_rerank:
            t0 = time.time()
            final_cands = tmd_rerank(
                [(cid, 0.0) for cid in reranked_ids],
                bank_vectors,
                cpe_ids,
                query_lane='lane_0',  # Placeholder
                lambda_weight=0.7
            )
            latencies['tmd_rerank'].append(time.time() - t0)

            final_ids = [cid for cid, _ in final_cands[:K_final]]
        else:
            final_ids = reranked_ids[:K_final]
            latencies['tmd_rerank'].append(0.0)

        for k in [1, 5, 10]:
            if k <= len(final_ids):
                stage_metrics['stage4_tmd'][f'hit@{k}'].append(
                    1.0 if target_idx in final_ids[:k] else 0.0
                )

        latencies['total'].append(time.time() - t_start)

    # Aggregate results
    results = {
        'config': config,
        'stages': {},
        'latency': {}
    }

    for stage_name, metrics in stage_metrics.items():
        results['stages'][stage_name] = {
            metric: float(np.mean(values)) * 100  # Convert to percentage
            for metric, values in metrics.items()
        }

    for lat_name, lat_values in latencies.items():
        results['latency'][lat_name] = {
            'p50': float(np.percentile(lat_values, 50) * 1000),  # ms
            'p95': float(np.percentile(lat_values, 95) * 1000),
            'p99': float(np.percentile(lat_values, 99) * 1000)
        }

    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print("HYBRID EVALUATION RESULTS")
    print(f"{'='*60}")

    # Stage-by-stage results
    for stage_name, metrics in results['stages'].items():
        print(f"\n{stage_name.upper()}:")
        for metric, value in sorted(metrics.items()):
            print(f"  {metric:20s}: {value:6.2f}%")

    # Latency breakdown
    print(f"\n{'='*60}")
    print("LATENCY BREAKDOWN (milliseconds)")
    print(f"{'='*60}")

    for component, lats in results['latency'].items():
        print(f"\n{component}:")
        print(f"  P50: {lats['p50']:6.2f} ms")
        print(f"  P95: {lats['p95']:6.2f} ms")
        print(f"  P99: {lats['p99']:6.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Evaluation")
    parser.add_argument('--model', default='artifacts/lvm/models_phase3/run_1000ctx_pilot/best_val_hit5.pt')
    parser.add_argument('--index', default='artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
    parser.add_argument('--vectors', default='artifacts/wikipedia_500k_corrected_vectors.npz')
    parser.add_argument('--validation', default='artifacts/lvm/data_phase3_tmd/validation_sequences_ctx100.npz')
    parser.add_argument('--K0-dense', type=int, default=1000, help='Top-K from dense retrieval')
    parser.add_argument('--K0-lvm', type=int, default=1000, help='Top-K from LVM retrieval')
    parser.add_argument('--K0-fused', type=int, default=1000, help='Top-K after RRF fusion')
    parser.add_argument('--K1', type=int, default=50, help='Top-K after LVM re-rank')
    parser.add_argument('--K-final', type=int, default=10, help='Final top-K after TMD')
    parser.add_argument('--RRF-K', type=int, default=60, help='RRF constant')
    parser.add_argument('--nprobe', type=int, default=16, help='FAISS nprobe')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--no-lvm-rerank', action='store_true', help='Skip LVM re-ranking')
    parser.add_argument('--use-tmd', action='store_true', help='Enable TMD re-ranking')
    parser.add_argument('--output', default='artifacts/evals/hybrid_results.json')

    args = parser.parse_args()

    # Load components
    print("Loading components...")
    model, model_config = load_lvm_model(args.model, device=args.device)
    orchestrator = IsolatedVecTextVectOrchestrator()
    faiss_index = load_faiss_index(args.index, nprobe=args.nprobe)
    bank_vectors, cpe_ids = load_vector_bank(args.vectors)
    val_data = load_validation_data(args.validation)

    # Evaluation config
    eval_config = {
        'K0_dense': args.K0_dense,
        'K0_lvm': args.K0_lvm,
        'K0_fused': args.K0_fused,
        'K1': args.K1,
        'K_final': args.K_final,
        'RRF_K': args.RRF_K,
        'nprobe': args.nprobe,
        'use_lvm_rerank': not args.no_lvm_rerank,
        'use_tmd_rerank': args.use_tmd
    }

    # Run evaluation
    results = evaluate_hybrid(
        model=model,
        orchestrator=orchestrator,
        faiss_index=faiss_index,
        val_data=val_data,
        bank_vectors=bank_vectors,
        cpe_ids=cpe_ids,
        config=eval_config,
        device=args.device
    )

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
