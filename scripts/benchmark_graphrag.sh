#!/usr/bin/env bash
# GraphRAG Benchmark Reference Script
# Created: 2025-10-05
# Purpose: Reproduce all GraphRAG benchmarks with correct parameters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default configuration
FAISS_NPZ="${FAISS_NPZ_PATH:-artifacts/fw10k_vectors.npz}"
FAISS_INDEX="${FAISS_INDEX_PATH:-artifacts/fw10k_ivf_flat_ip.index}"
N_QUERIES="${N_QUERIES:-200}"
TOPK="${TOPK:-10}"

# GraphRAG tuning parameters (Phase 1+2 fixes)
GR_RRF_K="${GR_RRF_K:-60}"              # RRF k parameter
GR_GRAPH_WEIGHT="${GR_GRAPH_WEIGHT:-1.0}"  # Graph signal weight
GR_SEED_TOP="${GR_SEED_TOP:-10}"        # Number of expansion seeds
GR_SIM_WEIGHT="${GR_SIM_WEIGHT:-1.0}"   # Query similarity weight

# TMD configuration
TMD_ALPHA="${TMD_ALPHA:-0.3}"           # TMD reranking alpha (optimal from Oct 4)

# Performance tuning
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FAISS_NUM_THREADS=1
export PYTHONPATH=.

usage() {
    cat <<EOF
Usage: $0 [OPTION]

Run GraphRAG benchmarks with various configurations.

OPTIONS:
    baseline        Run baseline comparison (vec, BM25, lex, GraphRAG)
    with-tmd        Run comprehensive benchmark including TMD reranking
    graphrag-only   Run GraphRAG validation (local + hybrid modes)
    tune-weights    Run grid search over GR_GRAPH_WEIGHT values
    all             Run all benchmarks
    help            Show this help message

ENVIRONMENT VARIABLES:
    FAISS_NPZ_PATH       Path to NPZ file (default: artifacts/fw10k_vectors.npz)
    FAISS_INDEX_PATH     Path to FAISS index (default: artifacts/fw10k_ivf_flat_ip.index)
    N_QUERIES            Number of queries (default: 200)
    TOPK                 Top-K results (default: 10)
    GR_RRF_K             RRF k parameter (default: 60)
    GR_GRAPH_WEIGHT      Graph signal weight (default: 1.0)
    GR_SEED_TOP          Expansion seeds (default: 10)
    GR_SIM_WEIGHT        Query similarity weight (default: 1.0)
    TMD_ALPHA            TMD alpha parameter (default: 0.3)

EXAMPLES:
    # Run baseline benchmark
    $0 baseline

    # Run with custom graph weight
    GR_GRAPH_WEIGHT=0.5 $0 graphrag-only

    # Run comprehensive with TMD
    $0 with-tmd

    # Tune graph weights
    $0 tune-weights

EOF
    exit 0
}

benchmark_baseline() {
    echo "=== BASELINE BENCHMARK ==="
    echo "Backends: vecRAG, BM25, Lexical, GraphRAG (local+hybrid)"
    echo "Dataset: Self-retrieval (ontology data)"
    echo "Queries: $N_QUERIES"
    echo ""

    TIMESTAMP=$(date +%Y%m%d_%H%M)
    OUTPUT="RAG/results/baseline_${TIMESTAMP}.jsonl"

    ./.venv/bin/python RAG/bench.py \
        --dataset self \
        --n "$N_QUERIES" \
        --topk "$TOPK" \
        --backends vec,bm25,lex,graphrag_local,graphrag_hybrid \
        --npz "$FAISS_NPZ" \
        --index "$FAISS_INDEX" \
        --out "$OUTPUT"

    echo ""
    echo "✓ Baseline benchmark complete"
    echo "Results: $OUTPUT"
    echo "Summary: $(ls -t RAG/results/summary_*.md | head -1)"
}

benchmark_with_tmd() {
    echo "=== COMPREHENSIVE BENCHMARK WITH TMD ==="
    echo "Backends: vecRAG, BM25, Lexical, TMD-Rerank, GraphRAG (local+hybrid)"
    echo "TMD Alpha: $TMD_ALPHA"
    echo "Queries: $N_QUERIES"
    echo ""

    TIMESTAMP=$(date +%Y%m%d_%H%M)
    OUTPUT="RAG/results/comprehensive_tmd_${TIMESTAMP}.jsonl"

    ./.venv/bin/python RAG/bench.py \
        --dataset self \
        --n "$N_QUERIES" \
        --topk "$TOPK" \
        --backends vec,bm25,lex,vec_tmd_rerank,graphrag_local,graphrag_hybrid \
        --npz "$FAISS_NPZ" \
        --index "$FAISS_INDEX" \
        --out "$OUTPUT"

    echo ""
    echo "✓ Comprehensive benchmark complete"
    echo "Results: $OUTPUT"
    echo "Summary: $(ls -t RAG/results/summary_*.md | head -1)"
}

benchmark_graphrag_only() {
    echo "=== GRAPHRAG VALIDATION BENCHMARK ==="
    echo "Phase 1+2 Fixes Applied:"
    echo "  - Safety: Re-rank only within vector candidates"
    echo "  - Scale: Graph uses RRF scores (GR_GRAPH_WEIGHT=$GR_GRAPH_WEIGHT)"
    echo "  - Query-sim: Added query similarity term (GR_SIM_WEIGHT=$GR_SIM_WEIGHT)"
    echo "  - Seeds: Expanded to $GR_SEED_TOP top results"
    echo "  - RRF k: $GR_RRF_K"
    echo ""

    TIMESTAMP=$(date +%Y%m%d_%H%M)
    OUTPUT="RAG/results/graphrag_validation_${TIMESTAMP}.jsonl"

    ./.venv/bin/python RAG/bench.py \
        --dataset self \
        --n "$N_QUERIES" \
        --topk "$TOPK" \
        --backends vec,graphrag_local,graphrag_hybrid \
        --npz "$FAISS_NPZ" \
        --index "$FAISS_INDEX" \
        --out "$OUTPUT"

    echo ""
    echo "✓ GraphRAG validation complete"
    echo "Results: $OUTPUT"
    echo "Summary: $(ls -t RAG/results/summary_*.md | head -1)"
}

benchmark_tune_weights() {
    echo "=== GRAPH WEIGHT TUNING ==="
    echo "Testing GR_GRAPH_WEIGHT values: 0.25, 0.5, 1.0, 2.0, 5.0"
    echo ""

    for weight in 0.25 0.5 1.0 2.0 5.0; do
        echo "--- Testing GR_GRAPH_WEIGHT=$weight ---"
        TIMESTAMP=$(date +%Y%m%d_%H%M)
        OUTPUT="RAG/results/graphrag_weight_${weight}_${TIMESTAMP}.jsonl"

        GR_GRAPH_WEIGHT=$weight ./.venv/bin/python RAG/bench.py \
            --dataset self \
            --n "$N_QUERIES" \
            --topk "$TOPK" \
            --backends vec,graphrag_hybrid \
            --npz "$FAISS_NPZ" \
            --index "$FAISS_INDEX" \
            --out "$OUTPUT"

        echo ""
    done

    echo "✓ Weight tuning complete"
    echo "Compare summaries:"
    ls -lt RAG/results/summary_*.md | head -5
}

main() {
    case "${1:-help}" in
        baseline)
            benchmark_baseline
            ;;
        with-tmd)
            benchmark_with_tmd
            ;;
        graphrag-only)
            benchmark_graphrag_only
            ;;
        tune-weights)
            benchmark_tune_weights
            ;;
        all)
            benchmark_baseline
            echo ""
            benchmark_graphrag_only
            echo ""
            benchmark_with_tmd
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo ""
            usage
            ;;
    esac
}

main "$@"
