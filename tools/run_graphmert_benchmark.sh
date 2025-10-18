#!/bin/bash
# Run GraphMERT-LVM 10k Benchmark
# ================================
#
# Usage:
#   # Single GPU
#   ./tools/run_graphmert_benchmark.sh 1
#
#   # 8 GPUs
#   ./tools/run_graphmert_benchmark.sh 8
#
#   # 40 GPUs
#   ./tools/run_graphmert_benchmark.sh 40

set -e

# Number of GPUs (default: 1)
NGPUS=${1:-1}

echo "=========================================="
echo "GraphMERT-LVM 10k Benchmark"
echo "=========================================="
echo "GPUs: $NGPUS"
echo "Dataset: 10k Wikipedia sequences"
echo "Epochs: 3"
echo "Batch size per GPU: 32"
echo ""

# Activate venv
source .venv/bin/activate

# Single GPU
if [ "$NGPUS" -eq 1 ]; then
    echo "Running on single GPU..."
    python app/lvm/train_graphmert_lvm_benchmark.py \
        --data artifacts/lvm/training_sequences_ctx5_10k.npz \
        --epochs 3 \
        --batch-size 32 \
        --lr 1e-4 \
        --device cuda:0
else
    # Multi-GPU with DDP
    echo "Running on $NGPUS GPUs with DDP..."
    torchrun \
        --standalone \
        --nproc_per_node=$NGPUS \
        app/lvm/train_graphmert_lvm_benchmark.py \
        --data artifacts/lvm/training_sequences_ctx5_10k.npz \
        --epochs 3 \
        --batch-size 32 \
        --lr 1e-4
fi

echo ""
echo "✓ Benchmark complete! Check artifacts/lvm/models/graphmert_lvm_benchmark/"
