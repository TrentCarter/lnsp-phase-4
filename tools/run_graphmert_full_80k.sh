#!/bin/bash
# Train GraphMERT-LVM on FULL 80k Dataset
# ========================================
#
# Using 40 GPUs with DDP for maximum parallelization!
#
# Dataset: 80,629 Wikipedia sequences
# Epochs: 10 (can adjust based on convergence)
# Total training samples: 806,290 forward passes

set -e

NGPUS=40
EPOCHS=10
BATCH_SIZE=32  # Per GPU = 1,280 total batch size across 40 GPUs!

echo "=========================================="
echo "GraphMERT-LVM FULL Training (80k dataset)"
echo "=========================================="
echo "GPUs: $NGPUS"
echo "Dataset: 80,629 Wikipedia sequences"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((NGPUS * BATCH_SIZE))"
echo ""
echo "🚀 GO BIG MODE ACTIVATED!"
echo ""

# Activate venv
source .venv/bin/activate

# Create output directory
OUTPUT_DIR="artifacts/lvm/models/graphmert_lvm_full_80k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Run with DDP on 40 GPUs
torchrun \
    --standalone \
    --nproc_per_node=$NGPUS \
    app/lvm/train_graphmert_lvm_benchmark.py \
    --data artifacts/lvm/training_sequences_ctx5.npz \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4 \
    --n-layers 12 \
    --n-heads 8 \
    --d-ff 2048 \
    --dropout 0.1 \
    --lambda-decay 0.6 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✓ Training Complete!"
echo "=========================================="
echo "Results: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/benchmark_results.json | jq '.'"
echo ""
echo "Check training time:"
echo "  cat $OUTPUT_DIR/benchmark_results.json | jq '.total_training_time'"
echo ""
echo "Check final cosine similarity:"
echo "  cat $OUTPUT_DIR/benchmark_results.json | jq '.history[-1].val_cosine'"
