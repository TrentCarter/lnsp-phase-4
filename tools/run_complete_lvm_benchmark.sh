#!/bin/bash
#
# Complete LVM Training & Benchmark Pipeline
# ============================================
#
# This script runs the entire pipeline:
# 1. Trains all 4 LVM models (8-16 hours)
# 2. Creates OOD test set from new Wikipedia articles
# 3. Runs comprehensive benchmark (in-dist + OOD)
# 4. Generates comparison report
#
# Usage:
#   bash tools/run_complete_lvm_benchmark.sh

set -e

echo "================================================================================"
echo "COMPLETE LVM TRAINING & BENCHMARK PIPELINE"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Train all 4 models (LSTM, AMN, GRU, Transformer) - 8-16 hours"
echo "  2. Create OOD test set (500 new Wikipedia articles) - 30 mins"
echo "  3. Benchmark all models (in-dist + OOD) - 10 mins"
echo "  4. Generate comparison report"
echo ""
echo "Total estimated time: 9-17 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 1: TRAINING ALL 4 LVM MODELS"
echo "================================================================================"
echo ""

bash tools/train_all_4_lvms.sh

echo ""
echo "================================================================================"
echo "STEP 2: CREATING OUT-OF-DISTRIBUTION TEST SET"
echo "================================================================================"
echo ""

./.venv/bin/python tools/create_ood_test_set.py \
    --start-article 8471 \
    --num-articles 500

echo ""
echo "================================================================================"
echo "STEP 3: COMPREHENSIVE BENCHMARK"
echo "================================================================================"
echo ""

./.venv/bin/python tools/benchmark_all_lvms_comprehensive.py

echo ""
echo "================================================================================"
echo "✅ COMPLETE PIPELINE FINISHED!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Training logs: logs/lvm_training_*/"
echo "  - Benchmark report: artifacts/lvm/benchmark_results_*.md"
echo "  - Benchmark data: artifacts/lvm/benchmark_results_*.json"
echo ""
