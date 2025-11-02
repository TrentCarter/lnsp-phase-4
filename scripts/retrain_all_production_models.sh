#!/bin/bash
#
# Retrain All Production Models with 5CAT Validation
# ===================================================
#
# Convenience script to retrain all 4 production models sequentially.
# Uses clean 584k data and integrates 5CAT testing.
#
# Usage:
#   ./scripts/retrain_all_production_models.sh

set -e

echo "============================================"
echo "Retrain All Production Models"
echo "============================================"
echo ""
echo "This will train 4 models sequentially:"
echo "  1. Transformer Optimized (recommended)"
echo "  2. GRU"
echo "  3. LSTM"
echo "  4. AMN"
echo ""
echo "Each model will:"
echo "  - Train for 20 epochs"
echo "  - Run 5CAT validation every 5 epochs"
echo "  - Alert if backward bias detected"
echo "  - Save best models"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting training sequence..."
echo ""

# Train Transformer (best overall performer)
echo "============================================"
echo "1/4: Training Transformer"
echo "============================================"
./scripts/train_with_5cat_validation.sh transformer 20 mps
echo ""

# Train GRU
echo "============================================"
echo "2/4: Training GRU"
echo "============================================"
./scripts/train_with_5cat_validation.sh gru 20 mps
echo ""

# Train LSTM
echo "============================================"
echo "3/4: Training LSTM"
echo "============================================"
./scripts/train_with_5cat_validation.sh lstm 20 mps
echo ""

# Train AMN
echo "============================================"
echo "4/4: Training AMN"
echo "============================================"
./scripts/train_with_5cat_validation.sh amn 20 mps
echo ""

echo "============================================"
echo "✅ ALL MODELS RETRAINED!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Review training logs in artifacts/lvm/models/*_5cat_*/"
echo "  2. Run comprehensive 5CAT tests on all models"
echo "  3. Update production symlinks (transformer_v0.pt, etc.)"
echo "  4. Restart LVM services with new models"
echo ""
