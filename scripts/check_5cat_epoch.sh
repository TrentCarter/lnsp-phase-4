#!/bin/bash
#
# Run mini-5CAT check on a specific model checkpoint
# Usage: ./scripts/check_5cat_epoch.sh <model_path> [max_samples]
#
# Examples:
#   ./scripts/check_5cat_epoch.sh artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt 500
#   ./scripts/check_5cat_epoch.sh artifacts/lvm/models/transformer_directional_v3/best_model.pt
#

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [max_samples]"
    echo ""
    echo "Examples:"
    echo "  $0 artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt 500"
    echo "  $0 artifacts/lvm/models/transformer_directional_v3/best_model.pt"
    exit 1
fi

MODEL_PATH="$1"
MAX_SAMPLES="${2:-500}"  # Default 500 samples for fast checks

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "Running 5CAT Check"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Max samples: $MAX_SAMPLES"
echo ""

LOG_FILE="/tmp/5cat_check_$(date +%Y%m%d_%H%M%S).log"

./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model "$MODEL_PATH" \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps \
  --max-samples "$MAX_SAMPLES" \
  | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 5CAT Check Complete!"
echo "=========================================="
echo "Results saved to: $LOG_FILE"
echo ""
echo "Quick interpretation:"
echo "  ✅ PASS: Margin(+1) > 0, k=+1 is peak, gates passing"
echo "  ⚠️  WARN: Margin(+1) near 0, or k=+2/+3 drift"
echo "  ❌ FAIL: Margin(+1) < 0, k=-1 is peak (backward prediction)"
echo ""
