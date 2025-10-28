#!/bin/bash
# Evaluate All Completed Mamba Models
# Runs retrieval evaluation on Mamba-S, Mamba-H, Mamba-Sandwich, and Mamba-GR

set -e

# Configuration
EVAL_NPZ="artifacts/lvm/eval_v2_ready_aligned.npz"
PAYLOAD="artifacts/wikipedia_584k_payload.npy"
FAISS_INDEX="artifacts/wikipedia_584k_ivf_flat_ip.index"
SHARDS="artifacts/article_shards.pkl"
DEVICE="mps"
NPROBE=64

echo "================================================================================
Mamba Phase-5 Model Evaluation
================================================================================
Evaluation Dataset: $EVAL_NPZ
Payload: $PAYLOAD
FAISS Index: $FAISS_INDEX
Shards: $SHARDS
Device: $DEVICE
nprobe: $NPROBE
================================================================================"

# Check dependencies
echo ""
echo "Checking dependencies..."
for file in "$EVAL_NPZ" "$PAYLOAD" "$FAISS_INDEX" "$SHARDS"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
    echo "✅ Found: $file"
done

echo ""
echo "Checking model checkpoints..."
MODELS=(
    "mamba_s:Mamba-S (Pure SSM)"
    "mamba_hybrid_local:Mamba-H (Hybrid 80/20)"
    "mamba_sandwich:Mamba-Sandwich (Attn→SSM→Attn)"
    "mamba_gr:Mamba-GR (SSM + GRU)"
)

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_desc <<< "$model_info"
    checkpoint="artifacts/lvm/models/$model_name/best.pt"
    if [ ! -f "$checkpoint" ]; then
        echo "❌ Missing: $checkpoint ($model_desc)"
        exit 1
    fi
    echo "✅ Found: $checkpoint ($model_desc)"
done

echo ""
echo "================================================================================
Starting Evaluations (Sequential)
================================================================================
"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model 1: Mamba-S
echo "1/4 Evaluating Mamba-S (Pure SSM)..."
echo "----------------------------------------"
./.venv/bin/python tools/eval_mamba_models.py \
    --model artifacts/lvm/models/mamba_s/best.pt \
    --eval-npz "$EVAL_NPZ" \
    --payload "$PAYLOAD" \
    --faiss "$FAISS_INDEX" \
    --device "$DEVICE" \
    --out "artifacts/lvm/eval_mamba_s_${TIMESTAMP}.json"

echo ""
echo "✅ Mamba-S evaluation complete!"
echo ""

# Model 2: Mamba-H
echo "2/4 Evaluating Mamba-H (Hybrid 80/20)..."
echo "----------------------------------------"
./.venv/bin/python tools/eval_mamba_models.py \
    --model artifacts/lvm/models/mamba_hybrid_local/best.pt \
    --eval-npz "$EVAL_NPZ" \
    --payload "$PAYLOAD" \
    --faiss "$FAISS_INDEX" \
    --device "$DEVICE" \
    --out "artifacts/lvm/eval_mamba_h_${TIMESTAMP}.json"

echo ""
echo "✅ Mamba-H evaluation complete!"
echo ""

# Model 3: Mamba-Sandwich
echo "3/4 Evaluating Mamba-Sandwich (Attn→SSM→Attn)..."
echo "----------------------------------------"
./.venv/bin/python tools/eval_mamba_models.py \
    --model artifacts/lvm/models/mamba_sandwich/best.pt \
    --eval-npz "$EVAL_NPZ" \
    --payload "$PAYLOAD" \
    --faiss "$FAISS_INDEX" \
    --device "$DEVICE" \
    --out "artifacts/lvm/eval_mamba_sandwich_${TIMESTAMP}.json"

echo ""
echo "✅ Mamba-Sandwich evaluation complete!"
echo ""

# Model 4: Mamba-GR
echo "4/4 Evaluating Mamba-GR (SSM + GRU)..."
echo "----------------------------------------"
./.venv/bin/python tools/eval_mamba_models.py \
    --model artifacts/lvm/models/mamba_gr/best.pt \
    --eval-npz "$EVAL_NPZ" \
    --payload "$PAYLOAD" \
    --faiss "$FAISS_INDEX" \
    --device "$DEVICE" \
    --out "artifacts/lvm/eval_mamba_gr_${TIMESTAMP}.json"

echo ""
echo "✅ Mamba-GR evaluation complete!"
echo ""

echo "================================================================================
All Evaluations Complete!
================================================================================

Results saved to:
  artifacts/lvm/eval_mamba_s_${TIMESTAMP}.json
  artifacts/lvm/eval_mamba_h_${TIMESTAMP}.json
  artifacts/lvm/eval_mamba_sandwich_${TIMESTAMP}.json
  artifacts/lvm/eval_mamba_gr_${TIMESTAMP}.json

To view results:
  cat artifacts/lvm/eval_mamba_s_${TIMESTAMP}.json | jq '.'

To compare all results:
  python tools/compare_mamba_results.py --timestamp $TIMESTAMP

================================================================================
"
