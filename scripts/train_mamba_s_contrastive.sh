#!/bin/bash
set -euo pipefail

echo "================================================================================"
echo "Mamba-S Contrastive Training (Phase 5.1)"
echo "================================================================================"
echo "Fix: InfoNCE + projection head for generalization to unseen articles"
echo "Training data: artifacts/lvm/train_payload_aligned.npz (396k sequences)"
echo "Val data: artifacts/lvm/val_payload_aligned.npz (99k sequences)"
echo "Device: mps"
echo "Save dir: artifacts/lvm/models/mamba_s_contrastive"
echo "Log: logs/mamba_s_contrastive_$(date +%Y%m%d_%H%M%S).log"
echo "================================================================================"
echo ""
echo "Expected gates (from contractor):"
echo "  Eval cosine ≥ 0.50 by epoch 2 (was 0.22 with AR-only)"
echo "  Contain@50 ≥ 60%"
echo "  Eff@5 ≥ 0.68"
echo "  R@5 ≥ 40%"
echo "  P95 ≤ 1.45ms"
echo ""
echo "Starting training..."
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_contrastive.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --val-split 0.2 \
  --d-model 768 \
  --n-layers 8 \
  --d-state 128 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --batch-size 256 \
  --grad-accum-steps 4 \
  --lambda-con 0.7 \
  --lambda-ar 0.3 \
  --temperature 0.07 \
  --article-dropout 0.2 \
  --span-corruption 0.1 \
  --epochs 20 \
  --lr 1e-3 \
  --weight-decay 0.02 \
  --warmup-steps 1000 \
  --early-stop-patience 3 \
  --device mps \
  --save-dir artifacts/lvm/models/mamba_s_contrastive

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
echo "Model: artifacts/lvm/models/mamba_s_contrastive/best.pt"
echo ""
echo "Next steps:"
echo "  1. Check history.json for epoch 2 val_cosine (gate: ≥0.50)"
echo "  2. Run unified eval on 5.2k samples"
echo "  3. Verify gates: Contain@50≥60%, R@5≥40%, Eff@5≥0.68, P95≤1.45ms"
echo "  4. If passed, train Sandwich/H with same regimen"
echo "================================================================================"
