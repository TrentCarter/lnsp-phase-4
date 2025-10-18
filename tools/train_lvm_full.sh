#!/bin/bash
# Full LVM-T training with corrected vectors

cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

./.venv/bin/python app/lvm/train_transformer.py \
  --data artifacts/lvm/training_sequences_ctx5.npz \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 4 \
  --device mps \
  --output-dir artifacts/lvm/models/transformer_corrected_80k \
  --tau 0.07
