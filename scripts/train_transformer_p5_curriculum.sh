#!/usr/bin/env bash
set -euo pipefail

# P5: Curriculum + positional scalar; pure MSE until Stage C (optional tiny adaptive dir)
# Requires: tools/compute_forward_distinctness.py, tools/build_curriculum_splits.py

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

TRAIN_NPZ=${1:-artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz}
VAL_NPZ=${2:-artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz}
OOD_NPZ=${3:-artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz}
ART_NPZ=${4:-artifacts/wikipedia_584k_fresh.npz}
DEVICE=${DEVICE:-mps}

echo "[P5] Computing forward distinctness…"
$PY tools/compute_forward_distinctness.py --npz "$TRAIN_NPZ"

SCORES_NPZ="${TRAIN_NPZ%.*}_forward_scores.npz"

echo "[P5] Building curriculum splits…"
$PY tools/build_curriculum_splits.py --train-npz "$TRAIN_NPZ" --scores-npz "$SCORES_NPZ"

BASE_DIR="artifacts/lvm/models/transformer_p5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

POS=0.03

echo "[P5] Stage A (top30, epochs 1–4)…"
$PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 4 --batch-size 32 --device "$DEVICE" \
  --positional-scalar $POS \
  --curriculum forward_top_30 --curriculum-scores "$SCORES_NPZ" \
  --lambda-dir 0.0 --lambda-fut 0.0 --lambda-ac 0.0 \
  --output-dir "$BASE_DIR/stageA"

echo "[P5] 5CAT after Stage A…"
$PY $TEST --model "$BASE_DIR/stageA/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 2000 | tee "$BASE_DIR/stageA_5cat.json"

echo "[P5] Stage B (top70, epochs 5–10)…"
$PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 10 --batch-size 32 --device "$DEVICE" \
  --positional-scalar $POS \
  --curriculum forward_top_70 --curriculum-scores "$SCORES_NPZ" \
  --lambda-dir 0.0 --lambda-fut 0.0 --lambda-ac 0.0 \
  --resume "$BASE_DIR/stageA/best_model.pt" \
  --output-dir "$BASE_DIR/stageB"

echo "[P5] 5CAT after Stage B…"
$PY $TEST --model "$BASE_DIR/stageB/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 2000 | tee "$BASE_DIR/stageB_5cat.json"

echo "[P5] Stage C (full, epochs 11–20). Adaptive dir ON if needed…"
$PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 20 --batch-size 32 --device "$DEVICE" \
  --positional-scalar $POS \
  --curriculum full \
  --lambda-dir 0.002 --adaptive-dir --margin-dir 0.01 \
  --lambda-fut 0.0 --lambda-ac 0.0 \
  --resume "$BASE_DIR/stageB/best_model.pt" \
  --output-dir "$BASE_DIR/stageC"

echo "[P5] 5CAT after Stage C…"
$PY $TEST --model "$BASE_DIR/stageC/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 5000 | tee "$BASE_DIR/stageC_5cat.json"

echo "[P5] DONE → $BASE_DIR"
