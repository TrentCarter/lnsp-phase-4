#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
train_phase2.sh \
  --model memory_gru --context 500 --routing tmd16 \
  --alpha 0.05 --tau 0.07 [--negatives soft|hard --neg-lo 0.6 --neg-hi 0.8] \
  --lr 1e-4 --wd 1e-4 --sched cosine --warmup 1 \
  --batch 64 --accum 4 \
  --earlystop hit5 --patience 3 \
  --seqs data/gwom_sequences.jsonl --vecdb data/vector_index.npz \
  --save artifacts/lvm/models_phase2/run_500ctx_routingA
USAGE
}

# Defaults
MODEL=memory_gru
CTX=500
ROUTING=tmd16
ALPHA=0.03
TAU=0.07
NEG=none
NEG_LO=0.6
NEG_HI=0.8
LR=1e-4
WD=1e-4
SCHED=cosine
WARMUP=1
BATCH=64
ACCUM=4
EARLY=hit5
PATIENCE=3
SEQS=""
VECDB=""
SAVE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL=$2; shift 2;;
    --context) CTX=$2; shift 2;;
    --routing) ROUTING=$2; shift 2;;
    --alpha) ALPHA=$2; shift 2;;
    --tau) TAU=$2; shift 2;;
    --negatives) NEG=$2; shift 2;;
    --neg-lo) NEG_LO=$2; shift 2;;
    --neg-hi) NEG_HI=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --wd) WD=$2; shift 2;;
    --sched) SCHED=$2; shift 2;;
    --warmup) WARMUP=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --accum) ACCUM=$2; shift 2;;
    --earlystop) EARLY=$2; shift 2;;
    --patience) PATIENCE=$2; shift 2;;
    --seqs) SEQS=$2; shift 2;;
    --vecdb) VECDB=$2; shift 2;;
    --save) SAVE=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

[[ -z "$SEQS" || -z "$VECDB" || -z "$SAVE" ]] && { echo "Missing --seqs/--vecdb/--save"; exit 2; }
mkdir -p "$SAVE"

# Environment echoes for reproducibility
printf "\n[Train] model=%s ctx=%s routing=%s alpha=%s tau=%s neg=%s [%s,%s] lr=%s wd=%s batch=%s accum=%s early=%s patience=%s\n" \
  "$MODEL" "$CTX" "$ROUTING" "$ALPHA" "$TAU" "$NEG" "$NEG_LO" "$NEG_HI" "$LR" "$WD" "$BATCH" "$ACCUM" "$EARLY" "$PATIENCE"

# Call your Python trainer. The trainer must:
#  - L2-normalize targets & predictions
#  - Predict delta and reconstruct y_hat = l2(x + Δ̂)
#  - Implement mixed loss + optional InfoNCE with α, τ
#  - Early stop on Hit@5
#  - Save best_val_hit5.pt and training_history.json in $SAVE

python3 train.py \
  --model "$MODEL" --context "$CTX" --routing "$ROUTING" \
  --alpha "$ALPHA" --tau "$TAU" --negatives "$NEG" --neg-lo "$NEG_LO" --neg-hi "$NEG_HI" \
  --lr "$LR" --wd "$WD" --sched "$SCHED" --warmup "$WARMUP" \
  --batch "$BATCH" --grad-accum "$ACCUM" \
  --earlystop-metric "$EARLY" --patience "$PATIENCE" \
  --sequences "$SEQS" --vector-bank "$VECDB" \
  --save-dir "$SAVE"

# Optional: quick tail of metrics
[[ -f "$SAVE/training_history.json" ]] && jq '{epoch:.epoch, hit1:.metrics.val.hit1, hit5:.metrics.val.hit5, hit10:.metrics.val.hit10}' "$SAVE/training_history.json" || true
