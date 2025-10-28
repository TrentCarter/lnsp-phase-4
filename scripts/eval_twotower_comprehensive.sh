#!/bin/bash
set -e

# Comprehensive Two-Tower Evaluation
# Includes: diagnostics, oracle check, separation check, retrieval, scoring

CHECKPOINT="${CHECKPOINT:-artifacts/lvm/models/twotower_mamba_s/epoch2.pt}"
EVAL_NPZ_DEFAULT="artifacts/lvm/eval_clean_disjoint.npz"
EVAL_NPZ="$EVAL_NPZ_DEFAULT"
DEVICE="${DEVICE:-cpu}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--eval-npz PATH] [--checkpoint PATH] [--device DEVICE]

Defaults:
  --checkpoint    $CHECKPOINT
  --eval-npz      $EVAL_NPZ_DEFAULT
  --device        $DEVICE

Override --eval-npz explicitly if you need to run against a different split
(for example a quarantined leaked eval set).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --eval-npz)
      shift
      if [ $# -eq 0 ]; then
        echo "Missing value for --eval-npz" >&2
        exit 1
      fi
      EVAL_NPZ="$1"
      ;;
    --checkpoint)
      shift
      if [ $# -eq 0 ]; then
        echo "Missing value for --checkpoint" >&2
        exit 1
      fi
      CHECKPOINT="$1"
      ;;
    --device)
      shift
      if [ $# -eq 0 ]; then
        echo "Missing value for --device" >&2
        exit 1
      fi
      DEVICE="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

echo "================================================================================"
echo "TWO-TOWER EPOCH 1 COMPREHENSIVE EVALUATION"
echo "================================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Eval NPZ: $EVAL_NPZ"
echo "Device: $DEVICE"
echo "================================================================================"
echo ""

if [ ! -f "$EVAL_NPZ" ]; then
    echo "❌ Eval NPZ not found: $EVAL_NPZ"
    exit 1
fi

if [[ "$EVAL_NPZ" == *"LEAKED_EVAL_SETS"* ]]; then
    echo "⚠️  Using quarantined eval data. Results must be labeled accordingly."
    echo ""
fi

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "================================================================================"
echo "STEP 1: DIAGNOSTICS"
echo "================================================================================"
echo ""

# 1a. Separation check
echo "1a. Checking cosine separation on val mini-batch..."
./.venv/bin/python tools/check_twotower_separation.py \
    --checkpoint "$CHECKPOINT" \
    --eval-npz "$EVAL_NPZ" \
    --n-samples 256 \
    --device "$DEVICE"
echo ""

echo "================================================================================"
echo "STEP 2: EMIT VECTORS"
echo "================================================================================"
echo ""

# 2a. Emit payload vectors (P tower)
echo "2a. Emitting payload vectors (P tower)..."
./.venv/bin/python tools/emit_payload_vectors.py \
    --checkpoint "$CHECKPOINT" \
    --npz "$EVAL_NPZ" \
    --out artifacts/eval/p_ep1.npy \
    --device "$DEVICE"
echo ""

# 2b. Oracle check on P index
echo "2b. Oracle check: query P index with its own vectors..."
./.venv/bin/python tools/oracle_check_p_index.py \
    --payload-vectors artifacts/eval/p_ep1.npy \
    --n-samples 500
echo ""

# 2c. Emit query vectors (Q tower)
echo "2c. Emitting query vectors (Q tower)..."
./.venv/bin/python tools/emit_query_vectors.py \
    --checkpoint "$CHECKPOINT" \
    --eval-npz "$EVAL_NPZ" \
    --out artifacts/eval/q_ep1.npy \
    --device "$DEVICE"
echo ""

echo "================================================================================"
echo "STEP 3: RETRIEVAL (FLAT INDEX)"
echo "================================================================================"
echo ""

./.venv/bin/python tools/retrieve_twotower.py \
    --payload-vectors artifacts/eval/p_ep1.npy \
    --queries artifacts/eval/q_ep1.npy \
    --topk 50 \
    --metric ip \
    --out artifacts/eval/flat_hits_ep1.jsonl
echo ""

echo "================================================================================"
echo "STEP 4: SCORING & GATE CHECK"
echo "================================================================================"
echo ""

./.venv/bin/python tools/score_twotower_retrieval.py \
    --hits artifacts/eval/flat_hits_ep1.jsonl \
    --out artifacts/eval/flat_scores_ep1.json
echo ""

echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo "Results: artifacts/eval/flat_scores_ep1.json"
echo "================================================================================"
echo ""

# Display gate decision
echo "================================================================================"
echo "GATE DECISION"
echo "================================================================================"
R5=$(cat artifacts/eval/flat_scores_ep1.json | grep -oP '"R@5":\s*\K[0-9.]+')
echo "R@5 = $R5%"
echo ""

if (( $(echo "$R5 > 5.0" | bc -l) )); then
    echo "✅ GATE PASSED: R@5 > 5%"
    echo ""
    echo "Next steps:"
    echo "1. Continue to epoch 2"
    echo "2. Run clean Phase-2 eval on held-out articles"
    echo "3. Target: R@5 ≥ 30%, Contain@50 ≥ 50%, MRR ≥ 0.20"
else
    echo "❌ GATE FAILED: R@5 ≤ 5%"
    echo ""
    echo "Fallback A: Check normalization + metric"
    echo "  - Verify both towers output unit-norm vectors"
    echo "  - Ensure FAISS uses inner product (IP)"
    echo ""
    echo "Fallback B: P-regression to corpus space"
    echo "  - Add regression term: || normalize(p̂) - normalize(p_target) ||²"
    echo "  - Weight: 0.1 (start small)"
    echo "  - Only P sees gradient"
fi

echo "================================================================================"
