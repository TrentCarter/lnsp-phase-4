#!/bin/bash
set -e

# Two-Tower Epoch 1 Evaluation
# Gate: R@5 > 5% on leaked eval set with FLAT index

CHECKPOINT="artifacts/lvm/models/twotower_mamba_s/epoch1.pt"
EVAL_NPZ="artifacts/lvm/eval_v2_payload_aligned.npz"
DEVICE="${DEVICE:-cpu}"

echo "================================================================================"
echo "TWO-TOWER EPOCH 1 GATE CHECK"
echo "================================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Eval NPZ: $EVAL_NPZ"
echo "Device: $DEVICE"
echo "Gate: R@5 > 5% (FLAT index, leaked eval)"
echo "================================================================================"
echo ""

# Step 1: Emit payload vectors (P tower)
echo "1/4 Emitting payload vectors (P tower)..."
./.venv/bin/python tools/emit_payload_vectors.py \
    --checkpoint "$CHECKPOINT" \
    --npz "$EVAL_NPZ" \
    --out artifacts/eval/p_ep1.npy \
    --device "$DEVICE"
echo ""

# Step 2: Emit query vectors (Q tower)
echo "2/4 Emitting query vectors (Q tower)..."
./.venv/bin/python tools/emit_query_vectors.py \
    --checkpoint "$CHECKPOINT" \
    --eval-npz "$EVAL_NPZ" \
    --out artifacts/eval/q_ep1.npy \
    --device "$DEVICE"
echo ""

# Step 3: Retrieve (FLAT index, IP metric)
echo "3/4 Retrieving with FLAT index..."
./.venv/bin/python tools/retrieve_twotower.py \
    --payload-vectors artifacts/eval/p_ep1.npy \
    --queries artifacts/eval/q_ep1.npy \
    --topk 50 \
    --metric ip \
    --out artifacts/eval/flat_hits_ep1.jsonl
echo ""

# Step 4: Score
echo "4/4 Scoring retrieval..."
./.venv/bin/python tools/score_twotower_retrieval.py \
    --hits artifacts/eval/flat_hits_ep1.jsonl \
    --out artifacts/eval/flat_scores_ep1.json
echo ""

echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo "Results: artifacts/eval/flat_scores_ep1.json"
echo "================================================================================"
