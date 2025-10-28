#!/bin/bash
# Go/No-Go Commands for Epoch 4 Evaluation
# Run these commands IN ORDER after training completes

set -e

echo "========================================"
echo "EPOCH 4 GO/NO-GO CHECKLIST"
echo "========================================"
echo ""

# ============================================================
# STEP 1: Check Gate Metrics (PRIMARY GATES)
# ============================================================
echo "STEP 1: Checking Gate Metrics..."
echo "========================================"

if [ ! -f artifacts/lvm/eval_epoch4/metrics.json ]; then
    echo "❌ ERROR: metrics.json not found"
    echo "   Run evaluation first: ./scripts/eval_epoch4_pipeline.sh"
    exit 1
fi

# Extract metrics
R5=$(jq -r '.["R@5"]' artifacts/lvm/eval_epoch4/metrics.json)
MRR=$(jq -r '.MRR' artifacts/lvm/eval_epoch4/metrics.json)

echo "📊 Metrics:"
jq . artifacts/lvm/eval_epoch4/metrics.json
echo ""

# Check gates
PASS_R5=$(echo "$R5 >= 0.30" | bc -l)
PASS_MRR=$(echo "$MRR >= 0.20" | bc -l)

if [ "$PASS_R5" -eq 1 ] || [ "$PASS_MRR" -eq 1 ]; then
    echo "✅ PASS - Quality gates met!"
    echo "   R@5 = $(echo "$R5 * 100" | bc)%"
    echo "   MRR = $MRR"
    echo ""
    echo "🎯 Proceed to STEP 4 (Tag & Ship)"
    exit 0
fi

echo "⚠️  FAIL - Quality gates not met"
echo "   R@5 = $(echo "$R5 * 100" | bc)% (need >= 30%)"
echo "   MRR = $MRR (need >= 0.20)"
echo ""

# ============================================================
# STEP 2: Check if JUST SHORT (reranker might help)
# ============================================================
echo "STEP 2: Checking if reranker can help..."
echo "========================================"

JUST_SHORT=$(echo "$R5 >= 0.27 && $R5 < 0.30" | bc -l)

if [ "$JUST_SHORT" -eq 1 ]; then
    echo "📈 JUST SHORT (R@5 = 0.27-0.29) - Reranker recommended"
    echo ""
    echo "Run these commands:"
    echo ""
    echo "# 2a. Train reranker"
    echo "./.venv/bin/python tools/reranker_train.py \\"
    echo "  --hits artifacts/lvm/eval_epoch4/hits50_ep4.jsonl \\"
    echo "  --scores artifacts/lvm/eval_epoch4/scores_ep4.json \\"
    echo "  --epochs 10 \\"
    echo "  --hidden 128 \\"
    echo "  --out artifacts/lvm/reranker/mlp_v0.pt"
    echo ""
    echo "# 2b. Apply reranker"
    echo "./.venv/bin/python tools/reranker_apply.py \\"
    echo "  --hits artifacts/lvm/eval_epoch4/hits50_ep4.jsonl \\"
    echo "  --model artifacts/lvm/reranker/mlp_v0.pt \\"
    echo "  --out artifacts/lvm/eval_epoch4/reranked_ep4.jsonl"
    echo ""
    echo "# 2c. Recompute metrics"
    echo "./.venv/bin/python tools/compute_metrics.py \\"
    echo "  --hits artifacts/lvm/eval_epoch4/reranked_ep4.jsonl \\"
    echo "  --truth artifacts/lvm/eval_clean_disjoint.npz \\"
    echo "  --out artifacts/lvm/eval_epoch4/metrics_reranked_ep4.json"
    echo ""
    exit 0
fi

# ============================================================
# STEP 3: Significantly short - need Near-Miss Bank + Epoch 5
# ============================================================
echo "STEP 3: Significantly short - Near-Miss Bank recommended"
echo "========================================"
echo ""
echo "Run these commands:"
echo ""
echo "# 3a. Mine near-miss bank"
echo "./.venv/bin/python tools/mine_nearmiss_bank.py \\"
echo "  --pvec artifacts/eval/p_train_ep3.npy \\"
echo "  --qvec artifacts/eval/q_train_ep3.npy \\"
echo "  --index artifacts/lvm/eval_epoch4/train_index_ep3.faiss \\"
echo "  --topk 4096 \\"
echo "  --out artifacts/corpus/near_miss_bank_ep4.npy"
echo ""
echo "# 3b. Resume training (Epoch 5)"
echo "./.venv/bin/python app/lvm/train_twotower_fast.py \\"
echo "  --resume artifacts/lvm/models/twotower_fast/epoch4.pt \\"
echo "  --train-npz artifacts/lvm/train_clean_disjoint.npz \\"
echo "  --same-article-k 3 \\"
echo "  --nearmiss-bank artifacts/corpus/near_miss_bank_ep4.npy \\"
echo "  --p-cache-npy artifacts/eval/p_train_ep3.npy \\"
echo "  --epochs 1 \\"
echo "  --batch-size 256 \\"
echo "  --lr 5e-5 \\"
echo "  --device cpu \\"
echo "  --save-dir artifacts/lvm/models/twotower_ep5"
echo ""
echo "# 3c. Re-evaluate Epoch 5"
echo "# (Create eval_epoch5_pipeline.sh similar to epoch4)"
echo ""

# ============================================================
# STEP 4: Tag & Ship (Called manually after PASS)
# ============================================================
echo "========================================"
echo "STEP 4 (Manual): Tag & Ship"
echo "========================================"
echo ""
echo "After achieving PASS:"
echo ""
echo "# Tag retriever"
echo "git tag -a retriever-v0-ep4 -m \"Two-tower retriever (R@5=${R5_PCT}%, MRR=${MRR})\""
echo ""
echo "# If reranker used, tag it too"
echo "git tag -a reranker-v0 -m \"MLP reranker (boost R@5 by +XYpp)\""
echo ""
echo "# Wire into vecRAG"
echo "echo 'LNSP_RETRIEVER_MODEL=artifacts/lvm/models/twotower_fast/epoch4.pt' >> .env"
echo "echo 'LNSP_RERANKER_MODEL=artifacts/lvm/reranker/mlp_v0.pt' >> .env  # if used"
echo "./scripts/restart_vecrag.sh"
echo ""
