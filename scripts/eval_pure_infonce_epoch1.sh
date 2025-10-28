#!/bin/bash
# Automatic Epoch 1 Evaluation for Pure InfoNCE
# Run this after epoch 1 completes

set -euo pipefail

echo "================================================================================"
echo "Pure InfoNCE Epoch 1 Evaluation"
echo "================================================================================"
echo ""

# Check if epoch 1 completed
if [ ! -f artifacts/lvm/models/mamba_s_pure_infonce/best.pt ]; then
    echo "❌ Checkpoint not found! Training not complete."
    exit 1
fi

echo "✅ Checkpoint found"
echo ""

# 1. Check pos/neg separation
echo "================================================================================"
echo "1/3 Positive/Negative Cosine Separation"
echo "================================================================================"
echo ""

./.venv/bin/python tools/inspect_batch_cosines.py \
    --checkpoint artifacts/lvm/models/mamba_s_pure_infonce/best.pt \
    --sample 2048 \
    --out artifacts/lvm/epoch1_posneg_stats.json

echo ""
echo "✅ Pos/neg check complete"
echo ""

# 2. Run retrieval evaluation (IVF index, nprobe=64)
echo "================================================================================"
echo "2/3 Retrieval Evaluation (IVF, nprobe=64)"
echo "================================================================================"
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
    --checkpoint artifacts/lvm/models/mamba_s_pure_infonce/best.pt \
    --eval-npz artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_payload_aligned.npz \
    --payload artifacts/wikipedia_584k_payload.npy \
    --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
    --device cpu \
    --nprobe 64 \
    --out artifacts/lvm/pure_infonce_epoch1_ivf.json

echo ""
echo "✅ IVF evaluation complete"
echo ""

# 3. Decision gate check
echo "================================================================================"
echo "DECISION GATE"
echo "================================================================================"
echo ""

python3 << 'PYTHON'
import json

# Load results
with open('artifacts/lvm/pure_infonce_epoch1_ivf.json') as f:
    ivf_results = json.load(f)

with open('artifacts/lvm/epoch1_posneg_stats.json') as f:
    posneg = json.load(f)

# Extract metrics
r5_ivf = ivf_results.get('metrics', {}).get('recall_5', 0) * 100
separation = posneg['separation']['delta']
auc = posneg['separation']['auc']

print(f"Retrieval:")
print(f"  R@5 (IVF nprobe=64): {r5_ivf:.1f}%")
print()
print(f"Separation:")
print(f"  Δ (pos - neg): {separation:.4f}")
print(f"  AUC: {auc:.4f}")
print()

# Decision
if r5_ivf > 0:
    print("✅ GATE PASSED: Non-zero retrieval!")
    print(f"   R@5 = {r5_ivf:.1f}% > 0%")
    if r5_ivf >= 5:
        print("   → Strong signal, continue training to completion")
    else:
        print("   → Weak but positive, train 2 more epochs and re-evaluate")
elif separation >= 0.10:
    print("⚠️  MIXED: Zero retrieval but positive separation")
    print(f"   Δ = {separation:.4f} ≥ 0.10")
    print("   → Check with FLAT index (may be ANN issue)")
else:
    print("❌ GATE FAILED: Zero retrieval AND no separation")
    print(f"   R@5 = {r5_ivf:.1f}%")
    print(f"   Δ = {separation:.4f} < 0.10")
    print("   → Pivot to Two-Tower architecture (Option 4)")

PYTHON

echo ""
echo "================================================================================"
echo "Evaluation Complete"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - IVF retrieval: artifacts/lvm/pure_infonce_epoch1_ivf.json"
echo "  - Pos/neg stats: artifacts/lvm/epoch1_posneg_stats.json"
echo ""
