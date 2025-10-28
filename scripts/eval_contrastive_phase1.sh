#!/bin/bash
# Phase 1 Evaluation: Compare InfoNCE vs AR-only on leaked eval
# Goal: Prove contrastive learning improves retrieval (0% → X%)

set -euo pipefail

echo "================================================================================"
echo "Phase 1 Evaluation: InfoNCE vs AR-only (Leaked Eval Comparison)"
echo "================================================================================"
echo ""
echo "Goal: Prove contrastive learning improves retrieval"
echo "  AR-only baseline: 0% R@5 (memorizes, doesn't transfer)"
echo "  InfoNCE + AR: Expected 20-40% R@5 (learns semantics, transfers!)"
echo ""
echo "Eval set: LEAKED_EVAL_SETS/eval_v2_ready.npz (90% overlap)"
echo "Note: Using same leaked eval for both models (fair comparison)"
echo ""
echo "================================================================================"
echo ""

# Check files exist
echo "Checking files..."
if [ ! -f "artifacts/lvm/models/mamba_s_contrastive/best.pt" ]; then
    echo "❌ Contrastive model not found!"
    exit 1
fi

if [ ! -f "artifacts/lvm/models/mamba_s_poc/best.pt" ]; then
    echo "❌ AR-only (POC) model not found!"
    exit 1
fi

if [ ! -f "artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_ready.npz" ]; then
    echo "❌ Leaked eval set not found!"
    exit 1
fi

echo "✅ All files found"
echo ""

# Run contrastive eval
echo "================================================================================"
echo "1/2 Evaluating InfoNCE + AR (Contrastive)"
echo "================================================================================"
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --eval-npz artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_ready.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 7140 \
  --nprobe 64 \
  --out artifacts/lvm/phase1_contrastive_leaked_eval.json

echo ""
echo "✅ Contrastive eval complete"
echo ""

# Run AR-only eval
echo "================================================================================"
echo "2/2 Evaluating AR-only (POC Baseline)"
echo "================================================================================"
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
  --eval-npz artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_ready.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 7140 \
  --nprobe 64 \
  --out artifacts/lvm/phase1_ar_only_leaked_eval.json

echo ""
echo "✅ AR-only eval complete"
echo ""

# Compare results
echo "================================================================================"
echo "COMPARISON RESULTS"
echo "================================================================================"
echo ""

python3 << 'PYTHON'
import json

# Load results
with open('artifacts/lvm/phase1_contrastive_leaked_eval.json') as f:
    contrastive = json.load(f)

with open('artifacts/lvm/phase1_ar_only_leaked_eval.json') as f:
    ar_only = json.load(f)

# Extract metrics
def get_metrics(result):
    m = result.get('metrics', {})
    return {
        'r5': m.get('recall_5', 0) * 100,
        'r10': m.get('recall_10', 0) * 100,
        'contain50': m.get('contain_50', 0) * 100,
        'eff5': m.get('eff_5', 0),
    }

c_metrics = get_metrics(contrastive)
a_metrics = get_metrics(ar_only)

print("| Model | R@5 | R@10 | Contain@50 | Eff@5 | Interpretation |")
print("|-------|-----|------|------------|-------|----------------|")
print(f"| AR-only | {a_metrics['r5']:.1f}% | {a_metrics['r10']:.1f}% | {a_metrics['contain50']:.1f}% | {a_metrics['eff5']:.3f} | Memorizes episodes |")
print(f"| InfoNCE + AR | {c_metrics['r5']:.1f}% | {c_metrics['r10']:.1f}% | {c_metrics['contain50']:.1f}% | {c_metrics['eff5']:.3f} | Learns semantics |")
print("")

improvement = c_metrics['r5'] - a_metrics['r5']
print(f"📊 Improvement: {improvement:+.1f}pp R@5")
print("")

if c_metrics['r5'] > 10:
    print("✅ SUCCESS: Contrastive learning improves retrieval!")
    print("   InfoNCE prevents memorization and enables transfer")
    print("")
    print("Next step: Phase 2 (fresh Wikipedia articles for clean eval)")
elif c_metrics['r5'] > 5:
    print("⚠️  PARTIAL SUCCESS: Some improvement, but below target")
    print("   Consider triage: stronger contrastive (λ_con=0.85) or more epochs")
else:
    print("❌ FAILURE: Contrastive not helping")
    print("   Check: projection head, InfoNCE loss, batch negatives")
    
PYTHON

echo ""
echo "================================================================================"
echo "Phase 1 Evaluation Complete"
echo "================================================================================"
echo "Results saved to:"
echo "  - artifacts/lvm/phase1_contrastive_leaked_eval.json"
echo "  - artifacts/lvm/phase1_ar_only_leaked_eval.json"
echo ""
