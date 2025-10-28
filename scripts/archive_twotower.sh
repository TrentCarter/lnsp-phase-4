#!/bin/bash
set -e

echo "============================================"
echo "ARCHIVING TWO-TOWER MAMBA CHECKPOINTS"
echo "============================================"
echo "Reason: Below v0 ship gates after Epoch 5"
echo "Decision: Ship baseline stack (AMN/GRU + reranker)"
echo "============================================"
echo ""

# Create archive directory with timestamp
ARCHIVE_DIR="artifacts/archive/twotower_mamba_2025-10-28"
mkdir -p "$ARCHIVE_DIR"

echo "1. Copying checkpoints..."
# Copy Epoch 3, 4, 5 checkpoints
if [ -f artifacts/lvm/models/twotower_fast/epoch3.pt ]; then
  cp artifacts/lvm/models/twotower_fast/epoch3.pt "$ARCHIVE_DIR/"
  echo "   ✅ Epoch 3 checkpoint (622 MB)"
fi

if [ -f artifacts/lvm/models/twotower_fast/epoch4.pt ]; then
  cp artifacts/lvm/models/twotower_fast/epoch4.pt "$ARCHIVE_DIR/"
  echo "   ✅ Epoch 4 checkpoint (622 MB)"
fi

if [ -f artifacts/lvm/models/twotower_polish/epoch5.pt ]; then
  cp artifacts/lvm/models/twotower_polish/epoch5.pt "$ARCHIVE_DIR/"
  echo "   ✅ Epoch 5 checkpoint (622 MB)"
fi

echo ""
echo "2. Copying evaluation results..."
# Copy evaluation metrics
if [ -f artifacts/lvm/eval_epoch4_samearticle/metrics.json ]; then
  cp artifacts/lvm/eval_epoch4_samearticle/metrics.json "$ARCHIVE_DIR/metrics_epoch4.json"
  echo "   ✅ Epoch 4 metrics"
fi

if [ -f artifacts/lvm/eval_epoch5_polish/metrics.json ]; then
  cp artifacts/lvm/eval_epoch5_polish/metrics.json "$ARCHIVE_DIR/metrics_epoch5.json"
  echo "   ✅ Epoch 5 metrics"
fi

echo ""
echo "3. Creating summary documentation..."
cat > "$ARCHIVE_DIR/SUMMARY.md" << 'EOF'
# Two-Tower Mamba Archive — October 28, 2025

## Decision Summary

**Status:** ❌ NO-GO for v0 ship
**Reason:** Below minimum thresholds after polish training (Epoch 5)
**Action:** Ship baseline retriever stack (AMN_v0 + GRU_v0 + vector reranker)

---

## Performance Results

### Epoch 4 (K=3 same-article negatives)
- **R@5:** 17.2% (target: ≥25%) - **6.8pp short**
- **Contain@50:** 76.8% (target: ≥82%) - **5.2pp short**
- **MRR:** 0.1185 (target: ≥0.20)
- **Training time:** 15.6 minutes on MPS
- **Eval setup:** Article-disjoint (train: 1061-4227, eval: 7637-7690)

### Epoch 5 (K=5 same-article negatives, polish training)
- **R@5:** 18.4% (target: ≥25%) - **6.6pp short**
- **Contain@50:** 76.6% (target: ≥82%) - **5.4pp short**
- **MRR:** 0.1181 (target: ≥0.20)
- **Training time:** 25 minutes on MPS
- **Improvement over Epoch 4:** +1.2pp on R@5, -0.2pp on Contain@50 (minimal)

### Kill-Switch Evaluation
**Criteria:** Contain@50 < 82% OR R@5 < 25% → STOP
**Result:** ❌ KILL-SWITCH TRIGGERED (both thresholds failed)

---

## Architecture Details

- **Type:** Two-Tower Retrieval (separate Q and P encoders)
- **Backbone:** Mamba-S with bidirectional processing
- **Loss:** InfoNCE with temperature τ=0.07
- **Parameters:** ~12M per tower
- **Training data:** 394k pairs with same-article + near-miss negatives

### Negative Sampling Strategy
- **Epoch 4:** K=3 same-article + K=1 near-miss (from FAISS mining)
- **Epoch 5:** K=5 same-article + K=1 near-miss (increased hard negatives)
- **Near-miss bank:** 394k entries from Epoch 3 FAISS mining

---

## Root Cause Analysis

**Problem:** Cross-article generalization failure on zero-shot article transfer

**Evidence:**
1. 76.6% containment → 23.4% of queries have NO gold answer in top-50
2. 18.4% R@5 → poor early precision despite decent containment
3. Minimal improvement from K=3→5 (+1.2pp) suggests capability gap, not data issue

**Hypothesis:** Mamba-S backbone insufficient for cross-article semantic transfer in retrieval tasks

**Validation:** Article-disjoint eval (train on 1061-4227, test on 7637-7690) shows model cannot generalize to unseen articles despite same-domain data

---

## What We Learned

### ✅ Successes
1. **Fast MPS training:** 15-25 minutes per epoch (vs 45-90 min expected)
2. **Infrastructure:** Complete two-tower pipeline working (Q/P encoding, FAISS mining, evaluation)
3. **Bug fixes:** Global ID tracking, same-article negatives, tensor shapes all correct
4. **Kill-switch enforcement:** Automated decision framework prevented bad ship

### ⚠️ Failures
1. **Cross-article retrieval:** Model cannot transfer learned representations to new articles
2. **Containment gap:** 5.4pp below threshold indicates fundamental limitation
3. **Polish training:** K=3→5 change yielded minimal improvement (+1.2pp R@5)

### 💡 Insights for Future Work
1. **Two-tower may need different architecture:** Consider Transformer or hybrid designs
2. **Evaluation strategy validated:** Article-disjoint eval correctly identified capability gap
3. **Baseline stack is proven:** AMN/GRU single-tower models achieve 63%+ OOD cosine
4. **Containment critical:** Must hit 82%+ for production confidence

---

## Checkpoints in This Archive

1. **epoch3.pt** (622 MB) - Pre-polish checkpoint (K=3 negatives, 394k samples)
2. **epoch4.pt** (622 MB) - Post-training checkpoint (R@5: 17.2%, Contain@50: 76.8%)
3. **epoch5.pt** (622 MB) - Polish training checkpoint (R@5: 18.4%, Contain@50: 76.6%)

All checkpoints include:
- Q-tower and P-tower state dicts
- Optimizer state
- Training metadata (epoch, loss, lr)

---

## Baseline Stack for v0 Ship

### Primary Model: AMN_v0
- **OOD Cosine:** 0.6375 (best generalization)
- **Latency:** 0.62 ms (fastest)
- **Memory:** 5.8 MB (smallest)

### Fallback Model: GRU_v0
- **In-Dist Cosine:** 0.5920 (best accuracy)
- **OOD Cosine:** 0.6295 (excellent generalization)
- **Latency:** 2.11 ms (acceptable for batch)

### Retriever
- **Embeddings:** GTR-T5 768D (vec2text-compatible)
- **Index:** FAISS IVF-Flat (cosine via IP on L2-norm)
- **Configuration:** per-lane nlist≈√N, nprobe=8

### Reranker
- **Type:** Vector-only MLP (2 layers)
- **Features:** cosine(q,p), margin vs best, per-article local context, diversity prior
- **Lift:** Expected 3-5pp improvement on R@5

---

## References

- **Release Documentation:** `docs/PROD/Release_v0_Retriever.md`
- **Model Cards:** `docs/ModelCards/{AMN_v0,GRU_v0}.md`
- **Training Code:** `app/lvm/train_twotower_fast.py`
- **Evaluation Code:** `/tmp/eval_epoch{4,5}_complete.sh`
- **Performance Leaderboard:** `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`

---

## Future Considerations

If revisiting two-tower retrieval in future releases:

1. **Alternative architectures:**
   - Transformer encoder (proven for cross-document tasks)
   - Hybrid Mamba+Attention (combine speed + global context)
   - Larger Mamba variants (Mamba-M or Mamba-L)

2. **Training improvements:**
   - Article-balanced sampling (ensure cross-article exposure)
   - Curriculum learning (easy→hard article transitions)
   - Multi-task learning (same-article + cross-article objectives)

3. **Evaluation rigor:**
   - Keep article-disjoint split (critical for generalization test)
   - Add intermediate checkpoints (diagnose where learning stalls)
   - Track per-article statistics (identify problematic article types)

---

## Archive Metadata

- **Date:** October 28, 2025
- **Total Size:** ~1.9 GB (3 checkpoints + metrics)
- **Retention:** Permanent (learning reference)
- **Owner:** Retrieval Platform
EOF

echo "   ✅ Created SUMMARY.md"

echo ""
echo "4. Creating restore script..."
cat > "$ARCHIVE_DIR/restore_checkpoint.sh" << 'EOF'
#!/bin/bash
# Restore a specific epoch checkpoint for analysis

EPOCH=${1:-5}
CHECKPOINT="epoch${EPOCH}.pt"

if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ Checkpoint not found: $CHECKPOINT"
  echo "Available checkpoints:"
  ls -lh epoch*.pt
  exit 1
fi

echo "Restoring $CHECKPOINT for analysis..."
cp "$CHECKPOINT" ../../lvm/models/twotower_restored.pt
echo "✅ Restored to artifacts/lvm/models/twotower_restored.pt"
echo ""
echo "Load in Python:"
echo "  checkpoint = torch.load('artifacts/lvm/models/twotower_restored.pt')"
echo "  q_tower.load_state_dict(checkpoint['q_tower'])"
echo "  p_tower.load_state_dict(checkpoint['p_tower'])"
EOF
chmod +x "$ARCHIVE_DIR/restore_checkpoint.sh"
echo "   ✅ Created restore_checkpoint.sh"

echo ""
echo "============================================"
echo "ARCHIVE COMPLETE"
echo "============================================"
echo "Location: $ARCHIVE_DIR"
echo ""
ls -lh "$ARCHIVE_DIR"
echo ""
echo "Total archive size:"
du -sh "$ARCHIVE_DIR"
echo ""
echo "✅ Two-tower Mamba checkpoints archived successfully"
echo "✅ Ready to ship v0 baseline stack"
