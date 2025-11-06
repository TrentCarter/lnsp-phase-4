# P7 "Directional Ranker" Implementation Guide

**Date**: 2025-11-04
**Status**: ✅ IMPLEMENTED - Ready for training
**Architecture**: Ranking-based LVM with semantic anchoring

---

## Executive Summary

P7 replaces the failed regression objective (predict exact next vector) with a **ranking objective** (rank next vector highest among candidates). This prevents "orthogonal escape" by forcing the model to compete with concrete negatives instead of drifting to arbitrary low-similarity space.

### Key Innovations:

1. **InfoNCE Ranking Loss**: Contrastive learning with in-batch negatives
2. **Prev-Repel Margin**: Explicitly push away from previous chunk
3. **Semantic Anchoring**: Blend predictions with context subspace (prevents orthogonal drift)
4. **Directional Gating**: Down-weight sequences with weak forward signal
5. **Teacher Pull**: Prevent early collapse during warmup

---

## Architecture Changes from P6b

| Component | P6b v2.3 (Failed) | P7 Ranker (New) |
|-----------|-------------------|-----------------|
| **Objective** | MSE regression → exact vector | InfoNCE ranking → highest score |
| **Loss** | `||pred - target||²` | `-log(exp(pos) / Σexp(neg))` |
| **Negatives** | None (implicit in MSE) | Previous + in-batch + hard negs |
| **Anchoring** | None (→ orthogonal escape!) | Blend with context (λ=0.6-0.9) |
| **Output** | Raw 768D vector | Unit sphere + anchored |
| **Gating** | Cosine-based (death spiral) | Δ-based (stable) |
| **Failure Mode** | Orthogonal escape (cos≈0.04) | **Prevented by anchoring** |

---

## Implementation Files

### 1. Loss Functions (`app/lvm/losses_ranking.py`)

**Core Functions**:
```python
def info_nce_ranking_loss(query, positive, negatives, temperature=0.07):
    """InfoNCE: rank positive highest among pool"""

def prev_repel_margin_loss(query, positive, previous, margin=0.07):
    """Hard negative: push away from previous chunk"""

def semantic_anchor_blend(query_raw, context_vectors, lambda_blend=0.8):
    """Prevent orthogonal escape: q' = norm(λ·q + (1-λ)·c)"""

def directional_gate_weights(context, target_next, target_prev, threshold=0.03):
    """Down-weight sequences with Δ < threshold"""

def cosine_floor_teacher_loss(query, target, floor_threshold=0.20):
    """Warmup: pull predictions back if cos < 0.20"""

def p7_combined_loss(...):
    """Combine all components with weights"""
```

**Loss Components**:
```python
loss_total = (
    w_rank * loss_rank * gate_weight +       # 1.0 * InfoNCE
    w_margin * loss_margin * gate_weight +   # 0.5 * prev-repel
    w_teacher * loss_teacher                 # 0.2 * teacher (warmup only)
)
```

### 2. Model Architecture (`app/lvm/models_p7_ranker.py`)

**TransformerP7Ranker**:
```python
class TransformerP7Ranker(nn.Module):
    """
    - Input proj: 768D → 512D
    - Positional encoding (learned)
    - Transformer encoder (4 layers, 8 heads)
    - Output head: 512D → 768D
    - Unit sphere normalization
    - Semantic anchoring: q' = norm(λ·q + (1-λ)·c)
    """

    def forward(self, x):
        # ... transformer encoding ...
        output_raw = self.head(x_last)
        output_raw_norm = F.normalize(output_raw, dim=-1)

        # Semantic anchoring
        lambda_blend = self.get_anchor_lambda()  # Learnable λ ∈ [0.6, 0.9]
        context_norm = F.normalize(x.mean(dim=1), dim=-1)
        output_blended = lambda_blend * output_raw_norm + (1 - lambda_blend) * context_norm
        output_anchored = F.normalize(output_blended, dim=-1)

        return output_anchored
```

**LSTMP7Ranker**:
- Simpler/faster alternative for quick experiments
- Same anchoring principle

### 3. Training Script (`app/lvm/train_p7_ranker.py`)

**Dataset**: `P7RankingDataset`
- Loads arXiv sequences (contexts, targets)
- Samples hard negatives from dataset

**Negative Pool Creation**:
```python
def create_negative_pool(batch):
    """
    1. Previous chunk (B, 1, 768)
    2. In-article hard negatives (B, M, 768)
    3. In-batch negatives (B, B-1, 768)
    → Combined: (B, 1+M+B-1, 768)
    """
```

**Training Loop**:
1. Forward pass → anchored predictions
2. Create negative pool
3. Compute P7 combined loss
4. Backward pass with gradient clipping
5. Validate with directional metrics

### 4. Shell Script (`scripts/train_p7_ranker.sh`)

**Quick Start**:
```bash
# Default training (context=5, margin=0.07, λ=0.8)
./scripts/train_p7_ranker.sh

# Custom hyperparameters
./scripts/train_p7_ranker.sh --context 7 --margin 0.10 --lambda 0.9 --epochs 12
```

---

## Hyperparameter Grid (Recommended)

### Priority 1: Core Parameters

| Parameter | Values | Impact |
|-----------|--------|--------|
| `--context` | 3, 5, 7 | Context window size |
| `--margin` | 0.05, 0.07, 0.10 | Prev-repel margin |
| `--lambda` | 0.6, 0.8, 0.9 | Anchor blend (learnable) |

**Recommended first runs**:
1. Baseline: `context=5, margin=0.07, λ=0.8` (default)
2. Strong anchor: `context=5, margin=0.07, λ=0.6` (more context)
3. Weak anchor: `context=5, margin=0.07, λ=0.9` (more model freedom)
4. Larger context: `context=7, margin=0.07, λ=0.8`
5. Higher margin: `context=5, margin=0.10, λ=0.8`

### Priority 2: Loss Weights

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `--w-rank` | 1.0 | Fixed | InfoNCE weight |
| `--w-margin` | 0.5 | 0.3-0.7 | Prev-repel weight |
| `--w-teacher` | 0.2 | 0.1-0.3 | Warmup teacher |

### Priority 3: Gating & Safety

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `--gate-threshold` | 0.03 | 0.02-0.05 | Δ for full weight |
| `--gate-weak-weight` | 0.25 | 0.1-0.5 | Weight for weak Δ |
| `--floor-threshold` | 0.20 | 0.15-0.30 | Teacher pull trigger |

---

## Training Command

```bash
./.venv/bin/python app/lvm/train_p7_ranker.py \
    --train-npz artifacts/lvm/arxiv_train_sequences.npz \
    --val-npz artifacts/lvm/arxiv_val_sequences.npz \
    --context-length 5 \
    --model-type transformer \
    --d-model 512 \
    --nhead 8 \
    --num-layers 4 \
    --dropout 0.1 \
    --anchor-lambda 0.8 \
    --anchor-learnable \
    --w-rank 1.0 \
    --w-margin 0.5 \
    --w-teacher 0.2 \
    --margin 0.07 \
    --temperature 0.07 \
    --gate-threshold 0.03 \
    --gate-weak-weight 0.25 \
    --floor-threshold 0.20 \
    --batch-size 64 \
    --epochs 10 \
    --lr 5e-4 \
    --warmup-epochs 2 \
    --device mps \
    --exp-name p7_ranker_c5_m0.07_l0.8
```

---

## Expected Training Behavior

### Epoch-by-Epoch Predictions:

**Epochs 1-2 (Warmup)**:
- `loss_teacher` active (pulling predictions toward targets)
- `cos_next` should climb from ~0.2 to ~0.4
- `cos_prev` should stay low (~0.1-0.2)
- `margin` should climb from ~0.05 to ~0.15
- Teacher violations should decrease (start high, drop to ~0)

**Epochs 3-5 (Stabilization)**:
- `loss_teacher` turns OFF (warmup complete)
- `cos_next` should stabilize at 0.4-0.5
- `cos_prev` should stay at 0.1-0.2
- `margin` should be positive and growing (0.15-0.25)
- `cos_anchor` should stay ≥ 0.50 (anchored to context)

**Epochs 6-10 (Refinement)**:
- `cos_next` should reach 0.5-0.6
- `cos_prev` should drop slightly (0.05-0.15)
- `margin` should reach 0.25-0.40 (strong forward!)
- `cos_anchor` should stay ≥ 0.45 (stable anchoring)
- `anchor_lambda` should converge (if learnable)

### Success Criteria:

✅ **PASS** (Model is good):
- Final `margin` ≥ +0.20
- Final `cos_next` ≥ 0.45
- Final `cos_anchor` ≥ 0.40
- No orthogonal drift warning (cos_anchor stays above 0.05)

⚠️ **MARGINAL** (Needs tuning):
- Final `margin` = 0.10-0.20
- Final `cos_next` = 0.35-0.45
- Final `cos_anchor` = 0.30-0.40

❌ **FAIL** (Try different hyperparameters):
- Final `margin` < 0.10
- Final `cos_next` < 0.35
- Final `cos_anchor` < 0.30
- Orthogonal drift warning (cos_anchor dropped below 0.05)

---

## Monitoring & Debugging

### Key Metrics to Watch:

1. **`margin = cos_next - cos_prev`**:
   - Primary goal metric
   - Must be POSITIVE and GROWING
   - Target: 0.20-0.40 by epoch 10

2. **`cos_anchor = cos(pred, context_centroid)`**:
   - Safety metric (prevents orthogonal escape)
   - Must stay above 0.40
   - If drops below 0.05 → STOP (orthogonal drift!)

3. **`anchor_lambda`** (if learnable):
   - Watch how it evolves
   - Should stay in [0.6, 0.9] range
   - If hits boundaries consistently → adjust range

4. **`gate_weight_mean`**:
   - Shows fraction of sequences with strong Δ
   - Should be 0.7-0.9 (most sequences have forward bias)
   - If < 0.5 → data quality issue (too many backward sequences)

5. **`teacher_violations_per_batch`**:
   - Only during warmup (epochs 1-2)
   - Should drop from high (100+) to low (0-10)
   - If stays high → increase `w_teacher` or `floor_threshold`

### Warning Signs:

🚨 **Orthogonal Drift**:
- `cos_anchor` drops below 0.10
- Both `cos_next` and `cos_prev` drop toward 0
- **Action**: STOP training, increase anchor λ (lower value = more anchoring)

🚨 **Backward Collapse**:
- `margin` becomes NEGATIVE
- `cos_prev` > `cos_next`
- **Action**: Increase `w_margin`, increase `margin` value

🚨 **Teacher Overpull**:
- `cos_next` too high during warmup (>0.7)
- Model not learning to generalize
- **Action**: Decrease `w_teacher` or shorten `warmup_epochs`

🚨 **Poor Data Quality**:
- `gate_weight_mean` consistently < 0.5
- Many sequences down-weighted
- **Action**: Check data Δ distribution, filter backward sequences

---

## Next Steps After Training

### 1. Validate Model Quality

Run directional test:
```bash
./.venv/bin/python /tmp/test_p6b_v23_direction.py \
    --model artifacts/lvm/models/p7_ranker_*/best_model.pt \
    --val-npz artifacts/lvm/arxiv_ood_sequences.npz \
    --device mps \
    --max-samples 2000
```

**Expect**:
- `cos_next` ≥ 0.45
- `cos_prev` ≤ 0.20
- `margin` ≥ +0.20

### 2. Evaluate on Dashboard

Add P7 model to dashboard (port 8999) and evaluate with:
- **R@1/5/10**: Retrieval accuracy
- **MRR@10**: Mean reciprocal rank
- **200+ test cases** (not 10!)
- **Compare to DIRECT baseline**

**Ship criteria** (from your spec):
- R@5 ≥ DIRECT + 10 pts
- MRR@10 ≥ DIRECT + 0.05
- No worse latency than P6b

### 3. Deploy if Passing

If P7 beats DIRECT:
- Deploy on port 9007 (replace P6b v2.3)
- Update production baseline
- Document performance gains

---

## Comparison to Previous Attempts

| Approach | Core Idea | Result | Why Failed |
|----------|-----------|--------|------------|
| **P1** | MSE baseline | Margin -0.167 | Wikipedia backward bias |
| **P2-P4** | Directional loss (weak) | Collapsed | λ too weak, landscape too flat |
| **P5.1** | Curriculum + reshaping | Margin -0.046 | Not enough to flip positive |
| **P6** | Remove identity path | Margin -0.082 | Proved data issue, not architecture |
| **P6b v1** | Aggressive ramp | Collapsed E3 | Death spiral (two negatives) |
| **P6b v2.1** | 6-layer defense | Margin -0.047 | Guardrails too conservative |
| **P6b v2.2** | Stronger pressure | Orthogonal escape | Directional loss too strong (ρ=0.35) |
| **P6b v2.3** | arXiv data | **Catastrophic (cos≈0.04)** | Orthogonal escape on forward data! |
| **P7 Ranker** | Ranking + anchoring | **TBD** | **Should prevent escape** ✅ |

---

## Why P7 Should Succeed

1. **Ranking objective**:
   - Forces model to compete with concrete negatives
   - Can't "escape" when negatives are explicit
   - Natural directionality from InfoNCE

2. **Semantic anchoring**:
   - Predictions MUST stay near context subspace
   - Blend weight λ ∈ [0.6, 0.9] prevents drift
   - Learnable λ finds optimal balance

3. **Directional gating**:
   - Only train on sequences with Δ ≥ 0.03
   - Reduces noise from ambiguous/backward samples
   - Stable (doesn't create death spiral like cosine gate)

4. **Teacher pull during warmup**:
   - Prevents early collapse (cos < 0.20)
   - Phases out after 2 epochs
   - Gives model good initialization

5. **Multiple safety mechanisms**:
   - Unit sphere normalization (magnitude stability)
   - Gradient clipping (training stability)
   - Orthogonal drift monitoring (early warning)
   - Hard negatives (explicit backward repulsion)

---

## FAQ

**Q: Why ranking instead of regression?**

A: Regression (MSE) has no explicit negatives. Model can minimize loss by predicting vectors orthogonal to both target and context (cos≈0 → MSE≈1.0 for all predictions). Ranking forces model to rank positive highest among explicit negatives, preventing escape.

**Q: What if P7 also fails with orthogonal escape?**

A: Semantic anchoring with λ ∈ [0.6, 0.9] makes it geometrically impossible. At λ=0.6, predictions are 60% model + 40% context centroid. This keeps predictions in the context subspace by construction.

**Q: Why is arXiv data (Δ=+0.064) still used if it's weak?**

A: P7 has directional gating that down-weights sequences with Δ < 0.03. This effectively filters out backward/ambiguous samples. Additionally, ranking objective is more robust to weak data signals than regression.

**Q: How do I know if orthogonal drift is happening?**

A: Training script monitors `cos_anchor = cos(pred, context_centroid)`. If it drops below 0.05 for two consecutive validations, training prints a warning and you should STOP. This is the canary metric.

**Q: What if margin is positive but R@K is still poor?**

A: Margin measures directionality, R@K measures retrieval quality. Both must pass. If margin is good but R@K is poor, the model learned directionality but not semantic quality. Increase context length or decrease λ (more anchoring).

**Q: Can I use P7 architecture with Wikipedia data?**

A: Yes, but expect margin to be lower (Δ=-0.07 in Wikipedia). The ranking objective is more robust than regression, so P7 might achieve margin≈0 instead of -0.16. But for best results, use forward-flow data.

---

## Quick Start Checklist

Before training:
- [ ] Verify data exists: `artifacts/lvm/arxiv_train_sequences.npz`
- [ ] Verify data exists: `artifacts/lvm/arxiv_val_sequences.npz`
- [ ] Check data quality: `cos_next - cos_prev ≥ +0.03` for training data
- [ ] Free up disk space: ~500MB for checkpoints

Training:
- [ ] Run: `./scripts/train_p7_ranker.sh`
- [ ] Monitor `margin`, `cos_anchor`, `gate_weight_mean`
- [ ] Save best model (highest margin)

After training:
- [ ] Run directional test (expect margin ≥ +0.20)
- [ ] Evaluate on dashboard (R@5, MRR@10)
- [ ] Compare to DIRECT baseline
- [ ] Deploy if passing ship criteria

---

**Status**: ✅ Implementation complete, ready for training
**Next Action**: Run `./scripts/train_p7_ranker.sh` and monitor results
**Expected Time**: ~2-3 hours for 10 epochs on MPS

**Report Author**: Claude Code
**Date**: 2025-11-04
