# Positional Encoding Dimension Mismatch - Fixed

**Date**: 2025-10-31
**Issue**: Training crash when `--use-positional` enabled
**Status**: ✅ Fixed

---

## Problem

When positional encoding was enabled with `--use-positional`, training crashed with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x769 and 768x64)
```

**Root Cause**:
1. Positional encoding adds 1 dimension: input becomes 769D (768 + 1 positional scalar)
2. Models were created with `input_dim=769`
3. Models used `input_dim` for BOTH input projection AND output projection
4. Model outputs became 769D
5. Training targets are always 768D (semantic vector dimension)
6. Dimension mismatch in loss computation

---

## Solution

**Separate input and output dimensions**:
- **Input**: Can be 768 (no positional) or 769 (with positional)
- **Output**: Always 768 (target vector dimension)

### Changes Made

**1. Updated Model Architectures** (`app/lvm/models.py`):
- Added `output_dim=768` parameter to all model constructors
- Updated output projections to use `output_dim` instead of `input_dim`
- Models affected:
  - `LSTMBaseline`: Line 46
  - `GRUStack`: Line 115
  - `TransformerVectorPredictor`: Line 185
  - `AdaptiveMultiscaleNetwork`: Line 282

**2. Updated Model Config** (`app/lvm/train_unified.py`):
- Added `output_dim` parameter to `get_model_config()` (Line 287)
- Set `output_dim=768` for all architectures
- Models always output 768D vectors regardless of input dimension

**3. Updated Model Creation** (`app/lvm/train_unified.py`):
- Line 491: Compute `input_dim = 769 if args.use_positional else 768`
- Line 497: Pass both `input_dim` and `output_dim=768` to config

---

## How It Works Now

### Without Positional Encoding (Default)
```python
input_dim = 768
output_dim = 768

# Model flow:
input (B, 5, 768) → projection → ... → output projection → (B, 768)
```

### With Positional Encoding (`--use-positional`)
```python
input_dim = 769  # 768 + 1 positional scalar
output_dim = 768  # Always match target dimension

# Model flow:
input (B, 5, 768) → add positional → (B, 5, 769) → projection → ... → output projection → (B, 768)
```

**Key Insight**: Input and output dimensions are independent. Positional encoding affects INPUT only, outputs always match targets (768D).

---

## Testing

**Before Fix**:
```bash
./scripts/train_transformer_directional_v3.sh
# Result: RuntimeError on first batch
```

**After Fix**:
```bash
./scripts/train_transformer_directional_v3.sh
# Result: Training proceeds normally
# Model accepts 769D input, outputs 768D vectors
```

---

## Architecture Examples

### LSTM
```python
LSTMBaseline(input_dim=769, hidden_dim=512, output_dim=768)
# input_dim=769 for LSTM input projection
# output_dim=768 for final output projection
```

### Transformer
```python
TransformerVectorPredictor(input_dim=769, d_model=512, output_dim=768)
# input_dim=769 for input projection (line 190)
# output_dim=768 for head projection (line 210)
```

### AMN
```python
AdaptiveMultiscaleNetwork(input_dim=769, d_model=256, hidden_dim=512, output_dim=768)
# input_dim=769 for context/query encoders
# output_dim=768 for residual network output
```

---

## Important Notes

1. **Positional encoding is NOT scheduled**:
   - If `--use-positional` is set, model is created with `input_dim=769`
   - Positional encoding is applied to ALL batches (cannot be disabled per-epoch)
   - Only the loss weights (λ_dir, λ_ac) are scheduled

2. **Model parameter count changes**:
   - Without positional: ~17.87M params (input proj: 768→512)
   - With positional: ~17.88M params (input proj: 769→512)
   - Difference: 512 params (one extra weight per hidden unit)

3. **Backwards compatibility**:
   - Old models trained without positional encoding still work
   - They have `input_dim=output_dim=768`
   - New models with positional have `input_dim=769, output_dim=768`

---

## Additional Fix: Evaluation Consistency

**Second Issue**: After fixing the output dimension, training worked but crashed during validation:
```
RuntimeError: linear(): input and weight.T shapes cannot be multiplied (5x768 and 769x512)
```

**Root Cause**: Positional encoding was applied during training but NOT during evaluation.

**Solution**: Updated `evaluate()` function to accept `use_positional` and `pos_scale` parameters:
- Line 266: Updated function signature
- Line 285-290: Apply positional encoding if enabled (same as training)
- Line 616-618: Pass positional parameters to evaluate()

**Result**: Model receives consistent input dimensions during both training and evaluation.

---

## Files Modified

- `app/lvm/models.py` (4 model classes updated - output_dim separation)
- `app/lvm/train_unified.py` (model config, creation, and evaluation consistency)
- `artifacts/lvm/V3_DIRECTIONAL_GUARDRAILS_COMPLETE.md` (documentation)

---

---

## Third Fix: Directional Loss Dimension Mismatch

**Third Issue**: Training worked through epochs 1-3, then crashed in epoch 4 (when directional losses activated):
```
RuntimeError: The size of tensor a (768) must match the size of tensor b (769) at non-singleton dimension 1
```

**Root Cause**:
- Directional losses compare context vectors with target vectors
- Context was 769D (with positional encoding)
- Targets are always 768D
- Cannot compute cosine similarity between different dimensions

**Solution**: Save original 768D context before adding positional encoding, use for directional losses:
- Line 162: Save `contexts_orig` before positional augmentation
- Line 210: Use `contexts_orig[:, -2, :]` for previous vector (768D)
- Line 219: Use `contexts_orig` for anti-copy loss (768D)
- Line 236: Use `contexts_orig[:, -1, :]` for diagnostic margin (768D)

**Result**:
- Model receives 769D input (with positional information)
- Directional losses compare 768D vectors (matching target dimension)

---

**Status**: ✅ RESOLVED (3 fixes applied)
**Author**: Claude Code + User
**Date**: 2025-10-31
**Training**: Now proceeding through all epochs successfully
