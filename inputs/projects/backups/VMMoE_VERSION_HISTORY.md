# VMMoE Configuration Version History

## Version 1.0.0 - Baseline (August 13, 2025)
**File**: `Project_81325_v1p0_VMMoE_Stable.json`

### Configuration:
- **Epochs**: 5
- **Steps/epoch**: 20  
- **Total steps**: 100
- **Learning rate**: 1e-06
- **Batch size**: 4
- **Dataset**: 500/60,000 vectors (0.8%)
- **Validation split**: 10%

### Results:
- **Training time**: 53.1 seconds
- **Final train loss**: 0.09218
- **Best validation loss**: 0.09186
- **Checkpoints**: `best_model.pth`, `20250813T213753_SN000011_VMamba_epoch4.pth`

### Status: ✅ **SUCCESSFUL BASELINE**
First working VMMoE training configuration that fixed d_ap increasing issues.

---

## Version 1.1.0 - Production (August 14, 2025)
**File**: `Project_81425_v1p1_VMMoE_Stable.json`

### Configuration:
- **Epochs**: 50 (+900%)
- **Steps/epoch**: 500 (+2400%)
- **Total steps**: 25,000 (+24,900%)  
- **Learning rate**: 5e-05 (+4900%)
- **Batch size**: 16 (+300%)
- **Dataset**: 60,000/60,000 vectors (100%)
- **Validation split**: 5% (-50%)
- **Warmup steps**: 2,500 (+2400%)

### Improvements:
- 🚀 **250× more training steps** for better convergence
- 📈 **50× higher learning rate** for faster learning
- 💾 **120× more training data** (full dataset)
- 🔧 **Optimized batch size** for hardware utilization
- ⚡ **Proper warmup schedule** for training stability

### Expected Results:
- **Training time**: 2-3 hours
- **Better convergence**: More training steps + proper LR
- **Higher quality**: Full dataset utilization
- **More stable**: Larger batches + proper warmup

### Status: 🚀 **READY FOR PRODUCTION RUN**

---

## Usage Commands:

### Run v1.0.0 (Baseline):
```bash
./.venv/bin/python3 -m app.vmmoe.training.trainer --project_config inputs/projects/Project_81325_v1p0_VMMoE_Stable.json
```

### Run v1.1.0 (Production):
```bash
./.venv/bin/python3 -m app.vmmoe.training.trainer --project_config inputs/projects/Project_81425_v1p1_VMMoE_Stable.json
```

### Run with custom vector limit:
```bash
./.venv/bin/python3 -m app.vmmoe.training.trainer --project_config inputs/projects/Project_81425_v1p1_VMMoE_Stable.json --max_vectors 10000
```

---

## File Naming Convention:
`Project_[MMDDYY]_v[MAJOR]p[MINOR]_[NAME].json`

- **MMDDYY**: Creation date (081325, 081425)
- **vXpY**: Version (v1p0 = v1.0, v1p1 = v1.1)  
- **NAME**: Descriptive name

---

## Next Version Planning:

### Version 1.2.0 - Advanced (Future):
Potential improvements:
- Increase to 12+ layers
- Add MoE experts (1 → 16)
- Experiment with different optimizers
- Add early stopping
- Implement mixed precision training