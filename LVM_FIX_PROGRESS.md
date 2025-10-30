# LVM Fix Progress Log

**Date Started:** 2025-10-29

This file tracks the progress of fixing the LVM garbage output bug.

---

### Task: Phase 1, Task 1.1 - Moment Matching Test

**Objective:** Diagnose the distribution mismatch between the GTR-T5 encoder and the LVM model output.

**Status:** Complete

**Log:**
- **2025-10-29:** Created `tools/test_moment_matching.py` to perform the test.
- **2025-10-29:** Debugged and successfully executed the script.

**Findings:**
- The test confirms the hypothesis from `LVM_FIX_TASKLIST.md`.
- The LVM's pre-normalized output exhibits a significant loss of per-dimension variance compared to the GTR-T5 encoder's output, confirming a mode collapse problem.
- The L2-normalization step masks the overall standard deviation issue but does not restore the critical per-dimension variance required by the vec2text decoder.

---

### Task: Phase 1, Task 1.2 - Mean Vector Baseline

**Objective:** Calculate a baseline cosine similarity by comparing target vectors to the global mean vector.

**Status:** Complete

**Log:**
- **2025-10-29:** Created and executed `tools/test_mean_vector_baseline.py`.

**Findings:**
- The script calculated an average cosine similarity of **0.4247**.
- **Note:** This result significantly differs from the `~75-80%` baseline mentioned in `LVM_FIX_TASKLIST.md`. This suggests the documentation may be outdated. The current GRU model's performance (0.5754) is still notably better than this calculated baseline.

---

### Task: Phase 1, Task 1.3 - Decoder A/B/C Test

**Objective:** Test the decoder's sensitivity to different vector distributions.

**Status:** Complete

**Log:**
- **2025-10-29:** Created and executed `tools/test_decoder_distributions.py`.

**Findings:**
- **Test A (Encoder -> Decoder):** Good reconstruction, as expected.
- **Test B (L2-Normalized Encoder -> Decoder):** Surprisingly, also good reconstruction. This **contradicts** the hypothesis in `LVM_FIX_TASKLIST.md` and proves the decoder is robust to normalization.
- **Test C (LVM Pre-Normalized -> Decoder):** Produced gibberish.
- **Conclusion:** The root cause is confirmed to be **mode collapse** in the LVM model itself, not the final L2 normalization step. The model is producing semantically poor vectors.

---

### Task: Phase 2, Task 2.1 - Split Output Heads

**Objective:** Modify all LVM models to return both raw and normalized outputs.

**Status:** Complete

**Log:**
- **2025-10-29:** Modified the `forward` method in `GRUStack`, `LSTMStack`, `TransformerVectorPredictor`, and `AttentionMixtureNetwork` in `app/lvm/models.py` to include the `return_raw` flag and return both vector types.

---

### Task: Phase 2, Task 2.2 - Moment-Matching Loss

**Objective:** Add a moment-matching loss to the training loop to prevent variance collapse.

**Status:** Complete

**Log:**
- **2025-10-29:** Modified `app/lvm/loss_utils.py` to accept pre-computed target statistics.
- **2025-10-29:** Modified `app/lvm/train_unified.py` to compute and pass these statistics to the loss function.

---

### Task: Phase 2, Task 2.3 - InfoNCE Contrastive Loss

**Objective:** Replace MSE with InfoNCE as the primary loss function to better prevent mode collapse.

**Status:** Complete

**Log:**
- **2025-10-29:** Modified the default loss weights in `app/lvm/train_unified.py` to enable InfoNCE, moment-matching, and variance losses.

---

### Task: Phase 2, Task 2.4 - Hard Negatives in Batch

**Objective:** Update the dataset to include hard negatives for more effective contrastive learning.

**Status:** Complete

**Log:**
- **2025-10-29:** Modified `VectorSequenceDataset` in `app/lvm/train_unified.py` to sample hard negatives.
- **2025-10-29:** Updated `train_epoch` to pass hard negatives to the loss function.
- **2025-10-29:** Updated `_info_nce` and `compute_losses` in `app/lvm/loss_utils.py` to incorporate hard negatives.

---

### Debugging: Simplified Training Experiment

**Objective:** Verify the basic training pipeline by overfitting a simple model on a small dataset.

**Status:** Complete

**Log:**
- **2025-10-29:** Created `tools/train_simple_lstm.py` to train a single-layer LSTM with MSE loss on 1000 samples.
- **2025-10-29:** Successfully ran the script. The model overfit as expected (loss decreased, cosine similarity increased), confirming the training pipeline is working correctly.

**Next Action:** Train the simple LSTM model on the full dataset to check performance at scale.
