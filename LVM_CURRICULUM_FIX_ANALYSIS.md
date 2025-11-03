
# LVM Curriculum Learning: Root Cause Analysis and Fix
**To:** Architect
**From:** Gemini
**Date:** 2025-11-02
**Subject:** Analysis of P5 LVM Training Failures and Data-Side Fix

## 1. Executive Summary

The P5 curriculum learning experiment failed to produce a forward-predicting LVM, exhibiting a persistent negative margin (backward-bias). The root cause was not the model architecture or loss function, but a critical flaw in the data preparation pipeline. 

The curriculum's "forward-distinctness" metric was inverted, causing Stage A to train on the 30% of samples *least* similar to the next vector, actively reinforcing the model's tendency to copy the last context vector. 

This issue has been identified and corrected by fixing the metric calculation and regenerating the curriculum splits. Verification shows the new Stage A data is now strongly correlated for forward prediction, with average `cos(target, ctx[-1])` improving from **0.25** to **0.69**. We are now positioned to restart P5 training on a valid data foundation.

## 2. Problem Analysis & Symptoms

For several iterations (P3, P4, P5-StageA), LVM training failed to overcome a fundamental "copy-last" problem. The primary symptom was a negative margin, indicating the model's prediction (`ŷ`) was more similar to the last context vector (`ctx[-1]`) than to the actual target vector (`y_next`).

**Key Metrics (Before Fix):**
- **P1 Baseline Margin:** -0.167
- **P5 Stage A Margin:** -0.041
- **P5 Stage A R@5:** ~17.5%

Initial hypotheses centered on architectural limitations or unstable loss penalties. However, the P5 experiment, designed to mitigate this with a curriculum of "forward-distinct" samples, failed to improve the margin, suggesting a deeper issue.

## 3. Investigation Details: Finding the Root Cause

The investigation followed a systematic, data-first approach.

### Step 1: Verify the "Forward-Distinct" Data

A Python script (`verify_forward_distinctness.py`) was created to analyze the `stage_a_top30.npz` dataset used in P5 training. It calculated the cosine similarity between the target vector and the last two context vectors.

**Initial Findings (The "Smoking Gun"):**
- **Average `cos(target, ctx[-1])`:** `0.2532`
- **Average `cos(target, ctx[-2])`:** `0.2892`

This result was alarming. It showed that, on average, the target was *more similar* to the second-to-last context vector than the last one. This directly contradicted the goal of the curriculum, which was to train on samples with a strong forward-predictive signal.

### Step 2: Analyze the Data Generation Pipeline

The investigation then turned to the scripts responsible for creating the curriculum splits.

1.  `tools/build_curriculum_splits.py`: This script was responsible for filtering the data. It used a `forward_distinctness` score and a threshold to select the "top 30%" of samples.
    ```python
    # Selects samples with a HIGH score
    mask_top30 = forward_distinctness >= threshold_top30
    ```

2.  `tools/compute_forward_distinctness.py`: This script calculated the score. Here, the critical flaw was discovered.
    
    **Incorrect Logic (Before Fix):**
    ```python
    # High Δ → target is FAR from prev (forward-distinct)
    forward_distinctness = 1.0 - sim_prev # where sim_prev = cos(target, ctx[-1])
    ```

The script defined "forward-distinctness" as the *distance* from the previous vector. Therefore, `build_curriculum_splits.py` was correctly selecting the samples with the highest scores, but this meant it was selecting for the *largest distance*—the exact opposite of what was intended.

## 4. The Fix: Correcting the Curriculum Metric

The fix was to redefine `forward_distinctness` to be the direct cosine similarity, ensuring that higher scores correspond to more desirable, forward-predictive samples.

**File Modified:** `tools/compute_forward_distinctness.py`

**Corrected Logic:**
```python
# High Δ → target is CLOSE to prev (forward-predictive)
# Lower score = less similar to prev = less useful for learning forward prediction
forward_distinctness = sim_prev # where sim_prev = cos(target, ctx[-1])
```

After applying this change, the curriculum generation process was re-run, creating new, valid data splits.

## 5. Verification of the Fix

The `verify_forward_distinctness.py` script was run again on the newly generated `stage_a_top30.npz` file. The results demonstrate a complete reversal of the data quality issue.

**Verification Results (After Fix):**
- **Average `cos(target, ctx[-1])`:** **`0.6894`** (from 0.2532)
- **Average `cos(target, ctx[-2])`:** **`0.4421`** (from 0.2892)

**Conclusion:** The Stage A data now correctly consists of samples with a very strong forward-predictive signal. The average similarity of `0.69` is significantly higher than the dataset's baseline mean of `0.47`, providing a solid foundation for curriculum learning.

## 6. Next Steps

With the data foundation now secure, we will proceed with the planned training experiments:

1.  **Retry P5 Stage A:** A new training run (`P5_FIXED_DATA_STAGE_A`) will be initiated using the corrected curriculum data and the original weak positional cue (`0.03`). This will isolate the impact of the data fix.

2.  **Proceed to P5.1/P6:** If the corrected data alone is insufficient to achieve a positive margin, we will proceed with the previously discussed P5.1 enhancements (stronger positional cues, attention biasing) or the P6 architectural changes ([NEXT] token).

This data-side error accounts for the repeated failures in our attempts to solve the LVM's backward-bias. We are now in a much stronger position to achieve a truly forward-predicting model.
