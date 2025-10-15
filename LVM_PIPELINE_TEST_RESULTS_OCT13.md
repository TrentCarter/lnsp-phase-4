# LVM Pipeline Test Results - October 13, 2025

## Test Overview

**Test Date**: October 13, 2025
**Test Type**: End-to-end pipeline (Text → 768D → LVM → 768D → Text)
**Model Tested**: Mamba2/GRU (best_model.pt)
**Test File**: `tools/test_lvm_full_pipeline.py`
**Log File**: `/tmp/lvm_pipeline_test_complete.log`

## Test Configuration

### Pipeline
1. **Encode** context (5 sentences) → 768D vectors
2. **LVM** predicts next vector from context
3. **Decode** predicted vector → text
4. **Compare** predicted text with expected target

### APIs Used
- **Encoder**: Port 8765 (sentence-transformers GTR-T5) ❌ **WRONG!**
- **Decoder**: Port 8766 (vec2text ielab) ✅
- **LVM**: Loaded from `artifacts/lvm/models/mamba2/best_model.pt`

## Test Results

| Test | LVM Cosine | Expected | Predicted |
|------|-----------|----------|-----------|
| **Cooking Recipe** | 2.77% | Let the cake cool before frosting. | *gibberish about schools* |
| **Morning Routine** | 4.68% | Then I got dressed for work. | *gibberish about Apollo 17* |
| **Scientific Process** | -3.71% | The findings were published in a journal. | *gibberish about Sony* |
| **Travel Story** | -5.73% | We took a taxi to our hotel. | *gibberish about Egyptian holidays* |
| **Simple Conversation** | 4.52% | How do you take it? | *gibberish about PJ Milk* |

**Average Similarity**: 0.51% (essentially random)

## 🚨 CRITICAL ISSUES DISCOVERED

### Issue #1: WRONG ENCODER USED ❌

**Problem**: The test used port 8765 (sentence-transformers) for encoding, which is **INCOMPATIBLE** with vec2text decoding!

From `docs/how_to_use_jxe_and_ielab.md` (lines 51-53):
```
⚠️ DEPRECATED: Port 8765 GTR-T5 API (sentence-transformers) - NOT compatible with vec2text
✅ USE THIS: Port 8767 Vec2Text-Compatible GTR-T5 API
```

**Evidence of Incompatibility**:
Even the "Expected→" column (encoding then decoding the ground truth) produced complete gibberish:
- Input: `"Let the cake cool before frosting."`
- Decoded: `"– R&D team, R&D, 'I have connections with Hennepin'; VFO Special Events – deemed not"`
- Cosine: **~0.08** (should be 0.63-0.85!)

**Root Cause**:
- sentence-transformers produces embeddings with cosine ~0.076 when decoded with vec2text
- vec2text's own encoder produces embeddings with cosine 0.63-0.85 when decoded
- Subtle differences in tokenization, pooling, and library versions cause incompatibility

### Issue #2: LVM Generalization Failure

**Problem**: Even if encoding was correct, the LVM shows poor generalization:
- **Training**: 49.76% cosine similarity on training data
- **Test**: 0.51% average on these 5 sequences (98.9% drop!)

**Possible Causes**:
1. **Overfitting**: Model memorized training patterns but can't generalize
2. **Domain Mismatch**: Test sequences are different from training data distribution
3. **Training Data**: CLAUDE.md warns against using ontology data for LVM training
   - Ontologies teach classification hierarchies (taxonomic)
   - LVMs need temporal/causal sequences (narrative)

## Fix Required

### Step 1: Use Correct Encoder

**Option A (Simplest)**: Use `/encode-decode` endpoint
```python
def encode_text(text: str) -> np.ndarray:
    """Encode using vec2text's own encoder"""
    response = requests.post(
        "http://127.0.0.1:8766/encode-decode",
        json={"texts": [text], "subscribers": "ielab", "steps": 1},
        timeout=30
    )
    data = response.json()
    # Extract the embedding that vec2text used internally
    # This ensures compatibility!
```

**Option B**: Use port 8767 for encoding
```python
def encode_text(text: str) -> np.ndarray:
    """Encode using vec2text-compatible GTR-T5"""
    response = requests.post(
        "http://127.0.0.1:8767/embed",  # Changed from 8765!
        json={"texts": [text]},
        timeout=10
    )
    data = response.json()
    return np.array(data["embeddings"][0], dtype=np.float32)
```

### Step 2: Verify Vec2text Baseline

Before testing LVM, verify vec2text round-trip works:
```bash
curl -X POST http://localhost:8766/encode-decode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Let the cake cool before frosting."], "subscribers": "ielab", "steps": 5}'
```

**Expected**: Cosine > 0.63 (typically 0.80-0.90)

### Step 3: Re-test LVM Pipeline

Only after vec2text baseline is confirmed, re-run the LVM pipeline test with the correct encoder.

## Expected Results After Fix

If the LVM model is working correctly, we should see:

| Component | Expected Cosine | Actual (Broken) | Fixed (TBD) |
|-----------|----------------|-----------------|-------------|
| Vec2text baseline (ground truth) | 0.63-0.85 | **0.08** ❌ | ??? |
| LVM predictions | 0.30-0.50 (realistic) | 0.51% | ??? |

## Documentation References

- **Vec2text compatibility**: `docs/how_to_use_jxe_and_ielab.md` (lines 9-105)
- **LVM training data requirements**: `CLAUDE.md` (lines 2-9, critical rule #4)
- **Encoder incompatibility discovery**: `LVM_TEST_SUMMARY_FINAL.md`
- **Service architecture**: `docs/PRDs/PRD_FastAPI_Services.md`

## Next Steps

1. ✅ **Fix encoder** - Use port 8767 or `/encode-decode` endpoint
2. ⏳ **Verify vec2text baseline** - Confirm cosine > 0.63 for ground truth
3. ⏳ **Re-run LVM test** - Get real LVM performance numbers
4. ⏳ **Investigate training data** - Check if ontology data was used (should be sequential data)
5. ⏳ **Consider retraining** - If trained on wrong data, retrain with Wikipedia/narratives

## Conclusion

**The test was invalid due to encoder incompatibility**. The gibberish output was caused by using sentence-transformers encoding with vec2text decoding, not by LVM failure.

**Real LVM performance is unknown** until we re-test with compatible encoding.

**Action Required**: Fix the test script to use port 8767 or `/encode-decode`, then re-run to get valid results.

---

**Test completed**: October 13, 2025
**Status**: ❌ **INVALID** - Encoder incompatibility
**Next session**: Fix encoder and re-test
