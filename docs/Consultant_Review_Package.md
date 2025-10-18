# Consultant Review Package: Vec2Text Alignment Issue

**Date**: 2025-10-16
**Issue**: Text→768D→Text round-trip produces gibberish despite good vector similarity

---

## Quick Summary

**Problem**: GTR-T5 encoding works perfectly (768D vectors), but vec2text decoding produces semantically unrelated gibberish.

**Metrics**:
- Cosine similarity: 0.86 (good)
- Text quality: Gibberish (bad)
- Example: "The quick brown fox" → [768D vector] → "a member of the icon workshops has evoking this theory..."

**Impact**: Cannot validate LVM predictions via human-readable text (but vector-space training still works)

---

## Required Files for Review

### 1. Main Report
**Location**: `docs/Vec2Text_Alignment_Issue_Report.md`
**Purpose**: Comprehensive analysis of the issue with test results, hypotheses, and questions

### 2. Test Evidence
**Location**: `/tmp/lvm_test_output.log`
**Purpose**: Complete test output showing gibberish decoded text
**Key Lines**: 18-31 (round-trip test results)

### 3. Test Script
**Location**: `tools/test_lvm_pipeline.py`
**Purpose**: Reproducible test showing the issue
**Key Lines**:
- Lines 30-92: GTR-T5 ↔ vec2text round-trip test
- Lines 66-90: Actual decoding calls

### 4. Encoder Implementation
**Location**: `src/vectorizer.py`
**Purpose**: GTR-T5 encoder configuration
**Key Lines**: 137-162 (encode method with normalize_embeddings=True)

**Alternative Location**: `src/adapters/lightrag/embedder_gtr.py`
**Purpose**: Secondary GTR-T5 implementation
**Key Lines**: 48-83 (embed_batch method)

### 5. Decoder Implementation
**Location**: `app/vect_text_vect/vec_text_vect_isolated.py` (35KB file)
**Purpose**: Vec2text orchestrator with subprocess isolation
**Note**: This is the main vec2text wrapper

**Location**: `app/vect_text_vect/vec2text_processor.py`
**Purpose**: Vec2text processor with inversion logic
**Key Lines**: 176-199 (_run_inversion method)

### 6. Configuration Documentation
**Location**: `docs/how_to_use_jxe_and_ielab.md`
**Purpose**: Official vec2text usage guide
**Note**: Documents JXE vs IELab decoders and expected behavior

---

## Reproduction Command

```bash
# Run the complete test (takes ~5 minutes)
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./.venv/bin/python tools/test_lvm_pipeline.py

# View results
cat /tmp/lvm_test_output.log
```

Expected output:
- Step 1: Shows gibberish decoded text (lines 18-31)
- Step 2: Extracts 200 chunks from database
- Step 3: Trains LVM transformer (50 epochs)
- Step 4: Tests generative prediction (0.1267 cosine similarity)

---

## Key Technical Details

### Encoder Configuration
- Model: `sentence-transformers/gtr-t5-base`
- Normalization: L2 (unit vectors)
- Device: CPU (T5+MPS has known issues)
- Precision: float32

### Decoder Configuration
- Model: IELab vec2text corrector
- Teacher: Same GTR-T5 base
- Device: CPU
- Steps tested: 1 and 5 (both produce gibberish)
- Environment: Subprocess isolation with `VEC2TEXT_FORCE_PROJECT_VENV=1`

### Test Results
| Input | Output (5 steps) | Quality |
|-------|------------------|---------|
| "The quick brown fox jumps over the lazy dog." | "a member of the icon workshops has evoking this theory..." | ✗ Gibberish |
| "Machine learning models process sequential data efficiently." | "re-edited the syringes. The plant atchison was not a member..." | ✗ Gibberish |
| "Photosynthesis converts sunlight into chemical energy in plants." | "pedagogy of a cinnature art practice. Chairperson of this writing..." | ✗ Gibberish |

---

## Questions for Consultant

1. **Is the IELab vec2text model compatible with `sentence-transformers/gtr-t5-base`?**
   - What GTR-T5 version/checkpoint was used for training?

2. **Is 0.86 cosine similarity sufficient for semantic reconstruction?**
   - What quality should we expect with 1, 5, and 20 steps?

3. **Was IELab trained on a specific domain?**
   - Does it require fine-tuning for Wikipedia content?

4. **Could subprocess isolation introduce numerical issues?**
   - Should we test direct (in-process) invocation?

5. **Is there a normalization mismatch?**
   - Both use L2 normalization, but could there be subtle differences?

6. **Should we use JXE decoder instead of IELab?**
   - Are there known-good configurations for Wikipedia data?

---

## Current Status

✅ **LVM training proceeding**: Vector-space operations work correctly (don't require text decoding)

⚠️ **Text validation blocked**: Cannot verify predictions via human-readable text

🔍 **Investigation needed**: Consultant review to determine root cause and solution

---

## File Tree

```
/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/
├── docs/
│   ├── Vec2Text_Alignment_Issue_Report.md  ← Main report
│   ├── Consultant_Review_Package.md        ← This file
│   └── how_to_use_jxe_and_ielab.md         ← Vec2text docs
├── tools/
│   └── test_lvm_pipeline.py                ← Test script
├── src/
│   ├── vectorizer.py                       ← GTR-T5 encoder
│   └── adapters/lightrag/
│       └── embedder_gtr.py                 ← Alt GTR-T5 impl
├── app/vect_text_vect/
│   ├── vec_text_vect_isolated.py           ← Vec2text orchestrator
│   └── vec2text_processor.py               ← Vec2text processor
└── /tmp/
    └── lvm_test_output.log                 ← Test output
```

---

## Next Steps

1. **Consultant reviews files** (see "Required Files for Review" above)
2. **Consultant identifies root cause** (see "Questions for Consultant" above)
3. **Implement fix** (may require model update, config change, or fine-tuning)
4. **Re-run test** to verify fix
5. **Resume LVM training** with working text validation

---

## Contact Information

For questions about this package:
- Main report: `docs/Vec2Text_Alignment_Issue_Report.md`
- Test script: `tools/test_lvm_pipeline.py`
- Test output: `/tmp/lvm_test_output.log`
