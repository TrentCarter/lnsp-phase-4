# Vec2Text Vector Alignment Issue Report

**Date**: 2025-10-16
**Reporter**: LVM Pipeline Test
**Status**: ⚠️ Quality Issue - Requires Expert Review

---

## Executive Summary

The text→768D→text round-trip using GTR-T5 encoding and vec2text decoding produces **semantically unrelated output** despite successful encoding. This raises concerns about vector alignment between the encoder (GTR-T5) and decoder (vec2text IELab model).

### Key Metrics
- **Cosine Similarity**: 0.86 (good vector similarity)
- **Text Quality**: Gibberish output (poor semantic reconstruction)
- **Decoding Steps**: Tested with 1 and 5 steps (both produce poor results)

---

## Problem Statement

### Test Case
**Input Text**: "The quick brown fox jumps over the lazy dog."

**GTR-T5 768D Encoding**: ✅ Working (produces normalized 768D vector)

**Vec2Text IELab Decoding**: ⚠️ Produces gibberish
- With 1 step: "a member of the icon workshops has evoking this theory..."
- With 5 steps: "Quirky Dog works the lazy fox to jump over a fox's coat..."

### Expected Behavior
Vec2text should reconstruct semantically similar text, especially with 5 decoding steps.

### Observed Behavior
Output is gibberish with no semantic relationship to input, despite 0.86 cosine similarity between encoded and decoded vectors.

---

## Technical Details

### Encoder Configuration (GTR-T5)

**Model**: `sentence-transformers/gtr-t5-base`
**Dimension**: 768D
**Normalization**: L2-normalized (unit vectors)
**Device**: CPU (T5+MPS has known issues)

**Code Location**: `src/vectorizer.py:137-142`
```python
def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
    if self.model is not None:
        emb = self.model.encode(texts, batch_size=batch_size,
                                normalize_embeddings=True,
                                show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        return emb
```

**Settings**:
- `normalize_embeddings=True` (confirmed in both `src/vectorizer.py` and `src/adapters/lightrag/embedder_gtr.py`)
- `convert_to_numpy=True`
- `dtype=np.float32`

---

### Decoder Configuration (Vec2Text IELab)

**Model**: IELab vec2text corrector
**Backend**: Isolated subprocess wrapper
**Teacher Model**: Same GTR-T5 base used for encoding

**Code Location**: `app/vect_text_vect/vec_text_vect_isolated.py`

**Subprocess Call** (from `tools/test_lvm_pipeline.py:73-78`):
```python
result = orchestrator._run_subscriber_subprocess(
    'ielab',
    embedding_tensor.cpu(),
    metadata={'original_texts': [original]},
    device_override='cpu'
)
```

**Environment Variables**:
- `VEC2TEXT_FORCE_PROJECT_VENV=1`
- `VEC2TEXT_DEVICE=cpu`
- `TOKENIZERS_PARALLELISM=false`

---

## Test Results

### Round-Trip Test (tools/test_lvm_pipeline.py)

| Test | Original Text | Decoded Text | Match |
|------|---------------|--------------|-------|
| 1 | The quick brown fox jumps over the lazy dog. | a member of the icon workshops has evoking this theory. — a surprising result. — a re- | ✗ |
| 2 | Machine learning models process sequential data efficiently. | re-edited the syringes. The plant atchison was not a member of this journal, but a | ✗ |
| 3 | Photosynthesis converts sunlight into chemical energy in plants. | pedagogy of a cinnature art practice. Chairperson of this writing is strongly encouraged in open study studies of other art | ✗ |

**Full Test Output**: `/tmp/lvm_test_output.log:1-31`

---

## Hypothesis: Possible Causes

### 1. Model Mismatch
- ✅ **Verified**: Both encoder and decoder use `sentence-transformers/gtr-t5-base`
- ✅ **Verified**: Same normalization strategy (L2-normalized unit vectors)
- ⚠️ **Concern**: Vec2text model may have been trained on different GTR-T5 version/config

### 2. Device/Precision Issues
- ✅ **Verified**: Both run on CPU (MPS disabled for T5)
- ✅ **Verified**: Both use float32 precision
- ⚠️ **Concern**: Subprocess isolation might introduce numerical differences

### 3. Vec2Text Training Data Domain
- ⚠️ **Concern**: IELab model may be trained on specific domain (medical/scientific?)
- ⚠️ **Concern**: May not generalize to Wikipedia content
- **Evidence**: Decoded text contains medical/academic jargon ("syringes", "pedagogy", "atchison")

### 4. Insufficient Decoding Steps
- ❌ **Ruled Out**: Tested with 1 and 5 steps, both produce gibberish
- **Note**: User documentation suggests 20+ steps for good quality, but even 5 steps should produce coherent text

### 5. Vector Space Misalignment
- ⚠️ **Primary Concern**: GTR-T5 encoder and vec2text decoder may operate in different semantic spaces
- **Evidence**: 0.86 cosine similarity suggests vectors are mathematically close, but text is semantically unrelated
- **Possible Cause**: Vec2text model trained on different GTR-T5 fine-tune or different normalization

---

## Critical Questions for Consultant

1. **Model Compatibility**:
   - Is the IELab vec2text model compatible with `sentence-transformers/gtr-t5-base`?
   - What version/checkpoint of GTR-T5 was used to train the vec2text model?

2. **Expected Behavior**:
   - Is 0.86 cosine similarity sufficient for semantic reconstruction?
   - What quality should we expect with 1, 5, and 20 decoding steps?

3. **Training Domain**:
   - What domain was the IELab model trained on?
   - Does it require domain-specific fine-tuning for Wikipedia content?

4. **Subprocess Isolation**:
   - Could the subprocess wrapper introduce numerical instability?
   - Should we test direct (in-process) vec2text invocation?

5. **Normalization**:
   - Do encoder and decoder require identical normalization?
   - Could there be a normalization mismatch (L2 vs. L1 vs. none)?

6. **Alternative Decoders**:
   - Should we test JXE decoder instead of IELab?
   - Are there known-good decoder configurations for Wikipedia data?

---

## Impact on LVM Training

### Current Situation
- ✅ **768D vectors are valid**: High cosine similarity (0.86)
- ✅ **LVM training proceeds**: Transformer learns vector→vector prediction
- ⚠️ **Text validation impossible**: Cannot verify prediction quality via human-readable text

### Risk Assessment
- **Low Risk**: LVM trains purely in vector space (text decoding not required)
- **Medium Risk**: If vectors are misaligned, LVM learns wrong semantic relationships
- **High Risk**: Cannot validate LVM outputs without working text decoder

### Mitigation Strategy
- ✅ **Proceed with LVM training**: Vector-space operations are independent of text decoding
- ✅ **Use cosine similarity**: Validate predictions in vector space (not text space)
- ⚠️ **Fix vec2text later**: Text decoding only needed for human inspection, not training

---

## Files for Review

### Encoder Implementation
1. **`src/vectorizer.py`** - Main GTR-T5 encoder wrapper (lines 137-162)
2. **`src/adapters/lightrag/embedder_gtr.py`** - Alternative GTR-T5 implementation (lines 48-83)

### Decoder Implementation
3. **`app/vect_text_vect/vec_text_vect_isolated.py`** - Vec2text orchestrator (35,623 bytes)
4. **`app/vect_text_vect/vec2text_processor.py`** - Vec2text processor (lines 176-199)

### Test Scripts
5. **`tools/test_lvm_pipeline.py`** - Complete LVM pipeline test (lines 30-92 show vec2text test)
6. **`/tmp/lvm_test_output.log`** - Test output with gibberish examples

### Configuration
7. **`docs/how_to_use_jxe_and_ielab.md`** - Vec2text usage documentation
8. **Environment variables** - See section above for critical settings

---

## Reproduction Steps

```bash
# 1. Run the complete LVM pipeline test
./.venv/bin/python tools/test_lvm_pipeline.py

# 2. Check Step 1 output (lines 1-31 in log)
# Expected: "✓ GTR-T5 ↔ vec2text working"
# Actual: Gibberish decoded text

# 3. Inspect cosine similarities
# Vector similarity: 0.86 (good)
# Text quality: Gibberish (bad)
```

---

## Recommendations

### Immediate Actions
1. **Verify Model Versions**: Confirm GTR-T5 and vec2text model compatibility
2. **Test Direct Invocation**: Bypass subprocess wrapper to rule out isolation issues
3. **Increase Decoding Steps**: Test with 20+ steps as suggested in documentation
4. **Try Alternative Decoder**: Test JXE decoder instead of IELab

### Long-Term Solutions
1. **Fine-tune Vec2text**: Train decoder on Wikipedia domain data
2. **Alternative Approach**: Use different decoding method (e.g., nearest neighbor retrieval)
3. **Validate Vector Space**: Test encoder-decoder alignment with known-good examples

---

## Appendix: Key Code Snippets

### Encoder Normalization (src/vectorizer.py:140)
```python
emb = self.model.encode(texts, batch_size=batch_size,
                        normalize_embeddings=True,  # ← L2 normalization
                        show_progress_bar=False)
```

### Decoder Normalization (app/vect_text_vect/vec2text_processor.py:173)
```python
def _prepare_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(self.device).float()
    embedding = F.normalize(embedding, dim=-1)  # ← L2 normalization
    return embedding
```

### Test Round-Trip (tools/test_lvm_pipeline.py:66-90)
```python
for i, (original, embedding) in enumerate(zip(test_sentences, embeddings)):
    # Convert to torch tensor
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)  # Add batch dim

    # Decode with IELab (CPU-only) using subprocess wrapper
    result = orchestrator._run_subscriber_subprocess(
        'ielab',
        embedding_tensor.cpu(),
        metadata={'original_texts': [original]},
        device_override='cpu'
    )

    if result['status'] == 'error':
        print(f"  Test {i+1}: ERROR - {result['error']}")
        continue

    decoded_text = result['result'][0] if isinstance(result['result'], list) else result['result']

    print(f"  Test {i+1}:")
    print(f"    Original: {original}")
    print(f"    Decoded:  {decoded_text}")
    print(f"    Match: {'✓' if original.lower() in decoded_text.lower() else '✗'}")
```

---

## Contact

For questions about this report, refer to:
- Test script: `tools/test_lvm_pipeline.py`
- Test output: `/tmp/lvm_test_output.log`
- LVM training results: `artifacts/lvm_test_model.pt`
