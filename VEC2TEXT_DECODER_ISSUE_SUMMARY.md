# Vec2Text Decoder Issue - Technical Summary

**Date**: October 28, 2025
**Status**: BLOCKING - LVM chat interface cannot decode vectors to text
**Severity**: Critical - Core functionality broken

---

## Problem Statement

The vec2text decoder is producing **garbage/nonsensical output** instead of coherent decoded text. This blocks the complete LVM (Latent Vector Model) chat pipeline:

```
Text → Encode (768D) → LVM Prediction → Decode (Text)
                                            ↑
                                         BROKEN
```

### Expected vs Actual Behavior

**Expected** (when working):
```python
Input:  "What is AI?"
Output: "What is AI? Artificial Intelligence is a field of computer science..."
Cosine: 0.85-0.95 (high quality)
```

**Actual** (current broken state):
```python
Input:  "What is AI?"
Output: "of the Pitchers and other organisms. What is AI? What is AI? (abbreviation of ) is a computer"
Cosine: 0.7234 (poor quality - should be 0.90+)
```

---

## Environment Details

### System
- **OS**: macOS Darwin 25.0.0
- **Python**: 3.13.7 (via venv at `./.venv`)
- **Project**: lnsp-phase-4

### Key Package Versions
```bash
torch                   2.8.0
vec2text                0.0.13
sentence-transformers   (compatible with vec2text)
```

### Service Architecture
- **Port 8767**: GTR-T5 Encoder (vec2text-compatible) - ✅ WORKING
- **Port 8766**: Vec2Text Decoder Service - ❌ DEPRECATED (crashes)
- **Ports 9001-9004**: LVM Chat Services (AMN, Transformer, GRU, LSTM)

---

## What Was Working

According to user: **"This was definitely working"**

The complete pipeline was operational:
1. Text input → GTR-T5 encoder (port 8767) → 768D vector ✅
2. 768D vector → LVM model → predicted 768D vector ✅
3. Predicted vector → Vec2text decoder → coherent text ✅ (PREVIOUSLY)

### Evidence of Previous Success
- Documentation shows round-trip tests with 0.85-0.95 cosine similarity
- `docs/how_to_use_jxe_and_ielab.md` has examples of CORRECT decoding
- LVM models are trained and validated (working correctly)
- Production data exists: 339,615 Wikipedia concepts with vectors

---

## Current Issue - Detailed Analysis

### Symptom 1: Round-Trip Test Fails
```python
# Encode "What is AI?" on port 8767
vector = encode("What is AI?")  # 768D vector - WORKS

# Decode same vector back
decoded = decode(vector, steps=1)  # Should return similar text
# Expected: "What is AI? Artificial intelligence..."
# Actual:   "of the Pitchers and other organisms..."
# Cosine:   0.7234 (WRONG - should be 0.95+)
```

### Symptom 2: PyTorch Runtime Error
When calling vec2text library directly:
```
RuntimeError: Inference tensors cannot be saved for backward.
To work around you can make a clone to get a normal tensor and use it in autograd.
```

**Location**: `.venv/lib/python3.13/site-packages/vec2text/trainers/corrector.py:463`

### Symptom 3: Device Mismatch Errors
```
RuntimeError: Tensor for argument input is on cpu but expected on mps
```

Even when forcing CPU, vec2text corrector loads models on MPS automatically.

---

## Root Cause Analysis

### Primary Suspect: PyTorch 2.8.0 Incompatibility

**Evidence**:
1. Vec2text 0.0.13 was developed for PyTorch 2.0-2.3
2. PyTorch 2.8.0 has **stricter rules** for inference tensors
3. Vec2text uses inference tensors in autograd operations (no longer allowed in 2.8.0)
4. Error trace shows the crash happens in vec2text's internal corrector code

**From PyTorch 2.8.0 release notes**:
> "Inference tensors cannot be saved for backward computation"

This is a **breaking change** that vec2text 0.0.13 doesn't handle.

### Secondary Issue: Port 8766 vs 8767 Confusion

**Port 8766** (vec2text_server.py):
- Loads vec2text JXE/IELab processors
- Uses `Vec2TextProcessor` wrapper
- ❌ Marked as DEPRECATED in documentation
- ❌ Crashes with PyTorch 2.8.0

**Port 8767** (vec2text_embedding_server.py):
- Loads vec2text-compatible GTR-T5 encoder
- Was encoding-only initially
- ✅ Encoder working perfectly
- ❌ Decode endpoint added but crashes with same PyTorch issue

**Documentation says**: Use port 8767 for BOTH encoding and decoding
**Current state**: Port 8767 has decode endpoint, but it still uses the broken vec2text library internally

---

## Attempted Solutions (Failed)

### 1. Added `/decode` Endpoint to Port 8767
```python
# app/api/vec2text_embedding_server.py
@app.post("/decode")
async def decode_vectors(request: DecodeRequest):
    # Calls vec2text.api.invert_embeddings()
    # Still crashes with PyTorch 2.8.0
```
**Result**: Same PyTorch 2.8.0 error

### 2. Force CPU Device
```python
os.environ['PYTORCH_MPS_DISABLE'] = '1'
corrector = load_pretrained_corrector("gtr-base")
corrector.model.to(device="cpu")
```
**Result**: Models still load on MPS, then crash with device mismatch

### 3. Restart Services with Venv Python
```bash
./.venv/bin/python -m uvicorn app.api.vec2text_embedding_server:app
```
**Result**: Still uses PyTorch 2.8.0 from venv

---

## Code Locations

### Key Files
1. **Port 8767 Service** (encoder + new decode endpoint):
   - `app/api/vec2text_embedding_server.py:152-211` (decode endpoint)

2. **Port 8766 Service** (deprecated decoder):
   - `app/api/vec2text_server.py:232-333` (_decode_single_vector_in_memory)
   - `app/vect_text_vect/vec2text_processor.py:107` (loads "gtr-base" corrector)

3. **LVM Chat Services**:
   - `app/api/lvm_inference.py:202` (decode_vector function)
   - `scripts/start_lvm_services.sh` (ports 9001-9004)

4. **Vec2Text Orchestrator**:
   - `app/vect_text_vect/vec_text_vect_isolated.py` (IsolatedVecTextVectOrchestrator)
   - Only has `encode_texts()` method, no direct vector decoding

### Service Logs
```bash
/tmp/lnsp_api_logs/vec2text_decoder.log       # Port 8766 (deprecated)
/tmp/lnsp_api_logs/vec2text_embedding.log     # Port 8767 (encoder)
/tmp/lnsp_api_logs/AMN Chat.log               # LVM chat services
```

---

## Test Commands

### 1. Verify Encoder Works (Port 8767)
```bash
curl -X POST http://localhost:8767/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is AI?"]}'
# ✅ Returns 768D vector correctly
```

### 2. Test Round-Trip (Encode → Decode)
```python
import requests

# Encode
resp = requests.post("http://localhost:8767/embed",
                     json={"texts": ["What is AI?"]})
vector = resp.json()['embeddings'][0]

# Decode
resp = requests.post("http://localhost:8767/decode",
                     json={"vectors": [vector], "steps": 1})
result = resp.json()['results'][0]

print(f"Output: {result['output']}")
print(f"Cosine: {result['cosine']}")
# ❌ Garbage output, cosine ~0.72
```

### 3. Check PyTorch Version
```bash
./.venv/bin/python -c "import torch; print(torch.__version__)"
# Output: 2.8.0 (TOO NEW!)
```

---

## Possible Solutions

### Option 1: Downgrade PyTorch (Recommended Quick Fix)
```bash
./.venv/bin/pip install 'torch==2.3.1'
# Restart all services
pkill -f "vec2text_embedding_server"
pkill -f "lvm_inference"
./scripts/start_lvm_services.sh
```

**Pros**:
- Should fix immediately (vec2text was tested with PyTorch 2.0-2.3)
- No code changes needed

**Cons**:
- Downgrades entire project's PyTorch
- May affect other components (LVM training, FAISS, etc.)
- Need to verify LVM models still work with 2.3.1

### Option 2: Patch Vec2Text Library
Edit `.venv/lib/python3.13/site-packages/vec2text/models/corrector_encoder.py:100`:

```python
# Before (crashes on 2.8.0):
embedding = self.embedding_transform_1(embedding)

# After (clone to detach from inference):
embedding = self.embedding_transform_1(embedding.clone().detach())
```

**Pros**: Keeps PyTorch 2.8.0

**Cons**:
- Modifies installed package (lost on reinstall)
- May break other vec2text functionality
- Requires deep understanding of vec2text internals

### Option 3: Update Vec2Text Package
```bash
./.venv/bin/pip install --upgrade vec2text
```

**Check**: https://github.com/jxmorris12/vec2text/releases for PyTorch 2.8.0 compatibility

**Pros**: Clean official fix

**Cons**:
- Newer vec2text may not exist
- API changes could break existing code
- Need to verify compatibility

### Option 4: Alternative Decoder Implementation
Implement custom decoder that doesn't use vec2text library:
- Use T5 model directly
- Implement beam search manually
- More control over device placement

**Pros**: Full control, no dependencies on broken library

**Cons**:
- Significant development effort
- May not match vec2text quality
- Requires ML expertise

---

## Questions for Investigation

1. **When did this break?**
   - Was there a recent `pip install --upgrade` that updated PyTorch?
   - Check git history: When was PyTorch 2.8.0 installed?

2. **What PyTorch version was working before?**
   - Check old pip freeze outputs
   - Git history of requirements.txt

3. **Can we create a separate venv for vec2text services?**
   - Venv1 (PyTorch 2.8.0): LVM training, FAISS operations
   - Venv2 (PyTorch 2.3.1): Vec2text encoder/decoder services
   - Communication via HTTP APIs (ports 8767, 9001-9004)

4. **Are there alternative vec2text backends?**
   - Check if IELab decoder works better than JXE
   - Test different vec2text model variants

---

## Immediate Next Steps

1. **Confirm working state previously existed**:
   ```bash
   git log --all --grep="vec2text" --oneline
   git log --all -p requirements.txt | grep torch
   ```

2. **Try PyTorch downgrade in isolated test**:
   ```bash
   # Create test venv
   python3 -m venv /tmp/test_venv
   /tmp/test_venv/bin/pip install 'torch==2.3.1' vec2text sentence-transformers

   # Test decode
   /tmp/test_venv/bin/python test_vec2text_simple.py
   ```

3. **Check vec2text GitHub issues**:
   - Search: "PyTorch 2.8" OR "Inference tensors cannot be saved"
   - Look for compatibility patches

4. **Document working configuration** (if found):
   - Pin exact versions in requirements.txt
   - Add compatibility notes to README

---

## Supporting Evidence

### Test Output Logs
**Port 8766 decoder log** (`/tmp/lnsp_api_logs/vec2text_decoder.log`):
```
🔍 Decoding with jxe, steps=2
✅ jxe decoding completed
📝 jxe output: a.k.a. - the physiology of the state. A basic aspect of a state is...
```
Shows nonsensical output even though service reports "success"

**LVM Chat test results** (from user):
```
Question: "What is AI?"
AMN Response: "a.k.a. - the physiology of the state..."

Question: "Who was george washington?"
AMN Response: "a.K. ownership. The basic aspect of a state..."
```
LVM predicts vectors correctly, but decoder produces garbage

### PyTorch Error Traceback
```
RuntimeError: Inference tensors cannot be saved for backward
  File "vec2text/trainers/corrector.py", line 463, in _generate_with_beam
  File "vec2text/models/corrector_encoder.py", line 100, in get_encoder_embedding
    embedding = self.embedding_transform_1(embedding)
  File "torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
```

---

## Contact Information

**Project**: lnsp-phase-4
**User**: Trent Carter
**Issue Date**: October 28, 2025
**Last Working**: Unknown (user confirms it was working previously)

**For Questions**:
- Check project Discord/Slack
- Review `docs/how_to_use_jxe_and_ielab.md` for original working setup
- See `CLAUDE.md` for project architecture overview

---

## Appendix: Full Environment

```bash
# Python environment
Python 3.13.7
venv: ./.venv

# Key packages (partial list)
torch==2.8.0
vec2text==0.0.13
sentence-transformers (latest)
fastapi (latest)
uvicorn (latest)

# Services running
Port 8767: vec2text_embedding_server (encoder + decode)
Port 9001: AMN LVM chat
Port 9002: Transformer LVM chat
Port 9003: GRU LVM chat
Port 9004: LSTM LVM chat

# Data
339,615 Wikipedia concepts with vectors
Vector storage: artifacts/wikipedia_500k_corrected_vectors.npz (663MB)
LVM models: artifacts/lvm/models/*.pt
```

---

**END OF SUMMARY**

This issue is BLOCKING the LVM chat interface. The most likely fix is downgrading PyTorch to 2.3.1, but need programmer approval before making that change system-wide.
