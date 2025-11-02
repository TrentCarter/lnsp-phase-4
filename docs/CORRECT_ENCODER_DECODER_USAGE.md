# CORRECT Encoder/Decoder Usage

**Date**: 2025-10-31
**Status**: ✅ PRODUCTION STANDARD

---

## 🔴 CRITICAL: Port 8766 is NOT compatible with Port 8767

Despite both claiming to use "GTR-T5", the decoder on port 8766 produces **gibberish** when given vectors from the encoder on port 8767 (cosine similarity ~0.05, nearly orthogonal).

---

## ✅ PRODUCTION SERVICES: Ports 7001 (Encode) and 7002 (Decode)

**NEW**: FastAPI services that expose the CORRECT orchestrator-based encode/decode functionality.

### Starting the Services

```bash
# Start encoder on port 7001
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app --host 127.0.0.1 --port 7001 &

# Start decoder on port 7002
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app --host 127.0.0.1 --port 7002 &

# Check health
curl http://localhost:7001/health
curl http://localhost:7002/health
```

### Usage Example (HTTP API)

```python
import requests

text = "The Eiffel Tower was built in 1889."

# Step 1: Encode via port 7001
encode_resp = requests.post("http://localhost:7001/encode", json={"texts": [text]})
vector = encode_resp.json()["embeddings"][0]

# Step 2: Decode via port 7002
decode_resp = requests.post("http://localhost:7002/decode", json={
    "vectors": [vector],
    "subscriber": "ielab",
    "steps": 5,
    "original_texts": [text]
})
decoded_text = decode_resp.json()["results"][0]

print(f"Original: {text}")
print(f"Decoded:  {decoded_text}")
# Expected: 80-100% keyword matches, meaningful output
```

**Result**: 80-100% keyword matches with semantically meaningful output.

---

## ✅ CORRECT Method: Use IsolatedVecTextVectOrchestrator (Direct Python)

### Full Working Example

```python
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import torch

# Initialize orchestrator (do this ONCE per session)
orchestrator = IsolatedVecTextVectOrchestrator(steps=5, debug=False)

# Example text
text = "The Eiffel Tower was built in 1889."

# Step 1: Encode text → 768D vector
vectors = orchestrator.encode_texts([text])
print(f"Encoded shape: {vectors.shape}")  # torch.Size([1, 768])
print(f"Vector norm: {torch.norm(vectors[0]).item():.4f}")  # 1.0000

# Step 2: Decode vector → text
result = orchestrator._run_subscriber_subprocess(
    'ielab',  # Use 'ielab' or 'jxe'
    vectors.cpu(),
    metadata={'original_texts': [text]},
    device_override='cpu'
)

if result['status'] == 'success':
    decoded_text = result['result'][0]
    print(f"Original:  {text}")
    print(f"Decoded:   {decoded_text}")
    # Output will have keyword matches and semantic similarity
else:
    print(f"Error: {result}")
```

### Expected Output

```
Original:  The Eiffel Tower was built in 1889.
Decoded:   The Eiffel Tower was built in Paris for the World's Fair in 1889...
```

Note: Decoded text won't be identical but will contain key concepts (Eiffel, Tower, 1889, etc.)

---

## ❌ WRONG Methods (DO NOT USE)

### Wrong #1: Using Port 8767 + Port 8766

```python
import requests

# ❌ DON'T DO THIS!
text = "The Eiffel Tower was built in 1889."

# Encode via port 8767
encode_resp = requests.post(
    "http://localhost:8767/embed",
    json={"texts": [text]}
)
vector = encode_resp.json()["embeddings"][0]

# Decode via port 8766
decode_resp = requests.post(
    "http://localhost:8766/decode",
    json={"vectors": [vector], "steps": 3, "decoder": "jxe"}
)
result = decode_resp.json()["results"][0]["subscribers"]["gtr → jxe"]

print(f"Decoded: {result['output']}")
# Output: "knowledge of how to work within one pavilion or heritage pavilion..."
# Cosine: 0.123 (gibberish - no semantic match!)
```

**Why this fails:** Port 8766 decoder expects vectors from a DIFFERENT encoder than port 8767 provides.

---

## Port Reference Table

| Port | Service | Compatible With | Status | Purpose |
|------|---------|-----------------|--------|---------|
| 7001 | Orchestrator Encoder (FastAPI) | Port 7002 decoder | ✅ PRODUCTION | **Encoding for full pipeline** |
| 7002 | Orchestrator Decoder (FastAPI) | Port 7001 encoder | ✅ PRODUCTION | **Decoding from port 7001** |
| 8767 | GTR-T5 Encoder | ??? (not 7002 or 8766!) | ⚠️ Partial | Standalone encoding (NOT for full pipeline) |
| 8766 | Vec2Text Decoder | ??? (not 8767!) | ❌ BROKEN | DO NOT USE |
| N/A | IsolatedVecTextVectOrchestrator | Self-contained | ✅ CORRECT | **Direct Python usage** |

---

## When to Use Each Method

### Use Ports 7001/7002 (PRODUCTION FastAPI Services)

**Start/Stop Scripts** (RECOMMENDED for production):
```bash
# Start all services (includes 7001 encoder + 7002 decoder + LVM chat services)
./scripts/start_lvm_services.sh

# Stop all services
./scripts/stop_lvm_services.sh

# Check health
curl http://localhost:7001/health
curl http://localhost:7002/health
```

**Use cases:**
- ✅ **PRODUCTION deployments** (persistent services)
- ✅ **Web applications** needing encode/decode via HTTP API
- ✅ **LVM chat services** (port 9000-9006 also started)
- ✅ **Multi-user environments** (shared encoding/decoding service)

### Use Orchestrator Directly (Python Integration)

**Use cases:**
- ✅ **Training LVM models** (need encode/decode)
- ✅ **Research notebooks** (Jupyter/Python scripts)
- ✅ **Batch processing** (single-process workflows)
- ✅ **Quality evaluation** (round-trip testing)

### Use Port 8767 Alone (ONLY if you don't need decoding)

- ✅ Batch encoding for FAISS indexing (vectors only, no decode)
- ✅ Similarity search (compare vectors, no text output)
- ⚠️ NEVER use port 8767 if you need to decode vectors back to text

### Never Use Port 8766

- ❌ Do NOT use for any purpose with this project
- ❌ Not compatible with port 8767 encoder
- ❌ Not compatible with port 7001 encoder
- ❌ Not compatible with orchestrator encoder
- ❌ Produces gibberish output (cosine ~0.05)

---

## Testing Your Setup

### Quick Test Script

```python
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

orchestrator = IsolatedVecTextVectOrchestrator(steps=5, debug=False)

# Test encode-decode round trip
test_text = "The cat sat on the mat."
vectors = orchestrator.encode_texts([test_text])
result = orchestrator._run_subscriber_subprocess(
    'ielab',
    vectors.cpu(),
    metadata={'original_texts': [test_text]},
    device_override='cpu'
)

decoded = result['result'][0] if result['status'] == 'success' else 'ERROR'
print(f"Original: {test_text}")
print(f"Decoded:  {decoded}")

# Check for keyword matches
if 'cat' in decoded.lower() and 'mat' in decoded.lower():
    print("✅ Pipeline working correctly!")
else:
    print("❌ Pipeline broken - decoded text has no keyword matches")
```

### Expected Success Criteria

- Decoded text contains at least 50% of keywords from original
- Semantic meaning is preserved (even if not word-for-word)
- No random gibberish or unrelated topics

---

## Troubleshooting

### Problem: Decoded text is gibberish

**Symptom**: Output like "knowledge of how to work within one pavilion..."
**Cause**: Using port 8766 decoder
**Fix**: Use `orchestrator._run_subscriber_subprocess()` instead

### Problem: "No module named 'vec_text_vect_isolated'"

**Symptom**: ImportError when importing orchestrator
**Fix**: Add to Python path:
```python
import sys
sys.path.insert(0, 'app/vect_text_vect')
```

Or use:
```bash
PYTHONPATH=. python your_script.py
```

---

## Summary

**ONLY ONE CORRECT METHOD:**

```python
orchestrator = IsolatedVecTextVectOrchestrator()
vectors = orchestrator.encode_texts([text])
result = orchestrator._run_subscriber_subprocess('ielab', vectors.cpu())
decoded = result['result'][0]
```

**DO NOT use port 8766 for ANYTHING.**
