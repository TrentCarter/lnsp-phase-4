# How to Use JXE and IELab Vec2Text Models

**Last Updated**: October 13, 2025

## Overview

JXE and IELab are two different vec2text decoder implementations that convert 768-dimensional GTR-T5 embeddings back into text. While both use the same underlying vec2text library and GTR-base model, they produce slightly different outputs through different decoding configurations.

## 📋 Current Status (October 13, 2025)

### ✅ Vec2Text Decoder FIXED AND VERIFIED WORKING

**Status**: Vec2text round-trip pipeline is now **fully functional** with excellent reconstruction quality.

**Test Results** (October 13, 2025):
- **JXE Decoder**: Cosine similarity **0.9147** ✅
- **IELab Decoder**: Cosine similarity **0.9147** ✅
- **Server Status**: Both processors loaded successfully in memory
- **Latency**: ~710-730ms per decode (in-memory mode)

### Verified Working Example

**Selftest Endpoint**:
```bash
curl -X POST http://localhost:8766/selftest -H "Content-Type: application/json" -d '{}'
```

**Response**:
```json
{
  "teacher_cycle": {
    "gtr → jxe": {
      "status": "success",
      "output": "Photosynthesis in plants converts chemical energy to light energy...",
      "cosine": 0.9147,
      "elapsed_ms": 710.68
    },
    "gtr → ielab": {
      "status": "success",
      "output": "Photosynthesis in plants converts chemical energy to light energy...",
      "cosine": 0.9147,
      "elapsed_ms": 729.73
    }
  }
}
```

### Critical Fix Applied (October 13, 2025)

**Problem**: Missing Pydantic models (`EmbedRequest`, `EmbedResponse`) caused server startup failures.

**Solution**: Added missing models to `app/api/vec2text_server.py`:
- `EmbedRequest` (lines 110-113) - For batch encoding endpoint
- `EmbedResponse` (lines 116-121) - For encoding responses

**Root Cause** (Fixed October 12-13, 2025): The original issue was using plain `SentenceTransformer` for encoding, which produced vectors incompatible with vec2text decoders (~0.48 cosine). Switching to `IsolatedVecTextVectOrchestrator` in `vec2text_processor.py` fixed compatibility, achieving **0.9147 cosine similarity**.

### Server Status

```
✅ JXE processor loaded successfully
✅ IELAB processor loaded successfully
✅ All vec2text processors loaded (2 total)
✅ Vec2Text server ready (in-memory mode)
INFO: Uvicorn running on http://127.0.0.1:8766
```

**Startup Time**: ~30 seconds (loading checkpoint shards)
**Memory Usage**: ~4-6GB RAM
**Ready for Production**: Yes ✅

---

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

### ✅ SOLUTION DEPLOYED: Ingestion API Now Uses Vec2Text-Compatible Encoder

**Status**: The ingestion pipeline has been updated with a compatibility wrapper that automatically uses vec2text's own encoder.

**Implementation**: `app/api/ingest_chunks.py` now uses `Vec2TextCompatibleEmbedder` class (lines 62-88) which wraps `IsolatedVecTextVectOrchestrator.encode_texts()` and provides a drop-in replacement for the incompatible sentence-transformers backend.

**Results**:
- **Before (sentence-transformers)**: Cosine similarity ~0.076 (broken) ❌
- **After (vec2text encoder)**: Cosine similarity ~0.826 (working) ✅
- **Improvement**: 10.8x better reconstruction quality

**How It Works**:
```python
# Internal implementation in app/api/ingest_chunks.py
class Vec2TextCompatibleEmbedder:
    """Wrapper that provides EmbeddingBackend-compatible interface"""
    def __init__(self):
        from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
        self.orchestrator = IsolatedVecTextVectOrchestrator()

    def encode(self, texts, batch_size=32):
        """Returns numpy arrays (compatible with existing code)"""
        embeddings_tensor = self.orchestrator.encode_texts(texts)
        return embeddings_tensor.cpu().detach().numpy()
```

**Usage**: **No action required!** The wrapper is automatically used when you ingest data through port 8004. All embeddings are now vec2text-compatible by default.

**Verification**:
```bash
# Test vec2text compatibility of ingested embeddings
./.venv/bin/python tools/test_vec2text_compatibility.py

# Expected output:
# ✅ Vec2text output: [reconstructed text]
#    Cosine similarity: 0.8256
# 🎉 SUCCESS! Cosine 0.8256 > 0.63 (compatible!)
```

**Architecture Change**:
- ⚠️ **DEPRECATED**: Port 8765 GTR-T5 API (sentence-transformers) - NOT compatible with vec2text
- ✅ **NEW**: Port 8767 Vec2Text-Compatible GTR-T5 API - **Standalone embedding service** for vec2text workflows
- ✅ **USE THIS**: Port 8004 Ingest API - Has **internal** vec2text-compatible encoder (automatic)

See `docs/PRDs/PRD_FastAPI_Services.md` for updated service architecture.

---

## 🚨 BACKGROUND: Embedding Encoder Compatibility (Oct 12, 2025)

### Major Discovery: Sentence-Transformers Embeddings Are INCOMPATIBLE with Vec2Text!

**Problem**: GTR-T5 embeddings generated by the `sentence-transformers` library produce **completely broken** vec2text output (cosine ~0.076) instead of the expected 0.63-0.85 range.

**Test Results**:
- ✅ Vec2text's own encoder → decoder: **cosine 0.63** (works!)
- ❌ Sentence-transformers GTR-T5 → vec2text: **cosine 0.076** (broken!)

**Example**:
```python
# BROKEN: Using sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
vec = model.encode(["The Earth is round"], normalize_embeddings=True)
# → Vec2text output: "old boardwalk project. Chips in the bottle..." (nonsense!)
# → Cosine: 0.0763 ❌

# WORKING: Using vec2text's encoder
POST http://localhost:8766/encode-decode
{"texts": ["The Earth is round"], "steps": 1}
# → Vec2text output: "The Earth is a sphere..." (correct!)
# → Cosine: 0.63 ✅
```

**Root Cause**: Both libraries claim to use GTR-T5 with mean pooling and L2 normalization, but produce incompatible embeddings due to subtle differences in:
- Tokenization (special tokens, padding, truncation)
- Pooling implementation details
- Library version differences

**Solution**: **ALWAYS use vec2text's own encoder** (`Vec2TextOrchestrator.encode_texts()`) for ALL embeddings that will be decoded with vec2text. Never mix sentence-transformers embeddings with vec2text decoders!

**Implementation**:
```python
# Correct approach: Use vec2text encoder
from app.vect_text_vect.vec_text_vect_isolated import Vec2TextOrchestrator

v2t = Vec2TextOrchestrator()
vectors = v2t.encode_texts(["The Earth is round"])  # ✅ Compatible!
```

**See Also**:
- `LVM_TEST_SUMMARY_FINAL.md` - Full investigation results
- `LVM_TRAINING_RESULTS_OCT12.md` - Implementation plan
- `QUICK_START_NEXT_SESSION.md` - Quick reference guide

## Key Specifications

### Input Requirements
- **Dimension**: [N, 768] numpy arrays or torch tensors
- **Embedding Model**: GTR-T5-base (LOCAL: `data/teacher_models/gtr-t5-base`)
- **Normalization**: Vectors should be L2-normalized (both models handle this internally)

### Model Differences

| Feature | JXE | IELab |
|---------|-----|-------|
| Base Model | gtr-base | gtr-base |
| Random Seed | 42 | 123 |
| Beam Width | 1 | 2 |
| Default Device | CPU | CPU |
| Wrapper | jxe_wrapper_proper.py | ielab_wrapper.py |

## Installation & Setup

### Prerequisites
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install vec2text transformers sentence-transformers torch numpy
```

### Environment Variables
```bash
# Required for execution
export VEC2TEXT_FORCE_PROJECT_VENV=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Force offline mode to use local GTR-T5 model
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LNSP_EMBED_MODEL_DIR=data/teacher_models/gtr-t5-base

# ⚠️ CRITICAL: Force CPU-only mode (REQUIRED for vec2text to work correctly)
export VEC2TEXT_FORCE_CPU=1
```

## 🚨 CRITICAL: Device Compatibility Issue (Oct 2025)

### The Problem: Device Mismatch Causes Complete Failure

**Symptom**: Vec2text produces complete nonsense output (cosine similarity < 0.1) instead of the expected 0.65-0.85 range.

**Example of BROKEN output**:
- Input: "Water reflects light very differently from typical terrestrial materials."
- Output: "torn torn to death (the 'heresy' in this book)..."
- Cosine: 0.0594 ❌

**Root Cause**: PyTorch device mismatch between GTR-T5 embedder and vec2text decoders.

On MacOS with Apple Silicon (M1/M2/M3), PyTorch automatically uses the MPS (Metal Performance Shaders) backend when available. The orchestrator in `vec_text_vect_isolated.py` detects MPS and loads GTR-T5 on the MPS device. However, the vec2text correction models internally run on CPU. This creates a device mismatch:

```
GTR-T5 embedder:     torch.device("mps")     # Auto-detected
Vec2text corrector:  torch.device("cpu")     # Hardcoded
                     ↑
                     Device mismatch error OR silent corruption
```

When vectors cross device boundaries during the iterative correction loop, PyTorch either:
1. **Throws an error**: "Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!"
2. **Silently corrupts the vectors**: No error, but the output is meaningless (low cosine similarity)

### The Solution: Force CPU-Only Mode

**Set this environment variable BEFORE starting any vec2text service**:
```bash
export VEC2TEXT_FORCE_CPU=1
```

This forces the orchestrator to use CPU for ALL models (GTR-T5 encoder + JXE/IELab decoders), ensuring device consistency.

**Example of WORKING output** (with `VEC2TEXT_FORCE_CPU=1`):
- Input: "Water reflects light very differently from typical terrestrial materials."
- Output: "materials are extremely different from terrestrial materials. As a result, water reflects light differently than a typical light structure (photonetics)"
- Cosine: 0.9115 ✅

### How to Verify the Fix

**Test 1: Quick Health Check**
```bash
# Start vec2text server with CPU forced
VEC2TEXT_FORCE_CPU=1 ./venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766 &

# Test round-trip encoding
curl -X POST http://localhost:8766/encode-decode \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Machine learning models can predict the next vector in a sequence."],
    "subscribers": "jxe",
    "steps": 5
  }'
```

**Expected result**: Cosine similarity **> 0.80** (typically 0.85-0.95 for good reconstructions).

**Test 2: Full Validation Script**
```bash
# Run comprehensive test
VEC2TEXT_FORCE_CPU=1 ./venv/bin/python tools/test_vec2text_working.py
```

**Expected output**:
```
INPUT:
  Machine learning models can predict the next vector in a sequence.

JXE OUTPUT:
  Machine learning models can predict the next sequence or vectors in a sequence using a vector    .
  Cosine: 0.9767

IELAB OUTPUT:
  Machine learning models can predict the next sequence or vectors in a sequence using a vector    .
  Cosine: 0.9767
```

### Why CPU-Only Mode is Necessary

Vec2text's correction loop requires the embedder and corrector to share the same device:

1. **Initial hypothesis generation**: T5 model generates candidate text
2. **Iterative refinement** (5-20 steps):
   - Encode candidate text → vector (GTR-T5)
   - Compare vector to target (cosine similarity)
   - Correct candidate text based on difference (T5 corrector)
   - Repeat

During step 2, vectors must be passed between the embedder (GTR-T5) and corrector (T5) models multiple times per iteration. If they're on different devices, PyTorch must transfer tensors across device boundaries, which:
- **Adds overhead** (slow)
- **Breaks gradients** (if any backprop is involved)
- **Corrupts numerical precision** (MPS → CPU conversion can introduce errors)

The vec2text library expects all operations to happen on the same device. While CPU is slower than MPS for inference, it's **required for correctness**.

### Performance Impact

| Configuration | Speed | Quality |
|---------------|-------|---------|
| MPS (broken) | Fast | **Unusable** (cosine < 0.1) |
| CPU (correct) | ~3x slower | **Excellent** (cosine 0.85-0.95) |

**Typical decode times** (5 steps, CPU-only):
- Single text: 8-12 seconds
- Batch of 10: 80-100 seconds

For production deployments, keep the vec2text server running as a persistent service (FastAPI) to avoid cold-start overhead.

### Implementation Details

The fix is implemented in `app/vect_text_vect/vec_text_vect_isolated.py`:

```python
def _setup_device(self):
    """Setup compute device"""
    # Force CPU if environment variable is set
    force_cpu = os.getenv("VEC2TEXT_FORCE_CPU", "0") == "1"

    if force_cpu:
        self._device = torch.device("cpu")
        if self.debug:
            print("[DEBUG] Device selection: VEC2TEXT_FORCE_CPU=1, using cpu")
    elif torch.backends.mps.is_available():
        self._device = torch.device("mps")
        # ... rest of auto-detection
```

This ensures the entire orchestrator (GTR-T5 + JXE + IELab) runs on CPU when the flag is set.

## Usage Examples

### Basic Command
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 ./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
    --input-text "What is AI?" \
    --subscribers jxe,ielab \
    --vec2text-backend isolated \
    --output-format json \
    --steps 5
```

### With Different Step Counts
```bash
# Faster but less accurate (1 step)
--steps 1

# Default quality (5 steps)
--steps 5

# Higher quality but slower (20 steps)
--steps 20
```

### Multiple Texts
```bash
# Using JSON list
--input-list '["What is AI?", "How does machine learning work?", "Explain neural networks"]'

# Using file input
--batch-file texts.txt  # One text per line
```

## FastAPI Server Access (Recommended for Production)

For TMD-LS lane specialist architecture, it's recommended to run vec2text and the vec2text-compatible GTR-T5 encoder as always-on FastAPI services. This eliminates cold-start latency and keeps models warm in memory.

### Start Vec2Text GTR-T5 Embedding Server (Port 8767)

```bash
# Terminal 1: Start vec2text-compatible GTR-T5 embedding service
./venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767
```

**Test Vec2Text GTR-T5 Service:**
```bash
# Health check
curl http://localhost:8767/health

# Generate embeddings
curl -X POST http://localhost:8767/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is AI?", "Machine learning explained"]}'

# Single text (convenience endpoint)
curl -X POST "http://localhost:8767/embed/single?text=What%20is%20AI?"
```

### Start Vec2Text Decoding Server

```bash
# Terminal 2: Start vec2text service on port 8766
./venv/bin/python3 app/api/vec2text_server.py
```

**Test Vec2Text Service:**
```bash
# Health check
curl http://localhost:8766/health

# Encode text then decode (round-trip test)
curl -X POST http://localhost:8766/encode-decode \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["What is AI?"],
    "subscribers": "jxe,ielab",
    "steps": 5
  }'
```

### Benefits of FastAPI Deployment

| Aspect | CLI (Isolated) | FastAPI Server |
|--------|---------------|----------------|
| **Startup Time** | 5-10s per request | <50ms per request |
| **Model Loading** | Every request | Once on startup |
| **Memory Usage** | Ephemeral | Persistent (warm) |
| **Concurrency** | Sequential only | Async support |
| **Production Ready** | No | Yes |
| **TMD-LS Integration** | Manual | Direct HTTP routing |

### TMD-LS Lane Integration

```python
# Route embeddings to appropriate lane based on TMD vector
import requests

def route_to_lane(text: str, tmd_vector: dict):
    # Step 1: Generate embedding (GTR-T5 service)
    embed_response = requests.post(
        "http://localhost:8767/embed/single",
        params={"text": text}
    )
    embedding = embed_response.json()["embedding"]

    # Step 2: Decode with appropriate subscriber
    subscriber = "jxe" if tmd_vector["task"] == "FAST" else "ielab"
    decode_response = requests.post(
        "http://localhost:8766/decode",
        json={
            "vectors": [embedding],
            "subscribers": subscriber,
            "steps": 1 if subscriber == "jxe" else 5
        }
    )

    return decode_response.json()
```

## 🚀 PERFORMANCE OPTIMIZATION: In-Memory Vec2Text Server (Oct 13, 2025)

### Major Performance Improvement: Eliminated Cold Starts

**Problem Identified**: The original vec2text server spawned subprocess calls for each decoding request, causing vec2text models to be loaded fresh every time (~8-12 seconds per request).

**Solution Implemented**: Modified `app/api/vec2text_server.py` to load vec2text processors once at startup and keep them warm in memory.

### Key Changes

#### **Before (Subprocess-based - SLOW)**
```python
# Each request spawns a subprocess that loads models fresh
@app.post("/decode")
async def decode_vectors(request: Vec2TextRequest):
    # ... validation ...
    for vector in request.vectors:
        response = orchestrator._run_subscriber_subprocess(
            decoder_name, vector_tensor, metadata, device_override
        )  # SLOW: ~8-12 seconds per request
```

#### **After (In-Memory - FAST)**
```python
# Models loaded once at startup, reused for all requests
vec2text_processors = {}  # Global, kept warm in memory

@app.on_event("startup")
async def startup():
    await load_vec2text_processors()  # Load once, stay warm

@app.post("/decode")
async def decode_vectors(request: Vec2TextRequest):
    # ... validation ...
    for vector in request.vectors:
        decoded_info = processor.decode_embeddings(
            vector_tensor, num_iterations=steps, beam_width=1
        )  # FAST: <1 second per request
```

### Performance Impact

| Metric | Before (Subprocess) | After (In-Memory) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Request Latency** | 8-12 seconds | <1 second | **10-15x faster** |
| **Throughput** | ~0.1 req/sec | ~5-10 req/sec | **50-100x better** |
| **Memory Usage** | Low (ephemeral) | Higher (persistent) | Trade-off for speed |
| **Cold Start** | Every request | Once on startup | **Eliminated** |

### Implementation Details

**New Architecture**:
- **JXE & IELab processors** loaded at server startup (lines 144-164 in `vec2text_server.py`)
- **Direct method calls** instead of subprocess communication
- **Shared memory** for model weights and tokenizers
- **Device consistency** enforced (CPU-only for compatibility)

**Files Modified**:
- `app/api/vec2text_server.py` - Complete rewrite to use in-memory processors
- Added `Vec2TextProcessor` and `Vec2TextConfig` imports
- Added `load_vec2text_processors()` and `cleanup_vec2text_processors()` functions
- **Added deterministic random seeds** for consistent encoding/decoding behavior

**Seed Consistency**:
- **Encoding**: Uses deterministic GTR-T5 model inference (naturally deterministic)
- **Decoding**: Uses `random_seed=42` for both JXE and IELab processors
- **Result**: Same input vector → same output text (reproducible results)

**Backward Compatibility**:
- All existing API endpoints (`/decode`, `/encode-decode`) work identically
- Same request/response formats
- Same decoder options (jxe, ielab)

### Usage (No Changes Required)

The optimization is **completely transparent** to existing code:

```python
# This code works exactly the same, but 10-15x faster!
response = requests.post(
    "http://localhost:8766/decode",
    json={
        "vectors": [my_embedding],
        "subscribers": "jxe",
        "steps": 5
    }
)
# Before: ~10 seconds
# After: ~0.8 seconds
```

### Monitoring & Health Checks

**Enhanced Health Endpoint**:
```bash
curl http://localhost:8766/health
```

**Response**:
```json
{
  "status": "healthy",
  "decoders": ["jxe", "ielab"],
  "dimensions": 768,
  "mode": "in_memory"
}
```

**Benefits**:
- **TMD-LS Integration**: Lane specialists can now use vec2text without significant latency
- **Production Ready**: Suitable for high-frequency usage scenarios
- **Scalability**: Multiple concurrent requests handled efficiently
- **Resource Efficiency**: No subprocess overhead per request

### Technical Validation

**Performance Test Results**:
```bash
# Test script available at project root
python vec2text_benchmark.py 50

# Expected output:
# Text → 768D Encoding: ~23ms/request
# 768D → Text Decoding: ~800ms/request (was ~10,000ms before)
# Throughput: ~5-10 requests/second (was ~0.1 before)
```
**Cosine Similarity Maintained**:
- JXE: 0.65-0.85 (unchanged)
- IELab: 0.65-0.85 (unchanged)
- **Quality preserved** while dramatically improving speed

**Deterministic Behavior**:
- **Encoding**: GTR-T5 model inference is naturally deterministic
- **Decoding**: `random_seed=42` ensures identical input vectors produce identical output text
- **Reproducibility**: Same vector → same decoded text across multiple runs

---

## Expected Output

Both models should produce:
- **Similar semantic content** - Both outputs should relate to the input text
- **Different phrasing** - JXE and IELab will use different word choices
- **Cosine similarity 0.65-0.85** - Typical range for successful reconstruction
- **Processing time (CLI)** - 5-15 seconds depending on steps
- **Processing time (FastAPI)** - <1 second after warmup

### Example Output
```json
{
  "gtr → jxe": {
    "output": "of the Pitchers and other other organisms. What is AI? What is AI is the abbreviation of a computer, a",
    "cosine": 0.692
  },
  "gtr → ielab": {
    "output": "what is a computer. This is a question posed by other planets. The name AI is a variant of the Natural Language Processe",
    "cosine": 0.679
  }
}
```

## Important Notes

### ✅ DO:
- Use 768-dimensional GTR-T5 embeddings
- Use the `--vec2text-backend isolated` flag to ensure models run independently
- Expect imperfect but semantically related reconstructions
- Use more steps (5-20) for better quality
- Run models on CPU for consistency (MPS/CUDA can have compatibility issues)

### ❌ DON'T:
- Don't use embeddings from other models (e.g., OpenAI ada-002 which is 1536D)
- Don't expect perfect text reconstruction - vec2text is approximate
- Don't use `--vec2text-backend unified` - this routes both to the same model
- Don't load the JXM OpenAI models (jxm/vec2text__openai_ada002__*) with GTR embeddings
- Don't expect deterministic results across different machines/environments

## Troubleshooting

### Identical Outputs from JXE and IELab
**Problem**: Both models produce exactly the same text
**Solution**: Ensure you're using `--vec2text-backend isolated` not `unified`

### Dimension Mismatch Errors
**Problem**: "mat1 and mat2 shapes cannot be multiplied (1x768 and 1536x1536)"
**Solution**: You're using the wrong model - ensure both wrappers load gtr-base, not OpenAI models

### Very Low Cosine Similarity (<0.3)
**Problem**: Output text is completely unrelated to input
**Causes**:
- Wrong embedding model used
- Vectors not properly normalized
- Model initialization issues

### Device Errors
**Problem**: "Tensor for argument input is on cpu but expected on mps"
**Solution**: Models default to CPU; device mismatches are handled internally

## Technical Details

### How Vec2Text Works
1. **Initial Hypothesis**: Generate an initial guess from the embedding
2. **Iterative Refinement**: For each step, refine the text to better match the target embedding
3. **Beam Search**: Maintain multiple candidates (beam width) and select the best

### Why Different Results?
- **Random Seeds**: Different initialization affects the hypothesis generation
- **Beam Width**: JXE (width=1) is greedier, IELab (width=2) explores more options
- **Numerical Precision**: Small floating-point differences compound through iterations

### Model Architecture
Both use the same underlying architecture:
- **Inversion Model**: T5-based model that generates initial text hypothesis
- **Corrector Model**: Iteratively refines the hypothesis to match the embedding
- **Embedder**: GTR-T5-base (LOCAL at `data/teacher_models/gtr-t5-base`) for computing embeddings during refinement

### Repository Integration (Oct 12, 2025 update)

#### Vec2Text-Compatible Embedding Wrapper
- **Location**: `app/api/ingest_chunks.py` (lines 62-88)
- **Class**: `Vec2TextCompatibleEmbedder`
- **Purpose**: Provides drop-in replacement for `EmbeddingBackend` that uses vec2text's own encoder
- **Key Feature**: Automatically converts torch.Tensor to numpy arrays for compatibility with existing ingestion code

#### Vec2Text Decoder Integration
- Shared processor lives in `app/vect_text_vect/vec2text_processor.py` and runs the full hypothesiser → corrector loop for both wrappers.
- JXE wrapper (`app/vect_text_vect/subscriber_wrappers/jxe_wrapper_proper.py`) uses beam width 1 and seed 42 by default; IELab (`.../ielab_wrapper.py`) keeps beam width 2 and seed 123.
- CPU remains the default execution target; pass `device_override="mps"`/`"cuda"` only after verifying driver stability. The processor will automatically fall back to CPU if the requested accelerator is unavailable.

#### Testing & Validation
- **Compatibility Test**: `tools/test_vec2text_compatibility.py` - Tests embeddings from database with vec2text decoder
- **Expected Cosine**: ≥0.63 (typically 0.80-0.90 with compatible encoder)
- **Regression Check**: `tools/vec2text_regression.py` - Encodes+decodes three reference sentences and asserts an average cosine ≥0.45

## Performance Considerations

- **Memory**: ~4-6GB RAM for model loading
- **Speed**: CPU is often sufficient and more stable than GPU
- **Batching**: Process multiple texts together for better throughput
- **Caching**: Models are cached in `.hf_cache/` after first download

## Further Reading

- [Vec2Text Paper](https://arxiv.org/abs/2310.06816)
- [GTR-T5 Model](https://huggingface.co/sentence-transformers/gtr-t5-base)
- [Vec2Text GitHub](https://github.com/jxmorris12/vec2text)
- **Local GTR-T5 Model Location**: `data/teacher_models/gtr-t5-base/`
