# How to Use JXE and IELab Vec2Text Models

## Overview

JXE and IELab are two different vec2text decoder implementations that convert 768-dimensional GTR-T5 embeddings back into text. While both use the same underlying vec2text library and GTR-base model, they produce slightly different outputs through different decoding configurations.

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
```

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

For TMD-LS lane specialist architecture, it's recommended to run vec2text and GTR-T5 as always-on FastAPI services. This eliminates cold-start latency and keeps models warm in memory.

### Start GTR-T5 Embedding Server

```bash
# Terminal 1: Start GTR-T5 embedding service on port 8765
./venv/bin/python3 app/api/gtr_embedding_server.py
```

**Test GTR-T5 Service:**
```bash
# Health check
curl http://localhost:8765/health

# Generate embeddings
curl -X POST http://localhost:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is AI?", "Machine learning explained"]}'

# Single text (convenience endpoint)
curl -X POST "http://localhost:8765/embed/single?text=What%20is%20AI?"
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
        "http://localhost:8765/embed/single",
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