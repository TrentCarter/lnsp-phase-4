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

## Expected Output

Both models should produce:
- **Similar semantic content** - Both outputs should relate to the input text
- **Different phrasing** - JXE and IELab will use different word choices
- **Cosine similarity 0.65-0.85** - Typical range for successful reconstruction
- **Processing time 5-15 seconds** - Depending on steps and text length

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