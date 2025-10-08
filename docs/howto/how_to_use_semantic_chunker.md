# How to Use the Semantic Chunker

**Version**: 1.0.0
**Date**: 2025-10-08
**Status**: Production Ready

---

## Overview

The LNSP Semantic Chunker provides three strategies for splitting text into concept-based chunks optimized for the TMD-LS pipeline:

1. **Semantic**: Fast embedding-based semantic boundary detection (uses GTR-T5)
2. **Proposition**: High-quality LLM-extracted atomic propositions
3. **Hybrid**: Semantic splitting + selective proposition refinement

All chunking modes preserve semantic coherence for accurate TMD domain routing.

---

## Installation

The semantic chunker is included in the standard LNSP installation. If you need to install dependencies manually:

```bash
# Install LlamaIndex and HuggingFace embeddings
./.venv/bin/pip install llama-index llama-index-embeddings-huggingface

# Verify installation
./.venv/bin/python -c "from src.semantic_chunker import SemanticChunker; print('✓ SemanticChunker installed')"
```

---

## Quick Start

### Option 1: Python API

```python
from src.semantic_chunker import SemanticChunker

# Initialize chunker
chunker = SemanticChunker()

# Chunk text
text = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
carbon dioxide as inputs.
"""

chunks = chunker.chunk(text)

# Inspect results
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.word_count} words")
    print(f"Text: {chunk.text[:100]}...")
    print()
```

### Option 2: FastAPI Endpoint

```bash
# Start chunking API server
./.venv/bin/uvicorn app.api.chunking:app --reload --port 8001

# Test with curl
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis is the process by which plants convert light energy...",
    "mode": "semantic",
    "max_chunk_size": 320
  }'
```

---

## Chunking Modes

### 1. Semantic Chunking (Default)

**Best for**: Fast processing, large-scale ingestion, general use

**How it works**:
- Uses GTR-T5 embeddings (same as LNSP vecRAG)
- Finds natural semantic boundaries between sentences
- Adaptive breakpoint detection (95th percentile threshold)

**Performance**:
- Speed: ~200ms for 1000 words
- Quality: Good semantic coherence
- Resource usage: Low (embedding inference only)

**Example**:

```python
from src.semantic_chunker import SemanticChunker

chunker = SemanticChunker(
    embed_model_name="sentence-transformers/gtr-t5-base",
    buffer_size=1,  # Sentences before/after to compare
    breakpoint_percentile_threshold=95,  # Similarity threshold
    min_chunk_size=500  # Minimum characters per chunk
)

chunks = chunker.chunk(text)
```

**Python API Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_model_name` | str | `gtr-t5-base` | HuggingFace embedding model |
| `buffer_size` | int | 1 | Sentences to compare |
| `breakpoint_percentile_threshold` | int | 95 | Similarity threshold |
| `min_chunk_size` | int | 500 | Minimum chars per chunk |

**REST API Parameters**:

```json
{
  "text": "Your text here...",
  "mode": "semantic",
  "max_chunk_size": 320,
  "min_chunk_size": 500,
  "metadata": {"source": "doc_123"}
}
```

---

### 2. Proposition Chunking

**Best for**: High-precision domains (Law, Medicine), research use

**How it works**:
- Uses local LLM (TinyLlama/Llama) to extract atomic propositions
- Each proposition is self-contained and indivisible
- Represents single factoids

**Performance**:
- Speed: ~2-3s for 1000 words (LLM inference)
- Quality: Excellent semantic precision
- Resource usage: Medium (requires Ollama)

**Requirements**:
- Ollama running on port 11434
- TinyLlama or Llama model installed

**Example**:

```python
from src.semantic_chunker import PropositionChunker

chunker = PropositionChunker(
    llm_endpoint="http://localhost:11434",
    llm_model="tinyllama:1.1b",
    max_propositions=50
)

chunks = chunker.chunk(text)

# Each chunk is an atomic proposition
for chunk in chunks:
    print(f"Proposition: {chunk.text}")
```

**Python API Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_endpoint` | str | `http://localhost:11434` | Ollama endpoint |
| `llm_model` | str | `tinyllama:1.1b` | LLM model name |
| `max_propositions` | int | 50 | Max propositions per doc |

**REST API Parameters**:

```json
{
  "text": "Your text here...",
  "mode": "proposition",
  "metadata": {"source": "doc_123"}
}
```

---

### 3. Hybrid Chunking

**Best for**: Balanced speed/quality, production use

**How it works**:
1. Stage 1: Fast semantic splitting
2. Stage 2: Proposition refinement for large chunks (> 150 words)
3. Keeps semantic chunks for small/medium chunks

**Performance**:
- Speed: ~500ms for 1000 words (mixed)
- Quality: Excellent for complex chunks, good for simple chunks
- Resource usage: Medium (embeddings + selective LLM)

**Example**:

```python
from src.semantic_chunker import HybridChunker

chunker = HybridChunker(
    embed_model_name="sentence-transformers/gtr-t5-base",
    llm_endpoint="http://localhost:11434",
    llm_model="tinyllama:1.1b",
    refine_threshold=150,  # Words threshold for refinement
    refine_domains=[11, 4, 6]  # Always refine Law, Medicine, Philosophy
)

chunks = chunker.chunk(text)
```

**Python API Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_model_name` | str | `gtr-t5-base` | Embedding model |
| `llm_endpoint` | str | `http://localhost:11434` | LLM endpoint |
| `llm_model` | str | `tinyllama:1.1b` | LLM model |
| `refine_threshold` | int | 150 | Words threshold for refinement |
| `refine_domains` | List[int] | `[]` | Domain codes to always refine |

**REST API Parameters**:

```json
{
  "text": "Your text here...",
  "mode": "hybrid",
  "force_refine": false,
  "metadata": {"domain_code": 11}
}
```

---

## REST API Reference

### Base URL

```
http://localhost:8001
```

### Endpoints

#### POST /chunk

Chunk text using specified mode.

**Request Body**:

```json
{
  "text": "Your text here...",
  "mode": "semantic",  // "semantic" | "proposition" | "hybrid"
  "max_chunk_size": 320,
  "min_chunk_size": 500,
  "metadata": {
    "document_id": "doc_123",
    "source": "biology_textbook",
    "domain_code": 0
  },
  "force_refine": false
}
```

**Response**:

```json
{
  "chunks": [
    {
      "text": "Photosynthesis is the process...",
      "chunk_id": "a1b2c3d4e5f6g7h8",
      "chunk_index": 0,
      "word_count": 45,
      "char_count": 312,
      "chunking_mode": "semantic",
      "metadata": {
        "document_id": "doc_123",
        "source": "biology_textbook",
        "embedding_model": "sentence-transformers/gtr-t5-base"
      }
    }
  ],
  "total_chunks": 3,
  "chunking_mode": "semantic",
  "statistics": {
    "total_chunks": 3,
    "mean_words": 45.2,
    "min_words": 32,
    "max_words": 67,
    "p95_words": 65,
    "chunking_modes": {"semantic": 3},
    "word_distribution": {
      "0-100": 3,
      "100-200": 0,
      "200-300": 0,
      "300+": 0
    }
  },
  "processing_time_ms": 124.5
}
```

#### GET /health

Health check endpoint.

**Response**:

```json
{
  "status": "healthy",
  "chunkers_loaded": {
    "semantic": true,
    "proposition": true,
    "hybrid": true
  },
  "version": "1.0.0"
}
```

#### GET /stats

Service statistics.

**Response**:

```json
{
  "total_requests": 1234,
  "total_chunks_created": 5678,
  "average_processing_time_ms": 152.3,
  "chunking_mode_usage": {
    "semantic": 800,
    "proposition": 200,
    "hybrid": 234
  }
}
```

---

## Integration with TMD-LS Pipeline

### Step 1: Chunk Text

```python
from src.semantic_chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk(document_text)
```

### Step 2: Route to TMD Router

```python
from src.tmd_router import route_concept

for chunk in chunks:
    # Extract TMD codes
    tmd = route_concept(chunk.text)

    # Attach TMD to chunk metadata
    chunk.metadata['domain_code'] = tmd['domain_code']
    chunk.metadata['task_code'] = tmd['task_code']
    chunk.metadata['modifier_code'] = tmd['modifier_code']
```

### Step 3: Extract CPESH

```python
from src.prompt_extractor import extract_cpe_from_text

for chunk in chunks:
    # Select lane specialist based on domain
    lane = select_lane(chunk.metadata['domain_code'])

    # Extract CPESH using lane specialist
    cpesh = extract_cpe_from_text(chunk.text, llm_model=lane['model'])

    # Validate with Echo Loop
    validation = validate_cpesh(chunk.text, cpesh)
```

### Complete Pipeline Example

```python
from src.semantic_chunker import HybridChunker
from src.tmd_router import route_concept
from src.prompt_extractor import extract_cpe_from_text
from src.echo_loop_validator import validate_cpesh

# Initialize chunker
chunker = HybridChunker()

# Process document
document_text = load_document("biology_textbook_chapter3.txt")
chunks = chunker.chunk(document_text)

print(f"Created {len(chunks)} chunks from document")

# Process each chunk
for chunk in chunks:
    # 1. Extract TMD
    tmd = route_concept(chunk.text)

    # 2. Extract CPESH using domain-specific LLM
    cpesh = extract_cpe_from_text(
        chunk.text,
        domain_code=tmd['domain_code']
    )

    # 3. Validate with Echo Loop
    validation = validate_cpesh(chunk.text, cpesh)

    if validation['valid']:
        print(f"✓ Chunk {chunk.chunk_index}: {cpesh['concept']}")
    else:
        print(f"✗ Chunk {chunk.chunk_index}: Re-queue (similarity={validation['cosine_similarity']})")
```

---

## Configuration

### Environment Variables

```bash
# Embedding model (used by semantic chunker)
export LNSP_EMBEDDER_PATH="sentence-transformers/gtr-t5-base"

# LLM endpoint (used by proposition/hybrid chunkers)
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="tinyllama:1.1b"

# Chunking API server
export CHUNKING_API_PORT=8001
export CHUNKING_API_HOST="127.0.0.1"
```

### Configuration File

Create `configs/chunking_config.json`:

```json
{
  "semantic": {
    "embed_model": "sentence-transformers/gtr-t5-base",
    "buffer_size": 1,
    "breakpoint_threshold": 95,
    "min_chunk_size": 500
  },
  "proposition": {
    "llm_endpoint": "http://localhost:11434",
    "llm_model": "tinyllama:1.1b",
    "max_propositions": 50
  },
  "hybrid": {
    "refine_threshold": 150,
    "refine_domains": [11, 4, 6]
  }
}
```

---

## Performance Tuning

### Speed Optimization

**For maximum speed** (10x-50x speedup):

```python
# Use semantic chunking only
chunker = SemanticChunker()
chunks = chunker.chunk(text)
```

**Expected throughput**: ~5,000 words/second

### Quality Optimization

**For maximum quality** (best accuracy):

```python
# Use proposition chunking
chunker = PropositionChunker(llm_model="llama3.1:8b")
chunks = chunker.chunk(text)
```

**Expected throughput**: ~300 words/second

### Balanced Configuration

**For production use** (good speed + quality):

```python
# Use hybrid chunking with optimized thresholds
chunker = HybridChunker(
    refine_threshold=200,  # Only refine very large chunks
    refine_domains=[11, 4]  # Only refine Law and Medicine
)
chunks = chunker.chunk(text)
```

**Expected throughput**: ~2,000 words/second

---

## Testing

### Run Unit Tests

```bash
# Run all chunker tests
./.venv/bin/pytest tests/test_semantic_chunker.py -v

# Run fast tests only (skip LLM tests)
./.venv/bin/pytest tests/test_semantic_chunker.py -v -m "not slow"

# Run specific test
./.venv/bin/pytest tests/test_semantic_chunker.py::TestSemanticChunker::test_chunk_short_text -v
```

### Manual Testing

```bash
# Test semantic chunker CLI
./.venv/bin/python src/semantic_chunker.py

# Test chunking API
./.venv/bin/uvicorn app.api.chunking:app --reload --port 8001

# Send test request
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_request.json
```

---

## Troubleshooting

### Issue: "LocalLlamaClient not available"

**Solution**: Install and start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull tinyllama:1.1b

# Start Ollama
ollama serve
```

### Issue: "Chunks too small/large"

**Solution**: Adjust thresholds

```python
# For larger chunks
chunker = SemanticChunker(
    min_chunk_size=1000  # Increase minimum
)

# For smaller chunks
chunker = SemanticChunker(
    breakpoint_percentile_threshold=90  # Lower threshold (more splits)
)
```

### Issue: "Out of memory during embedding"

**Solution**: Process in batches

```python
def chunk_large_document(text, batch_size=10000):
    """Chunk very large documents in batches."""
    chunks = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        batch_chunks = chunker.chunk(batch)
        chunks.extend(batch_chunks)
    return chunks
```

---

## Advanced Usage

### Custom Embedding Models

```python
# Use different embedding model
chunker = SemanticChunker(
    embed_model_name="BAAI/bge-large-en-v1.5"
)
```

### Domain-Specific Chunking

```python
# Configure hybrid chunker for specific domain
law_chunker = HybridChunker(
    refine_threshold=100,  # Lower threshold for legal text
    refine_domains=[11],   # Always refine Law domain
    llm_model="llama3.1:8b"  # Use larger model for legal
)

medical_chunker = HybridChunker(
    refine_threshold=120,
    refine_domains=[4],    # Always refine Medicine domain
    llm_model="phi3:mini"  # Use Phi3 for medical
)
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def chunk_documents(documents: List[str]) -> List[List[Chunk]]:
    """Chunk multiple documents in parallel."""
    chunker = SemanticChunker()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(chunker.chunk, documents))

    return results
```

---

## References

- [TMD-LS Architecture](../PRDs/PRD_TMD-LS.md)
- [TMD-LS Implementation Guide](../PRDs/PRD_TMD-LS_Implementation.md)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [GTR-T5 Embeddings](https://huggingface.co/sentence-transformers/gtr-t5-base)
- [Ollama Documentation](https://github.com/ollama/ollama)

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-08
**Maintained by**: LNSP Team
