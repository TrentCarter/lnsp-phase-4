# Semantic Chunker - Quick Reference

**Version**: 1.0.0 | **Status**: Production Ready ✅

---

## Installation

```bash
./.venv/bin/pip install llama-index llama-index-embeddings-huggingface
```

---

## Python API

### Semantic Chunker (Fast)

```python
from src.semantic_chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk(text)

# ~5,000 words/second
```

### Proposition Chunker (High Quality)

```python
from src.semantic_chunker import PropositionChunker

chunker = PropositionChunker(
    llm_endpoint="http://localhost:11434",
    llm_model="tinyllama:1.1b"
)
chunks = chunker.chunk(text)

# ~300 words/second (LLM-based)
```

### Hybrid Chunker (Balanced) ⭐ Recommended

```python
from src.semantic_chunker import HybridChunker

chunker = HybridChunker()
chunks = chunker.chunk(text)

# ~2,000 words/second
# Best speed/quality balance
```

---

## REST API

### Start Server

```bash
./.venv/bin/uvicorn app.api.chunking:app --reload --port 8001
```

### Endpoints

```bash
# Chunk text
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "semantic",
    "max_chunk_size": 320
  }'

# Health check
curl http://localhost:8001/health

# Statistics
curl http://localhost:8001/stats

# Documentation
open http://localhost:8001/docs
```

---

## Modes Comparison

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **Semantic** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Bulk ingestion, general use |
| **Proposition** | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Law, Medicine, research |
| **Hybrid** | ⚡⚡ Balanced | ⭐⭐⭐⭐ Very Good | **Production** (recommended) |

---

## Configuration

```bash
# Environment variables
export LNSP_EMBEDDER_PATH="sentence-transformers/gtr-t5-base"
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="tinyllama:1.1b"
export CHUNKING_API_PORT=8001
```

---

## TMD-LS Integration

```python
from src.semantic_chunker import HybridChunker
from src.tmd_router import route_concept
from src.prompt_extractor import extract_cpe_from_text

# 1. Chunk document
chunker = HybridChunker()
chunks = chunker.chunk(document_text)

# 2. Process each chunk
for chunk in chunks:
    # Extract TMD codes
    tmd = route_concept(chunk.text)

    # Extract CPESH with lane specialist
    cpesh = extract_cpe_from_text(chunk.text, domain_code=tmd['domain_code'])

    # Validate with Echo Loop
    validation = validate_cpesh(chunk.text, cpesh)
```

---

## Testing

```bash
# Run tests
./.venv/bin/pytest tests/test_semantic_chunker.py -v

# Skip slow LLM tests
./.venv/bin/pytest tests/test_semantic_chunker.py -m "not slow" -v

# Test CLI
./.venv/bin/python src/semantic_chunker.py
```

---

## Chunk Object

```python
@dataclass
class Chunk:
    text: str                    # Chunk text
    chunk_id: str                # Unique ID (MD5 hash)
    chunk_index: int             # Position in document
    word_count: int              # Number of words
    char_count: int              # Number of characters
    chunking_mode: str           # "semantic" | "proposition" | "hybrid"
    metadata: Dict[str, Any]     # Custom metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

---

## Performance Tips

### Maximum Speed

```python
# Use semantic only, larger min_chunk_size
chunker = SemanticChunker(min_chunk_size=1000)
```

### Maximum Quality

```python
# Use proposition with larger model
chunker = PropositionChunker(llm_model="llama3.1:8b")
```

### Production Balanced

```python
# Use hybrid with optimized thresholds
chunker = HybridChunker(
    refine_threshold=200,        # Only refine large chunks
    refine_domains=[11, 4]       # Only Law and Medicine
)
```

---

## Common Issues

### "LocalLlamaClient not available"

```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull tinyllama:1.1b
ollama serve
```

### Chunks too small

```python
# Increase minimum size
chunker = SemanticChunker(min_chunk_size=1000)
```

### Chunks too large

```python
# Lower breakpoint threshold (more splits)
chunker = SemanticChunker(breakpoint_percentile_threshold=90)
```

---

## Files

- **Code**: `src/semantic_chunker.py`
- **API**: `app/api/chunking.py`
- **Tests**: `tests/test_semantic_chunker.py`
- **Docs**: `docs/howto/how_to_use_semantic_chunker.md`
- **Summary**: `docs/SEMANTIC_CHUNKER_IMPLEMENTATION.md`

---

## Support

- Full documentation: `docs/howto/how_to_use_semantic_chunker.md`
- API docs: `http://localhost:8001/docs` (when server running)
- Implementation notes: `docs/SEMANTIC_CHUNKER_IMPLEMENTATION.md`

---

**Quick Start**: Use `HybridChunker()` for production - best balance of speed and quality! ⚡⭐
