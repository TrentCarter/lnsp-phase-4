# FastAPI Chunking Service - Quick Start Guide

**The easiest way to chunk your text!** 🚀

---

## 🎯 Three Ways to Use It

### **Option 1: Web Interface** ⭐ RECOMMENDED (No coding!)

**Best for**: Quick tests, visual feedback, beginners

#### Step 1: Start the server

```bash
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001 --reload
```

#### Step 2: Open your browser

```
http://127.0.0.1:8001/web
```

#### Step 3: Paste your text and click "Chunk Text"!

That's it! You'll see your chunks with statistics in a beautiful interface.

---

### **Option 2: curl (Command Line)**

**Best for**: Scripts, automation, command-line users

#### Quick Test:

```bash
# Start the server (in one terminal)
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001

# Send a request (in another terminal)
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "text": "Photosynthesis is the process by which plants convert light energy into chemical energy. This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and carbon dioxide as inputs. During photosynthesis, plants absorb light energy through chlorophyll molecules. Cellular respiration is the metabolic process that converts glucose into ATP energy. This process occurs in the mitochondria of cells.",
  "mode": "semantic",
  "min_chunk_size": 200
}
EOF
```

---

### **Option 3: Python Code**

**Best for**: Integration, programmatic use, advanced users

```python
import requests

# Start the server first!
# ./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001

# Your text
text = """
Your text here...
"""

# Call the API
response = requests.post(
    "http://127.0.0.1:8001/chunk",
    json={
        "text": text,
        "mode": "semantic",  # or "simple", "proposition", "hybrid"
        "min_chunk_size": 200
    }
)

# Get results
data = response.json()
print(f"Created {data['total_chunks']} chunks")

for chunk in data['chunks']:
    print(f"\nChunk {chunk['chunk_index'] + 1}:")
    print(f"  Words: {chunk['word_count']}")
    print(f"  Text: {chunk['text'][:100]}...")
```

---

## 🚀 Quick Start (Copy & Paste)

### Terminal 1: Start the Server

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Start the FastAPI server
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001 --reload
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### Terminal 2: Use the Web Interface

```bash
# Open in your browser
open http://127.0.0.1:8001/web
```

**Or use curl:**

```bash
# Save this as test_request.json
cat > test_request.json <<'EOF'
{
  "text": "Put your text here. It can be multiple sentences or paragraphs. The chunker will automatically find semantic boundaries and split your text into meaningful chunks.",
  "mode": "semantic",
  "min_chunk_size": 200
}
EOF

# Send request
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d @test_request.json | jq
```

---

## 📖 API Reference

### Endpoints

#### `GET /`
**Root endpoint** - Lists all available endpoints

```bash
curl http://127.0.0.1:8001/
```

Response:
```json
{
  "service": "LNSP Chunking API",
  "version": "1.0.0",
  "endpoints": {
    "chunk": "POST /chunk",
    "health": "GET /health",
    "stats": "GET /stats",
    "docs": "GET /docs",
    "web_ui": "GET /web"
  }
}
```

---

#### `GET /web`
**Web UI** - Interactive chunking interface

```bash
# Open in browser
open http://127.0.0.1:8001/web
```

---

#### `GET /health`
**Health check** - Check if service is running

```bash
curl http://127.0.0.1:8001/health
```

Response:
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

---

#### `GET /stats`
**Statistics** - Service usage statistics

```bash
curl http://127.0.0.1:8001/stats
```

Response:
```json
{
  "total_requests": 42,
  "total_chunks_created": 156,
  "average_processing_time_ms": 245.3,
  "chunking_mode_usage": {
    "semantic": 35,
    "simple": 5,
    "hybrid": 2
  }
}
```

---

#### `POST /chunk`
**Chunk text** - Main chunking endpoint

**Request:**
```json
{
  "text": "Your text here...",
  "mode": "semantic",
  "max_chunk_size": 320,
  "min_chunk_size": 200,
  "metadata": {
    "document_id": "doc_123",
    "source": "my_source"
  },
  "force_refine": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | ✅ Yes | - | Text to chunk (min 10 chars) |
| `mode` | string | No | `"semantic"` | `"semantic"`, `"simple"`, `"proposition"`, or `"hybrid"` |
| `max_chunk_size` | int | No | 320 | Max words per chunk (50-1000) |
| `min_chunk_size` | int | No | 500 | Min chars per chunk (100-2000) |
| `metadata` | object | No | `null` | Custom metadata to attach |
| `force_refine` | boolean | No | `false` | Force proposition refinement (hybrid mode only) |

**Response:**
```json
{
  "chunks": [
    {
      "text": "Chunk text here...",
      "chunk_id": "a1b2c3d4e5f6g7h8",
      "chunk_index": 0,
      "word_count": 45,
      "char_count": 312,
      "chunking_mode": "semantic",
      "metadata": {
        "document_id": "doc_123",
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
    "chunking_modes": {"semantic": 3}
  },
  "processing_time_ms": 124.5
}
```

---

#### `GET /docs`
**API Documentation** - Interactive Swagger UI

```bash
# Open in browser
open http://127.0.0.1:8001/docs
```

---

## 🎨 Chunking Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **semantic** | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | **Recommended** - Concept-aware chunking |
| **simple** | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | Quick word-count based chunking |
| **proposition** | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | High-precision atomic propositions |
| **hybrid** | ⚡⚡ Balanced | ⭐⭐⭐⭐ Very Good | Semantic + proposition refinement |

---

## 💡 Examples

### Example 1: Simple Request

```bash
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis converts light into energy. Cellular respiration converts glucose into ATP.",
    "mode": "semantic"
  }'
```

### Example 2: With Metadata

```bash
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "semantic",
    "metadata": {
      "document_id": "doc_123",
      "source": "biology_textbook",
      "chapter": 3
    }
  }'
```

### Example 3: Python Integration

```python
import requests

def chunk_text(text, mode="semantic"):
    response = requests.post(
        "http://127.0.0.1:8001/chunk",
        json={"text": text, "mode": mode, "min_chunk_size": 200}
    )
    return response.json()

# Use it
result = chunk_text("Your long text here...")
print(f"Created {result['total_chunks']} chunks")
```

### Example 4: Batch Processing

```python
import requests

texts = [
    "Text 1...",
    "Text 2...",
    "Text 3..."
]

for i, text in enumerate(texts):
    response = requests.post(
        "http://127.0.0.1:8001/chunk",
        json={"text": text, "mode": "semantic"}
    )
    data = response.json()
    print(f"Document {i+1}: {data['total_chunks']} chunks")
```

---

## 🔧 Troubleshooting

### "Connection refused"

**Problem**: Server not running

**Solution**:
```bash
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001 --reload
```

### "No chunks created"

**Problem**: Text too short (< min_chunk_size)

**Solution**: Lower `min_chunk_size` parameter:
```json
{
  "text": "Short text",
  "mode": "semantic",
  "min_chunk_size": 100
}
```

### "ImportError: LlamaIndex not available"

**Problem**: Missing dependencies

**Solution**:
```bash
./.venv/bin/pip install llama-index llama-index-embeddings-huggingface
```

---

## 📊 Web Interface Features

The web interface (`/web`) provides:

- ✅ **Live chunking** - Paste text and see results immediately
- ✅ **Mode selection** - Choose semantic, simple, or hybrid
- ✅ **Configurable** - Adjust min chunk size
- ✅ **Statistics** - Total chunks, avg words, processing time
- ✅ **Beautiful UI** - Modern, responsive design
- ✅ **No coding required** - Just paste and chunk!

---

## 🚀 Production Deployment

### Docker (Recommended)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.api.chunking:app", "--host", "0.0.0.0", "--port", "8001"]
```

```bash
# Build and run
docker build -t lnsp-chunking .
docker run -p 8001:8001 lnsp-chunking
```

### systemd Service

```ini
# /etc/systemd/system/lnsp-chunking.service
[Unit]
Description=LNSP Chunking API
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/lnsp-phase-4
ExecStart=/path/to/lnsp-phase-4/.venv/bin/uvicorn app.api.chunking:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lnsp-chunking
sudo systemctl start lnsp-chunking
```

---

## 📚 Related Documentation

- **Full Chunking Guide**: `docs/howto/how_to_use_semantic_chunker.md`
- **Implementation Details**: `docs/SEMANTIC_CHUNKER_IMPLEMENTATION.md`
- **Quick Reference**: `docs/CHUNKER_QUICK_REFERENCE.md`
- **API Docs** (when running): http://127.0.0.1:8001/docs

---

## ✨ Summary

**The FastAPI interface is the easiest way to use the semantic chunker!**

1. **Start server**: `./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001`
2. **Open web UI**: http://127.0.0.1:8001/web
3. **Paste your text** and click "Chunk Text"
4. **See results** instantly!

No coding required! 🎉

---

**Server currently running at**: http://127.0.0.1:8001
**Web UI**: http://127.0.0.1:8001/web
**API Docs**: http://127.0.0.1:8001/docs
