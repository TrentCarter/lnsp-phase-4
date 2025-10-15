# ✨ LNSP Semantic Chunker - Get Started in 30 Seconds

The easiest way to chunk your text into semantic segments!

---

## 🚀 Quick Start

### 1. Start the server

```bash
./start_chunking_api.sh
```

### 2. Open your browser

```
http://127.0.0.1:8001/web
```

### 3. Paste your text and click "Chunk Text"!

That's it! 🎉

---

## 📖 What You Get

✅ **Web Interface** - No coding required, just paste and chunk!
✅ **REST API** - Easy integration with any language
✅ **3 Chunking Modes** - Semantic (recommended), Simple, Hybrid
✅ **Fast** - Chunks 1000s of words per second
✅ **Accurate** - Respects concept boundaries

---

## 🎯 Three Ways to Use It

### Option 1: Web UI (Easiest!) ⭐

1. Start: `./start_chunking_api.sh`
2. Open: http://127.0.0.1:8001/web
3. Paste your text
4. Click "Chunk Text"
5. See results!

### Option 2: curl (Command Line)

```bash
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "text": "Your text here...",
  "mode": "semantic"
}
EOF
```

### Option 3: Python Code

```python
import requests

response = requests.post(
    "http://127.0.0.1:8001/chunk",
    json={"text": "Your text here...", "mode": "semantic"}
)

data = response.json()
print(f"Created {data['total_chunks']} chunks")
```

---

## 🔍 Chunking Modes

| Mode | Best For |
|------|----------|
| **semantic** ⭐ | Concept-aware chunking (recommended) |
| **simple** | Fast word-count based chunking |
| **hybrid** | High-quality with LLM refinement |

---

## 📚 Documentation

- **⭐ Complete Guide**: `CHUNKING_API_COMPLETE_GUIDE.md` (everything you need!)
- **Quick Reference**: `CHUNKING_QUICK_REFERENCE.md` (cheat sheet)
- **Settings Guide**: `CHUNKING_SETTINGS_GUIDE.md` (parameter tuning)
- **Web UI Guide**: `WEB_CHUNKER_GUIDE.md` (web interface)
- **API Docs** (when running): http://127.0.0.1:8001/docs

---

## 🛠️ Installation

If you haven't set up yet:

```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies
./.venv/bin/pip install llama-index llama-index-embeddings-huggingface

# Start the server
./start_chunking_api.sh
```

---

## ✨ Features

### Web Interface

- 🎨 Beautiful, modern UI
- ⚡ Real-time chunking
- 📊 Statistics and visualizations
- 🔧 Configurable settings
- 📱 Mobile-friendly

### API

- 🚀 Fast and reliable
- 📖 OpenAPI documentation
- 🔄 RESTful design
- 📈 Usage statistics
- ❤️ Health monitoring

---

## 📊 Example Output

```json
{
  "chunks": [
    {
      "text": "Photosynthesis is the process by which plants convert light energy...",
      "word_count": 45,
      "chunk_id": "a1b2c3d4",
      "chunking_mode": "semantic"
    }
  ],
  "total_chunks": 3,
  "statistics": {
    "mean_words": 45.2,
    "processing_time_ms": 124.5
  }
}
```

---

## 🆘 Troubleshooting

### Server won't start?

```bash
# Check if dependencies are installed
./.venv/bin/pip install llama-index llama-index-embeddings-huggingface

# Try starting manually
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001
```

### No chunks created?

Lower the `min_chunk_size` parameter to 100-200 chars.

### Need help?

Check the full guide: `docs/FASTAPI_CHUNKING_GUIDE.md`

---

## 🎉 That's It!

**Ready to chunk?**

```bash
./start_chunking_api.sh
```

Then open http://127.0.0.1:8001/web in your browser!

---

**Made with ❤️ for LNSP TMD-LS Pipeline**
