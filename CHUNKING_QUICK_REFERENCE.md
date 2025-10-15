# Semantic Chunking API - Quick Reference Card

## 🚀 Start Server
```bash
./start_chunking_api.sh
```
Open: **http://127.0.0.1:8001/web**

---

## 🎛️ Modes

| Mode | Speed | Use Case | Requires LLM |
|------|-------|----------|--------------|
| **Simple** | 0.5ms | Quick word-count splitting | ❌ |
| **Semantic** ⭐ | 50-100ms | Concept boundaries (recommended) | ❌ |
| **Proposition** | 266ms-2s | Atomic semantic units | ✅ |
| **Hybrid** | 100ms+ | Semantic + LLM refinement | ✅ |

---

## ⚙️ Key Parameters

### Min Chunk Size
- **Range**: 10-2000 characters
- **Default**: 100
- **Lower** = more chunks | **Higher** = fewer chunks

### Breakpoint Threshold (Semantic/Hybrid only)
- **Range**: 50-99
- **Default**: 85
- **Lower (70-80)** = aggressive splitting (more chunks)
- **Higher (85-95)** = conservative (fewer chunks)

---

## 🤖 LLM Models (Proposition/Hybrid)

| Model | Port | Speed | Best For |
|-------|------|-------|----------|
| **Granite3 MoE** ⚡ | 11437 | 266ms | Production |
| **TinyLlama** | 11435 | 716ms | Batch |
| **Phi3 Mini** | 11436 | 946ms | Precision |
| **Llama 3.1** | 11434 | 1614ms | Quality |

---

## 📡 API Endpoints

### POST /chunk
```bash
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text...", "mode": "semantic", "min_chunk_size": 100, "breakpoint_threshold": 85}'
```

### GET /health
```bash
curl http://127.0.0.1:8001/health
```

### GET /stats
```bash
curl http://127.0.0.1:8001/stats
```

---

## 🎯 Common Tasks

### Get 3 Chunks from 3 Concepts
```json
{
  "mode": "semantic",
  "min_chunk_size": 20,
  "breakpoint_threshold": 75
}
```

### Fast Chunking
```json
{
  "mode": "semantic",
  "min_chunk_size": 100,
  "breakpoint_threshold": 95
}
```

### High Quality
```json
{
  "mode": "hybrid",
  "llm_model": "llama3.1:8b",
  "min_chunk_size": 100
}
```

### Ultra Fast
```json
{
  "mode": "simple",
  "min_chunk_size": 100
}
```

---

## 🐛 Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| No chunks | Lower `min_chunk_size` to 20 |
| Too few chunks | Set `breakpoint_threshold=75` |
| Too many chunks | Set `breakpoint_threshold=95` |
| Slow processing | Use Semantic mode or Granite3 model |
| No LLM dropdown | Select Proposition/Hybrid mode |
| Model not working | Check Ollama: `curl http://localhost:11434/api/tags` |

---

## 📚 Full Documentation
- **Complete Guide**: `CHUNKING_API_COMPLETE_GUIDE.md`
- **Settings Guide**: `CHUNKING_SETTINGS_GUIDE.md`
- **Web UI Guide**: `WEB_CHUNKER_GUIDE.md`
- **30-Second Start**: `CHUNKING_README.md`

---

## ✅ Status
- ✅ Optimized (50x faster than initial implementation)
- ✅ 4 modes, multi-LLM support
- ✅ Production-ready
- ✅ Fully documented
