# 🎨 Web Chunker - Quick Guide

**The easiest way to chunk your text!**

---

## 🚀 Start the Server

```bash
./start_chunking_api.sh
```

Then open: **http://127.0.0.1:8001/web**

---

## 🎯 How to Use

### Step 1: Paste Your Text

Copy and paste any text into the large text box.

### Step 2: Configure Settings (Optional)

- **Chunking Mode**: Choose `Semantic` (recommended) for concept-aware chunking
- **Min Chunk Size**: Set to desired minimum (default: 100 chars)
  - **Smaller = More chunks** (20-50 chars for fine-grained)
  - **Larger = Fewer chunks** (200-500 chars for larger segments)
- **Breakpoint Threshold**: Control semantic boundary sensitivity (50-99)
  - **85** = Recommended for multi-concept texts ⭐
  - **75** = Very fine-grained splitting (many chunks)
  - **95** = Conservative (fewer, larger chunks)

### Step 3: Click "Chunk Text"

Watch it chunk in real-time!

---

## ⚙️ Settings Guide

### Min Chunk Size (Characters)

This controls how small chunks can be:

| Setting | Result | Use When |
|---------|--------|----------|
| **20-50** | Many small chunks | Want fine-grained segments |
| **100** (default) | Balanced chunks | General use ✅ |
| **200-500** | Larger chunks | Want complete paragraphs |

**Examples:**

```
Min = 20:  "Photosynthesis converts..." (1 sentence chunk)
Min = 100: "Photosynthesis converts... This occurs..." (2-3 sentences)
Min = 500: "Photosynthesis converts... [full paragraph]..." (complete concept)
```

### Chunking Modes

| Mode | How It Works | Best For |
|------|--------------|----------|
| **Semantic** ⭐ | Finds natural concept boundaries | General use (RECOMMENDED) |
| **Simple** | Word-count based splitting | Quick processing |
| **Hybrid** | Semantic + LLM refinement | High-precision needs |

---

## 💡 Tips

### Get More Chunks

✅ **Lower the Min Chunk Size** to 20-50 chars
```
20 chars → Many small chunks
50 chars → Medium chunks
100 chars → Balanced (default)
```

### Get Fewer Chunks

✅ **Raise the Min Chunk Size** to 200-500 chars
```
100 chars → Balanced (default)
200 chars → Larger chunks
500 chars → Fewest chunks
```

### Get Better Chunk Boundaries

✅ **Use Semantic mode** (default)
- Respects concept boundaries
- Doesn't split in middle of ideas
- Better for TMD-LS pipeline

---

## 📊 Understanding Results

### Statistics

- **Total Chunks**: How many chunks created
- **Avg Words/Chunk**: Average chunk size
- **Processing Time**: How long it took

### Each Chunk Shows

- **Words**: Number of words in chunk
- **Chars**: Number of characters
- **Mode**: How it was chunked (semantic/simple/hybrid)
- **Text**: The actual chunk content

---

## 🎓 Example Settings

### For Fine-Grained Analysis

```
Mode: Semantic
Min Chunk Size: 20
```

**Result**: Many small, concept-aware chunks

### For General Use (Recommended) ⭐

```
Mode: Semantic
Min Chunk Size: 100
```

**Result**: Balanced chunks with good concept boundaries

### For Large Segments

```
Mode: Semantic
Min Chunk Size: 300
```

**Result**: Fewer, larger chunks with complete concepts

---

## 🔧 Troubleshooting

### "No chunks created"

**Cause**: Min chunk size is larger than your text

**Fix**: Lower the min chunk size to 20-50

### "Only 1 chunk created"

**Cause**: Min chunk size is too large for your text

**Fix**: Lower the min chunk size

### "Too many small chunks"

**Cause**: Min chunk size is too small

**Fix**: Raise the min chunk size to 100-200

---

## ✨ Quick Settings for Common Cases

### Chunking a Short Paragraph (< 500 words)

```
Min Chunk Size: 50
Mode: Semantic
```

### Chunking a Long Article (> 1000 words)

```
Min Chunk Size: 200
Mode: Semantic
```

### Chunking for TMD-LS Pipeline

```
Min Chunk Size: 100
Mode: Semantic
```

### Maximum Precision (Law/Medicine)

```
Min Chunk Size: 100
Mode: Hybrid
```

---

## 🎯 Your Settings

Based on your selected text (3 concepts about Photosynthesis, Cellular Respiration, and Water Cycle):

**Recommended Settings:**

```
Min Chunk Size: 100
Mode: Semantic
```

**Expected Result**: ~2-3 chunks, one per concept

**To get more chunks**: Lower to 50
**To get fewer chunks**: Raise to 200

---

## 📖 Current Server

If the server is running, you can access:

- **Web UI**: http://127.0.0.1:8001/web
- **API Docs**: http://127.0.0.1:8001/docs
- **Health Check**: http://127.0.0.1:8001/health

---

## ✅ Summary

1. **Start**: `./start_chunking_api.sh`
2. **Open**: http://127.0.0.1:8001/web
3. **Paste** your text
4. **Set min chunk size** (20-500 chars)
   - **20-50** = Many chunks
   - **100** = Balanced (default) ✅
   - **200-500** = Fewer chunks
5. **Click** "Chunk Text"
6. **Done!** 🎉

---

**Now supports min chunk sizes from 10 to 2000 characters!**
