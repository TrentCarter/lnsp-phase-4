# 🎛️ Chunking Settings - Complete Guide

**How to get the perfect number of chunks for your text**

---

## ⚙️ Two Key Settings

### 1. **Min Chunk Size** (Characters)

Controls HOW SMALL chunks can be.

| Setting | Result |
|---------|--------|
| 20-50 | Many tiny chunks |
| 100 (default) | Balanced |
| 200-500 | Large chunks |

**Lower = More chunks**
**Higher = Fewer chunks**

---

### 2. **Breakpoint Threshold** (50-99)

Controls HOW SIMILAR text must be to stay in same chunk.

| Setting | Result |
|---------|--------|
| 50-70 | Many chunks (splits aggressively) |
| 85 | Balanced - Good for 3+ concepts ⭐ |
| 95 (default) | Fewer chunks (keeps similar text together) |

**Lower = More chunks**
**Higher = Fewer chunks**

---

## 🎯 Getting 3 Chunks from 3 Concepts

### Problem

Your text has 3 distinct concepts (Photosynthesis, Cellular Respiration, Water Cycle) but only creates 2 chunks.

### Solution

**Lower the breakpoint threshold to 85:**

```json
{
  "text": "Your 3-concept text...",
  "mode": "semantic",
  "min_chunk_size": 20,
  "breakpoint_threshold": 85
}
```

**Result**: 3 chunks! ✅

---

## 📊 Recommended Settings

### For 3+ Distinct Concepts

```
Min Chunk Size: 20-50
Breakpoint Threshold: 85
Mode: Semantic
```

**Expected**: One chunk per concept

---

### For Fine-Grained Analysis

```
Min Chunk Size: 20
Breakpoint Threshold: 75
Mode: Semantic
```

**Expected**: Many small chunks

---

### For General Use (Default)

```
Min Chunk Size: 100
Breakpoint Threshold: 95
Mode: Semantic
```

**Expected**: 2-3 chunks per document

---

### For Large Segments

```
Min Chunk Size: 300
Breakpoint Threshold: 95
Mode: Semantic
```

**Expected**: Fewest chunks possible

---

## 🔧 How to Adjust in Web UI

The web UI now shows a **"Breakpoint Threshold" slider**!

1. Open http://127.0.0.1:8001/web
2. Paste your text
3. Set **Min Chunk Size** to 20-100 (smaller = more chunks)
4. Adjust **Breakpoint Threshold** slider (50-99):
   - **85** for multi-concept texts ⭐
   - **75** for very fine-grained splitting
   - **95** for general use (default)
5. Click "Chunk Text"

**Or test via curl:**

```bash
curl -X POST http://127.0.0.1:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text...",
    "mode": "semantic",
    "min_chunk_size": 20,
    "breakpoint_threshold": 85
  }'
```

---

## 💡 Understanding Breakpoint Threshold

The semantic chunker compares sentences using embeddings:

- **95** (default): Only split when sentences are VERY different
  - Result: Keeps related concepts together
  - Best for: General documents

- **85**: Split when sentences are moderately different
  - Result: Separates distinct concepts
  - Best for: Multi-concept documents ⭐

- **75**: Split when sentences are slightly different
  - Result: Many fine-grained chunks
  - Best for: Detailed analysis

---

## 📈 Examples

### Example 1: Default Settings (2 chunks)

```json
{
  "min_chunk_size": 100,
  "breakpoint_threshold": 95
}
```

**Result**: Photosynthesis + Cellular Respiration + Water Cycle → **2 chunks**
- Chunk 1: Photosynthesis + Cellular Respiration
- Chunk 2: Water Cycle

**Why**: Cellular respiration and water cycle are semantically similar (both natural processes), so threshold=95 keeps them together.

---

### Example 2: Lower Threshold (3 chunks) ✅

```json
{
  "min_chunk_size": 20,
  "breakpoint_threshold": 85
}
```

**Result**: Photosynthesis + Cellular Respiration + Water Cycle → **3 chunks**
- Chunk 1: Photosynthesis
- Chunk 2: Cellular Respiration
- Chunk 3: Water Cycle

**Why**: threshold=85 is more sensitive to differences, so it separates the concepts.

---

### Example 3: Very Low Threshold (5+ chunks)

```json
{
  "min_chunk_size": 20,
  "breakpoint_threshold": 70
}
```

**Result**: Photosynthesis + Cellular Respiration + Water Cycle → **5+ chunks**
- Each paragraph or major point becomes its own chunk

**Why**: threshold=70 splits very aggressively.

---

## 🎯 Quick Settings Selector

**I want more chunks:**
- ✅ Lower `min_chunk_size` (50 → 20)
- ✅ Lower `breakpoint_threshold` (95 → 85)

**I want fewer chunks:**
- ✅ Raise `min_chunk_size` (100 → 300)
- ✅ Raise `breakpoint_threshold` (85 → 95)

**I want exactly N chunks for N concepts:**
- ✅ Set `breakpoint_threshold` to 85
- ✅ Set `min_chunk_size` to 20

---

## ⚡ All Modes Support These Settings

- ✅ **Simple**: Uses `min_chunk_size` (ignores breakpoint_threshold)
- ✅ **Semantic**: Uses both `min_chunk_size` and `breakpoint_threshold`
- ✅ **Hybrid**: Uses both (inherits from semantic)
- ✅ **Proposition**: Uses `min_chunk_size` (ignores breakpoint_threshold)

---

## 🚀 Test It Now

Open the web interface: http://127.0.0.1:8001/web

Then experiment with:
1. Min chunk size: Try 20, 50, 100, 200
2. Breakpoint threshold: Try 75, 85, 95
3. See how many chunks you get!

---

## ✅ Summary

**Problem**: 3 concepts → 2 chunks
**Solution**: Set `breakpoint_threshold=85`
**Result**: 3 concepts → 3 chunks ✅

**Quick Settings for 3 Concepts:**
```json
{
  "min_chunk_size": 20,
  "breakpoint_threshold": 85,
  "mode": "semantic"
}
```

---

**Now you have full control over chunking!** 🎉
