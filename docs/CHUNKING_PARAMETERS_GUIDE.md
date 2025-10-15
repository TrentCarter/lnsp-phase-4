# Chunking Parameters Guide

Complete guide to controlling chunk generation via FastAPI parameters.

---

## 🎛️ Available Parameters

| Parameter | Type | Default | Range | Effect |
|-----------|------|---------|-------|--------|
| `mode` | string | `semantic` | simple, semantic, proposition, hybrid | Chunking strategy |
| `breakpoint_threshold` | int | `95` | 50-99 | **Lower = more chunks, Higher = fewer chunks** |
| `min_chunk_size` | int | `100` | 10-2000 | Minimum characters per chunk |
| `max_chunk_size` | int | `320` | 50-1000 | Maximum words per chunk |
| `llm_model` | string | `tinyllama:1.1b` | - | LLM for proposition/hybrid modes |

---

## 📊 Chunking Mode Comparison

Based on real testing with sample text (1,314 chars, 200 words):

| Mode | Breakpoint | Chunks | Time | Speed | Use Case |
|------|-----------|--------|------|-------|----------|
| **Simple** | N/A | 15 | 0.17ms | ⚡⚡⚡ | Fast, word-count based |
| **Semantic (95)** | 95 | 2 | 51ms | ⚡⚡ | Fewer, larger semantic chunks |
| **Semantic (75)** | 75 | 5 | 54ms | ⚡⚡ | Medium semantic chunks |
| **Semantic (50)** | 50 | 6 | 50ms | ⚡⚡ | More, smaller semantic chunks |
| **Proposition** | N/A | Varies | ~500ms+ | ⚡ | Atomic facts (LLM-extracted) |

---

## 🔧 Usage Examples

### 1. Simple Mode (Fastest, Most Chunks)

```bash
# Via CLI tool
./tools/pipeline_walkthrough.py \
  --mode simple \
  --max-chunk-size 100

# Via API
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "simple",
    "max_chunk_size": 100
  }'
```

**Result**: 15 chunks, 0.17ms (ultra-fast, word-count based splitting)

---

### 2. Semantic Mode - Default (Fewer Chunks)

```bash
# Via CLI tool
./tools/pipeline_walkthrough.py \
  --mode semantic \
  --breakpoint 95

# Via API
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "semantic",
    "breakpoint_threshold": 95,
    "min_chunk_size": 100
  }'
```

**Result**: 2 chunks, 51ms (high-quality semantic boundaries, fewer splits)

---

### 3. Semantic Mode - Medium Granularity

```bash
# Via CLI tool
./tools/pipeline_walkthrough.py \
  --mode semantic \
  --breakpoint 75 \
  --min-chunk-size 100

# Via API
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "semantic",
    "breakpoint_threshold": 75,
    "min_chunk_size": 100
  }'
```

**Result**: 5 chunks, 54ms (balanced semantic splitting)

---

### 4. Semantic Mode - Fine Granularity (More Chunks)

```bash
# Via CLI tool
./tools/pipeline_walkthrough.py \
  --mode semantic \
  --breakpoint 50 \
  --min-chunk-size 50

# Via API
curl -X POST http://localhost:8001/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "mode": "semantic",
    "breakpoint_threshold": 50,
    "min_chunk_size": 50
  }'
```

**Result**: 6 chunks, 50ms (many small semantic chunks)

---

## 🎯 How Breakpoint Threshold Works

The `breakpoint_threshold` controls **semantic boundary sensitivity**:

```
Breakpoint Scale:
50 ←───────────────────────────────────────────────────────────────→ 99
More chunks                                            Fewer chunks
(split aggressively)                                  (merge similar)
```

### Examples with Same Text (1,314 chars):

| Breakpoint | Chunks | Behavior |
|-----------|--------|----------|
| **50** | 6 | Split at even slight semantic shifts |
| **75** | 5 | Split at moderate semantic boundaries |
| **85** | 3-4 | Split only at clear topic changes |
| **95** | 2 | Merge everything with similar meaning |

---

## 📏 How Chunk Sizes Work

### `min_chunk_size` (characters)

Prevents creating tiny, meaningless chunks.

```bash
# Allow smaller chunks (more granular)
--min-chunk-size 50

# Require substantial chunks (less granular)
--min-chunk-size 200
```

### `max_chunk_size` (words)

Limits chunk length (semantic mode only).

```bash
# Shorter chunks
--max-chunk-size 100

# Longer chunks
--max-chunk-size 500
```

---

## 🚀 Complete Pipeline Examples

### Example 1: Fast Ingestion (Simple Mode)

```bash
./tools/pipeline_walkthrough.py \
  --input data/samples/sample_prompts_1.json \
  --mode simple \
  --max-chunk-size 150
```

**Use case**: Bulk ingestion where semantic precision isn't critical

---

### Example 2: Balanced Semantic (Recommended)

```bash
./tools/pipeline_walkthrough.py \
  --input data/samples/sample_prompts_1.json \
  --mode semantic \
  --breakpoint 75 \
  --min-chunk-size 100
```

**Use case**: Standard ingestion with good semantic boundaries

---

### Example 3: Fine-Grained Semantic

```bash
./tools/pipeline_walkthrough.py \
  --input data/samples/sample_prompts_1.json \
  --mode semantic \
  --breakpoint 50 \
  --min-chunk-size 50 \
  --max-chunk-size 200
```

**Use case**: Maximum granularity for detailed concept extraction

---

### Example 4: Coarse Semantic (Fewer Chunks)

```bash
./tools/pipeline_walkthrough.py \
  --input data/samples/sample_prompts_1.json \
  --mode semantic \
  --breakpoint 95 \
  --min-chunk-size 200
```

**Use case**: Large context windows, fewer API calls

---

## 📈 Performance Trade-offs

### Speed vs Quality

| Mode | Speed | Semantic Quality | Chunk Count | Use Case |
|------|-------|------------------|-------------|----------|
| **Simple** | ⚡⚡⚡ | ⭐ | High | Bulk processing |
| **Semantic (95)** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Low | Large contexts |
| **Semantic (75)** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | **Recommended** |
| **Semantic (50)** | ⚡⚡ | ⭐⭐⭐⭐ | High | Fine-grained |
| **Proposition** | ⚡ | ⭐⭐⭐⭐⭐ | Very High | Maximum precision |

---

## 🔍 Real-World Recommendations

### For Chatbot / RAG Systems

```bash
--mode semantic
--breakpoint 75
--min-chunk-size 100
--max-chunk-size 400
```

**Why**: Balanced chunk size, good semantic boundaries, fast enough

---

### For Training Data Generation

```bash
--mode proposition
--llm-model llama3.1:8b
```

**Why**: Atomic facts, highest quality, suitable for slower offline processing

---

### For Bulk Document Processing

```bash
--mode simple
--max-chunk-size 200
```

**Why**: Fastest, consistent chunk sizes, acceptable for large-scale ingestion

---

### For Scientific Papers / Technical Docs

```bash
--mode semantic
--breakpoint 85
--min-chunk-size 150
--max-chunk-size 500
```

**Why**: Preserves context, respects paragraph boundaries, larger chunks for complex topics

---

## 🧪 Testing Different Configurations

Use the pipeline walkthrough tool to compare settings:

```bash
# Test 1: Default semantic
./tools/pipeline_walkthrough.py --breakpoint 95

# Test 2: Medium semantic
./tools/pipeline_walkthrough.py --breakpoint 75

# Test 3: Fine semantic
./tools/pipeline_walkthrough.py --breakpoint 50

# Test 4: Simple mode
./tools/pipeline_walkthrough.py --mode simple
```

Compare the results:
- Chunk count
- Processing time
- Chunk size distribution

---

## 📊 Breakpoint Threshold Tuning Guide

### Starting Point: 75 (Recommended)

Start here for most use cases, then adjust:

```
Too many chunks? → Increase breakpoint (80, 85, 90)
Too few chunks?  → Decrease breakpoint (70, 65, 60)
```

### Example Tuning Session

```bash
# Baseline
./tools/pipeline_walkthrough.py --breakpoint 75  # → 5 chunks

# Too many chunks? Try higher
./tools/pipeline_walkthrough.py --breakpoint 85  # → 3 chunks

# Too few chunks? Try lower
./tools/pipeline_walkthrough.py --breakpoint 65  # → 7 chunks

# Just right!
./tools/pipeline_walkthrough.py --breakpoint 75  # → 5 chunks ✓
```

---

## 🎓 Summary

**Key takeaways**:
1. ✅ **Lower breakpoint = more chunks** (50 = many, 95 = few)
2. ✅ **Semantic mode** is best balance of speed + quality
3. ✅ **Simple mode** is fastest but ignores semantics
4. ✅ **Start with breakpoint=75** and tune from there
5. ✅ **All parameters** controllable via CLI or API

**Test command**:
```bash
./tools/pipeline_walkthrough.py --help
```
