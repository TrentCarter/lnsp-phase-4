# Chunking Tools - Quick Reference

Three simple tools to test the semantic chunker with your own text.

---

## 🚀 Quick Start (Copy & Paste)

### **Option 1: Super Simple** (`quick_chunk.py`) ⭐ RECOMMENDED

**Best for**: Quick tests, paste and run

```bash
# 1. Edit tools/quick_chunk.py
# 2. Replace MY_TEXT with your text
# 3. Run:
python tools/quick_chunk.py
```

**Example Output**:
```
================================================================================
SEMANTIC CHUNKER - Quick Results
================================================================================

📊 Input: 1316 chars, 200 words
✓ Created 2 chunks

────────────────────────────────────────────────────────────────────────────────
CHUNK 1
────────────────────────────────────────────────────────────────────────────────
118 words | 780 chars

Photosynthesis is the process by which plants convert light energy...
```

---

### **Option 2: Full Featured** (`chunk_my_text.py`)

**Best for**: Detailed analysis, comparing modes, statistics

```bash
# 1. Edit tools/chunk_my_text.py
# 2. Replace YOUR_TEXT_HERE with your text
# 3. Run:
python tools/chunk_my_text.py

# Or with comparison mode:
python tools/chunk_my_text.py --compare

# Or from command line:
python tools/chunk_my_text.py --text "Your text here"
```

**Features**:
- ✅ Detailed chunk information (ID, word count, char count)
- ✅ Statistics (mean, range, distribution)
- ✅ Comparison mode (simple vs semantic)
- ✅ Command-line text input

---

### **Option 3: Concept Test** (`test_chunker_concepts.py`)

**Best for**: Understanding how chunker handles multiple concepts

```bash
# Run pre-configured test with 1, 2, and 3 concepts:
python tools/test_chunker_concepts.py
```

**Shows**:
- How semantic chunking respects concept boundaries
- Side-by-side comparison of simple vs semantic modes
- Performance with different concept densities

---

## 📝 How to Use

### Quick Chunk (Simplest)

1. Open `tools/quick_chunk.py`
2. Find the section marked `PASTE YOUR TEXT HERE`
3. Replace the text between the triple quotes
4. Run: `python tools/quick_chunk.py`

**Example**:
```python
MY_TEXT = """
Your text goes here.
It can be multiple paragraphs.
The chunker will find semantic boundaries automatically.
"""
```

---

### Full Featured Chunk

1. Open `tools/chunk_my_text.py`
2. Find the section marked `PUT YOUR TEXT HERE`
3. Replace `YOUR_TEXT_HERE` with your text
4. Run: `python tools/chunk_my_text.py`

**Options**:
```bash
# Show detailed results
python tools/chunk_my_text.py

# Compare simple vs semantic modes
python tools/chunk_my_text.py --compare

# Chunk text from command line
python tools/chunk_my_text.py --text "Put your text in quotes"
```

---

## 🎯 Which Tool Should I Use?

| Tool | Use When | Features |
|------|----------|----------|
| **quick_chunk.py** | Just want to see chunks quickly | ⚡ Minimal output |
| **chunk_my_text.py** | Need detailed analysis | 📊 Full statistics |
| **test_chunker_concepts.py** | Want to understand chunking behavior | 🧪 Pre-configured tests |

---

## 💡 Examples

### Example 1: Test Your Article

```bash
# Edit quick_chunk.py and paste your article text
MY_TEXT = """
Your article text here...
"""

# Run it
python tools/quick_chunk.py
```

### Example 2: Compare Modes

```bash
# Use chunk_my_text.py with --compare flag
python tools/chunk_my_text.py --compare
```

You'll see:
- **Simple Mode**: Word-count based chunks (may split concepts)
- **Semantic Mode**: Concept-aware chunks (preserves boundaries)

### Example 3: Command-Line Quick Test

```bash
python tools/chunk_my_text.py --text "Photosynthesis converts light into energy. Cellular respiration converts glucose into ATP."
```

---

## 🔍 Understanding the Output

### Chunk Information

Each chunk shows:
- **Words**: Number of words in chunk
- **Characters**: Number of characters
- **Chunk ID**: Unique identifier (MD5 hash)
- **Text**: The actual chunk content

### Statistics

- **Total chunks**: How many chunks were created
- **Mean words**: Average words per chunk
- **Word count range**: Min and max words in chunks
- **Distribution**: Breakdown by size (0-100, 100-200, etc.)

---

## 🎓 Tips

### Getting Better Chunks

1. **Use semantic mode** (default in these tools) for concept-aware chunking
2. **Longer text works better** - semantic chunker needs enough context
3. **Clear paragraph breaks** help the chunker identify concept boundaries

### Minimum Text Length

- Semantic chunker needs **at least ~200 characters** per chunk
- For very short text (< 200 chars), you'll get 0 chunks
- Solution: Use simple mode for very short texts

### Adjusting Chunk Size

Edit the `min_chunk_size` parameter:

```python
# In quick_chunk.py or chunk_my_text.py
chunker = SemanticChunker(min_chunk_size=300)  # Larger chunks
chunker = SemanticChunker(min_chunk_size=100)  # Smaller chunks
```

---

## 🚨 Troubleshooting

### "No chunks created"

**Problem**: Text is too short (< min_chunk_size)

**Solution**:
```python
# Lower the minimum size
chunker = SemanticChunker(min_chunk_size=100)
```

### "LlamaIndex not available"

**Problem**: Missing dependency

**Solution**:
```bash
pip install llama-index llama-index-embeddings-huggingface
```

### "Import error"

**Problem**: Running from wrong directory

**Solution**:
```bash
# Run from project root
cd /path/to/lnsp-phase-4
python tools/quick_chunk.py
```

---

## 📚 More Information

- **Full Documentation**: `docs/howto/how_to_use_semantic_chunker.md`
- **Implementation Details**: `docs/SEMANTIC_CHUNKER_IMPLEMENTATION.md`
- **Quick Reference**: `docs/CHUNKER_QUICK_REFERENCE.md`

---

## ✨ Quick Command Reference

```bash
# Simplest: Edit and run
python tools/quick_chunk.py

# Full featured: Edit and run
python tools/chunk_my_text.py

# With comparison
python tools/chunk_my_text.py --compare

# Command line input
python tools/chunk_my_text.py --text "Your text"

# Pre-configured concept test
python tools/test_chunker_concepts.py
```

---

**Recommended**: Start with `quick_chunk.py` - it's the simplest! 🚀
