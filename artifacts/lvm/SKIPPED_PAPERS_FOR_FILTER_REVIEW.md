# Skipped Papers Analysis - Filter Review

**Date**: 2025-11-04
**Issue**: V2 pre-cleaning filter rejected 82% of papers (3,045 / 3,715)
**Root Cause**: Text files are **single-line** (entire paper on one line, no `\n` characters)

**Impact**: Filter processes entire 40KB-195KB paper as ONE "line" → applies line-level regex rules to whole paper → rejects everything

---

## 📊 Summary Statistics

- **Total papers downloaded**: 3,715
- **Papers processed successfully**: 619 (17%)
- **Papers skipped**: 3,045 (82%)
- **Final papers after deduplication**: 619
- **Total vectors generated**: 111,825 (768D)

**Filter rejection rate**: 82% is TOO AGGRESSIVE

---

## 📄 Sample: First 20 Skipped Papers

### Paper 1: 2510.27688v1
**Title**: Continuous Autoregressive Language Models
**File size**: 88KB
**Problem**: Entire paper on one line (0 newlines)

**Content excerpt** (first 500 chars):
```
Preprint CONTINUOUS AUTOREGRESSIVE LANGUAGE MODELS Chenze Shao1, Darren Li1,2, Fandong Meng1∗, Jie Zhou1 1WeChat AI, Tencent Inc 2Qiuzhen College, Tsinghua University ABSTRACT The efficiency of large language models (LLMs) is fundamentally limited by their sequential, token-by-token generation process. We argue that overcoming this bottleneck requires a new design axis for LLM scaling: increasing the semantic bandwidth of each generative step. To this end, we introduce Continuous Autore-
```

**Filter behavior**:
- Sees this as ONE line containing entire 88KB paper
- Applies alphanumeric ratio check: (alphanumeric chars / 88,000 chars) likely < 0.6
- **REJECTED** as "ASCII art" or "code-like"

**Verdict**: ❌ **FALSE POSITIVE** - This is clean prose, should be kept!

---

### Paper 2: 2510.27671v1
**Title**: MolChord: Structure-Sequence Alignment for Protein-Guided Drug Design
**File size**: 47KB
**Problem**: Single-line file

---

### Paper 3: 2510.27659v1
**Title**: Challenges in Credit Assignment for Multi-Agent Reinforcement Learning
**File size**: 64KB
**Problem**: Single-line file

---

### Papers 4-20
All have the same issue: **Text extraction created single-line files instead of multi-line**

---

## 🔧 Filter Problems Identified

### Problem 1: No Newline Handling
**Current code**:
```python
for line in raw_text.splitlines():
    line_stripped = line.strip()
    # ... apply filters
```

**When file has no newlines**:
- `splitlines()` returns `['<entire 88KB paper as one string>']`
- Filter processes entire paper as ONE line
- Alphanumeric ratio check: `sum(c.isalnum() for c in 88KB_string) / len(88KB_string)`
- Result: Rejects legitimate prose papers

### Problem 2: Alphanumeric Ratio Too Strict
```python
if total_chars > 20 and (alphanumeric_chars / total_chars) < 0.6:
    continue  # Reject line
```

**Issue**:
- Designed for filtering ASCII art lines like `|---|---|---|`
- When applied to 88KB of prose containing math symbols, citations, etc.
- Ratio drops below 60% → entire paper rejected

### Problem 3: Missing Pre-Splitting Logic
The filter assumes text is already line-by-line (paragraphs separated by `\n\n`).

PDFs extracted as **one giant line** need preprocessing:
1. Add newlines after sentence-ending punctuation
2. Add double newlines for paragraph breaks
3. **THEN** apply line-level filters

---

## ✅ Recommended Fixes

### Fix 1: Add Text Normalization (BEFORE filtering)
```python
def normalize_single_line_text(raw_text: str) -> str:
    """
    Handle text files with no newlines (entire paper on one line).
    Add newlines after sentences and double newlines for paragraphs.
    """
    # Add newline after sentence-ending punctuation followed by space and capital
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', raw_text)

    # Add double newline for section headers (e.g., "INTRODUCTION", "METHODS")
    text = re.sub(r'\n([A-Z\s]{3,})\n', r'\n\n\1\n\n', text)

    return text
```

### Fix 2: Relax Alphanumeric Ratio for Long Lines
```python
# Only apply strict ratio check to SHORT lines (likely ASCII art)
if 20 < total_chars < 200:  # Short lines only
    alphanumeric_chars = sum(c.isalnum() for c in line_stripped)
    if (alphanumeric_chars / total_chars) < 0.6:
        continue  # Reject as ASCII art
```

### Fix 3: Skip Alphanumeric Check for Very Long Lines
```python
# If line is > 1000 chars, it's likely a paragraph, not ASCII art
if total_chars > 1000:
    # Skip alphanumeric ratio check
    pass
```

---

## 🎯 Expected Impact of Fixes

**Current**:
- 82% rejection rate
- 619 papers processed

**After fixes**:
- **Expected: 40-50% rejection rate** (still filter out table-heavy papers)
- **Expected: 1,500-2,000 papers processed**
- **Expected: ~270k-360k vectors** (2.4x-3.2x increase)

---

## 📋 Action Items for Consultant

1. **Verify root cause**: Check if all skipped papers are single-line files:
   ```bash
   wc -l data/datasets/arxiv/pdfs/2510.276*.txt | head -20
   ```

2. **Implement Fix 1**: Add `normalize_single_line_text()` function

3. **Implement Fix 2 OR Fix 3**: Relax alphanumeric ratio check

4. **Test on sample**: Re-run ingestion on first 100 skipped papers, verify acceptance rate

5. **Re-ingest full dataset**: If test passes, re-run full 3,715 papers

---

## 📊 Current vs. Target Results

| Metric | Current (V2) | Target (V2.1) | Improvement |
|--------|-------------|---------------|-------------|
| Papers processed | 619 (17%) | 1,500-2,000 (40-54%) | 2.4-3.2x |
| Papers skipped | 3,045 (82%) | 1,700-2,200 (46-60%) | -43% |
| Total vectors | 111k | 270k-360k | 2.4-3.2x |
| Δ (forward bias) | +0.06 | +0.08-0.10 (target) | TBD |

---

**Generated**: 2025-11-04
**For**: Filter tuning consultation
**Next step**: Implement fixes and re-test on sample papers
