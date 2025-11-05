# arXiv Download Status & Strategy

**Date**: 2025-11-04
**Status**: Ready to start 50k download with category-separated strategy

---

## 🔍 Problem Identified

**Issue**: arXiv API multi-category queries stop at ~100-215 records
**Root Cause**: arXiv API appears to have limits/bugs with OR queries `(cat:cs.CL OR cat:cs.LG OR ...)`
**Evidence**:
- Multi-category (4 cats): 100 records (requested 500) ❌
- Single category (cs.CL): 300 records (requested 300) ✅
- Single category (cs.CL): 215 records (requested 1000) ⚠️

---

## ✅ Solution: Category-Separated Download

**Strategy**: Download each category separately, then combine
**Script**: `scripts/download_arxiv_multi_category.sh`
**Categories**:
- cs.CL (Computation and Language) - Target: 12,500 papers
- cs.LG (Machine Learning) - Target: 12,500 papers
- stat.ML (Statistics - Machine Learning) - Target: 12,500 papers
- cs.AI (Artificial Intelligence) - Target: 12,500 papers

**Total Target**: 50,000 papers

---

## 📊 Validation: Text Extraction Works!

**Test Results** (100 papers):
- ✅ PDF download: Working
- ✅ Text extraction (pymupdf): Working
- ✅ Full text size: 38KB-195KB per paper (NOT abstracts!)
- ✅ File format: JSONL with `fulltext_path` field

**Example**:
```
Paper: 2510.27688v1
Text file: data/datasets/arxiv/pdfs/2510.27688v1.txt
Size: 115KB (full paper, not 4KB abstract)
```

---

## ⏱️ Time Estimates

**Per Paper**:
- API fetch: ~0.5s (rate limiting)
- PDF download: ~1-2s
- Text extraction: ~0.5s
- Sleep delay: 1s (hardcoded in script)
- **Total**: ~3-4s per paper

**Full 50k Papers**:
- Optimistic: 150,000s = 41.7 hours
- Realistic: 180,000s = **50 hours** (2 days)
- With retries/errors: **60 hours** (2.5 days)

---

## 🚀 Recommended Next Steps

### Option 1: Start Full 50k Download (2-3 days)
```bash
nohup ./scripts/download_arxiv_multi_category.sh \
  > logs/arxiv_download_multicategory.log 2>&1 &

# Monitor:
tail -f logs/arxiv_download_multicategory.log
```

**Pros**:
- Gets full Phase 1 dataset (250k-500k vectors)
- Production-ready for LVM training
- Comprehensive coverage

**Cons**:
- 2-3 days wait time
- Requires stable machine/connection

---

### Option 2: Download Smaller Pilot (6-12 hours)
```bash
# Modify script to download 2,500 per category = 10k total
# Edit line: PER_CATEGORY=2500

nohup ./scripts/download_arxiv_multi_category.sh \
  > logs/arxiv_download_10k.log 2>&1 &
```

**Pros**:
- 6-12 hours (overnight)
- Still gets 50k-100k vectors (sufficient for pilot)
- Can validate Δ measurement sooner

**Cons**:
- May need to re-download later for full training

---

### Option 3: Use Existing Data (Immediate)
```bash
# Process the 100 full papers we already have
python tools/ingest_arxiv_to_npz_simple.py \
  --input data/datasets/arxiv/arxiv_partial_100.jsonl \
  --output artifacts/lvm/arxiv_100_papers_full.npz
```

**Pros**:
- Immediate (no wait)
- Validates full pipeline end-to-end
- Tests P6b v2.3 architecture

**Cons**:
- Only ~500-1,000 vectors (too small for production)
- Can't measure Δ reliably (need 5k+ sequences)

---

## 💡 My Recommendation

**Two-Phase Approach**:

1. **NOW (5 minutes)**: Process existing 100 papers
   - Validate ingestion pipeline
   - Test P6b v2.3 architecture
   - Confirm text extraction quality

2. **PARALLEL (overnight)**: Start 10k download
   - Modify script: `PER_CATEGORY=2500`
   - Run overnight (6-12 hours)
   - Gets 50k-100k vectors (sufficient for Phase 1)

3. **NEXT SESSION (after validation)**: Decide on full 50k
   - If Δ measurement good → continue to 50k
   - If architecture issues → fix before big download

---

## 📁 Current Status

**Downloaded So Far**:
- 433 text files in `data/datasets/arxiv/pdfs/`
- 100 papers with metadata in `arxiv_partial_100.jsonl`
- Text files: 38KB-195KB each (full papers ✅)

**Ready to Process**:
- ✅ Custom paragraph chunker (677x faster than alternatives)
- ✅ Ingestion pipeline (`ingest_arxiv_to_npz_simple.py`)
- ✅ Sequence creation (`create_arxiv_sequences_simple.py`)
- ✅ Δ measurement tool (`diagnose_data_direction.py`)

**Next Blocker**:
- User decision: Start 10k download now, or 50k download, or process existing 100?

---

## 🎯 Success Criteria

**Pilot (10k papers)**:
- Papers: 10,000
- Vectors: 50,000-100,000
- Sequences: 50,000-100,000
- Δ measurement: ≥+0.08 (confirm forward bias)
- Time: 6-12 hours

**Full (50k papers)**:
- Papers: 50,000
- Vectors: 250,000-500,000
- Sequences: 250,000-500,000
- Δ measurement: ≥+0.08
- Time: 2-3 days

---

**Prepared by**: Claude Code 4.5 Sonnet
**Session Date**: 2025-11-04
**Status**: Ready to start download (awaiting user decision)
