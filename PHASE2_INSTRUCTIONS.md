# Phase 2: 10k arXiv Download - Instructions

**Status**: Ready to start
**Time Required**: 6-12 hours (overnight)
**Output**: 10,000 papers → 50,000-100,000 vectors

---

## 🚀 **To Start Phase 2 (in a SEPARATE terminal window)**

### Method 1: Use the automated script (RECOMMENDED)

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./START_PHASE2_10K_DOWNLOAD.sh
```

**What it does**:
- Downloads 2,500 papers from each category (cs.CL, cs.LG, stat.ML, cs.AI)
- Extracts full text using PyMuPDF (38KB-195KB per paper)
- Combines all into `arxiv_full_10k_combined.jsonl.gz`
- Logs progress to `logs/arxiv_download_10k.log`

---

### Method 2: Manual command-by-command

If you prefer to run each category manually:

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Category 1: cs.CL (Computation and Language)
./.venv/bin/python scripts/data_downloading/download_arxiv.py \
  --categories cs.CL \
  --max-total 2500 \
  --batch-size 200 \
  --pdf --extract-text --extractor pymupdf \
  --out data/datasets/arxiv/arxiv_cs_CL.jsonl \
  > logs/arxiv_cs_CL.log 2>&1

# Category 2: cs.LG (Machine Learning)
./.venv/bin/python scripts/data_downloading/download_arxiv.py \
  --categories cs.LG \
  --max-total 2500 \
  --batch-size 200 \
  --pdf --extract-text --extractor pymupdf \
  --out data/datasets/arxiv/arxiv_cs_LG.jsonl \
  > logs/arxiv_cs_LG.log 2>&1

# Category 3: stat.ML (Statistics - ML)
./.venv/bin/python scripts/data_downloading/download_arxiv.py \
  --categories stat.ML \
  --max-total 2500 \
  --batch-size 200 \
  --pdf --extract-text --extractor pymupdf \
  --out data/datasets/arxiv/arxiv_stat_ML.jsonl \
  > logs/arxiv_stat_ML.log 2>&1

# Category 4: cs.AI (Artificial Intelligence)
./.venv/bin/python scripts/data_downloading/download_arxiv.py \
  --categories cs.AI \
  --max-total 2500 \
  --batch-size 200 \
  --pdf --extract-text --extractor pymupdf \
  --out data/datasets/arxiv/arxiv_cs_AI.jsonl \
  > logs/arxiv_cs_AI.log 2>&1

# Combine all
cat data/datasets/arxiv/arxiv_cs_*.jsonl data/datasets/arxiv/arxiv_stat_*.jsonl > data/datasets/arxiv/arxiv_full_10k_combined.jsonl
gzip data/datasets/arxiv/arxiv_full_10k_combined.jsonl
```

---

## 📊 **Monitor Progress**

### Check overall progress:
```bash
tail -f logs/arxiv_download_10k.log
```

### Check specific category:
```bash
tail -f logs/arxiv_download_cs_CL.log    # Category 1
tail -f logs/arxiv_download_cs_LG.log    # Category 2
tail -f logs/arxiv_download_stat_ML.log  # Category 3
tail -f logs/arxiv_download_cs_AI.log    # Category 4
```

### Count downloaded papers:
```bash
ls data/datasets/arxiv/pdfs/*.txt | wc -l
```

### Check disk usage:
```bash
du -sh data/datasets/arxiv/pdfs/
```

---

## ⏱️ **Expected Timeline**

| Time | Status | What's Happening |
|------|--------|------------------|
| 0:00 | Start | cs.CL download begins |
| 2:00 | 25% | cs.CL complete, cs.LG starts |
| 4:00 | 50% | cs.LG complete, stat.ML starts |
| 6:00 | 75% | stat.ML complete, cs.AI starts |
| 8:00 | 100% | cs.AI complete, combining files |
| 8:05 | Done | arxiv_full_10k_combined.jsonl.gz ready |

**Note**: Times are estimates. Actual time depends on arXiv API speed and network.

---

## ✅ **Success Criteria**

When complete, you should have:

- ✅ `data/datasets/arxiv/arxiv_full_10k_combined.jsonl.gz` (~50MB compressed)
- ✅ ~10,000 lines in the JSONL file
- ✅ ~10,000 `.txt` files in `data/datasets/arxiv/pdfs/`
- ✅ ~10,000 `.pdf` files in `data/datasets/arxiv/pdfs/`
- ✅ PDFs directory: ~2-3GB total

---

## 🛑 **To Stop/Cancel**

If you need to stop the download:

```bash
# Find the process
ps aux | grep download_arxiv

# Kill it
kill <PID>
```

**Note**: Partial downloads are saved, so you can resume later!

---

## 🔍 **Troubleshooting**

### Download stops unexpectedly
```bash
# Check logs for errors
cat logs/arxiv_download_10k.log

# Resume from the failed category
# (just re-run the command for that category)
```

### Disk space running low
```bash
# Check available space
df -h .

# You need ~3-5GB free for 10k papers
```

### Network errors
- The script has built-in 3-second delays between batches
- If you get rate-limited, wait 5 minutes and resume

---

## 📋 **After Phase 2 Completes**

Next steps (run in main terminal):

```bash
# 1. Ingest to NPZ with custom paragraph chunker
python tools/ingest_arxiv_to_npz_simple.py \
  --input data/datasets/arxiv/arxiv_full_10k_combined.jsonl.gz \
  --output artifacts/lvm/arxiv_10k_full.npz \
  --encoder-url http://localhost:7001/encode

# 2. Create training sequences
python tools/create_arxiv_sequences_simple.py \
  --input artifacts/lvm/arxiv_10k_full.npz \
  --output artifacts/lvm/arxiv_10k_sequences.npz

# 3. Measure Δ (validate forward bias)
python tools/tests/diagnose_data_direction.py \
  artifacts/lvm/arxiv_10k_sequences.npz
```

**Expected results**:
- 50,000-100,000 vectors
- Δ ≥ +0.08 (confirms forward bias)
- Ready for P6b v2.3 training!

---

## 📁 **File Locations**

| File | Location | Size | Purpose |
|------|----------|------|---------|
| Combined JSONL | `data/datasets/arxiv/arxiv_full_10k_combined.jsonl.gz` | ~50MB | Metadata + paths |
| Text files | `data/datasets/arxiv/pdfs/*.txt` | 38KB-195KB each | Full paper text |
| PDFs | `data/datasets/arxiv/pdfs/*.pdf` | ~500KB each | Original PDFs |
| Main log | `logs/arxiv_download_10k.log` | Variable | Overall progress |
| Category logs | `logs/arxiv_download_*.log` | Variable | Per-category logs |

---

**Ready to start?** Run `./START_PHASE2_10K_DOWNLOAD.sh` in a separate terminal! 🚀

---

**Session**: 2025-11-04
**Phase 1 Status**: ✅ Complete (100 papers processed)
**Phase 2 Status**: ⏳ Ready to start
