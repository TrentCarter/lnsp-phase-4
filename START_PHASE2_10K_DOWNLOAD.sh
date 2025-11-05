#!/bin/bash
################################################################################
# PHASE 2: 10k arXiv Paper Download (Overnight)
# Run this in a SEPARATE terminal window
################################################################################

set -e

echo "=========================================="
echo "Phase 2: 10k arXiv Download (Overnight)"
echo "=========================================="
echo ""
echo "This will download 2,500 papers per category = 10,000 total"
echo "Estimated time: 6-12 hours"
echo "Output: data/datasets/arxiv/arxiv_full_10k_combined.jsonl.gz"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Navigate to project root
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Configuration
CATEGORIES=("cs.CL" "cs.LG" "stat.ML" "cs.AI")
PER_CATEGORY=2500  # 2500 x 4 = 10k total
BATCH_SIZE=200
EXTRACTOR="pymupdf"

OUTPUT_DIR="data/datasets/arxiv"
LOG_DIR="logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo ""
echo "=========================================="
echo "Starting Downloads"
echo "=========================================="
echo "Started: $(date)"
echo "Log: $LOG_DIR/arxiv_download_10k.log"
echo ""

# Create main log file
MAIN_LOG="$LOG_DIR/arxiv_download_10k.log"
echo "Phase 2: 10k arXiv Download Started at $(date)" > "$MAIN_LOG"
echo "" >> "$MAIN_LOG"

# Download each category separately
for i in "${!CATEGORIES[@]}"; do
    cat="${CATEGORIES[$i]}"
    cat_num=$((i+1))

    echo "[$cat_num/4] Downloading category: $cat (target: $PER_CATEGORY papers)"
    echo "[$cat_num/4] Category: $cat - Started at $(date)" >> "$MAIN_LOG"

    cat_safe=$(echo $cat | tr '.' '_')
    cat_log="$LOG_DIR/arxiv_download_${cat_safe}.log"

    # Run download
    ./.venv/bin/python scripts/data_downloading/download_arxiv.py \
        --categories "$cat" \
        --max-total $PER_CATEGORY \
        --batch-size $BATCH_SIZE \
        --pdf --extract-text --extractor $EXTRACTOR \
        --out "$OUTPUT_DIR/arxiv_${cat_safe}.jsonl" \
        > "$cat_log" 2>&1

    cat_status=$?

    if [ $cat_status -eq 0 ]; then
        downloaded=$(wc -l < "$OUTPUT_DIR/arxiv_${cat_safe}.jsonl" | tr -d ' ')
        echo "  ✓ $cat complete: $downloaded papers downloaded"
        echo "  Status: SUCCESS ($downloaded papers)" >> "$MAIN_LOG"
    else
        echo "  ✗ $cat FAILED (see $cat_log)"
        echo "  Status: FAILED" >> "$MAIN_LOG"
    fi

    echo "" >> "$MAIN_LOG"
done

echo ""
echo "=========================================="
echo "Combining Files"
echo "=========================================="

# Combine all JSONL files
echo "Combining all category files..."
cat "$OUTPUT_DIR"/arxiv_cs_*.jsonl "$OUTPUT_DIR"/arxiv_stat_*.jsonl > "$OUTPUT_DIR/arxiv_full_10k_combined.jsonl" 2>/dev/null || true

if [ -f "$OUTPUT_DIR/arxiv_full_10k_combined.jsonl" ]; then
    total_papers=$(wc -l < "$OUTPUT_DIR/arxiv_full_10k_combined.jsonl" | tr -d ' ')
    echo "✓ Combined $total_papers papers total"

    # Compress
    echo "Compressing..."
    gzip -f "$OUTPUT_DIR/arxiv_full_10k_combined.jsonl"

    final_size=$(du -h "$OUTPUT_DIR/arxiv_full_10k_combined.jsonl.gz" | awk '{print $1}')
    echo "✓ Final file: arxiv_full_10k_combined.jsonl.gz ($final_size)"
fi

echo ""
echo "=========================================="
echo "Download Statistics"
echo "=========================================="

# Count text files
txt_count=$(ls "$OUTPUT_DIR/pdfs/"*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "Text files extracted: $txt_count"

# Disk usage
pdf_size=$(du -sh "$OUTPUT_DIR/pdfs/" 2>/dev/null | awk '{print $1}')
echo "PDF directory size: $pdf_size"

echo ""
echo "=========================================="
echo "✅ Phase 2 Complete!"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Output file: $OUTPUT_DIR/arxiv_full_10k_combined.jsonl.gz"
echo "Total papers: $total_papers"
echo "Main log: $MAIN_LOG"
echo ""
echo "Next steps:"
echo "  1. Ingest to NPZ: python tools/ingest_arxiv_to_npz_simple.py ..."
echo "  2. Create sequences: python tools/create_arxiv_sequences_simple.py ..."
echo "  3. Measure Δ: python tools/tests/diagnose_data_direction.py ..."
echo "=========================================="

# Append to main log
echo "Phase 2 Complete at $(date)" >> "$MAIN_LOG"
echo "Total papers: $total_papers" >> "$MAIN_LOG"
