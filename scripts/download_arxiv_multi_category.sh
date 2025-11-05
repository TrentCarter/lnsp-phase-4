#!/bin/bash
# Download arXiv papers from multiple categories separately, then combine

set -e

CATEGORIES=("cs.CL" "cs.LG" "stat.ML" "cs.AI")
PER_CATEGORY=12500  # 12500 x 4 = 50k total
BATCH_SIZE=200
EXTRACTOR="pymupdf"

OUTPUT_DIR="data/datasets/arxiv"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Category arXiv Download"
echo "=========================================="
echo "Target: $PER_CATEGORY papers per category"
echo "Total: $((PER_CATEGORY * ${#CATEGORIES[@]})) papers"
echo "Started: $(date)"
echo ""

# Download each category separately
for cat in "${CATEGORIES[@]}"; do
    echo "--- Downloading category: $cat ---"
    cat_safe=$(echo $cat | tr '.' '_')
    ./.venv/bin/python scripts/data_downloading/download_arxiv.py \
        --categories "$cat" \
        --max-total $PER_CATEGORY \
        --batch-size $BATCH_SIZE \
        --pdf --extract-text --extractor $EXTRACTOR \
        --out "$OUTPUT_DIR/arxiv_${cat_safe}.jsonl"
    echo "✓ $cat complete"
    echo ""
done

# Combine all JSONL files
echo "--- Combining all files ---"
cat "$OUTPUT_DIR"/arxiv_cs_*.jsonl "$OUTPUT_DIR"/arxiv_stat_*.jsonl > "$OUTPUT_DIR/arxiv_full_50k_combined.jsonl"
gzip "$OUTPUT_DIR/arxiv_full_50k_combined.jsonl"

echo ""
echo "=========================================="
echo "✅ Download Complete!"
echo "=========================================="
echo "Output: $OUTPUT_DIR/arxiv_full_50k_combined.jsonl.gz"
echo "Completed: $(date)"
