#!/usr/bin/env bash
# ===================================================================
# LNSP Phase-4: Comprehensive VecRAG Ingestion
# ===================================================================
# Ensures ALL items from the input file are ingested into vecRAG
# with multiple fallback strategies and comprehensive error recovery.
#
# Usage: ./scripts/ingest_comprehensive.sh [path/to/data.jsonl]
#
# Environment variables:
#   BATCH_SIZE    - Batch size for processing (default: 100)
#   RESUME_FROM   - Resume from entry N (default: 0)
#   NO_DOCKER     - Skip docker checks (default: 0)
#   AUTO_RESUME   - Auto-resume from existing vectors without prompting (1 to enable)
#   FRESH_START   - Always start fresh, removing existing vectors (1 to enable)
#   SKIP_NEO4J    - Skip Neo4j startup prompts and proceed without graph features (1 to enable)
#
# Service Management:
#   - Automatically checks if PostgreSQL is running (required)
#   - Attempts to start PostgreSQL if not running (via brew)
#   - Checks if Neo4j is running (optional for graph features)
#   - Offers to start Neo4j via brew services if installed
#
# Examples:
#   ./scripts/ingest_comprehensive.sh                          # Default: process fw10k_chunks.jsonl
#   AUTO_RESUME=1 ./scripts/ingest_comprehensive.sh           # Auto-resume without prompting
#   FRESH_START=1 ./scripts/ingest_comprehensive.sh           # Start fresh without prompting
#   SKIP_NEO4J=1 ./scripts/ingest_comprehensive.sh            # Skip Neo4j, proceed without graph features
#   BATCH_SIZE=500 ./scripts/ingest_comprehensive.sh          # Use larger batch size
#   FRESH_START=1 SKIP_NEO4J=1 make ingest-all                # Fresh start without Neo4j
# ===================================================================

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Load env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Configuration
INPUT_FILE=${1:-artifacts/fw10k_chunks.jsonl}
ART_DIR=${ART_DIR:-artifacts}
NPZ_PATH=${NPZ_PATH:-$ART_DIR/comprehensive_vectors.npz}
BATCH_SIZE=${BATCH_SIZE:-100}
RESUME_FROM=${RESUME_FROM:-0}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Comprehensive VecRAG Ingestion Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Count total items in file
TOTAL_ITEMS=$(wc -l < "$INPUT_FILE")
echo -e "${GREEN}📊 Input file: $INPUT_FILE${NC}"
echo -e "${GREEN}📊 Total items to process: $TOTAL_ITEMS${NC}"
echo ""

# Check required services using shared script
source "$ROOT_DIR/scripts/check_services.sh"
if ! check_all_services; then
    echo -e "${RED}❌ Required services not available. Exiting.${NC}"
    exit 1
fi

echo ""

# Check for existing vectors and offer resume option
EXISTING_COUNT=0
AUTO_RESUME=${AUTO_RESUME:-}
FRESH_START=${FRESH_START:-}

if [ -f "$NPZ_PATH" ]; then
    EXISTING_COUNT=$(python3 -c "import numpy as np; print(np.load('$NPZ_PATH')['vectors'].shape[0])" 2>/dev/null || echo 0)
    if [ "$EXISTING_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}📁 Found existing vectors: $EXISTING_COUNT items${NC}"

        # Check for auto-resume or fresh-start flags
        if [ "$AUTO_RESUME" = "1" ]; then
            RESUME_FROM=$EXISTING_COUNT
            echo -e "${BLUE}   Auto-resuming from item $RESUME_FROM${NC}"
        elif [ "$FRESH_START" = "1" ]; then
            echo -e "${YELLOW}   Fresh start requested - removing existing vectors${NC}"
            rm -f "$NPZ_PATH"
        else
            # Interactive prompt
            read -p "   Do you want to resume from existing vectors? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                RESUME_FROM=$EXISTING_COUNT
                echo -e "${BLUE}   Resuming from item $RESUME_FROM${NC}"
            else
                echo -e "${YELLOW}   Starting fresh ingestion (existing vectors will be overwritten)${NC}"
                rm -f "$NPZ_PATH"
            fi
        fi
    fi
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting Comprehensive Ingestion${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Prepare tracking directory
mkdir -p "$ART_DIR"

# Run comprehensive ingestion
echo -e "${GREEN}▶️  Processing ALL $TOTAL_ITEMS items with fallback strategies${NC}"
echo -e "${GREEN}   Batch size: $BATCH_SIZE, Resume from: $RESUME_FROM${NC}"
echo ""

./.venv/bin/python -m src.ingest_all_items \
    --input-path "$INPUT_FILE" \
    --file-type "${FILE_TYPE:-jsonl}" \
    --batch-size "$BATCH_SIZE" \
    --resume-from "$RESUME_FROM" \
    --write-pg \
    --write-neo4j \
    --faiss-out "$NPZ_PATH" \
    # batch-id will be auto-generated as UUID

# Verify output
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

FINAL_COUNT=0
if [ -f "$NPZ_PATH" ]; then
    FINAL_COUNT=$(python3 -c "import numpy as np; print(np.load('$NPZ_PATH')['vectors'].shape[0])" 2>/dev/null || echo 0)
fi

if [ "$FINAL_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No vectors generated${NC}"
    exit 1
fi

# Calculate success rate
SUCCESS_RATE=$(python3 -c "print(f'{($FINAL_COUNT / $TOTAL_ITEMS * 100):.2f}')" 2>/dev/null || echo "N/A")

echo -e "${GREEN}✅ Ingestion Statistics:${NC}"
echo -e "   • Total items in file: $TOTAL_ITEMS"
echo -e "   • Vectors generated: $FINAL_COUNT"
echo -e "   • Success rate: $SUCCESS_RATE%"
echo -e "   • Vectors saved to: $NPZ_PATH"

# Check ingestion report
REPORT_FILE="$ART_DIR/ingestion_report_comprehensive-$TIMESTAMP.json"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo -e "${BLUE}📊 Detailed report available: $REPORT_FILE${NC}"

    # Extract key stats from report
    FAILED_COUNT=$(python3 -c "import json; print(json.load(open('$REPORT_FILE'))['statistics']['failed'])" 2>/dev/null || echo "N/A")
    if [ "$FAILED_COUNT" != "N/A" ] && [ "$FAILED_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Failed items: $FAILED_COUNT${NC}"
        echo -e "${YELLOW}   Check report for details on failed items${NC}"
    fi
fi

# Database summaries (best-effort)
echo ""
if command -v psql >/dev/null 2>&1; then
    echo -e "${BLUE}🔎 PostgreSQL status:${NC}"
    psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -c "
        SELECT 'CPE entries' as type, COUNT(*) as count FROM cpe_entry
        UNION ALL
        SELECT 'Vector entries', COUNT(*) FROM cpe_vectors;
    " 2>/dev/null || echo -e "${YELLOW}   Could not query PostgreSQL${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Comprehensive ingestion complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Review the ingestion report: $REPORT_FILE"
echo -e "  2. Build FAISS index: make build-faiss"
echo -e "  3. Test retrieval: make api PORT=8080"
echo ""

# Create summary file for tracking
SUMMARY_FILE="$ART_DIR/comprehensive_ingestion_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Comprehensive VecRAG Ingestion Summary
======================================
Timestamp: $(date)
Input file: $INPUT_FILE
Total items: $TOTAL_ITEMS
Vectors created: $FINAL_COUNT
Success rate: $SUCCESS_RATE%
Output: $NPZ_PATH
Report: $REPORT_FILE
EOF

echo -e "${GREEN}Summary saved to: $SUMMARY_FILE${NC}"