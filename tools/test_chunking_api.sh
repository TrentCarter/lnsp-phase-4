#!/bin/bash
# Simple script to test the chunking API
# Usage: ./tools/test_chunking_api.sh "Your text here"

# Default text if none provided
TEXT="${1:-Photosynthesis is the process by which plants convert light energy into chemical energy. This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and carbon dioxide as inputs. During photosynthesis, plants absorb light energy through chlorophyll molecules. This energy is used to convert carbon dioxide and water into glucose and oxygen. The glucose serves as food for the plant, while oxygen is released as a byproduct. Cellular respiration is the metabolic process that converts glucose into ATP energy. This process occurs in the mitochondria of cells. During cellular respiration, glucose is broken down through glycolysis, the Krebs cycle, and the electron transport chain. Oxygen is consumed and carbon dioxide is produced as a waste product.}"

# API endpoint
API_URL="http://127.0.0.1:8001/chunk"

echo "=========================================="
echo "Testing Chunking API"
echo "=========================================="
echo ""
echo "Sending text to API..."
echo ""

# Create JSON payload
cat > /tmp/chunk_request.json <<EOF
{
  "text": "$TEXT",
  "mode": "semantic",
  "min_chunk_size": 200
}
EOF

# Send request and pretty print response
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d @/tmp/chunk_request.json | jq '
  {
    total_chunks: .total_chunks,
    mode: .chunking_mode,
    processing_time_ms: .processing_time_ms,
    chunks: .chunks | map({
      chunk_index,
      word_count,
      text: .text[0:150] + "..."
    })
  }
'

echo ""
echo "=========================================="
echo "✓ Done"
echo "=========================================="
