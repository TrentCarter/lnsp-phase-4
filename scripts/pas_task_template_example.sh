#!/bin/bash
# Example PAS Task Template with KPI Receipt Emission
# Add this post-task step to your PAS job templates

set -e

# Task metadata (provided by PAS)
TASK_ID="${1:-1287}"
LANE_ID="${2:-4202}"
ARTIFACTS_DIR="${3:-artifacts/t${TASK_ID}}"

# Ensure artifacts directory exists
mkdir -p "$ARTIFACTS_DIR"

# ==================================================
# YOUR TASK EXECUTION HERE
# ==================================================
# Example: Run code generation, tests, etc.
# ... (task-specific commands)

# ==================================================
# POST-TASK: EMIT KPI RECEIPT
# ==================================================

echo "=== Emitting KPI Receipt ==="

# Run KPI receipt emitter
python -m services.plms.kpi_emit \
    --task-id "$TASK_ID" \
    --lane "$LANE_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --output "$ARTIFACTS_DIR/kpi_receipt.json"

KPI_EXIT_CODE=$?

# Check if KPIs passed
if [ $KPI_EXIT_CODE -eq 0 ]; then
    echo "✓ All KPIs passed - task validation successful"
    exit 0
else
    echo "✗ KPI validation failed - check $ARTIFACTS_DIR/kpi_receipt.json for details"
    exit 1
fi
