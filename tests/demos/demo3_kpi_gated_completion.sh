#!/bin/bash
# Demo 3: KPI-Gated Completion
# Shows validation blocking completion on KPI failures
# Duration: <2 minutes

set -e

DEMO_DIR="/tmp/plms_demo3"
TASK_ID=9999
LANE_ID=4202  # Code-Docs lane (readability KPI)

echo "=== Demo 3: KPI-Gated Completion ==="
echo "Task ID: $TASK_ID"
echo "Lane: $LANE_ID (Code-Docs)"
echo ""

# Setup demo artifacts directory
echo "Step 1: Setup demo artifacts"
echo "=========================================="
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR/artifacts/t${TASK_ID}"
ARTIFACTS_DIR="$DEMO_DIR/artifacts/t${TASK_ID}"

echo "Created artifacts directory: $ARTIFACTS_DIR"
echo ""

# Create failing scenario: poor readability (college-level text)
echo "Step 2: Create FAILING test case (poor readability)"
echo "=========================================="
cat > "$ARTIFACTS_DIR/generated_docs.md" <<'EOF'
The implementation of the aforementioned functionality necessitates the utilization
of sophisticated algorithmic constructs and methodological paradigms. Consequently,
the architectural framework must accommodate multifarious computational exigencies
while maintaining adherence to established software engineering principles.
EOF

# Create stub echo result (passing)
cat > "$ARTIFACTS_DIR/echo_result.json" <<'EOF'
{
  "cosine_similarity": 0.89,
  "threshold": 0.82
}
EOF

echo "Generated docs with poor readability (college-level text)"
echo "Content preview:"
head -n 2 "$ARTIFACTS_DIR/generated_docs.md"
echo ""

# Run KPI receipt emitter (should FAIL on readability)
echo "Step 3: Run KPI validation (should FAIL)"
echo "=========================================="
set +e  # Don't exit on error
python -m services.plms.kpi_emit \
    --task-id "$TASK_ID" \
    --lane "$LANE_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --output "$ARTIFACTS_DIR/kpi_receipt.json"

KPI_EXIT_CODE=$?
set -e

echo ""
echo "KPI validation exit code: $KPI_EXIT_CODE"
echo ""

if [ $KPI_EXIT_CODE -eq 0 ]; then
    echo "❌ FAIL: Expected KPI validation to fail, but it passed!"
    exit 1
fi

echo "✓ KPI validation correctly failed (exit code $KPI_EXIT_CODE)"
echo ""

# Show receipt with violations
echo "Step 4: Inspect KPI receipt"
echo "=========================================="
cat "$ARTIFACTS_DIR/kpi_receipt.json" | jq '.'
echo ""

READABILITY=$(cat "$ARTIFACTS_DIR/kpi_receipt.json" | jq -r '.kpis[] | select(.name == "readability") | .value')
READABILITY_PASS=$(cat "$ARTIFACTS_DIR/kpi_receipt.json" | jq -r '.kpis[] | select(.name == "readability") | .pass')

echo "Readability score: $READABILITY (threshold: ≤10.0)"
echo "Readability pass:  $READABILITY_PASS"
echo ""

if [ "$READABILITY_PASS" != "false" ]; then
    echo "❌ FAIL: Expected readability to fail, but it passed!"
    exit 1
fi

echo "✓ Readability KPI correctly failed"
echo ""

# Now fix the issue: simplify text
echo "Step 5: Fix issue and re-validate"
echo "=========================================="
cat > "$ARTIFACTS_DIR/generated_docs.md" <<'EOF'
This module helps you manage your tasks. To use it, import the Task class
and create a new instance. Then call the run() method to start the task.
EOF

echo "Generated simpler docs (grade-school level)"
echo "Content preview:"
head -n 2 "$ARTIFACTS_DIR/generated_docs.md"
echo ""

# Re-run KPI validation (should PASS now)
set +e
python -m services.plms.kpi_emit \
    --task-id "$TASK_ID" \
    --lane "$LANE_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --output "$ARTIFACTS_DIR/kpi_receipt_fixed.json"

KPI_EXIT_CODE=$?
set -e

echo ""
echo "KPI validation exit code: $KPI_EXIT_CODE"
echo ""

if [ $KPI_EXIT_CODE -ne 0 ]; then
    echo "❌ FAIL: Expected KPI validation to pass after fix, but it failed!"
    cat "$ARTIFACTS_DIR/kpi_receipt_fixed.json" | jq '.'
    exit 1
fi

echo "✓ KPI validation passed after fix (exit code 0)"
echo ""

# Show fixed receipt
echo "Step 6: Inspect fixed KPI receipt"
echo "=========================================="
cat "$ARTIFACTS_DIR/kpi_receipt_fixed.json" | jq '.'
echo ""

READABILITY_FIXED=$(cat "$ARTIFACTS_DIR/kpi_receipt_fixed.json" | jq -r '.kpis[] | select(.name == "readability") | .value')
READABILITY_PASS_FIXED=$(cat "$ARTIFACTS_DIR/kpi_receipt_fixed.json" | jq -r '.kpis[] | select(.name == "readability") | .pass')

echo "Readability score: $READABILITY_FIXED (threshold: ≤10.0)"
echo "Readability pass:  $READABILITY_PASS_FIXED"
echo ""

if [ "$READABILITY_PASS_FIXED" != "true" ]; then
    echo "❌ FAIL: Expected readability to pass after fix!"
    exit 1
fi

echo "✓ Readability KPI now passes"
echo ""

# Cleanup
rm -rf "$DEMO_DIR"

echo "=== Demo 3 Complete ==="
echo "✓ KPI validation blocks completion on failure"
echo "✓ Receipt shows violation with logs path"
echo "✓ After fix, validation passes"
echo ""
echo "💡 Value: \"Echo is green\" isn't enough - real KPIs gate Done"
echo "   HMI shows KPI violations with severity + logs links"
echo ""
echo "📊 HMI Integration:"
echo "   - Red banner: \"Task blocked - KPI violations detected\""
echo "   - List each failing KPI with:"
echo "     - Name, threshold, actual value, operator"
echo "     - Link to logs for debugging"
echo "   - Green banner after fix: \"All KPIs passed\""
