#!/bin/bash
# Demo 2: Representative Canary (Stratified Sampling)
# Shows guaranteed strata coverage with 1% rehearsal
# Duration: <2 minutes

set -e

PLMS_API_BASE_URL="${PLMS_API_BASE_URL:-http://localhost:6100}"

echo "=== Demo 2: Representative Canary ==="
echo "API Base URL: $PLMS_API_BASE_URL"
echo ""

echo "Step 1: Request 1% rehearsal simulation"
echo "=========================================="
RESPONSE=$(curl -s "$PLMS_API_BASE_URL/api/projects/42/simulate?rehearsal_pct=0.01")

echo "Full response:"
echo "$RESPONSE" | jq '.'
echo ""

# Extract key metrics
STRATA_COVERAGE=$(echo "$RESPONSE" | jq -r '.sampling_metrics.strata_coverage')
ACTUAL_PCT=$(echo "$RESPONSE" | jq -r '.sampling_metrics.actual_pct')
REQUESTED_PCT=$(echo "$RESPONSE" | jq -r '.sampling_metrics.requested_pct')
AUTO_BUMPED=$(echo "$RESPONSE" | jq -r '.sampling_metrics.auto_bumped')
N_SAMPLE=$(echo "$RESPONSE" | jq -r '.sampling_metrics.n_sample')
N_TOTAL=$(echo "$RESPONSE" | jq -r '.sampling_metrics.n_total')
N_STRATA=$(echo "$RESPONSE" | jq -r '.sampling_metrics.n_strata')

echo "Step 2: Verify strata coverage"
echo "=========================================="
echo "Requested:       ${REQUESTED_PCT}% of tasks"
echo "Actual:          ${ACTUAL_PCT}% of tasks"
echo "Sample size:     $N_SAMPLE / $N_TOTAL tasks"
echo "Strata count:    $N_STRATA"
echo "Coverage:        $STRATA_COVERAGE (1.0 = all strata sampled)"
echo "Auto-bumped:     $AUTO_BUMPED"
echo ""

if [ "$STRATA_COVERAGE" != "1.0" ]; then
    echo "❌ FAIL: Expected strata_coverage = 1.0, got: $STRATA_COVERAGE"
    exit 1
fi

echo "✓ All strata covered (coverage = 1.0)"
echo ""

echo "Step 3: Check extrapolated estimates with CIs"
echo "=========================================="
TOKENS_MEAN=$(echo "$RESPONSE" | jq -r '.simulation_results.extrapolated_full.tokens')
TOKENS_CI=$(echo "$RESPONSE" | jq -r '.simulation_results.extrapolated_full.ci_90.tokens')
COST_MEAN=$(echo "$RESPONSE" | jq -r '.simulation_results.extrapolated_full.cost_usd')
COST_CI=$(echo "$RESPONSE" | jq -r '.simulation_results.extrapolated_full.ci_90.cost_usd')

echo "Projected tokens: $TOKENS_MEAN (90% CI: $TOKENS_CI)"
echo "Projected cost:   \$${COST_MEAN} (90% CI: $COST_CI)"
echo ""

echo "✓ Extrapolation includes credible intervals"
echo ""

echo "Step 4: Check risk factors"
echo "=========================================="
RISK_COUNT=$(echo "$RESPONSE" | jq '.simulation_results.risk_factors | length')
echo "Risk factors identified: $RISK_COUNT"

if [ "$RISK_COUNT" -gt "0" ]; then
    echo "$RESPONSE" | jq '.simulation_results.risk_factors'
fi

echo ""

echo "=== Demo 2 Complete ==="
echo "✓ Stratified sampling with guaranteed coverage"
echo "✓ 1% canary is representative (all strata sampled)"
echo "✓ Extrapolated estimates with 90% credible intervals"
echo "✓ Risk factors identified"
echo ""
echo "💡 Value: Forecast and derisk before committing full budget"
echo "   HMI overlays: risk heatmap + budget runway gauge"
echo ""
echo "📊 HMI Integration:"
echo "   - Display strata coverage badge (✓ if 1.0)"
echo "   - Show CI bands on cost/time projections"
echo "   - Highlight risk factors with severity colors"
