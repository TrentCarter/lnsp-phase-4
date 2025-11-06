#!/bin/bash
# Quick comparison of P7 grid results

echo "P7 Grid Results Comparison"
echo "=========================================="

for exp in exp1_baseline exp2_strong_anchor exp3_higher_margin exp4_larger_context; do
    if [ -f "${exp}_history.json" ]; then
        echo ""
        echo "$exp:"
        # Extract final epoch metrics
        python3 -c "
import json
with open('${exp}_history.json', 'r') as f:
    history = json.load(f)
    if history:
        final = history[-1]
        print(f\"  Epoch: {final.get('epoch', 'N/A')}\")
        val = final.get('val', {})
        print(f\"  Margin: {val.get('margin', 'N/A'):.4f}\")
        print(f\"  cos_next: {val.get('cos_next', 'N/A'):.4f}\")
        print(f\"  cos_prev: {val.get('cos_prev', 'N/A'):.4f}\")
        print(f\"  cos_anchor: {val.get('cos_anchor', 'N/A'):.4f}\")
        print(f\"  anchor_λ: {final.get('anchor_lambda', 'N/A'):.3f}\")
" 2>/dev/null || echo "  [Could not parse history]"
    else
        echo ""
        echo "$exp: [No results]"
    fi
done

echo ""
echo "=========================================="
