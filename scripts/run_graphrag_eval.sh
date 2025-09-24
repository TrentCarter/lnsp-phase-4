#!/bin/bash
# GraphRAG Evaluation Script - Reproducible end-to-end pipeline
# Runs: sanity checks → graph build → 20 queries → report generation

set -e  # Exit on any error

echo "=== GraphRAG Evaluation Pipeline ==="
echo "Started: $(date)"

# Configuration
CONFIG_FILE="configs/lightrag.yml"
QUERIES_FILE="eval/graphrag_20.txt"
RESULTS_FILE="eval/graphrag_runs.jsonl"
REPORT_FILE="eval/day12_graphrag_report.md"

# Sanity checks
echo "🔍 Running sanity checks..."

# Check environment
if [ -z "$LNSP_LLM_PROVIDER" ]; then
    echo "❌ LNSP_LLM_PROVIDER not set"
    exit 1
fi

if [ -z "$LNSP_LLM_MODEL" ]; then
    echo "❌ LNSP_LLM_MODEL not set"
    exit 1
fi

if [ "$LNSP_ALLOW_MOCK" = "1" ]; then
    echo "❌ LNSP_ALLOW_MOCK must be 0 for real evaluation"
    exit 1
fi

# Check LNSP_FUSED=0
if [ "$LNSP_FUSED" != "0" ]; then
    echo "❌ LNSP_FUSED must be 0 for pure 768D mode"
    exit 1
fi

# Check required files
for file in "$CONFIG_FILE" "artifacts/fw10k_chunks.jsonl" "artifacts/fw10k_vectors_768.npz" "artifacts/fw10k_ivf_768.index"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ Sanity checks passed"

# Check if graph already exists
if [ -f "artifacts/kg/nodes.jsonl" ] && [ -f "artifacts/kg/edges.jsonl" ]; then
    echo "📋 Graph already exists, skipping build..."
else
    echo "🏗️ Building knowledge graph..."

    python3 -m src.adapters.lightrag.build_graph \
        --config "$CONFIG_FILE" \
        --out-nodes artifacts/kg/nodes.jsonl \
        --out-edges artifacts/kg/edges.jsonl \
        --stats artifacts/kg/stats.json \
        --load-neo4j

    echo "✅ Graph build complete"
fi

# Run GraphRAG queries
echo "🚀 Running GraphRAG queries..."

PYTHONPATH=src:. python3 -m src.adapters.lightrag.graphrag_runner \
    --config "$CONFIG_FILE" \
    --lane L1_FACTOID \
    --query-file "$QUERIES_FILE" \
    --out "$RESULTS_FILE" \
    --persist-postgres

echo "✅ GraphRAG queries complete"

# Generate report
echo "📊 Generating evaluation report..."

python3 << 'EOF'
import json
import statistics
from pathlib import Path

# Load results
results_file = Path("eval/graphrag_runs.jsonl")
results = []
with open(results_file) as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

print(f"Loaded {len(results)} query results")

# Analysis
latencies = [r['latency_ms'] for r in results if 'latency_ms' in r]
tokens_total = [r.get('usage_total_tokens', 0) for r in results]
graph_nodes = [r.get('graph_nodes_used', 0) for r in results]
graph_edges = [r.get('graph_edges_used', 0) for r in results]

# Generate report
report = f"""# GraphRAG Evaluation Report
**Date:** {results[0]['timestamp'][:10] if results else 'Unknown'}
**Queries:** {len(results)}/20 completed
**Model:** {results[0].get('model', 'Unknown') if results else 'Unknown'}

## Performance Metrics

### Latency
- **P50:** {statistics.median(latencies) if latencies else 0:.0f}ms
- **P95:** {sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0:.0f}ms
- **Mean:** {statistics.mean(latencies) if latencies else 0:.0f}ms

### Token Usage
- **Total tokens/query:** {statistics.mean(tokens_total) if tokens_total else 0:.0f}
- **Prompt tokens/query:** {statistics.mean([r.get('usage_prompt_tokens', 0) for r in results]) if results else 0:.0f}
- **Completion tokens/query:** {statistics.mean([r.get('usage_completion_tokens', 0) for r in results]) if results else 0:.0f}

### Graph Usage
- **Avg nodes used:** {statistics.mean(graph_nodes) if graph_nodes else 0:.1f}
- **Avg edges used:** {statistics.mean(graph_edges) if graph_edges else 0:.1f}
- **Queries with graph:** {sum(1 for r in results if r.get('graph_nodes_used', 0) > 0)}/{len(results)}

## Sample Results

"""

# Add 3 sample results
for i, result in enumerate(results[:3]):
    report += f"""### Query {i+1}: {result['query'][:80]}...
**Response:** {result['response'][:200]}...
**Graph nodes:** {result.get('graph_nodes_used', 0)}
**Latency:** {result['latency_ms']}ms
**Tokens:** {result.get('usage_total_tokens', 0)}

"""

# Save report
with open("eval/day12_graphrag_report.md", 'w') as f:
    f.write(report)

print("Report saved to eval/day12_graphrag_report.md")
EOF

echo "✅ Evaluation complete!"
echo "📁 Results: $RESULTS_FILE"
echo "📊 Report: $REPORT_FILE"
echo "Finished: $(date)"
