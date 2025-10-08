#!/bin/bash

# LLM Speed Benchmark: Llama 3.1:8b vs TinyLlama 1.1b
# Tests tokens/sec for TMD-LS lane specialist architecture

set -e

echo "==================================================================="
echo "LLM SPEED BENCHMARK: Llama 3.1:8b vs TinyLlama 1.1b"
echo "==================================================================="
echo ""

# Test prompts of varying lengths
PROMPTS=(
    "What is AI?"
    "Explain the concept of machine learning in detail."
    "Write a comprehensive explanation of quantum computing, including its principles, applications, and current limitations. Cover quantum superposition, entanglement, and how quantum gates differ from classical logic gates."
)

PROMPT_NAMES=(
    "Short (3 tokens)"
    "Medium (~10 tokens)"
    "Long (~50 tokens)"
)

# Function to benchmark a model
benchmark_model() {
    local model=$1
    local port=$2
    local prompt=$3
    local prompt_name=$4

    echo "Testing $model on port $port"
    echo "Prompt: $prompt_name"
    echo "-------------------------------------------------------------------"

    # Make request and capture response
    response=$(curl -s http://localhost:$port/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"$prompt\", \"stream\": false}")

    # Parse response
    total_duration=$(echo "$response" | jq -r '.total_duration // 0')
    eval_duration=$(echo "$response" | jq -r '.eval_duration // 0')
    prompt_eval_duration=$(echo "$response" | jq -r '.prompt_eval_duration // 0')
    eval_count=$(echo "$response" | jq -r '.eval_count // 0')
    prompt_eval_count=$(echo "$response" | jq -r '.prompt_eval_count // 0')

    # Convert nanoseconds to seconds
    total_sec=$(echo "scale=3; $total_duration / 1000000000" | bc)
    eval_sec=$(echo "scale=3; $eval_duration / 1000000000" | bc)
    prompt_eval_sec=$(echo "scale=3; $prompt_eval_duration / 1000000000" | bc)

    # Calculate tokens/sec
    if [ "$eval_count" -gt 0 ] && [ "$eval_duration" -gt 0 ]; then
        tokens_per_sec=$(echo "scale=2; $eval_count * 1000000000 / $eval_duration" | bc)
    else
        tokens_per_sec="N/A"
    fi

    # Calculate prompt tokens/sec
    if [ "$prompt_eval_count" -gt 0 ] && [ "$prompt_eval_duration" -gt 0 ]; then
        prompt_tokens_per_sec=$(echo "scale=2; $prompt_eval_count * 1000000000 / $prompt_eval_duration" | bc)
    else
        prompt_tokens_per_sec="N/A"
    fi

    echo "  Total time: ${total_sec}s"
    echo "  Generation time: ${eval_sec}s"
    echo "  Prompt eval time: ${prompt_eval_sec}s"
    echo "  Tokens generated: $eval_count"
    echo "  Prompt tokens: $prompt_eval_count"
    echo "  ⚡ Generation speed: ${tokens_per_sec} tokens/sec"
    echo "  📖 Prompt processing: ${prompt_tokens_per_sec} tokens/sec"
    echo ""

    # Return tokens/sec for aggregation
    echo "$tokens_per_sec"
}

# Check if both servers are running
echo "Checking server availability..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Error: Llama 3.1:8b server not running on port 11434"
    echo "Start with: ollama serve"
    exit 1
fi

echo "✅ Llama 3.1:8b server running on port 11434"
echo ""

# Run benchmarks
echo "==================================================================="
echo "BENCHMARK RESULTS"
echo "==================================================================="
echo ""

llama_speeds=()
tinyllama_speeds=()

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    prompt_name="${PROMPT_NAMES[$i]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $((i+1))/3: $prompt_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    echo "🔹 LLAMA 3.1:8B"
    llama_speed=$(benchmark_model "llama3.1:8b" 11434 "$prompt" "$prompt_name")
    llama_speeds+=("$llama_speed")

    echo "🔸 TINYLLAMA 1.1B"
    tinyllama_speed=$(benchmark_model "tinyllama:1.1b" 11434 "$prompt" "$prompt_name")
    tinyllama_speeds+=("$tinyllama_speed")

    # Calculate speedup
    if [ "$llama_speed" != "N/A" ] && [ "$tinyllama_speed" != "N/A" ]; then
        speedup=$(echo "scale=2; $tinyllama_speed / $llama_speed" | bc)
        echo "⚡ TinyLlama is ${speedup}x faster for this test"
    fi
    echo ""
done

# Calculate averages
echo "==================================================================="
echo "SUMMARY"
echo "==================================================================="
echo ""

llama_total=0
llama_count=0
for speed in "${llama_speeds[@]}"; do
    if [ "$speed" != "N/A" ]; then
        llama_total=$(echo "$llama_total + $speed" | bc)
        llama_count=$((llama_count + 1))
    fi
done

tinyllama_total=0
tinyllama_count=0
for speed in "${tinyllama_speeds[@]}"; do
    if [ "$speed" != "N/A" ]; then
        tinyllama_total=$(echo "$tinyllama_total + $speed" | bc)
        tinyllama_count=$((tinyllama_count + 1))
    fi
done

if [ "$llama_count" -gt 0 ]; then
    llama_avg=$(echo "scale=2; $llama_total / $llama_count" | bc)
else
    llama_avg="N/A"
fi

if [ "$tinyllama_count" -gt 0 ]; then
    tinyllama_avg=$(echo "scale=2; $tinyllama_total / $tinyllama_count" | bc)
else
    tinyllama_avg="N/A"
fi

echo "Average Generation Speed:"
echo "  🔹 Llama 3.1:8b    : ${llama_avg} tokens/sec"
echo "  🔸 TinyLlama 1.1b  : ${tinyllama_avg} tokens/sec"
echo ""

if [ "$llama_avg" != "N/A" ] && [ "$tinyllama_avg" != "N/A" ]; then
    overall_speedup=$(echo "scale=2; $tinyllama_avg / $llama_avg" | bc)
    echo "⚡ Overall: TinyLlama is ${overall_speedup}x faster"
    echo ""

    # Compare to PRD claims
    echo "PRD_TMD-LS.md Claims:"
    echo "  • Llama ~200-300 tokens/sec"
    echo "  • TinyLlama ~600-800 tokens/sec"
    echo ""
    echo "Actual Results:"
    echo "  • Llama: ${llama_avg} tokens/sec"
    echo "  • TinyLlama: ${tinyllama_avg} tokens/sec"
fi

echo ""
echo "==================================================================="
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown CPU')"
echo "Date: $(date)"
echo "==================================================================="
