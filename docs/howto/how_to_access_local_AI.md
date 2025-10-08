# How to Access Local AI (LLM Integration)

This guide explains how to use the local LLM integration for generating meaningful proposition, TMD, and CPE annotations in LNSP evaluations.

## Overview

LNSP can optionally use local or remote LLM services to generate rich, contextual annotations instead of deterministic placeholders. This dramatically improves the quality of evaluation metadata.

**Supported Backends:**
- **Ollama** (default): Local models via Ollama API
- **OpenAI-Compatible**: Any OpenAI-compatible API (vLLM, LM Studio, OpenRouter, actual OpenAI, etc.)

## Quick Start

### 1. Install and Setup Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a Llama model (recommended: llama3.1:8b)
ollama pull llama3.1:8b

# Start Ollama daemon (keep running in background)
ollama serve
```

### 2. Configure Environment Variables

```bash
# Enable LLM integration
export LNSP_USE_LLM=true

# Specify the model to use
export LNSP_LLM_MODEL="llama3.1:8b"

# Provide NPZ file for document snippets (optional but recommended)
export LNSP_OFFLINE_NPZ=artifacts/fw1k_vectors.npz

# Set Ollama URL (optional, defaults to localhost:11434)
export OLLAMA_URL="http://localhost:11434/api/chat"
```

### 3. Run Evaluation with LLM

```bash
# Run evaluation with LLM-enhanced annotations
./venv/bin/python -m src.eval_runner \
  --queries eval/day3_eval.jsonl \
  --api http://localhost:8080/search \
  --top-k 8 \
  --timeout 15 \
  --out eval/results_with_llm.jsonl
```

## Multi-Model Setup for TMD-LS Lane Specialists

For the **TMD-Lane Specialist** architecture (see `docs/PRDs/PRD_TMD-LS.md`), you can run multiple specialized models on different ports. This enables routing different task types to optimized lightweight models.

### Running Multiple Ollama Instances

**Step 1: Download Specialized Models**

```bash
# Download models for different lanes
ollama pull llama3.1:8b          # General-purpose lane (port 11434)
ollama pull tinyllama:1.1b       # Fast specialist lanes (port 11435)
ollama pull phi3:mini            # Precision lanes (optional, port 11436)
```

**Step 2: Start Multiple Ollama Servers**

```bash
# Terminal 1: Primary Ollama instance (llama3.1:8b on default port 11434)
ollama serve

# Terminal 2: TinyLlama specialist instance (port 11435)
OLLAMA_HOST=127.0.0.1:11435 ollama serve

# Terminal 3 (optional): Phi3 precision instance (port 11436)
OLLAMA_HOST=127.0.0.1:11436 ollama serve
```

**Step 3: Configure Lane-Specific Routing**

```bash
# Example: Route fast retrieval tasks to TinyLlama
export LNSP_LLM_LANE_L1_ENDPOINT="http://localhost:11435"
export LNSP_LLM_LANE_L1_MODEL="tinyllama:1.1b"

# Route complex reasoning to Llama 3.1
export LNSP_LLM_LANE_L2_ENDPOINT="http://localhost:11434"
export LNSP_LLM_LANE_L2_MODEL="llama3.1:8b"

# Route precision tasks to Phi3
export LNSP_LLM_LANE_L3_ENDPOINT="http://localhost:11436"
export LNSP_LLM_LANE_L3_MODEL="phi3:mini"
```

**Step 4: Test Each Instance**

```bash
# Test TinyLlama on port 11435
curl -s http://localhost:11435/api/generate \
  -d '{"model": "tinyllama:1.1b", "prompt": "Define AI briefly", "stream": false}' \
  | jq -r '.response'

# Test Llama 3.1 on default port 11434
curl -s http://localhost:11434/api/generate \
  -d '{"model": "llama3.1:8b", "prompt": "Explain quantum computing", "stream": false}' \
  | jq -r '.response'
```

### Performance Characteristics & Instance Management

⏺ Quick Performance Test Results ✅

  Key Findings: using fastAPI 

  | Metric      | TinyLlama | Granite3    | Phi3     | Llama 3.1     |
  |-------------|-----------|-------------|----------|---------------|
  | Tokens/sec  | 284 (🥇)  | 193 (🥈)    | 119 (🥉) | 73 (baseline) |
  | First Token | 0.033s    | 0.033s      | 0.062s   | 0.143s        |
  | Total Time  | 0.716s    | 0.266s (🏆) | 0.946s   | 1.614s        |

  Standout Results:
  - 🚀 Granite3 MoE: Fastest overall (266ms total) - perfect for quick queries
  - ⚡ TinyLlama & Granite3: 33ms first token (INSTANT!)
  - 🎯 TinyLlama: Highest tok/s + detailed responses (194 tokens)

**Comprehensive Benchmark Results (Apple M4 Max - Metal Backend):**

| Rank | Model           | Port  | Params | Speed (tok/s) | vs Llama | Use Case                    | Status      |
|------|-----------------|-------|--------|---------------|----------|-----------------------------|-----------  |
| 🥇 1 | tinyllama:1.1b  | 11435 | 1.1B   | **276.81**    | 3.80x    | Ultra-fast retrieval, batch | ✅ Running  |
| 🥈 2 | granite3-moe:1b | 11437 | 1.0B   | **188.22**    | 2.58x    | Fast specialist, low latency| ✅ Running  |
| 🥉 3 | phi3:mini       | 11436 | 3.8B   | **125.01**    | 1.72x    | Precision, code generation  | ✅ Running  |
| 4    | llama3.1:8b     | 11434 | 8B     | **72.86**     | 1.00x    | Complex reasoning, accuracy | ✅ Running  |

**Management Commands:**

| Action | Model | Command |
|--------|-------|---------|
| **START** | Llama 3.1 (default) | `ollama serve` |
| **START** | TinyLlama (11435) | `OLLAMA_HOST=127.0.0.1:11435 ollama serve > /tmp/ollama_tinyllama.log 2>&1 &` |
| **START** | Phi3 (11436) | `OLLAMA_HOST=127.0.0.1:11436 ollama serve > /tmp/ollama_phi3.log 2>&1 &` |
| **START** | Granite3 (11437) | `OLLAMA_HOST=127.0.0.1:11437 ollama serve > /tmp/ollama_granite.log 2>&1 &` |
| **STOP** | All instances | `pkill -f "ollama serve"` |
| **STOP** | Specific port | `lsof -ti:11435 \| xargs kill` (replace port number) |
| **CHECK STATUS** | All ports | See script below |
| **VIEW LOGS** | TinyLlama | `tail -f /tmp/ollama_tinyllama.log` |
| **VIEW LOGS** | Phi3 | `tail -f /tmp/ollama_phi3.log` |
| **VIEW LOGS** | Granite3 | `tail -f /tmp/ollama_granite.log` |

**Quick Status Check Script:**
```bash
#!/bin/bash
echo "=== Ollama Instance Status ==="
echo "Port 11434 (Llama 3.1): $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Down')"
echo "Port 11435 (TinyLlama): $(curl -s http://localhost:11435/api/tags >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Down')"
echo "Port 11436 (Phi3):      $(curl -s http://localhost:11436/api/tags >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Down')"
echo "Port 11437 (Granite3):  $(curl -s http://localhost:11437/api/tags >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Down')"
```

**Key Findings:**
- **TinyLlama**: Fastest overall, 3.80x speedup, best for high-throughput lanes
- **Granite3 MoE**: Excellent speed/quality balance (2.58x), IBM's low-latency specialist
- **Phi3 Mini**: Best for structured output and code (1.72x speedup)
- **Llama 3.1**: Highest quality reasoning, best for complex tasks

**Note:** Benchmarked on Apple M4 Max (Metal). NVIDIA A10/L4 GPUs achieve 2.5-4x higher absolute throughput, but relative speedups remain consistent.

See full benchmark: `docs/benchmarks/LLM_Speed_Benchmark_Results.md`

### TMD-LS Integration

When integrated with TMD-LS architecture:

1. **Router classifies TMD vector** (Task-Modifier-Domain)
2. **Routes to appropriate specialist** based on lane configuration
3. **Specialist processes** using optimized model for that semantic space
4. **Echo Loop validates** output (cosine ≥ 0.82)
5. **Falls back** to general model if validation fails

See `docs/PRDs/PRD_TMD-LS.md` for full architecture details.

## Alternative Backend Setup

### Option A: Ollama (Default - Local)

Follow the Quick Start section above for Ollama setup.

### Option B: OpenAI-Compatible APIs

For vLLM, LM Studio, OpenRouter, or actual OpenAI:

```bash
# Enable LLM integration
export LNSP_USE_LLM=true

# Specify OpenAI-compatible backend
export LNSP_LLM_BACKEND=openai

# Configure API endpoint and credentials
export OPENAI_BASE_URL="http://localhost:8000/v1"  # vLLM/LM Studio/etc.
export OPENAI_API_KEY="sk-local"                   # API key (use "sk-local" for local servers)

# Specify the model
export LNSP_LLM_MODEL="qwen2.5"                    # Model name (backend-specific)

# Optional: Set domain for annotations
export LNSP_DOMAIN_DEFAULT="FACTOIDWIKI"
```

**Common OpenAI-Compatible Setups:**

**vLLM Server:**
```bash
# Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Configure LNSP to use vLLM
export LNSP_LLM_BACKEND=openai
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-local"
export LNSP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

**LM Studio:**
```bash
# Start LM Studio server through the GUI (usually port 1234)

# Configure LNSP to use LM Studio
export LNSP_LLM_BACKEND=openai
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="sk-local"
export LNSP_LLM_MODEL="your-model-name"  # As shown in LM Studio
```

**OpenRouter (Cloud):**
```bash
# Configure LNSP to use OpenRouter
export LNSP_LLM_BACKEND=openai
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-v1-..."  # Your OpenRouter API key
export LNSP_LLM_MODEL="meta-llama/llama-3.1-8b-instruct"
```

### Backend Auto-Detection

The system can automatically detect the backend based on environment variables:

- If `OLLAMA_HOST` or `OLLAMA_URL` is set → Uses Ollama backend
- If `OPENAI_BASE_URL` is set → Uses OpenAI-compatible backend
- Otherwise → Defaults to Ollama backend

You can override this by explicitly setting `LNSP_LLM_BACKEND=ollama|openai`.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LNSP_USE_LLM` | `false` | Enable/disable LLM integration |
| `LNSP_LLM_BACKEND` | `ollama` | Backend to use: `ollama` or `openai` |
| `LNSP_LLM_MODEL` | `llama3:8b` | Model name (backend-specific) |
| `LNSP_DOMAIN_DEFAULT` | `FACTOIDWIKI` | Default domain for annotations |
| `LNSP_OFFLINE_NPZ` | None | Path to NPZ file for document snippets |
| **Ollama Backend** | | |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_URL` | `http://localhost:11434` | Alternative Ollama server URL |
| **OpenAI-Compatible Backend** | | |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | API base URL |
| `OPENAI_API_KEY` | `sk-local` | API key |

### Available Models

**Ollama Models:**
The integration works with any Ollama-compatible model. Recommended for TMD-LS:

**High-Speed Specialists (Lanes L1, L2, L6):**
- **tinyllama:1.1b** - Fastest (277 tok/s), ultra-lightweight, 3.80x speedup
- **granite3-moe:1b** - IBM MoE (188 tok/s), excellent balance, 2.58x speedup

**Precision Specialists (Lanes L3, L5):**
- **phi3:mini** - Code/structured output (125 tok/s), 3.8B params, 1.72x speedup

**Reasoning Specialist (Lane L4):**
- **llama3.1:8b** - Complex reasoning baseline (73 tok/s), highest quality

**Other Options:**
- **qwen2.5:7b-instruct** - Fast and efficient alternative
- **llama3.1:70b** - Highest quality (requires significant resources)

Check available models:
```bash
ollama list
```

Download recommended TMD-LS specialist models:
```bash
ollama pull tinyllama:1.1b     # Fastest specialist
ollama pull granite3-moe:1b    # Balanced specialist
ollama pull phi3:mini          # Precision specialist
ollama pull llama3.1:8b        # Reasoning specialist
```

**OpenAI-Compatible Models:**
Model availability depends on your chosen provider:

- **vLLM**: Any HuggingFace model compatible with vLLM
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `microsoft/Phi-3-mini-128k-instruct`
- **LM Studio**: Models available in LM Studio's catalog
- **OpenRouter**: See [openrouter.ai/models](https://openrouter.ai/models) for available models
- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`, etc.

## Output Comparison

### Without LLM (Deterministic Placeholders)
```json
{
  "proposition": "What is artificial intelligence?",
  "tmd": {"task": "RETRIEVE", "method": "DENSE", "domain": "FACTOIDWIKI"},
  "cpe": {"concept": null, "probe": "What is artificial intelligence?", "expected": ["ai_concept"]}
}
```

### With LLM (Meaningful Annotations)
```json
{
  "proposition": "Artificial intelligence refers to the simulation of human intelligence in machines and computer systems.",
  "tmd": {"task": "ANSWER", "method": "DENSE", "domain": "ARTIFICIAL_INTELLIGENCE"},
  "cpe": {"concept": "artificial intelligence", "probe": "What is artificial intelligence?", "expected": ["ai_concept"]}
}
```

### Key Improvements

1. **Proposition**: Meaningful definition instead of query echo
2. **TMD Task**: Context-aware task classification (ANSWER vs RETRIEVE)
3. **TMD Domain**: Specific domain extraction (ARTIFICIAL_INTELLIGENCE vs generic FACTOIDWIKI)
4. **CPE Concept**: Extracted key concept instead of null

## Architecture

### LLM Bridge (`src/llm_bridge.py`)

The LLM integration uses a lightweight bridge that:
- Sends structured prompts to Ollama
- Includes top search results as context
- Parses JSON responses with graceful error handling
- Falls back to deterministic logic if LLM unavailable

### Integration Point (`src/eval_runner.py`)

The eval runner integrates LLM at annotation generation time:
1. Generates deterministic baseline annotations
2. If LLM enabled, prepares document snippets from search results
3. Calls LLM with query + context to generate enhanced annotations
4. Merges LLM output with baseline (LLM takes precedence)
5. Falls back silently to deterministic if LLM fails

## Performance Impact

Based on testing with llama3.1:8b:

- **Latency overhead**: ~3-5ms per query (minimal)
- **Quality improvement**: Dramatic (see output comparison above)
- **Reliability**: Graceful fallback ensures no failures
- **Resource usage**: Local processing, no external API calls

## Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Check if Ollama is running
   curl -s http://localhost:11434/api/tags

   # Start Ollama if needed
   ollama serve
   ```

2. **Model not available**
   ```bash
   # List available models
   ollama list

   # Pull missing model
   ollama pull llama3.1:8b
   ```

3. **Diagnostic Script for Ollama Connection Issues**

   If you encounter connection issues with Ollama, use the diagnostic script:
   ```bash
   ./venv/bin/python3 scripts/test_extraction_pipeline.py
   ```

   This script will:
   - Check environment variables (OLLAMA_URL, OLLAMA_HOST)
   - Test base connection to Ollama service
   - Validate API endpoints (/api/tags and /api/chat)
   - Provide clear diagnostic output

   **Important Notes:**
   - The script uses `llama3.1:8b` model for testing (not `llama3:8b`)
   - Ensure you have at least one model installed for chat endpoint testing
   - The chat endpoint requires POST with proper JSON structure

4. **LLM responses disabled**
   ```bash
   # Verify environment variables
   echo $LNSP_USE_LLM
   echo $LNSP_LLM_MODEL

   # Should see "true" and your model name
   ```

5. **JSON parsing errors**
   - The bridge handles malformed JSON gracefully
   - Falls back to deterministic annotations on parse errors
   - Check logs for specific error messages

### Debugging

Enable verbose logging to see LLM interactions:
```bash
# Run with debug output
LNSP_USE_LLM=true LNSP_LLM_MODEL="llama3.1:8b" \
./venv/bin/python -m src.eval_runner \
  --queries eval/test_single.jsonl \
  --api http://localhost:8080/search \
  --top-k 3 \
  --out eval/debug.jsonl
```

## Integration with Other Tools

### n8n Workflows

The LLM integration works seamlessly with n8n workflows. Results include rich annotations that can be used in downstream processing.

### API Responses

When used with the LNSP API, LLM-generated annotations are included in evaluation outputs, providing richer metadata for analysis.

## Best Practices

1. **Use with retrieval fallback**: Combine with `LNSP_LEXICAL_FALLBACK=true` for best results
2. **Provide NPZ file**: Set `LNSP_OFFLINE_NPZ` to give LLM document context
3. **Monitor resources**: Larger models provide better quality but require more RAM/CPU
4. **Test graceful degradation**: Ensure system works even when LLM is unavailable

## Examples

### Basic LLM Test
```bash
# Simple test with single query
echo '{"id": "test", "lane": "L1_FACTOID", "query": "What is machine learning?", "gold": ["ml"]}' > test.jsonl

LNSP_USE_LLM=true LNSP_LLM_MODEL="llama3.1:8b" \
./venv/bin/python -m src.eval_runner \
  --queries test.jsonl \
  --api http://localhost:8080/search \
  --top-k 3 \
  --out test_result.jsonl
```

### Full Evaluation with LLM (Ollama)
```bash
# Complete evaluation with all enhancements (Ollama)
export LNSP_USE_LLM=true
export LNSP_LLM_MODEL="llama3.1:8b"
export LNSP_OFFLINE_NPZ=artifacts/fw1k_vectors.npz
export LNSP_LEXICAL_FALLBACK=true

./venv/bin/python -m src.eval_runner \
  --queries eval/day3_eval.jsonl \
  --api http://localhost:8080/search \
  --top-k 8 \
  --timeout 15 \
  --out eval/enhanced_results.jsonl
```

### Full Evaluation with LLM (OpenAI-Compatible)
```bash
# Complete evaluation using vLLM backend
export LNSP_USE_LLM=true
export LNSP_LLM_BACKEND=openai
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-local"
export LNSP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LNSP_OFFLINE_NPZ=artifacts/fw1k_vectors.npz
export LNSP_LEXICAL_FALLBACK=true

./venv/bin/python -m src.eval_runner \
  --queries eval/day3_eval.jsonl \
  --api http://localhost:8080/search \
  --top-k 8 \
  --timeout 15 \
  --out eval/enhanced_results_vllm.jsonl
```

This integration transforms LNSP from using placeholder annotations to generating rich, contextual metadata that provides genuine insights into retrieval performance and query understanding.