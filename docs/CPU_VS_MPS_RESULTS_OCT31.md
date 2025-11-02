# CPU vs MPS Performance Results (October 31, 2025)

## Executive Summary

**Conclusion**: CPU is **2.93x FASTER** than MPS (Apple Silicon GPU) for vec2text decoding.

**Recommendation**: **Use CPU services on ports 7001/7002 for production.**

## Performance Results

### Latency Comparison

| Configuration | Avg Latency | Quality (ROUGE-L) | Throughput | Status |
|---------------|-------------|-------------------|------------|--------|
| **CPU → IELab** | **1,288ms** | 3.92/10 | 0.78 req/sec | ✅ **PRODUCTION** |
| **CPU → JXE** | **1,288ms** | 3.92/10 | 0.78 req/sec | ✅ **PRODUCTION** |
| MPS → IELab | 3,779ms | 3.92/10 | 0.26 req/sec | ⚠️ Not recommended |
| MPS → JXE | 3,785ms | 3.92/10 | 0.26 req/sec | ⚠️ Not recommended |

### Key Findings

1. ✅ **CPU is 2.93x faster** than MPS for vec2text decoding
2. ✅ **Quality is identical** across all configurations (3.92/10 ROUGE-L)
3. ✅ **JXE and IELab perform identically** on the same hardware
4. ⚠️ **MPS encoding is faster** (22ms vs 51ms) but **decoding destroys the advantage**

## Root Cause Analysis

### The Sequential Bottleneck

Vec2text decoding is **fundamentally sequential** and cannot benefit from GPU parallelism:

```python
# Vec2text's iterative refinement loop
for step in range(3):  # CANNOT parallelize this loop!
    candidate_text = t5_model(previous_output)  # Sequential dependency
    embedding = gtr_t5(candidate_text)
    if close_enough(embedding, target):
        break
    previous_output = candidate_text  # Each step depends on previous
```

### Batch Size Experiments

We tested if batching multiple vectors could improve MPS performance:

| Batch Size | Per-Item Decode Time | Insight |
|------------|---------------------|---------|
| 1 | 3,707ms | Baseline |
| 5 | 3,663ms | **No improvement!** |
| 10 | 3,668ms | **Still no improvement!** |

**Conclusion**: Per-item time stays constant at ~3,700ms regardless of batch size. This proves the bottleneck is **algorithmic, not hardware**.

## Why GPUs Can't Help

### 1. Iterative Refinement is Sequential

- Each iteration depends on the previous one (steps=3 means 3 serial passes)
- Beam width=1 means no beam-level parallelism
- Cannot parallelize across iterations

```
Step 1 → wait for completion → Step 2 → wait → Step 3 → done
```

Even with infinite GPU cores, you'd still wait for each step to finish before starting the next one.

### 2. GPU Parallelism Doesn't Apply

- GPUs excel at parallel operations on large batches
- Sequential operations hit memory/bandwidth bottlenecks
- MPS overhead (memory transfers, kernel launches) makes it worse
- No SIMD/parallel opportunities within a single sequential decode

### 3. CPU Wins with Different Parallelism

- 12 CPU cores can handle multiple **concurrent** requests (different vectors)
- Lower overhead for sequential operations
- Better for iterative algorithms like beam search
- Higher throughput through concurrency, not parallelism

## Architecture Analysis

### What CPUs Are Doing (12 cores at 100%)

✅ **Processing multiple independent decode requests in parallel**
- Each core runs one sequential decode operation
- Core 1: Decoding vector A (step 1 → step 2 → step 3)
- Core 2: Decoding vector B (step 1 → step 2 → step 3)
- Core 3: Decoding vector C (step 1 → step 2 → step 3)
- ...
- High throughput through concurrency

### What MPS Is Doing (barely used)

⚠️ **Trying to parallelize within a single sequential operation** (impossible!)
- Spending cycles on GPU memory transfers
- Waiting for sequential dependencies
- Cannot parallelize the iterative loop
- GPU sits mostly idle between iterations

## Test Methodology

### Test Files

1. **Quick Test** (`test_cpu_vs_mps_quick.py`)
   - 3 iterations per configuration
   - 3 test phrases
   - Total: 36 encode-decode cycles
   - Runtime: ~60 seconds

2. **Full Test** (`test_cpu_vs_mps_comparison.py`)
   - 10 iterations per configuration
   - 3 test phrases
   - Total: 120 encode-decode cycles
   - Runtime: ~3 minutes

3. **Batch Test** (`test_batch_sizes.py`)
   - Tests batch sizes: 1, 5, 10
   - Proves batching doesn't help
   - Runtime: ~45 seconds

### Test Environment

- **Hardware**: Apple Silicon M-series (M1/M2/M3)
- **OS**: macOS Sequoia (Darwin 25.0.0)
- **Python**: 3.13.7
- **PyTorch**: 2.6.0 with MPS support
- **Vec2Text**: Latest version
- **GTR-T5**: sentence-transformers/gtr-t5-base

### Test Phrases

```python
test_phrases = [
    "Artificial intelligence is a branch of computer science.",
    "Airplanes fly through the air using aerodynamic principles.",
    "Photosynthesis converts light energy into chemical energy in plants."
]
```

## Production Configuration

### Recommended Services (CPU)

```bash
# Start CPU encoder (port 7001)
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app \
  --host 127.0.0.1 --port 7001 --log-level error &

# Start CPU decoder (port 7002)
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app \
  --host 127.0.0.1 --port 7002 --log-level error &

# Or use convenience script
./scripts/start_lvm_services.sh
```

### Health Checks

```bash
# Check CPU encoder
curl http://localhost:7001/health
# Expected: {"status":"ok","service":"orchestrator_encoder","device":"cpu",...}

# Check CPU decoder
curl http://localhost:7002/health
# Expected: {"status":"ok","service":"orchestrator_decoder","device":"cpu",...}
```

### Usage Example

```python
import requests

text = "Artificial intelligence is a branch of computer science."

# Encode
encode_resp = requests.post("http://localhost:7001/encode",
                           json={"texts": [text]})
vector = encode_resp.json()["embeddings"][0]

# Decode
decode_resp = requests.post("http://localhost:7002/decode",
                           json={"vectors": [vector],
                                 "subscriber": "ielab",
                                 "steps": 3})
decoded = decode_resp.json()["results"][0]

print(f"Original: {text}")
print(f"Decoded:  {decoded}")
# Result: Meaningful output with semantic similarity
```

## What Would Help (But Doesn't Exist)

1. **Reduce steps**: Use `steps=1` instead of `steps=3`
   - ⚡ 3x faster
   - ⚠️ Lower quality output
   - Trade-off: speed vs accuracy

2. **Different decoder architecture**: Non-iterative methods
   - 🔬 Research problem
   - Would require new model training
   - Not available in current vec2text

3. **Larger beam width**: More parallelism within each step
   - Limited benefit (still bottlenecked by iteration)
   - Increases memory usage
   - Slight quality improvement

## References

- **Main Documentation**: `docs/how_to_use_jxe_and_ielab.md` (CPU vs MPS Performance Analysis section)
- **Project Guidance**: `CLAUDE.md` (CRITICAL: CORRECT ENCODER/DECODER section)
- **Test Files**:
  - `test_cpu_vs_mps_quick.py` (quick 3-iteration test)
  - `test_cpu_vs_mps_comparison.py` (full 10-iteration test)
  - `test_batch_sizes.py` (batching experiments)

## Conclusion

This isn't about fairness, tuning, or hardware limitations. **Vec2text's architecture fundamentally doesn't map to GPU parallelism**.

The iterative refinement loop requires sequential execution:

```
Step 1 → wait → Step 2 → wait → Step 3 → done
```

Even with infinite GPU cores, each step must wait for the previous one to complete. CPU's 2.93x advantage is **real and architectural**, not a configuration issue.

**Use CPU services (7001/7002) for production.** ✅
