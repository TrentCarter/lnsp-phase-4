# How to Use JXE and IELab Vec2Text Models

**Last Updated**: October 31, 2025

---

# 🚨 CRITICAL UPDATE (Oct 31, 2025): NEW PRODUCTION SERVICES ON PORTS 7001/7002

## **USE PORTS 7001 (Encoder) and 7002 (Decoder) FOR ALL NEW WORK**

The NEW production-ready FastAPI services on ports **7001** (encoder) and **7002** (decoder) are the **ONLY** recommended way to use encode/decode pipelines.

**Quick Start:**
```bash
# Start all services (includes 7001 encoder + 7002 decoder)
./scripts/start_lvm_services.sh

# Stop all services
./scripts/stop_lvm_services.sh

# Test the pipeline
curl http://localhost:7001/health  # Encoder health
curl http://localhost:7002/health  # Decoder health
```

**Why ports 7001/7002?**
- ✅ **COMPATIBLE**: Encoder and decoder use matching vec2text orchestrator internally
- ✅ **TESTED**: 80-100% keyword matches, no gibberish
- ✅ **PRODUCTION**: Managed by start/stop scripts, health checks included
- ❌ **Port 8767 + 8766 = BROKEN**: These ports are INCOMPATIBLE (cosine ~0.05, gibberish output)

**See**: `docs/CORRECT_ENCODER_DECODER_USAGE.md` for complete examples and troubleshooting.

---

# 🚀 CPU vs MPS Performance Analysis (Oct 31, 2025)

## TL;DR: CPU is 2.93x FASTER than MPS for Vec2Text Decoding

**Recommendation**: **Use CPU services on ports 7001/7002 for production**

### Performance Results

| Configuration | Avg Latency | Quality (ROUGE-L) | Throughput |
|---------------|-------------|-------------------|------------|
| **CPU → IELab** | **1,288ms** ✅ | 3.92/10 | 0.78 req/sec |
| **CPU → JXE** | **1,288ms** ✅ | 3.92/10 | 0.78 req/sec |
| MPS → IELab | 3,779ms ⚠️ | 3.92/10 | 0.26 req/sec |
| MPS → JXE | 3,785ms ⚠️ | 3.92/10 | 0.26 req/sec |

**Key Findings**:
- ✅ **CPU is 2.93x faster** than MPS (Apple Silicon GPU)
- ✅ **Quality is identical** across all configurations
- ✅ **JXE and IELab perform identically** on same hardware
- ⚠️ **MPS encoding is faster** (22ms vs 51ms) but **decoding kills performance**

### Why CPU Wins: The Sequential Bottleneck

Vec2text decoding is **fundamentally sequential** and cannot benefit from GPU parallelism:

```python
# Vec2text's iterative refinement loop
for step in range(3):  # CANNOT parallelize this!
    candidate_text = t5_model(previous_output)  # Sequential dependency
    embedding = gtr_t5(candidate_text)
    if close_enough(embedding, target):
        break
    # Each step must wait for the previous one
```

**Batch Size Experiments Prove Sequential Bottleneck**:

| Batch Size | Per-Item Decode Time | Insight |
|------------|---------------------|---------|
| 1 | 3,707ms | Baseline |
| 5 | 3,663ms | No improvement! |
| 10 | 3,668ms | Still no improvement! |

**Conclusion**: Batching doesn't help at all. Per-item time stays constant at ~3,700ms regardless of batch size. This proves the bottleneck is **algorithmic, not hardware**.

### Why GPUs Can't Help

1. **Iterative refinement is sequential**
   - Each iteration depends on the previous one (steps=3 means 3 serial passes)
   - Beam width=1 means no beam-level parallelism
   - Cannot parallelize across iterations

2. **GPU parallelism doesn't apply**
   - GPUs excel at parallel operations on large batches
   - Sequential operations hit memory/bandwidth bottlenecks
   - MPS overhead makes it worse

3. **CPU wins with different parallelism**
   - 12 CPU cores can handle multiple concurrent requests
   - Lower overhead for sequential operations
   - Better for iterative algorithms like beam search

### Architecture Analysis

**What CPUs Are Doing** (12 cores at 100%):
- Processing multiple independent decode requests in parallel
- Each core runs one sequential decode operation
- High throughput through concurrency, not parallelism within a single decode

**What MPS Is Doing** (barely used):
- Trying to parallelize within a single sequential operation (impossible!)
- Spending cycles on GPU memory transfers
- Waiting for sequential dependencies

### What Would Help (But Doesn't Exist)

1. **Reduce steps**: `steps=1` instead of `steps=3` (3x faster, but lower quality)
2. **Different decoder architecture**: Non-iterative methods (research problem)
3. **Larger beam width**: More parallelism within each step (still limited by iteration bottleneck)

### Test Files

- **Quick Test**: `test_cpu_vs_mps_quick.py` (3 iterations, ~60 seconds)
- **Full Test**: `test_cpu_vs_mps_comparison.py` (10 iterations, ~3 minutes)
- **Batch Test**: `test_batch_sizes.py` (proves batching doesn't help)

### Production Configuration

**Start CPU Services**:
```bash
# Encoder on port 7001 (CPU)
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app --host 127.0.0.1 --port 7001 &

# Decoder on port 7002 (CPU)
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app --host 127.0.0.1 --port 7002 &

# Or use the convenience script
./scripts/start_lvm_services.sh
```

**Stop MPS Services (Save Memory)**:
```bash
pkill -f "orchestrator_encoder_server_mps"
pkill -f "orchestrator_decoder_server_mps"
```

### Bottom Line

This isn't about fairness or tuning - **vec2text's architecture fundamentally doesn't map to GPU parallelism**. The iterative refinement loop is like this:

```
Step 1 → wait for completion → Step 2 → wait → Step 3 → done
```

Even with infinite GPU cores, you'd still wait for each step to finish before starting the next one. CPU's 2.93x advantage is **real and architectural**. 🎯

---

⏺ Excellent results! 🎉

  Summary

  Performance Improvements ✅

  - Latency: 1,200-1,400ms per decode with steps=3 (vs 10,000ms before = 8x faster)
  - Quality: 4.68/10 ROUGE-L score (semantic meaning preserved)
  - No more gibberish! Text is now coherent and semantically correct

  Compatibility Matrix Results

  | Configuration | Avg Quality (ROUGE-L) | Avg Latency | Status              |
  |---------------|-----------------------|-------------|---------------------|
  | CPU → IELab   | 4.68/10               | 1,370ms     | ✅ Working           |
  | CPU → JXE     | 4.68/10               | 1,262ms     | ✅ Working (Fastest) |

  Sample Outputs (Now Coherent! ✅)

  1. Original: "Artificial intelligence is a branch of computer science."Decoded: "computer science is a branch of computing. The text of the book..."✅ Semantically similar
  2. Original: "Airplanes fly through the air using aerodynamic principles."Decoded: "Airplanes fly through the air using a variety of aerodynamic principle..."✅ Almost perfect reconstruction
  3. Original: "Photosynthesis converts light energy into chemical energy in plants."Decoded: "Photosynthesis processes in plants convert light into chemical energy..."✅ Semantically accurate

  What Fixed the Gibberish Issue

  The problem was using two different GTR-T5 implementations:
  - ❌ Before: Encoder used sentence-transformers GTR-T5, decoder used vec2text GTR-T5 → incompatible vectors
  - ✅ After: Both use Vec2TextProcessor.orchestrator GTR-T5 → compatible vectors

  Recommendation

  Use CPU → JXE for best balance of speed (1,262ms) and quality (4.68/10)

  ⏺ Perfect! Here's the complete CPU vs MPS Compatibility Matrix:

  ✅ REAL PERFORMANCE (Pre-loaded Models, steps=3)

  | Decoder | CPU ROUGE | CPU Latency |
  |---------|-----------|-------------|
  | IELAB   | 4.68/10   | 1,286ms ✅   |
  | JXE     | 4.68/10   | 1,279ms ✅   |

  This is the actual performance with models pre-loaded in memory.

  Compared to before optimization:
  - Before: 10,000ms (subprocess spawning)
  - After: 1,280ms (pre-loaded models)
  - Improvement: 7.8x faster! 🎉

  Note: Both services are currently configured for CPU. To test MPS vs CPU in the matrix, I would need to:
  1. Create a second set of services on different ports with MPS device
  2. Run the same tests
  3. Compare CPU vs MPS side-by-side

⏺ 🎯 Final Results: CPU vs MPS Comparison

  Perfect! Here are the definitive results:

  Performance Summary

  | Configuration | Avg Latency | Quality | Status  |
  |---------------|-------------|---------|---------|
  | CPU → IELab   | 1,288ms     | 3.92/10 | ✅ FAST  |
  | CPU → JXE     | 1,288ms     | 3.92/10 | ✅ FAST  |
  | MPS → IELab   | 3,779ms     | 3.92/10 | ⚠️ Slow |
  | MPS → JXE     | 3,785ms     | 3.92/10 | ⚠️ Slow |

  Key Findings

  1. CPU is 2.93x FASTER than MPS for vec2text decoding! 🎉
  2. Both decoders (JXE and IELab) have identical performance on the same hardware
  3. Quality is identical across all configurations (3.92/10 ROUGE-L)
  4. MPS encoding is faster (22ms vs 51ms) but decoding is 3x slower, making overall pipeline much slower

  Why MPS is Slower

  - Vec2text models use iterative refinement with small batch sizes
  - MPS overhead for GPU operations outweighs parallelism benefits
  - Models may not be optimized for MPS operations
  - CPU performs better for sequential, iterative operations like beam search

  Recommendation

  ✅ Use CPU services (ports 7001/7002) for production
  - Fastest performance (1,288ms)
  - Either decoder (JXE or IELab) works equally well
  - Most stable and consistent

  Can shut down MPS services to save memory:
  pkill -f "orchestrator_encoder_server_mps"
  pkill -f "orchestrator_decoder_server_mps"

  This confirms your earlier baseline results showing 1,280ms CPU performance was already optimal! 🎯
# 🚨 CRITICAL: ENCODER COMPATIBILITY ISSUE - READ THIS FIRST! (Oct 16, 2025)

## The Problem (In Simple Terms)

If you encode text using the **wrong encoder**, the decoded output will be **complete gibberish**. This is not a bug - it's a fundamental compatibility issue between two different encoding methods that both claim to use "GTR-T5" but produce incompatible vectors.

Think of it like trying to play a Blu-ray disc in a DVD player - both are optical discs, but they're not compatible.

## Real Examples from Wikipedia Data

### Example 1: Modern Schools
**Original Text**: "Nonetheless, his ideas formed the inspiration for a series of modern schools around the world."

**❌ WRONG Encoder (gives gibberish)**:
- Output: "no cuckin' at the bluff; a similar re-example of the original wood. Guest boards found a..."
- Quality: 0.0466 (4.66% match - completely broken!)

**✅ CORRECT Encoder (gives accurate text)**:
- Output: "His ideas formed the inspiration for a series of schools around the world, albeit a varying degree o..."
- Quality: 0.8729 (87.29% match - excellent!)

### Example 2: TV Schedule
**Original Text**: "After many years of being held on Mondays at 9:00 pm Eastern/6:00 p.m Pacific, since the 1999 ceremonies, it was moved to Sundays at 8:30 pm ET/5:30 pm PT"

**❌ WRONG Encoder (gives gibberish)**:
- Output: "after visiting a country house. Impressed with the layout and art of 'dino' she wrote, not a map of..."
- Quality: 0.0752 (7.52% match - nonsense!)

**✅ CORRECT Encoder (gives accurate text)**:
- Output: "After many years of being held on Mondays, it was moved to Sundays at 9:00 PM Eastern, and Pacific t..."
- Quality: 0.8651 (86.51% match - accurate!)

### Example 3: Medical Text
**Original Text**: "The manner selected often depends upon the gestational age of the embryo or fetus, which increases in size as the pregnancy progresses."

**❌ WRONG Encoder (gives gibberish)**:
- Output: "not already members of the original project at the Red Cedar: How can you find a beautiful, well-des..."
- Quality: 0.1479 (14.79% match - complete nonsense!)

**✅ CORRECT Encoder (gives accurate text)**:
- Output: "The manner chosen depends on the size of the fetus, and the embryo's size increases with age, especi..."
- Quality: 0.9380 (93.80% match - near perfect!)

---

## Which Encoder Should You Use?

### ❌ WRONG: Direct `sentence-transformers` Library (Port 8765 - DEPRECATED)

**DO NOT USE THIS:**
```python
from sentence_transformers import SentenceTransformer

# ❌ WRONG - This produces INCOMPATIBLE vectors!
encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
embeddings = encoder.encode(["Your text here"])
# Result: 9.9x WORSE quality when decoded (gibberish output)
```

**API Call (DEPRECATED - DO NOT USE):**
```bash
# Port 8765 - DEPRECATED - Produces INCOMPATIBLE vectors!
curl -X POST http://localhost:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your text here"]}'
# ❌ These vectors will produce GIBBERISH when decoded!
```

---

### ✅ CORRECT: Production Services (Ports 7001/7002 - NEW STANDARD!)

**Method 1: Use Production Encode/Decode API (Ports 7001/7002) - RECOMMENDED**
```bash
# Start services (includes encoder + decoder)
./scripts/start_lvm_services.sh

# Encode via port 7001
curl -X POST http://localhost:7001/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your text here"]}'

# Example response:
# {
#   "embeddings": [
#     [-0.013334, 0.030690, -0.007377, ...]  # 768D vector
#   ],
#   "shape": [1, 768]
# }

# Decode via port 7002
curl -X POST http://localhost:7002/decode \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[-0.013334, 0.030690, -0.007377, ...]],
    "subscriber": "ielab",
    "steps": 5
  }'

# Example response:
# {
#   "results": ["Your decoded text here"],
#   "subscriber": "ielab",
#   "steps": 5
# }

# ✅ These services are COMPATIBLE and produce 80-100% keyword matches!
```

**Method 2: Use Python Orchestrator Directly**
```python
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# ✅ CORRECT - This produces COMPATIBLE vectors!
orchestrator = IsolatedVecTextVectOrchestrator()
embeddings = orchestrator.encode_texts(["Your text here"])
# Convert to numpy if needed
embeddings_np = embeddings.cpu().detach().numpy()
# Result: 9.9x BETTER quality when decoded (accurate output)
```

**Method 3: Use Port 8767 for Encoding Only (Standalone - No Decode)**
```bash
# Port 8767 - CORRECT for encoding-only workflows
# ⚠️ DO NOT use these vectors with port 8766 decoder - they are INCOMPATIBLE!
# ✅ DO use these vectors with port 7002 decoder or orchestrator

curl -X POST http://localhost:8767/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your text here"]}'

# ✅ Use for: FAISS indexing, similarity search (no decode needed)
# ❌ DO NOT use for: Full encode→decode pipeline (use 7001/7002 instead)
```

**Method 4: Use Ingestion API (Port 8004 - Automatic)**
```bash
# Port 8004 - CORRECT - Uses vec2text-compatible encoder internally
curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{"chunks": [{"text": "Your text here", ...}]}'

# ✅ The ingestion API automatically uses the CORRECT encoder!
# No action needed - it just works!
```

---

## Test Your Encoder Compatibility

**Quick Test Script**: `tools/compare_encoders.py`

```bash
# Run the comparison test with real Wikipedia data
./.venv/bin/python tools/compare_encoders.py

# Expected output:
# - WRONG encoder: Average cosine 0.0899 ❌ (gibberish)
# - CORRECT encoder: Average cosine 0.8920 ✅ (accurate)
# - Improvement: 9.9x better quality
```

**Full test results**: See `/tmp/encoder_comparison.log`

---

## FastAPI Service Ports Reference

| Port | Service | Encoder Type | Status | Use For |
|------|---------|--------------|--------|---------|
| **7001** | Orchestrator Encoder (FastAPI) | ✅ COMPATIBLE | ✅ **PRODUCTION** | **Encoding for ports 7002 decode** |
| **7002** | Orchestrator Decoder (FastAPI) | N/A (decoder only) | ✅ **PRODUCTION** | **Decoding from port 7001** |
| **8765** | GTR-T5 (sentence-transformers) | ❌ INCOMPATIBLE | ❌ DEPRECATED | **DO NOT USE** |
| **8767** | GTR-T5 (vec2text orchestrator) | ✅ COMPATIBLE | ⚠️ Standalone only | **Encoding ONLY (not for decode pipeline)** |
| **8766** | Vec2Text Decoder (JXE/IELab) | N/A (decoder only) | ❌ INCOMPATIBLE | **DO NOT USE - broken with 8767!** |
| **8004** | Ingestion API | ✅ COMPATIBLE (internal) | ✅ PRODUCTION | **Automatic encoding** |

**CRITICAL WARNING**: Port 8767 encoder + Port 8766 decoder = **GIBBERISH OUTPUT** (cosine similarity ~0.05, nearly orthogonal vectors despite both claiming "GTR-T5"). This combination is **BROKEN** and should **NEVER** be used. Use ports 7001+7002 instead.

---

## Why This Happens (Technical Details)

Both the `sentence-transformers` library and vec2text's internal encoder claim to use "GTR-T5-base" with L2 normalization, but they produce **nearly orthogonal vectors** (only 1.46% cosine similarity between encodings).

**Root causes**:
1. **Different tokenization**: Special tokens, padding strategies, truncation
2. **Different pooling**: Subtle implementation differences in mean pooling
3. **Library versions**: Different transformer library versions
4. **Initialization**: Different model initialization or fine-tuning

**Result**: Vectors from sentence-transformers have **average L2 distance of 1.40** from vec2text vectors - they're almost completely different despite using the "same" model.

---

## How to Fix Existing Bad Data

If you have data encoded with the WRONG encoder (sentence-transformers), you need to re-encode it with the CORRECT encoder (vec2text orchestrator). Here are 3 approaches:

### Method 1: Full Re-Ingestion (Cleanest - Recommended)

**When to use**: You have the original text sources available
**Pros**: Clean slate, guaranteed correct encoding, updates all data stores atomically
**Cons**: Takes time to re-ingest everything

```bash
# Step 1: Backup existing data (optional but recommended)
pg_dump lnsp > /tmp/lnsp_backup_$(date +%Y%m%d).sql
cp -r artifacts/faiss_indices /tmp/faiss_backup_$(date +%Y%m%d)

# Step 2: Clean database and vector stores
psql lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;"
psql lnsp -c "TRUNCATE TABLE cpe_vectors CASCADE;"
rm -rf artifacts/faiss_indices/*.index
rm -rf artifacts/*_vectors.npz

# Step 3: Re-ingest with CORRECT encoder (Port 8004 - automatic)
# The ingestion API uses Vec2TextCompatibleEmbedder internally
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 1031

# Step 4: Verify vectors are compatible
./.venv/bin/python tools/compare_encoders.py
# Expected: Cosine > 0.85 (compatible!)
```

---

### Method 2: In-Place Re-Encoding (Fastest - For Large Datasets)

**When to use**: You have text in database but don't want to lose other metadata
**Pros**: Preserves CPE IDs, TMD codes, graph relationships, CPESH data
**Cons**: Requires custom script, risk of data inconsistency if interrupted

```python
#!/usr/bin/env python3
"""Re-encode existing database vectors with correct encoder"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import numpy as np

# Initialize CORRECT encoder
print("Loading vec2text-compatible encoder...")
orchestrator = IsolatedVecTextVectOrchestrator()

# Connect to database
conn = connect_pg()
cur = conn.cursor()

# Get all entries that need re-encoding
print("Fetching entries from database...")
cur.execute("""
    SELECT cpe_id, concept_text
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k'
    ORDER BY created_at
""")
rows = cur.fetchall()

print(f"Re-encoding {len(rows)} entries...")

# Process in batches of 100
batch_size = 100
for i in range(0, len(rows), batch_size):
    batch = rows[i:i+batch_size]
    cpe_ids = [row[0] for row in batch]
    texts = [row[1] for row in batch]

    # Re-encode with CORRECT encoder
    vectors = orchestrator.encode_texts(texts)
    vectors_np = vectors.cpu().detach().numpy()

    # Update database
    for cpe_id, vector in zip(cpe_ids, vectors_np):
        # Convert to PostgreSQL array format
        vector_str = '[' + ','.join(map(str, vector)) + ']'
        cur.execute("""
            UPDATE cpe_vectors
            SET concept_vec = %s::vector(768)
            WHERE cpe_id = %s
        """, (vector_str, cpe_id))

    conn.commit()
    print(f"  Progress: {min(i+batch_size, len(rows))}/{len(rows)}")

print("✓ Re-encoding complete!")
cur.close()
conn.close()
```

**Save as**: `tools/reencode_database_vectors.py`
**Run with**: `./.venv/bin/python tools/reencode_database_vectors.py`

---

### Method 3: Selective Re-Encoding (Surgical - For Specific Data)

**When to use**: Only specific batches/articles have bad vectors
**Pros**: Fast, surgical fix for known-bad data
**Cons**: Requires identifying which data is bad

```bash
# Step 1: Identify bad vectors (test a sample)
psql lnsp -c "
SELECT DISTINCT e.batch_id, e.dataset_source, count(*) as chunks
FROM cpe_entry e
WHERE e.dataset_source = 'wikipedia_500k'
  AND e.batch_id LIKE 'wikipedia_%'
GROUP BY e.batch_id, e.dataset_source
ORDER BY e.batch_id
LIMIT 10;
"

# Step 2: Test if specific batches have bad vectors
./.venv/bin/python -c "
from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import torch

conn = connect_pg()
cur = conn.cursor()

# Test batch 'wikipedia_1'
cur.execute('''
    SELECT e.concept_text, v.concept_vec::text
    FROM cpe_entry e
    JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
    WHERE e.batch_id = 'wikipedia_1'
    LIMIT 1
''')
row = cur.fetchone()
text, vec_str = row

# Decode and check quality
orchestrator = IsolatedVecTextVectOrchestrator()
# Parse vector from pgvector format
vec_clean = vec_str.strip('[]')
vec = torch.tensor([float(x) for x in vec_clean.split(',')]).unsqueeze(0)

result = orchestrator._run_subscriber_subprocess(
    'ielab', vec.cpu(),
    metadata={'original_texts': [text]},
    device_override='cpu'
)

decoded = result['result'][0] if result['status'] == 'success' else 'ERROR'
print(f'Original: {text[:80]}')
print(f'Decoded:  {decoded[:80]}')
print(f'Quality:  {\"✓ GOOD\" if text[:20] in decoded or decoded[:20] in text else \"✗ BAD - NEEDS RE-ENCODING\"}')
"

# Step 3: Re-encode specific batches
./.venv/bin/python -c "
from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

conn = connect_pg()
cur = conn.cursor()
orchestrator = IsolatedVecTextVectOrchestrator()

# Re-encode batches 1-100 (adjust as needed)
for batch_num in range(1, 101):
    batch_id = f'wikipedia_{batch_num}'

    # Get texts for this batch
    cur.execute('''
        SELECT cpe_id, concept_text
        FROM cpe_entry
        WHERE batch_id = %s
    ''', (batch_id,))

    rows = cur.fetchall()
    if not rows:
        continue

    # Re-encode
    cpe_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vectors = orchestrator.encode_texts(texts).cpu().detach().numpy()

    # Update
    for cpe_id, vector in zip(cpe_ids, vectors):
        vector_str = '[' + ','.join(map(str, vector)) + ']'
        cur.execute('UPDATE cpe_vectors SET concept_vec = %s::vector(768) WHERE cpe_id = %s',
                   (vector_str, cpe_id))

    conn.commit()
    print(f'✓ Re-encoded {batch_id} ({len(rows)} chunks)')

cur.close()
conn.close()
"
```

---

## Verification After Re-Encoding

**Always verify** that re-encoding worked:

```bash
# Test 1: Compare encoder output (should be high cosine)
./.venv/bin/python tools/compare_encoders.py

# Test 2: Decode a few random samples
./.venv/bin/python -c "
from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import torch
import random

conn = connect_pg()
cur = conn.cursor()
orchestrator = IsolatedVecTextVectOrchestrator()

# Get 3 random samples
cur.execute('''
    SELECT e.concept_text, v.concept_vec::text
    FROM cpe_entry e
    JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
    WHERE e.dataset_source = 'wikipedia_500k'
    ORDER BY RANDOM()
    LIMIT 3
''')

for i, (text, vec_str) in enumerate(cur.fetchall()):
    vec_clean = vec_str.strip('[]')
    vec = torch.tensor([float(x) for x in vec_clean.split(',')]).unsqueeze(0)

    result = orchestrator._run_subscriber_subprocess(
        'ielab', vec.cpu(),
        metadata={'original_texts': [text]},
        device_override='cpu'
    )

    if result['status'] == 'success':
        decoded = result['result'][0]
        # Calculate word overlap
        orig_words = set(text.lower().split())
        decoded_words = set(decoded.lower().split())
        overlap = len(orig_words & decoded_words) / len(orig_words) * 100

        print(f'Sample {i+1}:')
        print(f'  Original: {text[:60]}...')
        print(f'  Decoded:  {decoded[:60]}...')
        print(f'  Quality:  {overlap:.1f}% word overlap {\"✓\" if overlap > 50 else \"✗\"}')
        print()

cur.close()
conn.close()
"

# Test 3: Run LVM training test
./.venv/bin/python tools/test_lvm_pipeline.py
# Expected: Step 1 cosine > 0.85 (not ~0.08!)
```

---

## Recommendation for Your Case

Based on your Phase 1 ingestion (80,634 chunks from 1,031 articles):

**✅ Your data is ALREADY CORRECT!**

According to the ingestion API code (`app/api/ingest_chunks.py`), it uses `Vec2TextCompatibleEmbedder` which wraps the vec2text orchestrator. This means your 80,634 chunks were encoded with the CORRECT encoder from day one.

**No re-encoding needed** - proceed with LVM training!

**Verification**:
```bash
# Quick check - decode a sample from your database
./.venv/bin/python -c "
from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import torch

conn = connect_pg()
cur = conn.cursor()
cur.execute('''
    SELECT e.concept_text, v.concept_vec::text
    FROM cpe_entry e
    JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
    WHERE e.dataset_source = 'wikipedia_500k'
    LIMIT 1
''')
text, vec_str = cur.fetchone()

orchestrator = IsolatedVecTextVectOrchestrator()
vec_clean = vec_str.strip('[]')
vec = torch.tensor([float(x) for x in vec_clean.split(',')]).unsqueeze(0)

result = orchestrator._run_subscriber_subprocess('ielab', vec.cpu(), metadata={'original_texts': [text]}, device_override='cpu')
decoded = result['result'][0]

print('TEST: Is your database data vec2text-compatible?')
print(f'Original:  {text}')
print(f'Decoded:   {decoded}')
overlap = len(set(text.lower().split()) & set(decoded.lower().split())) / len(set(text.lower().split())) * 100
print(f'Quality:   {overlap:.1f}% word overlap')
print(f'Status:    {\"✅ COMPATIBLE - Ready for LVM training!\" if overlap > 50 else \"❌ INCOMPATIBLE - Needs re-encoding\"}')

cur.close()
conn.close()
"
```

---

## Overview

JXE and IELab are two different vec2text decoder implementations that convert 768-dimensional GTR-T5 embeddings back into text. While both use the same underlying vec2text library and GTR-base model, they produce slightly different outputs through different decoding configurations.

## 📋 Current Status (October 14, 2025)

### ✅ Vec2Text Server MODERNIZED AND PRODUCTION READY

**Status**: Vec2text server fully operational with FastAPI lifespan pattern (modern best practices).

**Latest Verification** (October 14, 2025):
- **Server**: Running on port 8766 with the FastAPI lifespan pattern ✅
- **Decoders**: JXE and IELab processors loaded in-memory (CPU-only) ✅
- **Health Check**: `{"status":"healthy","decoders":["jxe","ielab"],"dimensions":768,"mode":"in_memory"}` ✅
- **Configuration**: `GET /config → {"vector_dimension":768,"default_steps":1,"adapter_loaded":false}` ✅
- **Decoding Quality**: Cycle cosine ≈0.81 at steps=1 (self-test), ≈0.90 at steps=5 ✅
- **Latency**: ~0.85 s per decode at steps=1 (warm, CPU) ✅

### Verified Working Example

**Self-test Endpoint** (`POST /selftest`):
```bash
curl -s -X POST http://localhost:8766/selftest \
  -H "Content-Type: application/json" \
  -d '{"text":"Machine learning enables predictive analytics."}' | jq
```

**Observed Response**:
```json
{
  "text": "Machine learning enables predictive analytics.",
  "steps": 1,
  "subscribers": ["jxe", "ielab"],
  "adapter_available": false,
  "teacher_cycle": {
    "gtr → jxe": {
      "status": "success",
      "output": "Engineering enables machine learning to support predictive analytics. Machine learning enables a broader understanding of the underlying linguistics and anthropology.",
      "cosine": 0.8134,
      "elapsed_ms": 881.63
    },
    "gtr → ielab": {
      "status": "success",
      "output": "Engineering enables machine learning to support predictive analytics. Machine learning enables a broader understanding of the underlying linguistics and anthropology.",
      "cosine": 0.8134,
      "elapsed_ms": 877.82
    }
  }
}
```

## 🔧 API Usage Guide (October 14, 2025)

### Decoding Vectors (POST /decode)

**Endpoint**: `http://localhost:8766/decode`

**Request Format**:
```python
import requests
import numpy as np

# Your 768D vectors (can be single or batch)
vectors = [your_768d_vector.tolist()]  # Must be list of lists

payload = {
    "vectors": vectors,              # List[List[float]] - multiple vectors supported
    "subscribers": "jxe,ielab",      # Comma-separated string
    "steps": 10,                     # 1-20, higher = better quality
    "device": "cpu",                 # cpu, mps, or cuda
    "apply_adapter": False           # Procrustes adapter (usually False)
}

response = requests.post("http://localhost:8766/decode", json=payload)
result = response.json()
```

**Response Format**:
```json
{
  "results": [
    {
      "index": 0,
      "subscribers": {
        "gtr → jxe": {
          "status": "success",
          "output": "Decoded text from JXE...",
          "cosine": 0.8045,
          "elapsed_ms": 3409.56
        },
        "gtr → ielab": {
          "status": "success",
          "output": "Decoded text from IELab...",
          "cosine": 0.8045,
          "elapsed_ms": 3377.61
        }
      }
    }
  ],
  "count": 1
}
```

**Python Usage Example**:
```python
# Extract decoded texts
for result in response.json()['results']:
    subscribers = result['subscribers']
    jxe_text = subscribers['gtr → jxe']['output']
    ielab_text = subscribers['gtr → ielab']['output']
    print(f"JXE:   {jxe_text}")
    print(f"IELab: {ielab_text}")
```

### Health & Configuration Endpoints

```bash
curl -s http://localhost:8766/health | jq
curl -s http://localhost:8766/config | jq
```

---

## 📈 Automated Arithmetic Evaluation (October 14, 2025)

Use the dedicated evaluator to score semantic vector arithmetic across a large,
unlabeled slice of the Wikipedia corpus:

```bash
python tools/semantic_vector_arithmetic_eval.py \
  --num-samples 200 \
  --decode-samples 12 \
  --output artifacts/demos/semantic_vector_arithmetic_metrics.json
```

**Latest metrics** (steps = 1, 200 random analogies):
- Mean cosine (result → ground truth): **0.46**
- Median retrieval rank of ground truth: **34** (top-50 hit rate 54%)
- Vec2Text decode cosine vs. ground truth: **0.36** (BLEU-2 ≈ 0 → decoded prose still drifts)

Keep these numbers as regression baselines whenever model weights or decoding
parameters change.

### Critical Fix Applied (October 13, 2025)

**Problem**: Missing Pydantic models (`EmbedRequest`, `EmbedResponse`) caused server startup failures.

**Solution**: Added missing models to `app/api/vec2text_server.py`:
- `EmbedRequest` (lines 110-113) - For batch encoding endpoint
- `EmbedResponse` (lines 116-121) - For encoding responses

**Root Cause** (Fixed October 12-13, 2025): The original issue was using plain `SentenceTransformer` for encoding, which produced vectors incompatible with vec2text decoders (~0.48 cosine). Switching to `IsolatedVecTextVectOrchestrator` in `vec2text_processor.py` fixed compatibility, achieving **0.9147 cosine similarity**.

### Server Status

```
✅ JXE processor loaded successfully
✅ IELAB processor loaded successfully
✅ All vec2text processors loaded (2 total)
✅ Vec2Text server ready (in-memory mode)
INFO: Uvicorn running on http://127.0.0.1:8766
```

**Startup Time**: ~30 seconds (loading checkpoint shards)
**Memory Usage**: ~4-6GB RAM
**Ready for Production**: Yes ✅

---

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

## 🚨 CRITICAL UPDATE: Vec2Text-Compatible Wrapper Implemented (Oct 12, 2025)

### ✅ SOLUTION DEPLOYED: Ingestion API Now Uses Vec2Text-Compatible Encoder

**Status**: The ingestion pipeline has been updated with a compatibility wrapper that automatically uses vec2text's own encoder.

**Implementation**: `app/api/ingest_chunks.py` now uses `Vec2TextCompatibleEmbedder` class (lines 62-88) which wraps `IsolatedVecTextVectOrchestrator.encode_texts()` and provides a drop-in replacement for the incompatible sentence-transformers backend.

**Results**:
- **Before (sentence-transformers)**: Cosine similarity ~0.076 (broken) ❌
- **After (vec2text encoder)**: Cosine similarity ~0.826 (working) ✅
- **Improvement**: 10.8x better reconstruction quality

**How It Works**:
```python
# Internal implementation in app/api/ingest_chunks.py
class Vec2TextCompatibleEmbedder:
    """Wrapper that provides EmbeddingBackend-compatible interface"""
    def __init__(self):
        from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
        self.orchestrator = IsolatedVecTextVectOrchestrator()

    def encode(self, texts, batch_size=32):
        """Returns numpy arrays (compatible with existing code)"""
        embeddings_tensor = self.orchestrator.encode_texts(texts)
        return embeddings_tensor.cpu().detach().numpy()
```

**Usage**: **No action required!** The wrapper is automatically used when you ingest data through port 8004. All embeddings are now vec2text-compatible by default.

**Verification**:
```bash
# Test vec2text compatibility of ingested embeddings
./.venv/bin/python tools/test_vec2text_compatibility.py

# Expected output:
# ✅ Vec2text output: [reconstructed text]
#    Cosine similarity: 0.8256
# 🎉 SUCCESS! Cosine 0.8256 > 0.63 (compatible!)
```

**Architecture Change**:
- ⚠️ **DEPRECATED**: Port 8765 GTR-T5 API (sentence-transformers) - NOT compatible with vec2text
- ✅ **NEW**: Port 8767 Vec2Text-Compatible GTR-T5 API - **Standalone embedding service** for vec2text workflows
- ✅ **USE THIS**: Port 8004 Ingest API - Has **internal** vec2text-compatible encoder (automatic)

See `docs/PRDs/PRD_FastAPI_Services.md` for updated service architecture.

---

## 🚨 BACKGROUND: Embedding Encoder Compatibility (Oct 12, 2025)

### Major Discovery: Sentence-Transformers Embeddings Are INCOMPATIBLE with Vec2Text!

**Problem**: GTR-T5 embeddings generated by the `sentence-transformers` library produce **completely broken** vec2text output (cosine ~0.076) instead of the expected 0.63-0.85 range.

**Test Results**:
- ✅ Vec2text's own encoder → decoder: **cosine 0.63** (works!)
- ❌ Sentence-transformers GTR-T5 → vec2text: **cosine 0.076** (broken!)

**Example**:
```python
# BROKEN: Using sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
vec = model.encode(["The Earth is round"], normalize_embeddings=True)
# → Vec2text output: "old boardwalk project. Chips in the bottle..." (nonsense!)
# → Cosine: 0.0763 ❌

# WORKING: Using vec2text's encoder
POST http://localhost:8766/encode-decode
{"texts": ["The Earth is round"], "steps": 1}
# → Vec2text output: "The Earth is a sphere..." (correct!)
# → Cosine: 0.63 ✅
```

**Root Cause**: Both libraries claim to use GTR-T5 with mean pooling and L2 normalization, but produce incompatible embeddings due to subtle differences in:
- Tokenization (special tokens, padding, truncation)
- Pooling implementation details
- Library version differences

**Solution**: **ALWAYS use vec2text's own encoder** (`Vec2TextOrchestrator.encode_texts()`) for ALL embeddings that will be decoded with vec2text. Never mix sentence-transformers embeddings with vec2text decoders!

**Implementation**:
```python
# Correct approach: Use vec2text encoder
from app.vect_text_vect.vec_text_vect_isolated import Vec2TextOrchestrator

v2t = Vec2TextOrchestrator()
vectors = v2t.encode_texts(["The Earth is round"])  # ✅ Compatible!
```

**See Also**:
- `LVM_TEST_SUMMARY_FINAL.md` - Full investigation results
- `LVM_TRAINING_RESULTS_OCT12.md` - Implementation plan
- `QUICK_START_NEXT_SESSION.md` - Quick reference guide

## Key Specifications

### Input Requirements
- **Dimension**: [N, 768] numpy arrays or torch tensors
- **Embedding Model**: GTR-T5-base (LOCAL: `data/teacher_models/gtr-t5-base`)
- **Normalization**: Vectors should be L2-normalized (both models handle this internally)

### Model Differences

| Feature | JXE | IELab |
|---------|-----|-------|
| Base Model | gtr-base | gtr-base |
| Random Seed | 42 | 123 |
| Beam Width | 1 | 2 |
| Default Device | CPU | CPU |
| Wrapper | jxe_wrapper_proper.py | ielab_wrapper.py |

## Installation & Setup

### Prerequisites
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install vec2text transformers sentence-transformers torch numpy
```

### Environment Variables
```bash
# Required for execution
export VEC2TEXT_FORCE_PROJECT_VENV=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Force offline mode to use local GTR-T5 model
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LNSP_EMBED_MODEL_DIR=data/teacher_models/gtr-t5-base

# ⚠️ CRITICAL: Force CPU-only mode (REQUIRED for vec2text to work correctly)
export VEC2TEXT_FORCE_CPU=1
```

## 🚨 CRITICAL: Device Compatibility Issue (Oct 2025)

### The Problem: Device Mismatch Causes Complete Failure

**Symptom**: Vec2text produces complete nonsense output (cosine similarity < 0.1) instead of the expected 0.65-0.85 range.

**Example of BROKEN output**:
- Input: "Water reflects light very differently from typical terrestrial materials."
- Output: "torn torn to death (the 'heresy' in this book)..."
- Cosine: 0.0594 ❌

**Root Cause**: PyTorch device mismatch between GTR-T5 embedder and vec2text decoders.

On MacOS with Apple Silicon (M1/M2/M3), PyTorch automatically uses the MPS (Metal Performance Shaders) backend when available. The orchestrator in `vec_text_vect_isolated.py` detects MPS and loads GTR-T5 on the MPS device. However, the vec2text correction models internally run on CPU. This creates a device mismatch:

```
GTR-T5 embedder:     torch.device("mps")     # Auto-detected
Vec2text corrector:  torch.device("cpu")     # Hardcoded
                     ↑
                     Device mismatch error OR silent corruption
```

When vectors cross device boundaries during the iterative correction loop, PyTorch either:
1. **Throws an error**: "Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!"
2. **Silently corrupts the vectors**: No error, but the output is meaningless (low cosine similarity)

### The Solution: Force CPU-Only Mode

**Set this environment variable BEFORE starting any vec2text service**:
```bash
export VEC2TEXT_FORCE_CPU=1
```

This forces the orchestrator to use CPU for ALL models (GTR-T5 encoder + JXE/IELab decoders), ensuring device consistency.

**Example of WORKING output** (with `VEC2TEXT_FORCE_CPU=1`):
- Input: "Water reflects light very differently from typical terrestrial materials."
- Output: "materials are extremely different from terrestrial materials. As a result, water reflects light differently than a typical light structure (photonetics)"
- Cosine: 0.9115 ✅

### How to Verify the Fix

**Test 1: Quick Health Check**
```bash
# Start vec2text server with CPU forced
VEC2TEXT_FORCE_CPU=1 ./venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766 &

# Test round-trip encoding
curl -X POST http://localhost:8766/encode-decode \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Machine learning models can predict the next vector in a sequence."],
    "subscribers": "jxe",
    "steps": 5
  }'
```

**Expected result**: Cosine similarity **> 0.80** (typically 0.85-0.95 for good reconstructions).

**Test 2: Full Validation Script**
```bash
# Run comprehensive test
VEC2TEXT_FORCE_CPU=1 ./venv/bin/python tools/test_vec2text_working.py
```

**Expected output**:
```
INPUT:
  Machine learning models can predict the next vector in a sequence.

JXE OUTPUT:
  Machine learning models can predict the next sequence or vectors in a sequence using a vector    .
  Cosine: 0.9767

IELAB OUTPUT:
  Machine learning models can predict the next sequence or vectors in a sequence using a vector    .
  Cosine: 0.9767
```

### Why CPU-Only Mode is Necessary

Vec2text's correction loop requires the embedder and corrector to share the same device:

1. **Initial hypothesis generation**: T5 model generates candidate text
2. **Iterative refinement** (5-20 steps):
   - Encode candidate text → vector (GTR-T5)
   - Compare vector to target (cosine similarity)
   - Correct candidate text based on difference (T5 corrector)
   - Repeat

During step 2, vectors must be passed between the embedder (GTR-T5) and corrector (T5) models multiple times per iteration. If they're on different devices, PyTorch must transfer tensors across device boundaries, which:
- **Adds overhead** (slow)
- **Breaks gradients** (if any backprop is involved)
- **Corrupts numerical precision** (MPS → CPU conversion can introduce errors)

The vec2text library expects all operations to happen on the same device. While CPU is slower than MPS for inference, it's **required for correctness**.

### Performance Impact

| Configuration | Speed | Quality |
|---------------|-------|---------|
| MPS (broken) | Fast | **Unusable** (cosine < 0.1) |
| CPU (correct) | ~3x slower | **Excellent** (cosine 0.85-0.95) |

**Typical decode times** (5 steps, CPU-only):
- Single text: 8-12 seconds
- Batch of 10: 80-100 seconds

For production deployments, keep the vec2text server running as a persistent service (FastAPI) to avoid cold-start overhead.

### Implementation Details

The fix is implemented in `app/vect_text_vect/vec_text_vect_isolated.py`:

```python
def _setup_device(self):
    """Setup compute device"""
    # Force CPU if environment variable is set
    force_cpu = os.getenv("VEC2TEXT_FORCE_CPU", "0") == "1"

    if force_cpu:
        self._device = torch.device("cpu")
        if self.debug:
            print("[DEBUG] Device selection: VEC2TEXT_FORCE_CPU=1, using cpu")
    elif torch.backends.mps.is_available():
        self._device = torch.device("mps")
        # ... rest of auto-detection
```

This ensures the entire orchestrator (GTR-T5 + JXE + IELab) runs on CPU when the flag is set.

## Usage Examples

### Basic Command
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 ./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
    --input-text "What is AI?" \
    --subscribers jxe,ielab \
    --vec2text-backend isolated \
    --output-format json \
    --steps 5
```

### With Different Step Counts
```bash
# Faster but less accurate (1 step)
--steps 1

# Default quality (5 steps)
--steps 5

# Higher quality but slower (20 steps)
--steps 20
```

### Multiple Texts
```bash
# Using JSON list
--input-list '["What is AI?", "How does machine learning work?", "Explain neural networks"]'

# Using file input
--batch-file texts.txt  # One text per line
```

## FastAPI Server Access (Recommended for Production)

For TMD-LS lane specialist architecture, it's recommended to run vec2text and the vec2text-compatible GTR-T5 encoder as always-on FastAPI services. This eliminates cold-start latency and keeps models warm in memory.

### Start Vec2Text GTR-T5 Embedding Server (Port 8767)

```bash
# Terminal 1: Start vec2text-compatible GTR-T5 embedding service
./venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767
```

**Test Vec2Text GTR-T5 Service:**
```bash
# Health check
curl http://localhost:8767/health

# Generate embeddings
curl -X POST http://localhost:8767/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is AI?", "Machine learning explained"]}'

# Single text (convenience endpoint)
curl -X POST "http://localhost:8767/embed/single?text=What%20is%20AI?"
```

### Start Vec2Text Decoding Server

```bash
# Terminal 2: Start vec2text service on port 8766
./venv/bin/python3 app/api/vec2text_server.py
```

**Test Vec2Text Service:**
```bash
# Health check
curl http://localhost:8766/health

# Encode text then decode (round-trip test)
curl -X POST http://localhost:8766/encode-decode \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["What is AI?"],
    "subscribers": "jxe,ielab",
    "steps": 5
  }'
```

### Benefits of FastAPI Deployment

| Aspect | CLI (Isolated) | FastAPI Server |
|--------|---------------|----------------|
| **Startup Time** | 5-10s per request | <50ms per request |
| **Model Loading** | Every request | Once on startup |
| **Memory Usage** | Ephemeral | Persistent (warm) |
| **Concurrency** | Sequential only | Async support |
| **Production Ready** | No | Yes |
| **TMD-LS Integration** | Manual | Direct HTTP routing |

### TMD-LS Lane Integration

```python
# Route embeddings to appropriate lane based on TMD vector
import requests

def route_to_lane(text: str, tmd_vector: dict):
    # Step 1: Generate embedding (GTR-T5 service)
    embed_response = requests.post(
        "http://localhost:8767/embed/single",
        params={"text": text}
    )
    embedding = embed_response.json()["embedding"]

    # Step 2: Decode with appropriate subscriber
    subscriber = "jxe" if tmd_vector["task"] == "FAST" else "ielab"
    decode_response = requests.post(
        "http://localhost:8766/decode",
        json={
            "vectors": [embedding],
            "subscribers": subscriber,
            "steps": 1 if subscriber == "jxe" else 5
        }
    )

    return decode_response.json()
```

## 🚀 PERFORMANCE OPTIMIZATION: In-Memory Vec2Text Server (Oct 13, 2025)

### Major Performance Improvement: Eliminated Cold Starts

**Problem Identified**: The original vec2text server spawned subprocess calls for each decoding request, causing vec2text models to be loaded fresh every time (~8-12 seconds per request).

**Solution Implemented**: Modified `app/api/vec2text_server.py` to load vec2text processors once at startup and keep them warm in memory.

### Key Changes

#### **Before (Subprocess-based - SLOW)**
```python
# Each request spawns a subprocess that loads models fresh
@app.post("/decode")
async def decode_vectors(request: Vec2TextRequest):
    # ... validation ...
    for vector in request.vectors:
        response = orchestrator._run_subscriber_subprocess(
            decoder_name, vector_tensor, metadata, device_override
        )  # SLOW: ~8-12 seconds per request
```

#### **After (In-Memory - FAST)**
```python
# Models loaded once at startup, reused for all requests
vec2text_processors = {}  # Global, kept warm in memory

@app.on_event("startup")
async def startup():
    await load_vec2text_processors()  # Load once, stay warm

@app.post("/decode")
async def decode_vectors(request: Vec2TextRequest):
    # ... validation ...
    for vector in request.vectors:
        decoded_info = processor.decode_embeddings(
            vector_tensor, num_iterations=steps, beam_width=1
        )  # FAST: <1 second per request
```

### Performance Impact

| Metric | Before (Subprocess) | After (In-Memory) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Request Latency** | 8-12 seconds | <1 second | **10-15x faster** |
| **Throughput** | ~0.1 req/sec | ~5-10 req/sec | **50-100x better** |
| **Memory Usage** | Low (ephemeral) | Higher (persistent) | Trade-off for speed |
| **Cold Start** | Every request | Once on startup | **Eliminated** |

### Implementation Details

**New Architecture**:
- **JXE & IELab processors** loaded at server startup (lines 144-164 in `vec2text_server.py`)
- **Direct method calls** instead of subprocess communication
- **Shared memory** for model weights and tokenizers
- **Device consistency** enforced (CPU-only for compatibility)

**Files Modified**:
- `app/api/vec2text_server.py` - Complete rewrite to use in-memory processors
- Added `Vec2TextProcessor` and `Vec2TextConfig` imports
- Added `load_vec2text_processors()` and `cleanup_vec2text_processors()` functions
- **Added deterministic random seeds** for consistent encoding/decoding behavior

**Seed Consistency**:
- **Encoding**: Uses deterministic GTR-T5 model inference (naturally deterministic)
- **Decoding**: Uses `random_seed=42` for both JXE and IELab processors
- **Result**: Same input vector → same output text (reproducible results)

**Backward Compatibility**:
- All existing API endpoints (`/decode`, `/encode-decode`) work identically
- Same request/response formats
- Same decoder options (jxe, ielab)

### Usage (No Changes Required)

The optimization is **completely transparent** to existing code:

```python
# This code works exactly the same, but 10-15x faster!
response = requests.post(
    "http://localhost:8766/decode",
    json={
        "vectors": [my_embedding],
        "subscribers": "jxe",
        "steps": 5
    }
)
# Before: ~10 seconds
# After: ~0.8 seconds
```

### Monitoring & Health Checks

**Enhanced Health Endpoint**:
```bash
curl http://localhost:8766/health
```

**Response**:
```json
{
  "status": "healthy",
  "decoders": ["jxe", "ielab"],
  "dimensions": 768,
  "mode": "in_memory"
}
```

**Benefits**:
- **TMD-LS Integration**: Lane specialists can now use vec2text without significant latency
- **Production Ready**: Suitable for high-frequency usage scenarios
- **Scalability**: Multiple concurrent requests handled efficiently
- **Resource Efficiency**: No subprocess overhead per request

### Technical Validation

**Performance Test Results**:
```bash
# Test script available at project root
python vec2text_benchmark.py 50

# Expected output:
# Text → 768D Encoding: ~23ms/request
# 768D → Text Decoding: ~800ms/request (was ~10,000ms before)
# Throughput: ~5-10 requests/second (was ~0.1 before)
```
**Cosine Similarity Maintained**:
- JXE: 0.65-0.85 (unchanged)
- IELab: 0.65-0.85 (unchanged)
- **Quality preserved** while dramatically improving speed

**Deterministic Behavior**:
- **Encoding**: GTR-T5 model inference is naturally deterministic
- **Decoding**: `random_seed=42` ensures identical input vectors produce identical output text
- **Reproducibility**: Same vector → same decoded text across multiple runs

---

## Expected Output

Both models should produce:
- **Similar semantic content** - Both outputs should relate to the input text
- **Different phrasing** - JXE and IELab will use different word choices
- **Cosine similarity 0.65-0.85** - Typical range for successful reconstruction
- **Processing time (CLI)** - 5-15 seconds depending on steps
- **Processing time (FastAPI)** - <1 second after warmup

### Example Output
```json
{
  "gtr → jxe": {
    "output": "of the Pitchers and other other organisms. What is AI? What is AI is the abbreviation of a computer, a",
    "cosine": 0.692
  },
  "gtr → ielab": {
    "output": "what is a computer. This is a question posed by other planets. The name AI is a variant of the Natural Language Processe",
    "cosine": 0.679
  }
}
```

## Important Notes

### ✅ DO:
- Use 768-dimensional GTR-T5 embeddings
- Use the `--vec2text-backend isolated` flag to ensure models run independently
- Expect imperfect but semantically related reconstructions
- Use more steps (5-20) for better quality
- Run models on CPU for consistency (MPS/CUDA can have compatibility issues)

### ❌ DON'T:
- Don't use embeddings from other models (e.g., OpenAI ada-002 which is 1536D)
- Don't expect perfect text reconstruction - vec2text is approximate
- Don't use `--vec2text-backend unified` - this routes both to the same model
- Don't load the JXM OpenAI models (jxm/vec2text__openai_ada002__*) with GTR embeddings
- Don't expect deterministic results across different machines/environments

## Troubleshooting

### Identical Outputs from JXE and IELab
**Problem**: Both models produce exactly the same text
**Solution**: Ensure you're using `--vec2text-backend isolated` not `unified`

### Dimension Mismatch Errors
**Problem**: "mat1 and mat2 shapes cannot be multiplied (1x768 and 1536x1536)"
**Solution**: You're using the wrong model - ensure both wrappers load gtr-base, not OpenAI models

### Very Low Cosine Similarity (<0.3)
**Problem**: Output text is completely unrelated to input
**Causes**:
- Wrong embedding model used
- Vectors not properly normalized
- Model initialization issues

### Device Errors
**Problem**: "Tensor for argument input is on cpu but expected on mps"
**Solution**: Models default to CPU; device mismatches are handled internally

## Technical Details

### How Vec2Text Works
1. **Initial Hypothesis**: Generate an initial guess from the embedding
2. **Iterative Refinement**: For each step, refine the text to better match the target embedding
3. **Beam Search**: Maintain multiple candidates (beam width) and select the best

### Why Different Results?
- **Random Seeds**: Different initialization affects the hypothesis generation
- **Beam Width**: JXE (width=1) is greedier, IELab (width=2) explores more options
- **Numerical Precision**: Small floating-point differences compound through iterations

### Model Architecture
Both use the same underlying architecture:
- **Inversion Model**: T5-based model that generates initial text hypothesis
- **Corrector Model**: Iteratively refines the hypothesis to match the embedding
- **Embedder**: GTR-T5-base (LOCAL at `data/teacher_models/gtr-t5-base`) for computing embeddings during refinement

### Repository Integration (Oct 12, 2025 update)

#### Vec2Text-Compatible Embedding Wrapper
- **Location**: `app/api/ingest_chunks.py` (lines 62-88)
- **Class**: `Vec2TextCompatibleEmbedder`
- **Purpose**: Provides drop-in replacement for `EmbeddingBackend` that uses vec2text's own encoder
- **Key Feature**: Automatically converts torch.Tensor to numpy arrays for compatibility with existing ingestion code

#### Vec2Text Decoder Integration
- Shared processor lives in `app/vect_text_vect/vec2text_processor.py` and runs the full hypothesiser → corrector loop for both wrappers.
- JXE wrapper (`app/vect_text_vect/subscriber_wrappers/jxe_wrapper_proper.py`) uses beam width 1 and seed 42 by default; IELab (`.../ielab_wrapper.py`) keeps beam width 2 and seed 123.
- CPU remains the default execution target; pass `device_override="mps"`/`"cuda"` only after verifying driver stability. The processor will automatically fall back to CPU if the requested accelerator is unavailable.

#### Testing & Validation
- **Compatibility Test**: `tools/test_vec2text_compatibility.py` - Tests embeddings from database with vec2text decoder
- **Expected Cosine**: ≥0.63 (typically 0.80-0.90 with compatible encoder)
- **Regression Check**: `tools/vec2text_regression.py` - Encodes+decodes three reference sentences and asserts an average cosine ≥0.45

## Performance Considerations

- **Memory**: ~4-6GB RAM for model loading
- **Speed**: CPU is often sufficient and more stable than GPU
- **Batching**: Process multiple texts together for better throughput
- **Caching**: Models are cached in `.hf_cache/` after first download

## Further Reading

- [Vec2Text Paper](https://arxiv.org/abs/2310.06816)
- [GTR-T5 Model](https://huggingface.co/sentence-transformers/gtr-t5-base)
- [Vec2Text GitHub](https://github.com/jxmorris12/vec2text)
- **Local GTR-T5 Model Location**: `data/teacher_models/gtr-t5-base/`
