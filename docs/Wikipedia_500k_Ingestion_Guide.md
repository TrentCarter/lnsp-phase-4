# Wikipedia 500k Ingestion Guide

## Quick Start

### 1. Start Required APIs

```bash
# Terminal 1: Episode Chunker
./.venv/bin/uvicorn app.api.episode_chunker:app --port 8900

# Terminal 2: Semantic Chunker
./.venv/bin/uvicorn app.api.chunking:app --port 8001

# Terminal 3: TMD Router
./.venv/bin/uvicorn app.api.tmd_router:app --port 8002

# Terminal 4: GTR-T5 Embeddings
./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --port 8767

# Terminal 5: Ingest API
./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004
```

### 2. Run Ingestion (Choose One)

#### Option A: Small Test (10 articles)
```bash
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 10
```

#### Option B: Full 500k in Batches (Recommended)
```bash
# 50 batches of 10,000 articles each
LNSP_TMD_MODE=hybrid ./tools/ingest_wikipedia_batched.sh 10000 500000
```

#### Option C: Custom Batch Size
```bash
# 10 batches of 50,000 articles each
LNSP_TMD_MODE=hybrid ./tools/ingest_wikipedia_batched.sh 50000 500000
```

### 3. Monitor Progress

In a separate terminal:
```bash
./tools/monitor_wikipedia_ingestion.sh
```

## Performance Estimates (Hybrid TMD Mode)

| Metric | Per Article | For 500k Articles |
|--------|-------------|-------------------|
| Processing Time | 2.68s | ~372 hours (15.5 days) |
| Chunks Created | ~13 chunks | ~6.5M chunks |
| Storage (PostgreSQL) | ~30 KB | ~15 GB |
| Storage (FAISS) | ~40 KB | ~20 GB |
| Total Storage | ~70 KB | ~40 GB |

## Features

### Hybrid TMD Mode
- **Domain**: Extracted once per article using LLM (~200ms)
- **Task/Modifier**: Heuristic classification per chunk (~0.5ms)
- **Speed**: 6.5x faster than full LLM mode
- **Accuracy**: 70-80% (heuristics) vs 95% (LLM), but Domain captures most semantic meaning

### Batched Ingestion Benefits
- ✅ **Checkpointing**: Resume from last completed batch
- ✅ **Progress Tracking**: Per-batch metrics and ETA
- ✅ **Error Recovery**: Isolates failures to single batches
- ✅ **Resource Management**: Controlled memory usage

### Monitoring Features
- Real-time concept count updates (every 30s)
- Rate calculation (chunks/sec, articles/sec)
- ETA to completion
- Storage usage tracking

## Troubleshooting

### APIs Not Running
```bash
# Check all APIs
curl http://localhost:8900/health  # Episode Chunker
curl http://localhost:8001/health  # Semantic Chunker
curl http://localhost:8002/health  # TMD Router
curl http://localhost:8767/health  # GTR-T5 Embeddings
curl http://localhost:8004/health  # Ingest API
```

### Resume From Failure
```bash
# Check checkpoint file
cat artifacts/ingestion_metrics/checkpoint.txt

# Batched script automatically resumes from last checkpoint
LNSP_TMD_MODE=hybrid ./tools/ingest_wikipedia_batched.sh 10000 500000
```

### View Batch Logs
```bash
# Latest batch
tail -f logs/wikipedia_ingestion/batch_$(cat artifacts/ingestion_metrics/checkpoint.txt).log

# All batch summaries
grep "✅ Batch" logs/wikipedia_ingestion/*.log
```

## Pipeline Architecture

```
Wikipedia Article (10k chars)
    ↓
[Episode Chunker :8900] → 5-10 episodes
    ↓
[Semantic Chunker :8001] → 13 chunks avg (320 chars each)
    ↓
[TMD Router :8002] → Domain (LLM) + Task/Modifier (heuristics)
    ↓
[GTR-T5 Embeddings :8767] → 768D vectors (batch)
    ↓
[Ingest API :8004] → PostgreSQL + FAISS (atomic)
```

## Data Validation

After ingestion completes:

```bash
# Verify concept count
psql lnsp -c "SELECT count(*) FROM cpe_entry;"

# Check dataset sources
psql lnsp -c "SELECT dataset_source, count(*) FROM cpe_entry GROUP BY dataset_source ORDER BY count DESC LIMIT 10;"

# Verify TMD distribution
psql lnsp -c "SELECT domain_code, count(*) FROM cpe_entry GROUP BY domain_code ORDER BY count DESC;"

# Check FAISS files
ls -lh artifacts/*.npz
```

## Next Steps After Ingestion

1. **Build FAISS Index**
   ```bash
   make build-faiss
   ```

2. **Verify Data Synchronization**
   ```bash
   ./scripts/verify_data_sync.sh
   ```

3. **Run Benchmarks**
   ```bash
   PORT=8094 make slo-grid
   ```

4. **Train LVM** (if desired)
   ```bash
   # LVM training requires sequential data chains
   # See: docs/LVM_TRAINING_CRITICAL_FACTS.md
   ```

## Important Notes

- ✅ **Wikipedia is SAFE for LVM training** (sequential, narrative data)
- ❌ **Do NOT use ontologies for LVM training** (taxonomic, not sequential)
- ✅ **Ontologies are for vecRAG/GraphRAG only**
- ⚠️ **This is a multi-day operation** - run in batches with monitoring
- 💾 **Ensure 50GB+ free disk space** before starting
- 🔄 **PostgreSQL must be running** throughout ingestion
- 📊 **FAISS files auto-save** after each batch (verified Oct 4, 2025)

## Configuration

### Environment Variables
- `LNSP_TMD_MODE`: `hybrid` (recommended) or `full`
- `LNSP_LLM_ENDPOINT`: LLM endpoint for TMD Domain extraction (default: http://localhost:11434)
- `LNSP_LLM_MODEL`: LLM model name (default: llama3.1:8b)

### Batch Script Variables
- `BATCH_SIZE`: Articles per batch (default: 10000)
- `TOTAL_ARTICLES`: Total to ingest (default: 500000)
- `LOG_DIR`: Log directory (default: logs/wikipedia_ingestion)
- `METRICS_DIR`: Metrics directory (default: artifacts/ingestion_metrics)

## Files Created

### During Ingestion
- `logs/wikipedia_ingestion/batch_N.log` - Per-batch logs
- `artifacts/ingestion_metrics/batch_N_metrics.json` - Per-batch metrics
- `artifacts/ingestion_metrics/checkpoint.txt` - Resume checkpoint
- `artifacts/pipeline_metrics.json` - Overall pipeline metrics

### After Completion
- `artifacts/ingestion_metrics/final_summary.json` - Final statistics
- `artifacts/*.npz` - FAISS vector files (auto-created)
- PostgreSQL `cpe_entry` table - All concepts with TMD codes

## Support

For issues or questions:
- Check logs in `logs/wikipedia_ingestion/`
- Review metrics in `artifacts/ingestion_metrics/`
- See `CLAUDE.md` for project-wide guidance
- Refer to `LNSP_LONG_TERM_MEMORY.md` for critical rules
