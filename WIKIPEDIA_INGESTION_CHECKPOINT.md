# Wikipedia Ingestion Checkpoint (2025-10-16)

## Status: PAUSED

### Current Progress
- **Started**: 2025-10-16 (afternoon)
- **Paused**: 2025-10-16 (evening)
- **Runtime**: ~2 hours
- **Articles processed**: 2,393 (from original 1,032 → now at 3,425)
- **Concepts generated**: 232,525 total (started with 80,634)
- **New concepts**: ~152,000

### Database State
```sql
-- Current state (PostgreSQL)
SELECT COUNT(*) as concepts,
       COUNT(DISTINCT batch_id) as batches,
       MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER)) as max_article
FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';

-- Results:
concepts:  232,525
batches:   3,413
max_article: 3,425
```

### Resume Point

**Next article to process**: 3,426

```bash
# To resume Wikipedia ingestion:
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Resume from article 3,426 (continue from where we left off)
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3426 \
  --limit 7000 \
  > logs/wikipedia_ingestion_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > /tmp/wikipedia_ingestion.pid
echo "Wikipedia ingestion resumed. PID: $(cat /tmp/wikipedia_ingestion.pid)"
```

### Monitoring Commands

```bash
# Check progress
tail -f logs/wikipedia_ingestion_*.log

# Check database growth
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"

# Check process status
ps aux | grep ingest_wikipedia_pipeline
```

### Performance Notes

- **Actual speed**: ~15-25s per article (slower than expected 2.68s)
- **Estimated time for 7,000 more articles**: ~30-40 hours
- **Reason**: TMD extraction + LLM overhead + API congestion

### Next Steps

1. **Tonight**: Run overnight training on current 232k concepts
2. **Tomorrow**: Resume Wikipedia ingestion from article 3,426
3. **After completion**: Retrain with larger dataset

### Files Created
- `artifacts/graphmert_lvm/leafy_chain_graphs_80k.npz` (456 MB) - Entity-linked training graphs
- `logs/wikipedia_ingestion_20251016/ingestion.log` - Full ingestion log
