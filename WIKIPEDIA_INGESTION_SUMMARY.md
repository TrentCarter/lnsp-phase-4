# Wikipedia Ingestion - Quick Reference Summary

**Last Updated**: October 23, 2025  
**Status**: ✅ Ready for next batch

---

## Current State

✅ **8,447 articles** successfully ingested (99.73% success rate)  
✅ **584,545 chunks** with GTR-T5 768D embeddings  
✅ **Article range**: 1-8,470 (missing: 421, 7151, 7691)  
✅ **Database**: PostgreSQL `lnsp` (tables: `cpe_entry`, `cpe_vectors`)

---

## Resume Ingestion Tonight

```bash
# Start next batch (10,000 articles, ~9.4 hours)
./tools/resume_wikipedia_ingestion.sh

# Or custom amount:
./tools/resume_wikipedia_ingestion.sh 5000  # Process 5k articles
```

**Next batch**: Articles 8,471-18,470  
**Estimated time**: ~9.4 hours (10k articles)

---

## Performance Metrics

- **3.4 seconds** per article average
- **8,222x database speedup** (COPY-based bulk ingestion)
- **279ms** embedding generation (GTR-T5 768D)
- **9ms** database write (down from 74,000ms!)

---

## Key Files

**Scripts**:
- `tools/ingest_wikipedia_bulk.py` - Production ingestion (all fixes applied)
- `tools/resume_wikipedia_ingestion.sh` - Resume script for next batch

**Documentation**:
- `docs/WIKIPEDIA_INGESTION_STATE.md` - Complete current state
- `docs/WIKIPEDIA_BULK_INGESTION_LESSONS.md` - All gotchas & solutions

**Source Data**:
- `data/datasets/wikipedia/wikipedia_500k.jsonl` - 500,000 articles

**Logs**:
- `logs/wikipedia_FINAL_fixed_20251023_183926.log` - Last successful run

---

## Critical Gotchas (All Fixed!)

1. ✅ **UTF-8 encoding** - Multi-layer sanitization applied
2. ✅ **JSON encoding** - `ensure_ascii=True` with control char removal
3. ✅ **NULL constraints** - Made CPESH/TMD fields nullable
4. ✅ **INSERT mismatch** - Added `NOW()` for `created_at`
5. ✅ **Transaction aborts** - Proactive ROLLBACK before batches
6. ✅ **Data preservation** - Always checks MAX article before resuming

---

## Verify Current State

```bash
# Article count
psql lnsp -c "SELECT COUNT(DISTINCT chunk_position->>'article_index') FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"

# Chunk count
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"

# Check for gaps
psql lnsp -c "
WITH expected AS (SELECT generate_series(1, 8470) AS article_id),
     actual AS (SELECT DISTINCT CAST(chunk_position->>'article_index' AS INTEGER) AS article_id 
                FROM cpe_entry WHERE dataset_source = 'wikipedia_500k')
SELECT e.article_id FROM expected e LEFT JOIN actual a ON e.article_id = a.article_id 
WHERE a.article_id IS NULL;
"
```

---

## Ready for LVM Training

The data is ready for Latent Vector Model training:

✅ 584,545 chunks with 768D embeddings  
✅ Temporal ordering preserved (article_index, chunk_index)  
✅ No CPESH required (vector sequences only)

**Next Steps**:
1. Continue ingestion to 100k+ articles
2. Extract sequential chunk chains for training
3. Train LVM models (LSTM, Transformer, etc.)

See `docs/LVM_DATA_MAP.md` for training pipeline details.

---

**Questions?** See `docs/WIKIPEDIA_BULK_INGESTION_LESSONS.md` for complete troubleshooting guide.
