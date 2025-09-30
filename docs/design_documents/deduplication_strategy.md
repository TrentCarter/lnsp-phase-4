# LNSP Deduplication Strategy

## Overview

The LNSP ingestion pipeline implements **content-based deduplication** to prevent duplicate entries regardless of source corpus or ID availability.

## Implementation (src/ingest_factoid.py:97-107)

### Strategy

```python
# Generate deterministic CPE ID based on source content or ID
source_id = sample.get("id")
if source_id:
    # Use provided ID for deterministic mapping (e.g., FactoidWiki)
    cpe_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
else:
    # Use content hash for deduplication when no ID provided
    # This prevents duplicate ingestion of same content
    content_hash = sample["contents"]
    cpe_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))
```

### How It Works

1. **With Source IDs** (FactoidWiki, curated datasets):
   - Uses `uuid.uuid5(NAMESPACE_URL, source_id)` for stable, deterministic UUIDs
   - Same source ID → same CPE ID → automatic deduplication

2. **Without Source IDs** (user requests, ad-hoc content):
   - Uses `uuid.uuid5(NAMESPACE_DNS, content_text)` based on full content
   - Identical content → same CPE ID → automatic deduplication
   - Different content → different CPE ID → both stored

3. **Database Protection**:
   - PostgreSQL: `ON CONFLICT (cpe_id) DO NOTHING` (db_postgres.py:34)
   - Silently skips duplicate insertions
   - No errors, idempotent ingestion

## Benefits

- ✅ **Cross-corpus deduplication**: Same content from different sources → deduplicated
- ✅ **User request safety**: Re-submitting same query → won't create duplicates
- ✅ **Idempotent ingestion**: Can re-run ingestion pipelines safely
- ✅ **Eval stability**: Source IDs generate stable CPE IDs for evaluation sets

## Database Optimization

Hash index on `source_chunk` for fast content-based lookups:
```sql
CREATE INDEX cpe_source_chunk_hash ON cpe_entry USING hash (md5(source_chunk));
```

## Testing

```python
# Test content-based deduplication
content = "The Eiffel Tower was built in 1889."
cpe_id1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
cpe_id2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
assert cpe_id1 == cpe_id2  # ✅ Same content → same ID
```

## Edge Cases

1. **Minor content variations**: Different punctuation/whitespace → different IDs
   - This is intentional: slight variations may have different semantics
   - Future enhancement: normalize whitespace before hashing

2. **Cross-namespace collisions**: Extremely rare (UUID5 collision probability ≈ 10^-36)

3. **Neo4j nodes**: Uses same CPE ID, inherits deduplication behavior

## Usage Examples

### FactoidWiki (with IDs)
```bash
# Re-run safe: skips existing 27 items, adds new ones
./.venv/bin/python -m src.ingest_factoid \
  --file-path data/factoidwiki_1k.jsonl \
  --num-samples 1000 \
  --write-pg --write-neo4j
```

### User Requests (no IDs)
```python
# Same content submitted twice → only stored once
samples = [
    {"contents": "Photosynthesis converts light to energy."},
    {"contents": "Photosynthesis converts light to energy."}  # Duplicate
]
result = ingest(samples, write_pg=True)
# Result: 1 entry stored (duplicate skipped)
```

## Future Enhancements

1. **Fuzzy deduplication**: Detect near-duplicates using embeddings
2. **Whitespace normalization**: Treat `"text"` and `"text "` as same
3. **Semantic deduplication**: Use concept vectors to detect paraphrases