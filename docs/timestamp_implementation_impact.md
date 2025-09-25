# Timestamp Implementation Impact Analysis

## Overview
To support indefinite CPESH cache persistence with proper audit trails, we need to add timestamp tracking (`created_at`, `last_accessed`) throughout the pipeline.

## Files and Functions Requiring Updates

### 1. Core Schema Updates

#### `src/schemas.py`
**Current State:** CPESH model has no timestamp fields
**Required Changes:**
```python
class CPESH(BaseModel):
    # Existing fields...
    created_at: Optional[str] = None      # ISO8601 timestamp
    last_accessed: Optional[str] = None   # ISO8601 timestamp
```

### 2. API Implementation

#### `src/api/retrieve.py`
**Current State:** Basic CPESH extraction without caching
**Required Changes:**

1. **Add cache initialization in `__init__()`:**
   - Load existing cache from JSONL with timestamps
   - Parse `created_at` and `last_accessed` fields

2. **Update `search()` method:**
   - When cache hit: Update `last_accessed` timestamp
   - When cache miss: Set `created_at` timestamp
   - Persist both timestamps to cache file

3. **New functions needed:**
   ```python
   def _load_cpesh_cache(self) -> Dict[str, Dict]
   def _save_cpesh_entry(self, doc_id: str, cpesh: CPESH, created_at: str, last_accessed: str)
   def _update_last_accessed(self, doc_id: str)
   ```

### 3. Database Schema Updates

#### `scripts/init_postgres.sql` & `scripts/init_pg.sql`
**Current State:** `cpe_entry` table has `created_at` but no CPESH-specific timestamps
**Required Changes:**

Option A - Separate CPESH cache table:
```sql
CREATE TABLE IF NOT EXISTS cpesh_cache (
    doc_id TEXT PRIMARY KEY,
    cpesh JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count INTEGER DEFAULT 1
);
```

Option B - Add to existing `cpe_entry`:
```sql
ALTER TABLE cpe_entry
ADD COLUMN cpesh_created_at TIMESTAMPTZ,
ADD COLUMN cpesh_last_accessed TIMESTAMPTZ,
ADD COLUMN cpesh_access_count INTEGER DEFAULT 0;
```

### 4. Cache File Format

#### `artifacts/cpesh_cache.jsonl`
**Current Format:** Not implemented yet
**Required Format:**
```json
{
  "doc_id": "550e8400-e29b-41d4-a716",
  "cpesh": {
    "concept": "...",
    "probe": "...",
    "expected": "...",
    "soft_negative": "...",
    "hard_negative": "...",
    "soft_sim": 0.82,
    "hard_sim": 0.45
  },
  "created_at": "2025-09-24T10:30:00Z",
  "last_accessed": "2025-09-24T14:45:00Z",
  "access_count": 3
}
```

### 5. Utility Functions

#### New file: `src/utils/timestamps.py`
```python
from datetime import datetime, timezone

def get_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()

def parse_iso_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts)
```

### 6. Test Updates

#### `tests/test_tmd_acceptance.py`
**Add new test class:**
```python
class TestCPESHTimestamps:
    def test_cache_entry_has_timestamps(self)
    def test_last_accessed_updates_on_hit(self)
    def test_created_at_persists(self)
```

#### New file: `tests/test_cpesh_cache.py`
- Test cache loading with timestamps
- Test cache persistence
- Test timestamp updates
- Test access counting

### 7. Configuration Updates

#### Environment Variables (already documented)
- `LNSP_CPESH_CACHE` - Cache file location
- `LNSP_CPESH_MAX_K` - Max items to enrich
- `LNSP_CPESH_TIMEOUT_S` - LLM timeout

### 8. Migration/Upgrade Path

#### New file: `scripts/migrate_cpesh_timestamps.py`
For existing deployments without timestamps:
```python
def migrate_cache_file(old_path: str, new_path: str):
    """Add timestamps to existing cache entries."""
    # Read old format
    # Add created_at = file modification time
    # Add last_accessed = created_at
    # Write new format
```

## Implementation Priority

1. **Phase 1 - Core Implementation** (Required for S5)
   - Update `src/schemas.py` with timestamp fields
   - Modify `src/api/retrieve.py` to handle timestamps
   - Create `src/utils/timestamps.py`
   - Update cache file format

2. **Phase 2 - Persistence** (Nice to have)
   - Database schema updates
   - Migration scripts
   - Backup/restore utilities

3. **Phase 3 - Analytics** (Future)
   - Access pattern analysis
   - Cache effectiveness metrics
   - Pruning recommendations

## Backward Compatibility

- Cache entries without timestamps treated as `created_at = now()`
- Gradual migration as entries are accessed
- No breaking changes to API contract

## Testing Strategy

1. Unit tests for timestamp utilities
2. Integration tests for cache operations
3. Performance tests to ensure timestamp overhead is minimal
4. Migration tests for existing caches

## Monitoring & Observability

Add metrics for:
- Cache hit rate over time
- Average age of cached entries
- Access frequency distribution
- Cache size growth rate

## Estimated Impact

- **Code changes:** ~200 lines across 5 files
- **New code:** ~150 lines (utilities + tests)
- **Database changes:** Optional (can use file-only initially)
- **Performance impact:** Minimal (<1ms per operation)
- **Storage impact:** ~100 bytes per cache entry for timestamps