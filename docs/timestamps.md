# Timestamp Schema Policy

## Overview

This document defines the timestamp schema policy for the LNSP pipeline, ensuring consistent timestamp handling across all storage layers (Text DB, Vector DB, Graph DB).

## Core Principles

1. **Indefinite Retention**: All data persists indefinitely until manually pruned
2. **Audit Trail**: Every record tracks creation and last access times
3. **ISO8601 Format**: All timestamps use UTC ISO8601 format
4. **No Auto-Expiration**: No TTL-based automatic deletion

## Timestamp Fields

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `created_at` | ISO8601 | When the record was first created/extracted | `2025-09-25T10:30:00Z` |
| `last_accessed` | ISO8601 | When the record was last read/retrieved | `2025-09-25T14:45:00Z` |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `modified_at` | ISO8601 | When the record content was updated | `2025-09-25T12:00:00Z` |
| `access_count` | Integer | Number of times accessed | `42` |
| `last_modified_by` | String | System/user that last modified | `cpesh_extractor` |

## Layer-Specific Implementation

### 1. Text Database (PostgreSQL)

```sql
-- Core timestamp columns for all tables
created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
last_accessed    TIMESTAMPTZ NOT NULL DEFAULT now(),
modified_at      TIMESTAMPTZ,
access_count     INTEGER DEFAULT 1

-- CPESH cache specific
CREATE TABLE IF NOT EXISTS cpesh_cache (
    doc_id           TEXT PRIMARY KEY,
    cpesh            JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed    TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count     INTEGER DEFAULT 1,
    quality_score    FLOAT,
    last_modified_by TEXT DEFAULT 'system'
);

-- Index for pruning queries
CREATE INDEX cpesh_cache_access_idx ON cpesh_cache (last_accessed);
CREATE INDEX cpesh_cache_quality_idx ON cpesh_cache (quality_score);
```

### 2. Vector Database (FAISS)

Since FAISS doesn't natively support metadata, timestamps are stored in companion structures:

```python
# Companion metadata for FAISS vectors
vector_metadata = {
    "doc_id": {
        "created_at": "2025-09-25T10:30:00Z",
        "last_accessed": "2025-09-25T14:45:00Z",
        "vector_idx": 12345,  # FAISS index position
        "access_count": 1
    }
}
```

Storage options:
- **Primary**: PostgreSQL `cpe_vectors` table with timestamp columns
- **Secondary**: NPZ file with parallel timestamp arrays
- **Cache**: In-memory dict with periodic JSONL persistence

### 3. Graph Database (Neo4j)

```cypher
// Node timestamps
CREATE (n:Concept {
    cpe_id: $cpe_id,
    created_at: datetime(),
    last_accessed: datetime(),
    access_count: 1
})

// Relationship timestamps
CREATE (a)-[r:RELATES_TO {
    created_at: datetime(),
    confidence: 0.85,
    last_verified: datetime()
}]->(b)

// Index for temporal queries
CREATE INDEX concept_created_idx FOR (n:Concept) ON (n.created_at);
CREATE INDEX concept_accessed_idx FOR (n:Concept) ON (n.last_accessed);
```

### 4. File-Based Cache (JSONL)

```json
{
    "doc_id": "550e8400-e29b-41d4-a716",
    "cpesh": {
        "concept": "Photosynthesis",
        "probe": "What process converts light to chemical energy?",
        "expected": "Photosynthesis",
        "soft_negative": "Cellular respiration",
        "hard_negative": "Nuclear fission"
    },
    "created_at": "2025-09-25T10:30:00Z",
    "last_accessed": "2025-09-25T14:45:00Z",
    "access_count": 3,
    "quality_score": 0.92
}
```

## Bit-Packing for Compact Storage

For high-volume storage where space is critical, timestamps can be bit-packed:

### Epoch Seconds (32-bit)
```python
def pack_timestamp(iso_string: str) -> int:
    """Pack ISO8601 to 32-bit epoch seconds."""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return int(dt.timestamp())

def unpack_timestamp(epoch_int: int) -> str:
    """Unpack 32-bit epoch to ISO8601."""
    dt = datetime.fromtimestamp(epoch_int, tz=timezone.utc)
    return dt.isoformat()
```

### Relative Timestamps (16-bit)
For access patterns, store relative time since creation:
```python
def pack_relative_days(created: str, accessed: str) -> int:
    """Pack days-since-creation into 16 bits (max 179 years)."""
    dt_created = datetime.fromisoformat(created.replace('Z', '+00:00'))
    dt_accessed = datetime.fromisoformat(accessed.replace('Z', '+00:00'))
    days = (dt_accessed - dt_created).days
    return min(days, 65535)  # Cap at 16-bit max
```

## Manual Pruning Rules

### Quality-Based Pruning
```sql
-- Remove low-quality entries not accessed recently
DELETE FROM cpesh_cache
WHERE quality_score < 0.5
  AND last_accessed < (now() - INTERVAL '30 days')
  AND access_count < 5;
```

### Usage-Based Pruning
```sql
-- Archive entries with no recent access
INSERT INTO cpesh_archive
SELECT * FROM cpesh_cache
WHERE last_accessed < (now() - INTERVAL '90 days');

-- Then remove from main cache
DELETE FROM cpesh_cache
WHERE last_accessed < (now() - INTERVAL '90 days');
```

### Curation-Based Pruning
```python
# Manual curation via quality metrics
def prune_by_curation(entries: List[Dict], keep_threshold: float = 0.7):
    """Keep only high-quality curated entries."""
    return [e for e in entries if e.get('quality_score', 0) >= keep_threshold]
```

## Migration Strategy

### Handling Missing Timestamps
```python
def migrate_entry(entry: Dict) -> Dict:
    """Add timestamps to legacy entries."""
    now = get_iso_timestamp()

    if 'created_at' not in entry:
        # Use file modification time or current time
        entry['created_at'] = now

    if 'last_accessed' not in entry:
        # Initialize to creation time
        entry['last_accessed'] = entry['created_at']

    if 'access_count' not in entry:
        entry['access_count'] = 1

    return entry
```

## Utility Functions

Located in `src/utils/timestamps.py`:

```python
from datetime import datetime, timezone
from typing import Optional

def get_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()

def parse_iso_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))

def update_access_timestamp(record: Dict) -> Dict:
    """Update last_accessed and increment counter."""
    record['last_accessed'] = get_iso_timestamp()
    record['access_count'] = record.get('access_count', 0) + 1
    return record

def get_age_days(created_at: str) -> int:
    """Get age of record in days."""
    created = parse_iso_timestamp(created_at)
    now = datetime.now(timezone.utc)
    return (now - created).days

def should_prune(record: Dict,
                 min_quality: float = 0.5,
                 max_age_days: int = 180,
                 min_access_count: int = 2) -> bool:
    """Determine if record should be pruned."""
    quality = record.get('quality_score', 1.0)
    age = get_age_days(record['created_at'])
    access = record.get('access_count', 0)

    if quality < min_quality and age > max_age_days:
        return True
    if access < min_access_count and age > max_age_days * 2:
        return True
    return False
```

## Performance Considerations

1. **Index Strategy**: Always index `created_at` and `last_accessed` for query performance
2. **Batch Updates**: Update `last_accessed` in batches to reduce write load
3. **Async Updates**: Consider async/deferred timestamp updates for high-throughput scenarios
4. **Compression**: Use bit-packing for archival storage where precision isn't critical

## Monitoring and Metrics

Track these timestamp-related metrics:

- **Cache Age Distribution**: Histogram of record ages
- **Access Pattern**: Heat map of access times
- **Pruning Rate**: Records pruned per day/week
- **Storage Growth**: Rate of new timestamp records
- **Access Frequency**: Top-N most accessed records

## Compliance and Audit

1. **Immutability**: `created_at` must never be modified after initial write
2. **Audit Log**: All pruning operations must be logged with timestamp and reason
3. **Backup**: Archive pruned records before deletion for compliance
4. **Verification**: Regular integrity checks on timestamp consistency

## Example Implementation

```python
class TimestampedCache:
    """Cache with full timestamp support."""

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def get(self, doc_id: str) -> Optional[Dict]:
        """Get entry and update access timestamp."""
        if doc_id in self.cache:
            entry = self.cache[doc_id]
            entry = update_access_timestamp(entry)
            self._persist_entry(doc_id, entry)
            return entry
        return None

    def put(self, doc_id: str, data: Dict) -> None:
        """Add new entry with timestamps."""
        now = get_iso_timestamp()
        entry = {
            **data,
            'created_at': now,
            'last_accessed': now,
            'access_count': 1
        }
        self.cache[doc_id] = entry
        self._persist_entry(doc_id, entry)

    def prune(self, **kwargs) -> int:
        """Prune entries based on policy."""
        to_prune = [
            doc_id for doc_id, entry in self.cache.items()
            if should_prune(entry, **kwargs)
        ]
        for doc_id in to_prune:
            self._archive_entry(doc_id, self.cache[doc_id])
            del self.cache[doc_id]
        return len(to_prune)
```

## Summary

This timestamp policy ensures:
- ✅ Data persists indefinitely (no auto-expiration)
- ✅ Full audit trail via `created_at` and `last_accessed`
- ✅ Manual control over pruning based on quality/usage/curation
- ✅ Consistent format (ISO8601) across all layers
- ✅ Backward compatibility via automatic migration
- ✅ Performance optimization via bit-packing when needed