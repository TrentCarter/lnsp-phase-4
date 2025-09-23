# LNSP API Response Schema

This document defines the canonical response format for the LNSP retrieval API.

## POST /search

### Request Schema

```json
{
  "q": "What is artificial intelligence?",      // Required: query string (1-512 chars)
  "lane": "L1_FACTOID",                        // Required: L1_FACTOID|L2_GRAPH|L3_SYNTH
  "top_k": 8                                   // Optional: 1-100, default 8
}
```

**Request Fields:**
- `q` (string, required): Natural language query, 1-512 characters
- `lane` (string, required): Retrieval lane - one of:
  - `L1_FACTOID`: Dense-only retrieval (default mode)
  - `L2_GRAPH`: Dense + graph expansion
  - `L3_SYNTH`: Full hybrid path (dense + graph + reranking)
- `top_k` (integer, optional): Number of results to return, 1-100, default 8

### Response Schema (P4 Format)

```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID for request tracking
  "lane": "L1_FACTOID",                                // Echo request lane
  "k": 8,                                              // Number of results requested
  "scores": [0.87, 0.82, 0.79, 0.76, 0.73, 0.71, 0.68, 0.65],  // Similarity scores
  "support_ids": [                                    // Array of CPE IDs (stable UUIDs)
    "a1b2c3d4e5f6g7h8",
    "b2c3d4e5f6g7h8a9",
    "c3d4e5f6g7h8a9b0",
    "d4e5f6g7h8a9b0c1",
    "e5f6g7h8a9b0c1d2",
    "f6g7h8a9b0c1d2e3",
    "g7h8a9b0c1d2e3f4",
    "h8a9b0c1d2e3f4g5"
  ]
}
```

**Alternative Format (Legacy Compatibility):**

```json
{
  "trace_id": "req_20250923_143022_abc123",     // Optional: request tracing ID
  "lane": "L1_FACTOID",                        // Echo request lane
  "mode": "DENSE",                             // DENSE|GRAPH|HYBRID (actual mode used)
  "k": 8,                                      // Number of results requested
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",  // Canonical: cpe_id (stable UUID)
      "doc_id": "enwiki:12345",                      // Optional: upstream document ID
      "score": 0.87,                                 // Optional: similarity/ranking score
      "support_ids": ["doc456", "doc789"],           // Optional: supporting document IDs
      "why": "Dense embedding match"                 // Optional: retrieval explanation
    }
  ]
}
```

**Response Fields:**
- `trace_id` (string, optional): Unique identifier for request tracing and debugging
- `lane` (string): Echo of the requested lane
- `mode` (string): Actual retrieval mode used (DENSE|GRAPH|HYBRID)
- `k` (integer): Number of results in the response
- `items` (array): Array of search results, each containing:
  - `id` (string, required): Canonical CPE UUID for stable evaluation
  - `doc_id` (string, optional): Upstream document identifier
  - `score` (float, optional): Similarity or ranking score (0.0-1.0)
  - `support_ids` (array, optional): Array of supporting document IDs
  - `why` (string, optional): Human-readable explanation of retrieval method

### Contract Guarantees

1. **Stable IDs**: The `id` field always equals `cpe_id` (UUID) for consistent evaluation
2. **Non-empty Response**: Response always includes `items` array (empty array if no results)
3. **Lane Routing**:
   - `L1_FACTOID` → dense retrieval
   - `L2_GRAPH` → graph expansion
   - `L3_SYNTH` → hybrid retrieval
4. **Lexical Fallback**: Configurable via `LNSP_LEXICAL_FALLBACK` environment variable

## CPE ID Specification

**Stable CPE ID Generation Rule:**
```
cpe_id = sha1("{dataset}:{doc_id}:{chunk_start}:{chunk_len}:{version}")[:16]
```

**Parameters:**
- `dataset`: Source dataset identifier (e.g., "factoidwiki")
- `doc_id`: Upstream document identifier
- `chunk_start`: Character offset where chunk begins
- `chunk_len`: Length of chunk in characters
- `version`: Schema version for backward compatibility

**Example:**
```python
import hashlib

def generate_cpe_id(dataset: str, doc_id: str, chunk_start: int, chunk_len: int, version: str = "v1") -> str:
    """Generate stable CPE ID using SHA1 hash."""
    content = f"{dataset}:{doc_id}:{chunk_start}:{chunk_len}:{version}"
    return hashlib.sha1(content.encode()).hexdigest()[:16]

# Example usage
cpe_id = generate_cpe_id("factoidwiki", "enwiki:12345", 0, 256, "v1")
# Returns: "a1b2c3d4e5f6g7h8"
```

**Enforcement:**
- All retrieval responses MUST use this stable ID format
- IDs are deterministic and reproducible across ingestion runs
- This ensures consistent evaluation metrics and traceability

### Environment Configuration

- `LNSP_LEXICAL_FALLBACK=0` (default): Dense-only L1, hybrid L2/L3
- `LNSP_LEXICAL_FALLBACK=1`: Enable lexical fallback override for L1

### Example Usage

```bash
# Dense-only retrieval
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"q": "What is machine learning?", "lane": "L1_FACTOID", "top_k": 5}'

# Graph-enhanced retrieval
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"q": "How do neural networks work?", "lane": "L2_GRAPH", "top_k": 10}'
```

### Error Responses

```json
{
  "error": "Invalid lane parameter",
  "code": "INVALID_LANE",
  "details": "Lane must be one of: L1_FACTOID, L2_GRAPH, L3_SYNTH"
}
```

Common error codes:
- `INVALID_LANE`: Invalid lane parameter
- `QUERY_TOO_LONG`: Query exceeds 512 characters
- `INVALID_TOP_K`: top_k outside valid range (1-100)
- `SERVICE_UNAVAILABLE`: Backend service unavailable