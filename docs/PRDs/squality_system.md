# Quality System PRD: Intelligent Quality Scoring (IQS)

## Overview

The Intelligent Quality Scoring (IQS) system enhances search relevance by computing document quality scores based on multiple real signals and blending them with cosine similarity scores to produce better-ranked search results.

## Architecture

### Components

1. **Quality Scoring Tool** (`tools/score_id_quality.py`)
   - Computes per-document quality scores from real signals
   - Outputs quality maps in JSONL and NPZ formats
   - Runs offline as a pre-processing step

2. **API Integration** (`src/api/retrieve.py`)
   - Loads quality scores at startup
   - Blends cosine similarity with quality scores
   - Re-ranks search results by final score

3. **Schema Updates** (`src/schemas.py`)
   - Adds `quality` and `final_score` fields to `SearchItem`
   - Enables transparent quality information in API responses

## Quality Scoring Algorithm

### Input Signals

The quality score is computed from four real, measurable signals:

#### 1. Text Health (40% weight, `--w_text`)
- **Length scoring**: Documents with 60-500 characters get optimal scores
- **Alphanumeric ratio**: Higher ratios indicate cleaner text
- **Control character penalty**: Penalizes documents with control characters
- Formula: `0.7*len_score + 0.3*alnum_ratio - 0.5*ctrl_pen`

#### 2. Graph Connectivity (30% weight, `--w_graph`)
- **Node degree**: Count of incoming/outgoing edges in knowledge graph
- **Soft saturation**: Uses formula `k/(k+5)` to prevent degree dominance
- **Fallback**: Returns 0.0 if graph edges file doesn't exist

#### 3. Duplicate Penalty (20% weight, `--w_dup`)
- **FAISS-based detection**: Uses vector similarity to find near-duplicates
- **Threshold**: Cosine similarity ≥ 0.999 considered duplicate
- **Penalty calculation**: `min(1.0, duplicate_count/3.0)`
- **Applied as**: `(1.0 - duplicate_penalty)` to reward uniqueness

#### 4. CPESH Margin (10% weight, `--w_cpesh`, optional)
- **Local LLM extraction**: Uses Llama to generate expected/negative answers
- **Contrastive scoring**: Measures separation between expected and negatives
- **Margin calculation**: `max(0.0, 1.0 - max(soft_sim, hard_sim))`
- **Requires**: `--use-cpesh` flag and local Llama server

### Final Quality Score

```
quality = w_text*text_health + w_graph*graph_degree + w_dup*(1-dup_penalty) + w_cpesh*cpesh_margin
```

All weights are configurable via command-line arguments (defaults: 0.4, 0.3, 0.2, 0.1).

## API Integration

### Startup Process
1. Load quality map from `artifacts/id_quality.jsonl`
2. Parse `LNSP_W_COS` (default 0.85) and `LNSP_W_QUALITY` (default 0.15) environment variables
3. Quality scores cached in memory for fast lookup

### Search Process
1. Perform vector similarity search as usual
2. For each result, lookup quality score by `doc_id`
3. Compute final score: `final_score = W_cos*cosine_sim + W_quality*quality`
4. Re-rank results by `final_score` (descending)
5. Return results with `quality` and `final_score` fields populated

### Environment Variables
- `LNSP_W_COS`: Weight for cosine similarity (default: 0.85)
- `LNSP_W_QUALITY`: Weight for quality score (default: 0.15)

## Usage

### Step 1: Generate Quality Scores
```bash
# Basic quality scoring (without CPESH)
python tools/score_id_quality.py \
  --npz artifacts/fw10k_vectors_768.npz \
  --index artifacts/fw10k_ivf_768.index \
  --edges artifacts/kg/edges.jsonl

# With CPESH margin computation (requires local Llama)
python tools/score_id_quality.py \
  --npz artifacts/fw10k_vectors_768.npz \
  --index artifacts/fw10k_ivf_768.index \
  --edges artifacts/kg/edges.jsonl \
  --use-cpesh
```

### Step 2: Start API Server
```bash
# Standard configuration
uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080

# Custom blend weights
LNSP_W_COS=0.7 LNSP_W_QUALITY=0.3 uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080
```

### Step 3: Test Search Results
```bash
curl -s -X POST http://localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"q":"Which ocean is largest?","top_k":5,"lane":"L1_FACTOID"}' | jq .
```

Expected response format:
```json
{
  "lane": "L1_FACTOID",
  "mode": "DENSE",
  "items": [
    {
      "id": "uuid-here",
      "doc_id": "doc123",
      "score": 0.85,
      "quality": 0.72,
      "final_score": 0.831,
      "concept_text": "The Pacific Ocean is the largest ocean...",
      "tmd_code": null,
      "lane_index": 0
    }
  ],
  "trace_id": "abc123"
}
```

## Files and Artifacts

### Input Files
- `artifacts/fw10k_vectors_768.npz`: Vector embeddings and metadata
- `artifacts/fw10k_ivf_768.index`: FAISS index for duplicate detection
- `artifacts/kg/edges.jsonl`: Knowledge graph edges (optional)

### Output Files
- `artifacts/id_quality.jsonl`: Per-document quality scores in JSONL format
- `artifacts/id_quality.npz`: Quality scores in NumPy format for fast loading

### JSONL Format
```json
{
  "doc_id": "doc123",
  "quality": 0.72,
  "features": {
    "text": 0.85,
    "graph": 0.60,
    "dup_penalty": 0.10,
    "cpesh_margin": 0.00
  }
}
```

## Configuration Options

### Quality Tool Arguments
- `--npz`: Path to vectors NPZ file (default: `artifacts/fw10k_vectors_768.npz`)
- `--index`: Path to FAISS index (default: `artifacts/fw10k_ivf_768.index`)
- `--edges`: Path to graph edges file (default: `artifacts/kg/edges.jsonl`)
- `--out-jsonl`: Output JSONL path (default: `artifacts/id_quality.jsonl`)
- `--out-npz`: Output NPZ path (default: `artifacts/id_quality.npz`)
- `--kdup`: Neighbors to check for duplicates (default: 5)
- `--w_text`: Text health weight (default: 0.4)
- `--w_graph`: Graph connectivity weight (default: 0.3)
- `--w_dup`: Duplicate penalty weight (default: 0.2)
- `--w_cpesh`: CPESH margin weight (default: 0.1)
- `--use-cpesh`: Enable CPESH margin computation

### API Environment Variables
- `LNSP_W_COS`: Cosine similarity weight in final scoring (default: 0.85)
- `LNSP_W_QUALITY`: Quality score weight in final scoring (default: 0.15)
- `LNSP_EMBED_MODEL_DIR`: Path to embedding model (for CPESH)
- `LNSP_LLM_ENDPOINT`: Local Llama endpoint (for CPESH)
- `LNSP_LLM_MODEL`: Llama model name (for CPESH)

## Validation and Testing

### Quality Score Validation
```bash
# Check that quality scores were generated
ls -la artifacts/id_quality.*

# Inspect a few quality scores
head -5 artifacts/id_quality.jsonl

# Verify no NaN values and scores in [0,1] range
python -c "
import json
scores = []
with open('artifacts/id_quality.jsonl') as f:
    for line in f:
        j = json.loads(line)
        scores.append(j['quality'])
print(f'Count: {len(scores)}')
print(f'Range: [{min(scores):.3f}, {max(scores):.3f}]')
print(f'Mean: {sum(scores)/len(scores):.3f}')
"
```

### API Response Validation
```bash
# Test different blend weights
LNSP_W_COS=1.0 LNSP_W_QUALITY=0.0 curl -s -X POST http://localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"q":"machine learning","top_k":3}' | jq '.items[].final_score'

LNSP_W_COS=0.5 LNSP_W_QUALITY=0.5 curl -s -X POST http://localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"q":"machine learning","top_k":3}' | jq '.items[].final_score'
```

### Health Checks
- Quality scores file exists and contains valid JSON
- All quality values are in [0, 1] range
- No NaN or infinite values in quality scores
- API responses include both `quality` and `final_score` fields
- Changing blend weights produces different result orderings

## Performance Considerations

### Memory Usage
- Quality map loaded once at startup and cached in memory
- ~8 bytes per document (doc_id string + float quality)
- For 10K documents: ~80KB additional memory

### Computation Time
- Quality scoring: O(n) where n = number of documents
- FAISS duplicate detection: O(n*k) where k = neighbors to check
- CPESH computation: O(n*LLM_latency) - can be slow
- API re-ranking: O(n*log(n)) sorting overhead

### Disk Usage
- JSONL output: ~100 bytes per document
- NPZ output: ~16 bytes per document (4 floats)
- For 10K documents: ~1MB total

## Integration Points

### Existing Systems
- **FAISS Index**: Used for duplicate detection via vector similarity
- **Knowledge Graph**: Optional graph connectivity scoring
- **Local Llama**: Optional CPESH margin computation
- **Embedding Models**: Reuses existing GTR-T5 embedder

### Future Extensions
- **Learning to Rank**: Quality features could feed into learned ranking models
- **Dynamic Updates**: Quality scores could be updated incrementally
- **Multi-Modal**: Additional quality signals from images, tables, etc.
- **A/B Testing**: Framework for testing different quality formulations

## Troubleshooting

### Common Issues

1. **Missing quality file**: API defaults quality to 0.5 if file doesn't exist
2. **FAISS dimension mismatch**: Ensure index dimension matches vector NPZ
3. **CPESH timeouts**: Increase timeout or disable `--use-cpesh` flag
4. **Memory issues**: Reduce batch size or disable expensive features

### Debug Commands
```bash
# Check quality file format
python -c "
import json
with open('artifacts/id_quality.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Invalid JSON at line {i+1}: {line}')
        if i > 10: break
"

# Verify API integration
curl -s http://localhost:8080/healthz | jq .

# Test quality impact
curl -s -X POST http://localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"q":"test query","top_k":1}' | jq '.items[0] | {score, quality, final_score}'
```

## Success Criteria

### Acceptance Gates
1. ✅ `artifacts/id_quality.jsonl` exists with valid JSON
2. ✅ All quality values in [0, 1] range with no NaN/infinite values
3. ✅ API `/search` endpoint returns `quality` and `final_score` fields
4. ✅ Changing `LNSP_W_COS`/`LNSP_W_QUALITY` affects result ordering
5. ✅ No synthetic/placeholder data - all signals derived from real vectors/text/graph

### Quality Metrics
- **Relevance**: Higher quality documents should rank higher for ambiguous queries
- **Diversity**: Duplicate penalty should reduce near-duplicate results
- **Stability**: Quality scores should be consistent across runs
- **Performance**: <10ms additional latency for quality re-ranking