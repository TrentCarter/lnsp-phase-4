# Runtime Environment Configuration
_Version: 1.0 (S3 Sprint - 2025-09-25)_

## Stack Overview

### Core Dependencies
- **Python**: 3.11+ (required for type hints and modern async)
- **Virtual Environment**: `./venv` (project-isolated)
- **Package Manager**: pip with requirements.txt

### Primary Services

#### 1. FastAPI Application
- **Framework**: FastAPI with Uvicorn ASGI server
- **Default Port**: 8092 (configurable)
- **Host**: 127.0.0.1 (localhost only for security)
- **Workers**: 1 (development), 4 (production)
- **Reload**: Enabled in development, disabled in production

#### 2. Vector Store (Faiss)
- **Library**: faiss-cpu or faiss-gpu
- **Index Location**: `artifacts/faiss/`
- **Memory Mode**: MMAP for large indices
- **Thread Pool**: 4 threads for parallel search
- **Dynamic nlist**: Auto-downshift when train_points < 40×nlist (see `docs/architecture.md` for policy)

#### 3. Cache Layer (CPESH)
- **Format**: JSONL with streaming support
- **Location**: `artifacts/cpesh_cache.jsonl`
- **Backup**: `artifacts/cpesh_cache_backup_{timestamp}.jsonl`
- **Access Pattern**: LRU with timestamp tracking

#### 4. Database Connections
- **PostgreSQL**:
  - Connection pool: 5-20 connections
  - Timeout: 30s
  - Extensions: pgvector for embeddings
- **Neo4j**:
  - Bolt protocol on port 7687
  - Connection pool: 5-10 connections

## Environment Variables

### Required
```bash
# API Configuration
LNSP_API_PORT=8092
LNSP_API_HOST=127.0.0.1
LNSP_API_WORKERS=1

# Faiss Configuration
LNSP_FAISS_INDEX_PATH=artifacts/faiss/index.ivf
LNSP_FAISS_NPROBE=16
LNSP_FAISS_METRIC=inner_product

# Cache Configuration
LNSP_CACHE_PATH=artifacts/cpesh_cache.jsonl
LNSP_CACHE_MAX_SIZE=50000
LNSP_CACHE_TTL_DAYS=14

# Database URLs (if using)
LNSP_PG_URL=postgresql://user:pass@localhost:5432/lnsp
LNSP_NEO4J_URL=bolt://localhost:7687
```

### Optional Performance Tuning
```bash
# Scoring Weights (S3 optimized)
LNSP_W_COS=0.85      # Cosine similarity weight
LNSP_W_QUALITY=0.15  # Quality score weight
LNSP_W_MARGIN=0.0    # Margin score weight

# Feature Flags
LNSP_USE_QUALITY=1       # Enable quality scoring
LNSP_USE_CPESH_MARGIN=0  # Disable margin scoring (S3)

# Test Mode
LNSP_TEST_MODE=1  # Use test data/reduced index
```

## Startup Sequence

### Development Mode
```bash
# 1. Activate virtual environment
source ./venv/bin/activate

# 2. Verify dependencies
python -m pip list | grep -E "fastapi|uvicorn|faiss"

# 3. Start API with hot reload
LNSP_W_COS=0.85 LNSP_W_QUALITY=0.15 LNSP_W_MARGIN=0.0 \
LNSP_USE_QUALITY=1 LNSP_USE_CPESH_MARGIN=0 \
./venv/bin/uvicorn src.api.retrieve:app \
  --host 127.0.0.1 \
  --port 8092 \
  --reload
```

### Production Mode
```bash
# 1. Load production config
source .env.production

# 2. Start with multiple workers
./venv/bin/uvicorn src.api.retrieve:app \
  --host 0.0.0.0 \
  --port ${LNSP_API_PORT} \
  --workers 4 \
  --log-level warning
```

## Health Checks

### Endpoints
- `GET /health` - Basic liveness check
- `GET /health/faiss` - Faiss index status
- `GET /cache/stats` - Cache statistics
- `GET /metrics/slo` - SLO compliance metrics

### Monitoring Script
```bash
#!/bin/bash
# health_check.sh
curl -s http://127.0.0.1:8092/health | jq .
curl -s http://127.0.0.1:8092/health/faiss | jq .
curl -s http://127.0.0.1:8092/cache/stats | jq .
```

## Resource Requirements

### Minimum (Development)
- RAM: 4GB
- CPU: 2 cores
- Disk: 10GB free
- Network: Localhost only

### Recommended (10k vectors)
- RAM: 8GB
- CPU: 4 cores
- Disk: 20GB free
- Network: Gigabit for distributed setup

### Production (100k+ vectors)
- RAM: 32GB+
- CPU: 8+ cores
- Disk: 100GB+ SSD
- Network: 10Gbps for cluster

## Troubleshooting

### Common Issues

1. **Faiss Segfault**
   - Check index compatibility with faiss version
   - Verify training data size (≥40×nlist)
   - Reduce nprobe if memory limited

2. **Slow Query Performance**
   - Warm up cache with common queries
   - Tune nprobe (lower = faster, less accurate)
   - Enable index preloading

3. **Cache Misses**
   - Check cache file permissions
   - Verify JSONL format integrity
   - Monitor cache size vs TTL

### Debug Mode
```bash
# Enable verbose logging
export LNSP_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

# Run with debugger
python -m debugpy --listen 5678 --wait-for-client \
  -m uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8092
```

## Performance Baselines

### 10k Vector Index
- Index build time: < 5 minutes
- Query latency P50: < 80ms
- Query latency P95: < 450ms
- Memory usage: < 2GB
- Cache hit rate: > 30%

### SLO Compliance
Monitor these metrics via `/metrics/slo`:
- Hit@1: ≥ 45%
- Hit@3: ≥ 55%
- Availability: ≥ 99.5%
- Error rate: < 1%