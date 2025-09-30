# Quick Start: P13 Echo Validation & P15 LVM Training

**TL;DR**: Run P13 validation, generate GWOM data, train Latent-Only Mamba, deploy inference API.

---

## Step 1: Execute P13 Echo Validation (Today)

### Test Run
```bash
# Validate 100 entries without DB update
./.venv/bin/python -m src.pipeline.p13_echo_validation \
  --limit 100 \
  --no-update-db
```

### Full Production Run
```bash
# Validate all 4,993 entries and update database
./.venv/bin/python -m src.pipeline.p13_echo_validation \
  --update-db \
  --report-out artifacts/p13_echo_report.json
```

### Expected Output
```
===============================================================================
VALIDATION SUMMARY
===============================================================================
Total Entries:       4,993
Validated:           4,993
Passed (≥0.82):     4,494 (90.0%)
Failed (<0.82):      499 (10.0%)

Mean Echo Score:     0.8524
Median Echo Score:   0.8631
P95 Echo Score:      0.9421

✅ QUALITY GATE: PASSED (≥90% pass rate)
===============================================================================
```

### Interpreting Results

| Pass Rate | Action |
|-----------|--------|
| ≥90% | ✅ Proceed to P15 training |
| 80-89% | ⚠️  Review failed lanes, consider selective re-interrogation |
| <80% | ❌ Re-run P5 with improved prompts |

---

## Step 2: Set Up GWOM Infrastructure (Week 1)

```bash
# Create directory structure
mkdir -p artifacts/gwom/{gwom_segments,checkpoints}

# Initialize data lake
cat > artifacts/gwom/gwom_manifest.jsonl <<EOF
{"version": "0.1", "created": "$(date -Iseconds)", "segments": [], "total_chains": 0}
EOF

# Initialize index
sqlite3 artifacts/gwom/gwom_index.db <<EOF
CREATE TABLE sequences (
    seq_id TEXT PRIMARY KEY,
    segment TEXT,
    method TEXT,
    length INT,
    quality_score REAL,
    coherence_mean REAL,
    created_at TEXT
);
CREATE INDEX idx_method ON sequences(method);
CREATE INDEX idx_quality ON sequences(quality_score);
EOF

echo "✅ GWOM infrastructure ready"
```

---

## Step 3: Generate GWOM Training Data (Week 2-3)

### Quick MVP (10K chains for testing)
```bash
# Generate 10K chains for MVP testing
./.venv/bin/python -m src.gwom.generate_all \
  --graphrag-chains 4200 \
  --wiki-chains 3800 \
  --ontology-chains 2000 \
  --min-length 5 \
  --max-length 15 \
  --coherence-threshold 0.70 \
  --out artifacts/gwom/gwom_active.jsonl

# Verify
wc -l artifacts/gwom/gwom_active.jsonl
# Expected: 10000 lines
```

### Full Production (250K chains)
```bash
# Generate 250K chains for full training
./.venv/bin/python -m src.gwom.generate_all \
  --graphrag-chains 100000 \
  --wiki-chains 100000 \
  --ontology-chains 50000 \
  --min-length 5 \
  --max-length 15 \
  --coherence-threshold 0.70 \
  --out artifacts/gwom/gwom_active.jsonl

# Pre-compute vectors
./.venv/bin/python -m src.gwom.vectorize_chains \
  --input artifacts/gwom/gwom_active.jsonl \
  --embedder gtr-t5-base \
  --out artifacts/gwom/gwom_vectors.npz

# Verify
./.venv/bin/python -c "
import numpy as np
data = np.load('artifacts/gwom/gwom_vectors.npz')
print(f'Chains: {data[\"vectors\"].shape[0]}')
print(f'Max length: {data[\"vectors\"].shape[1]}')
print(f'Dimension: {data[\"vectors\"].shape[2]}')
print(f'✅ Expected: (250000, 15, 768)')
"
```

---

## Step 4: Train Latent-Only Mamba (Week 4-5)

### Quick Toy Run (Sanity Check)
```bash
# Train on 1K chains for 3 epochs (should complete in ~30 mins)
./.venv/bin/python -m src.lvm.train_latent_mamba \
  --data artifacts/gwom/gwom_vectors.npz \
  --limit 1000 \
  --epochs 3 \
  --batch-size 32 \
  --lr 1e-4 \
  --model-out artifacts/lvm/toy_model.pt \
  --device cuda

# Check toy model works
./.venv/bin/python -m src.lvm.test_toy_model \
  --model artifacts/lvm/toy_model.pt
```

### Full Training Run
```bash
# Train on full 250K chains, 10 epochs
./.venv/bin/python -m src.lvm.train_latent_mamba \
  --data artifacts/gwom/gwom_vectors.npz \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --scheduler cosine \
  --curriculum-stages 3 \
  --model-out artifacts/lvm/latent_mamba_v1.pt \
  --checkpoint-dir artifacts/lvm/checkpoints \
  --device cuda \
  --wandb-project lnsp-p15

# Expected: ~8-12 hours on single GPU
```

### Expected Training Metrics

| Epoch | Train Loss | Val Loss | Next-Vec Cosine | Status |
|-------|-----------|----------|-----------------|--------|
| 1 | 0.45 | 0.48 | 0.65 | Baseline |
| 3 | 0.28 | 0.32 | 0.74 | Learning |
| 5 | 0.18 | 0.22 | 0.81 | ✅ Target reached |
| 10 | 0.12 | 0.18 | 0.84 | Converged |

**Success Gate**: Val next-vector cosine ≥0.80 by epoch 10

---

## Step 5: Deploy Inference API (Week 6)

### Start Inference Server
```bash
# Start uvicorn server
./.venv/bin/uvicorn src.lvm.inference_api:app \
  --host 127.0.0.1 \
  --port 8095 \
  --reload
```

### Test Inference
```bash
# Test query
curl -X POST http://localhost:8095/infer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does photosynthesis produce oxygen?"
  }'

# Expected response (< 2 seconds)
{
  "query": "How does photosynthesis produce oxygen?",
  "response": "During photosynthesis, light-dependent reactions split water molecules through photolysis, releasing oxygen gas as a byproduct.",
  "intermediate": {
    "query_vec_dim": 768,
    "lvm_output_vec_dim": 768,
    "retrieval_method": "faiss",
    "retrieval_cosine": 0.93,
    "vec2text_used": false
  },
  "latency_ms": 1847
}
```

### Stress Test
```bash
# Load test with 100 concurrent requests
./.venv/bin/python -m src.lvm.load_test \
  --endpoint http://localhost:8095/infer \
  --queries eval/test_queries.jsonl \
  --concurrency 100 \
  --duration 60s

# Target: P95 latency < 2s, throughput ≥50 req/s
```

---

## Step 6: Evaluation (Week 7-8)

### Automated Evaluation
```bash
# Run eval suite
./.venv/bin/python -m src.eval.eval_lvm \
  --api http://localhost:8095/infer \
  --queries eval/eval_queries.jsonl \
  --ground-truth eval/eval_ground_truth.jsonl \
  --metrics-out eval/lvm_metrics.json

# Expected metrics
{
  "retrieval_precision@1": 0.74,
  "retrieval_precision@5": 0.91,
  "echo_loop_cosine": 0.86,
  "vec2text_fallback_rate": 0.15,
  "mean_latency_ms": 1623,
  "p95_latency_ms": 1987
}
```

### Human Evaluation
```bash
# Generate evaluation UI
./.venv/bin/python -m src.eval.human_eval_ui \
  --port 8096

# Open http://localhost:8096 in browser
# Rate 100 random responses: Good / Acceptable / Poor
# Target: ≥80% "Good" or "Acceptable"
```

---

## Common Commands Reference

### P13 Echo Validation
```bash
# Full run
make p13-validate

# Test run
make p13-test

# View report
cat artifacts/p13_echo_report.json | jq '.pass_rate'
```

### GWOM Generation
```bash
# Generate GWOM data
make gwom-generate CHAINS=250000

# Check status
make gwom-status

# Rotate to Parquet
make gwom-rotate
```

### LVM Training
```bash
# Full training
make lvm-train

# Resume from checkpoint
make lvm-resume CHECKPOINT=epoch_5

# Monitor training
tensorboard --logdir artifacts/lvm/logs
```

### Inference
```bash
# Start server
make lvm-serve

# Test inference
make lvm-test-infer

# Stop server
make lvm-stop
```

---

## Monitoring Dashboards

### Training Metrics (W&B or TensorBoard)
- Loss curves (train/val)
- Next-vector cosine similarity
- Learning rate schedule
- Gradient norms

### Inference Metrics (Grafana or custom)
- Request latency (mean/p50/p95/p99)
- Throughput (req/s)
- Retrieval method distribution (Faiss vs Vec2Text)
- Error rate

### Data Quality Metrics
- GWOM coherence distribution
- P13 echo score distribution
- Failed entries by lane
- CPESH coverage by domain

---

## Troubleshooting

### P13 Validation Fails (<80% pass rate)
1. Check embedding model is GTR-T5 (not other models)
2. Verify probe questions are not empty
3. Review failed lanes - are they consistently low quality?
4. Consider re-running P5 with better prompts

### GWOM Generation Slow
1. Check Neo4j query performance (graph walks)
2. Enable caching for Wikipedia fetches
3. Parallelize across multiple workers
4. Consider using smaller chain lengths (5-10 instead of 5-15)

### LVM Training Not Converging
1. Reduce learning rate (1e-4 → 1e-5)
2. Increase batch size (64 → 128)
3. Check data quality (GWOM coherence ≥0.70)
4. Try simpler architecture (fewer layers)
5. Validate toy task works first

### Inference Latency >2s
1. Cache Faiss index in memory (mmap)
2. Reduce Vec2Text steps (20 → 5)
3. Use batching for multiple queries
4. Consider quantization (FP16 or INT8)
5. Deploy on GPU instead of CPU

### Vec2Text Quality Poor
1. Ensure using both JXE and IELab (ensemble)
2. Increase decoding steps (5 → 10)
3. Check embedding model matches (GTR-T5)
4. Validate vectors are L2-normalized

---

## Success Checklist

### P13 Validation ✓
- [ ] Test run (100 entries) passes
- [ ] Full run (4,993 entries) completes
- [ ] Pass rate ≥90%
- [ ] Report saved to artifacts/
- [ ] Database updated with echo_score

### GWOM Generation ✓
- [ ] Infrastructure set up (directories, index DB)
- [ ] GraphRAG walks implemented
- [ ] WikiSearch anchoring implemented
- [ ] Ontology traversal implemented
- [ ] 250K chains generated
- [ ] Mean coherence ≥0.75
- [ ] Vectors pre-computed (gwom_vectors.npz)

### LVM Training ✓
- [ ] Toy run (1K chains, 3 epochs) succeeds
- [ ] Full training (250K chains, 10 epochs) completes
- [ ] Val next-vector cosine ≥0.80
- [ ] Model saved (<100MB)
- [ ] Checkpoints saved every epoch

### Inference Deployment ✓
- [ ] Server starts successfully
- [ ] Test query returns response
- [ ] Latency <2s (P95)
- [ ] Faiss retrieval precision ≥70%
- [ ] Vec2Text fallback ≤20%
- [ ] Load test passes (100 concurrent)

### Evaluation ✓
- [ ] Automated eval runs
- [ ] Human eval ≥80% "Good/Acceptable"
- [ ] Metrics logged to dashboard
- [ ] A/B test vs baseline
- [ ] Production deployment approved

---

## Support & Resources

### Documentation
- **Full Plan**: `docs/PRDs/PRD_P15_Latent_LVM_Implementation_Plan.md`
- **Pipeline Status**: `docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md`
- **GWOM Design**: `docs/PRDs/PRD_GWOM_design_Options.md`
- **Vec2Text Usage**: `docs/how_to_use_jxe_and_ielab.md`

### Key Scripts
- **P13 Validation**: `src/pipeline/p13_echo_validation.py`
- **GWOM Generation**: `src/gwom/generate_all.py`
- **LVM Training**: `src/lvm/train_latent_mamba.py`
- **Inference**: `src/lvm/inference.py`

### External Resources
- **Mamba Paper**: https://arxiv.org/abs/2312.00752
- **Vec2Text Paper**: https://arxiv.org/abs/2310.06816
- **GTR-T5 Model**: https://huggingface.co/sentence-transformers/gtr-t5-base

---

**Last Updated**: 2025-09-30
**Version**: 1.0
**Status**: Ready to Execute
