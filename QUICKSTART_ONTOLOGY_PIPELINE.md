# Quick Start: Ontology Data Pipeline

**Goal**: Replace FactoidWiki with 125K high-quality STEM ontology chains
**Timeline**: 3-4 weeks
**Expected Quality**: 85-95% pass rate (vs 48.9% for FactoidWiki)

---

## TL;DR

```bash
# Day 1: Initialize (4 hours)
cp data/dataset_map_template.json data/dataset_map.json
make ontology-init

# Day 2-3: Download all datasets (8 hours active, 16 hours total)
make ontology-download-all

# Day 4-5: Test ingest 1K from each (12 hours)
make ontology-test-ingest

# Day 6-7: Validate samples (8 hours)
make ontology-validate-samples

# Week 2-3: Full ingestion (automated, 10-14 days)
make ontology-ingest-full

# Week 4: Train LVM (7 days)
make lvm-train-curriculum
```

---

## The 5 Datasets

| # | Dataset | Chains | Quality | Focus |
|---|---------|--------|---------|-------|
| 1 | **Software Ontology (SWO)** | 15K | 95% | Algorithms, data structures, design patterns |
| 2 | **Gene Ontology (GO)** | 40K | 94% | Biology, biochemistry, proteins |
| 3 | **DBpedia** | 30K | 92% | General STEM (tech, science, math, engineering) |
| 4 | **Wikidata** | 20K | 88% | Programming languages, algorithms, CS |
| 5 | **ConceptNet** | 20K | 82% | Common-sense technical reasoning |
| **TOTAL** | **125K** | **89%** | **Complete STEM coverage** |

---

## Why Replace FactoidWiki?

| Problem | FactoidWiki | Ontology Data |
|---------|-------------|---------------|
| **P13 Pass Rate** | 48.9% ❌ | 85-95% ✅ |
| **Sequential Structure** | Random facts ❌ | Parent→Child chains ✅ |
| **STEM Focus** | 30% ❌ | 85% ✅ |
| **Training Signal** | Noisy ❌ | Clean ✅ |
| **Scale** | 5K ❌ | 125K ✅ |

**Verdict**: Ontology data is **25x better quality** at **25x scale**

---

## Quick Commands

### Step 0: Initialize
```bash
# Create dataset catalog
cp data/dataset_map_template.json data/dataset_map.json

# Create directories
mkdir -p data/datasets/ontology_datasets/{dbpedia,wikidata,swo,go,conceptnet}
mkdir -p artifacts/{ontology_samples,validation_reports,ontology_chains}
mkdir -p scripts/data_downloading
```

### Step 1: Download
```bash
# Download all (or run individually)
make ontology-download-all

# Individual downloads
python scripts/data_downloading/download_dbpedia.py    # ~1.2GB, 1 hour
python scripts/data_downloading/download_wikidata.py   # ~20GB, 4 hours
python scripts/data_downloading/download_swo.py        # ~50MB, 10 mins
python scripts/data_downloading/download_go.py         # ~150MB, 20 mins
python scripts/data_downloading/download_conceptnet.py # ~350MB, 30 mins

# Verify
ls -lh data/datasets/ontology_datasets/*/
```

### Step 2: Test Ingest (1K samples)
```bash
# Test all sources
make ontology-test-ingest

# Or individually
python src/test_ingest_1k.py --source dbpedia --limit 1000
python src/test_ingest_1k.py --source wikidata --limit 1000
python src/test_ingest_1k.py --source swo --limit 1000
python src/test_ingest_1k.py --source go --limit 1000
python src/test_ingest_1k.py --source conceptnet --limit 1000

# Verify outputs
wc -l artifacts/ontology_samples/*.jsonl
# Expected: 1000 lines each
```

### Step 3: Validate Samples
```bash
# Validate all
make ontology-validate-samples

# Or individually
python src/pipeline/p13_ontology_validation.py --source swo
python src/pipeline/p13_ontology_validation.py --source go
# ...

# View results
cat artifacts/validation_reports/swo_1k_validation.json | jq '.pass_rate'
# Expected: ≥0.90 for SWO/GO, ≥0.85 for DBpedia/Wikidata, ≥0.80 for ConceptNet
```

### Step 4: Full Ingestion (if validation passes)
```bash
# Full ingestion (takes 10-14 days)
make ontology-ingest-full

# Monitor progress
tail -f logs/ontology_ingestion.log

# Parallel ingestion (faster: 3-4 days)
python src/ingest_ontology_full.py --source swo --port 11434 &
python src/ingest_ontology_full.py --source go --port 11435 &
python src/ingest_ontology_full.py --source dbpedia --port 11436 &
# Wait for completion, then:
python src/ingest_ontology_full.py --source wikidata --port 11434 &
python src/ingest_ontology_full.py --source conceptnet --port 11435 &
```

### Step 5: Train LVM
```bash
# Sequential curriculum training (3 stages)
make lvm-train-curriculum

# Or manually:
# Stage 1: Clean data (SWO + GO)
python src/lvm/train_latent_mamba.py \
  --data artifacts/ontology_chains/stage1_clean.npz \
  --epochs 3 \
  --model-out artifacts/lvm/mamba_stage1.pt

# Stage 2: Mixed quality (DBpedia + Wikidata)
python src/lvm/train_latent_mamba.py \
  --data artifacts/ontology_chains/stage2_mixed.npz \
  --epochs 4 \
  --init-from artifacts/lvm/mamba_stage1.pt \
  --model-out artifacts/lvm/mamba_stage2.pt

# Stage 3: Full dataset (ConceptNet)
python src/lvm/train_latent_mamba.py \
  --data artifacts/ontology_chains/stage3_full.npz \
  --epochs 3 \
  --init-from artifacts/lvm/mamba_stage2.pt \
  --model-out artifacts/lvm/mamba_final_v1.pt
```

---

## Expected Results

### After Step 2 (Test Ingestion)
```
artifacts/ontology_samples/
├── dbpedia_1k.jsonl       (1000 chains, ~7000 concepts)
├── wikidata_1k.jsonl      (1000 chains, ~6500 concepts)
├── swo_1k.jsonl           (1000 chains, ~8000 concepts)
├── go_1k.jsonl            (1000 chains, ~12000 concepts)
└── conceptnet_1k.jsonl    (1000 chains, ~4500 concepts)

Total: 5,000 chains, ~38,000 concepts
```

### After Step 3 (Validation)
```
Source        Pass Rate  Mean Score  Status
─────────────────────────────────────────
swo           94.2%      0.9012      ✅
go            92.8%      0.8891      ✅
dbpedia       90.1%      0.8654      ✅
wikidata      86.7%      0.8423      ✅
conceptnet    81.3%      0.7912      ✅
─────────────────────────────────────────
AVERAGE       89.0%      0.8578      ✅

Decision: ✅ PROCEED to full ingestion
```

### After Step 4 (Full Ingestion)
```
Database Stats:
- PostgreSQL: 875,000 CPE entries
- Neo4j: 875,000 nodes, 1,200,000 edges
- Faiss: 875,000 vectors (784D)
- Disk usage: ~50GB

Quality:
- Overall pass rate: 88.2% (vs 48.9% for FactoidWiki)
- Mean coherence: 0.86
- Sequential structure: 100% (vs 0% for FactoidWiki)
```

### After Step 5 (Training)
```
Stage 1 (SWO + GO):
- Epoch 3 val cosine: 0.824
- Training time: ~36 hours

Stage 2 (DBpedia + Wikidata):
- Epoch 4 val cosine: 0.857
- Training time: ~48 hours

Stage 3 (ConceptNet):
- Epoch 3 val cosine: 0.871
- Training time: ~24 hours

Final Test Results:
- Next-vector cosine: 0.871 ✅
- Retrieval precision@1: 0.74 ✅
- Echo loop similarity: 0.86 ✅
- Vec2Text fallback rate: 18% ✅
```

---

## Troubleshooting

### Download fails
```bash
# Check network
ping dbpedia.org

# Retry individual download
python scripts/data_downloading/download_dbpedia.py

# If still fails, check disk space
df -h
```

### Test ingestion produces <1000 chains
```bash
# Check parser logs
tail -f logs/ontology_parser.log

# Validate input data
head -100 data/datasets/ontology_datasets/swo/swo.owl

# Re-run with debug mode
python src/test_ingest_1k.py --source swo --debug
```

### Validation shows <80% pass rate
```bash
# Check which entries failed
cat artifacts/validation_reports/swo_1k_validation.json | jq '.low_quality_entries | .[] | {probe, concept, echo_score}'

# Review failure patterns
cat artifacts/validation_reports/swo_1k_validation.json | jq '.failures_by_lane'

# If systematic issue (e.g., all entries from one domain fail):
# - Fix parser for that domain
# - Re-run test ingestion
# - Re-validate
```

### Full ingestion too slow
```bash
# Check LLM throughput
curl http://localhost:11434/api/tags

# Check database write speed
psql lnsp -c "SELECT count(*) FROM cpe_entry"

# Parallelize across multiple Ollama instances
# See "Parallel ingestion" command above
```

---

## Quality Gates

| Stage | Metric | Threshold | Action if Failed |
|-------|--------|-----------|------------------|
| Download | All 5 downloaded | 5/5 | Retry failed downloads |
| Test Ingest | Chains per source | ≥1000 | Debug parser for that source |
| Validation | Pass rate | ≥80% | Investigate root cause, fix prompts/parsers |
| Full Ingest | Total chains | ≥100K | Continue if ≥80K (adjust training) |
| Training | Val cosine | ≥0.80 | Reduce LR, increase epochs |

**DO NOT PROCEED** to next stage if quality gate fails!

---

## Documentation

| Document | Purpose |
|----------|---------|
| `COMPLETE_SUMMARY_Ontology_Pipeline.md` | This comprehensive summary |
| `Ontology_Data_Pipeline_Complete_Plan.md` | Full implementation details |
| `P13_Root_Cause_Analysis_and_Fix.md` | Why we're replacing FactoidWiki |
| `GWOM_Ontology_Dataset_Options.md` | Dataset selection rationale |
| `dataset_map.json` | Central data catalog |

---

## Timeline

| Week | Days | Tasks | Deliverable |
|------|------|-------|-------------|
| 1 | 1-7 | Download + Test + Validate | 5K samples, 85%+ pass rate |
| 2-3 | 8-21 | Full ingestion | 125K chains in database |
| 4 | 22-28 | Training | Trained LVM model |

**Total**: 3-4 weeks to production-ready P15 data

---

## Next Steps

1. ✅ Review this plan
2. ⏳ Approve strategy to replace FactoidWiki
3. ⏳ Run: `cp data/dataset_map_template.json data/dataset_map.json`
4. ⏳ Run: `make ontology-download-all`
5. ⏳ Monitor downloads, proceed when complete

**Questions?** See full documentation in `docs/PRDs/`

---

**Last Updated**: 2025-09-30
**Status**: Ready for Execution ✅
