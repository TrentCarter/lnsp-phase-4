# 10K Ontology Ingestion Checklist
**Target:** 2K SWO + 8K WordNet = 10,000 concepts
**Components:** PostgreSQL + Neo4j + FAISS + CPESH + TMD + Graph

---

## ✅ PRE-FLIGHT CHECKS (Complete These BEFORE Starting)

### 1. Data Files Ready
- [x] SWO chains: `artifacts/ontology_chains/swo_chains.jsonl` (2,013 chains)
- [x] WordNet chains: `artifacts/ontology_chains/wordnet_chains_8k.jsonl` (8,000 chains)

### 2. Services Running
```bash
# PostgreSQL
psql lnsp -c "SELECT 1"

# Neo4j
cypher-shell -u neo4j -p password "RETURN 1"

# Ollama (CRITICAL for CPESH generation)
curl http://localhost:11434/api/tags
```

### 3. LLM Environment Set
```bash
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Verify
echo $LNSP_LLM_ENDPOINT
echo $LNSP_LLM_MODEL
```

### 4. Disk Space
```bash
# Need ~5GB free
df -h | grep -E "Avail|/Users"
```

---

## 📋 COMPLETE INGESTION PIPELINE

### Phase 1: SWO Ingestion (2,000 concepts)
**Estimated time:** 2-3 hours

**What gets created:**
- [ ] PostgreSQL `cpe_entry` rows (2,000) with `dataset_source='ontology-swo'`
- [ ] PostgreSQL `cpe_vectors` rows (2,000) with:
  - `concept_vec` (768D from GTR-T5)
  - `tmd_dense` (16D TMD features)
  - `fused_vec` (784D = 768 + 16)
- [ ] Neo4j `Concept` nodes (2,000) with `source='swo'`
- [ ] Neo4j `RELATES_TO` edges (ontology hierarchy)
- [ ] FAISS index updated (or created if first run)

**CPESH Components** (per concept):
- [ ] `concept_text` - Base concept
- [ ] `probe_text` - Query formulation
- [ ] `expected_text` - Positive examples
- [ ] `soft_negatives` - Near-misses (JSONB array)
- [ ] `hard_negatives` - Clear negatives (JSONB array)

**Command:**
```bash
./.venv/bin/python -m src.ingest_ontology_simple \
  --input artifacts/ontology_chains/swo_chains.jsonl \
  --write-pg \
  --write-neo4j \
  --write-faiss \
  --limit 2000
```

**Verification:**
```bash
# PostgreSQL count
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source='ontology-swo';"

# Neo4j count
cypher-shell -u neo4j -p password "MATCH (n:Concept {source:'swo'}) RETURN count(n);"

# Check CPESH completeness
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source='ontology-swo' AND jsonb_array_length(soft_negatives) > 0;"
```

---

### Phase 2: WordNet Ingestion (8,000 concepts)
**Estimated time:** 6-8 hours

**What gets created:**
- [ ] PostgreSQL `cpe_entry` rows (8,000) with `dataset_source='ontology-wordnet'`
- [ ] PostgreSQL `cpe_vectors` rows (8,000) with 784D fused vectors
- [ ] Neo4j `Concept` nodes (8,000) with `source='wordnet'`
- [ ] Neo4j `RELATES_TO` edges (hypernym relationships)
- [ ] FAISS index updated to 10,000 total vectors

**Command:**
```bash
./.venv/bin/python -m src.ingest_ontology_simple \
  --input artifacts/ontology_chains/wordnet_chains_8k.jsonl \
  --write-pg \
  --write-neo4j \
  --write-faiss \
  --limit 8000
```

**Verification:**
```bash
# PostgreSQL count
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source='ontology-wordnet';"

# Neo4j count
cypher-shell -u neo4j -p password "MATCH (n:Concept {source:'wordnet'}) RETURN count(n);"

# Total across both sources
psql lnsp -c "SELECT dataset_source, COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'ontology-%' GROUP BY dataset_source;"
```

---

## ✅ FINAL VERIFICATION (Must Check All)

### 1. PostgreSQL Data Integrity
```bash
# Total concept count (should be 10,000)
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'ontology-%';"

# Total vector count (should be 10,000)
psql lnsp -c "SELECT COUNT(*) FROM cpe_vectors v JOIN cpe_entry e USING (cpe_id) WHERE e.dataset_source LIKE 'ontology-%';"

# CPESH completeness (should be >95%)
psql lnsp -c "
SELECT
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE jsonb_array_length(soft_negatives) > 0) as with_soft_neg,
  COUNT(*) FILTER (WHERE jsonb_array_length(hard_negatives) > 0) as with_hard_neg
FROM cpe_entry
WHERE dataset_source LIKE 'ontology-%';
"

# TMD vector dimensions (should all be 16)
psql lnsp -c "
SELECT
  jsonb_array_length(tmd_dense) as tmd_dims,
  COUNT(*) as count
FROM cpe_vectors v
JOIN cpe_entry e USING (cpe_id)
WHERE e.dataset_source LIKE 'ontology-%'
GROUP BY tmd_dims;
"

# Concept vector dimensions (should all be 768)
psql lnsp -c "
SELECT
  jsonb_array_length(concept_vec) as concept_dims,
  COUNT(*) as count
FROM cpe_vectors v
JOIN cpe_entry e USING (cpe_id)
WHERE e.dataset_source LIKE 'ontology-%'
GROUP BY concept_dims;
"
```

### 2. Neo4j Graph Structure
```bash
# Total node count (should be 10,000)
cypher-shell -u neo4j -p password "
MATCH (n:Concept)
WHERE n.source IN ['swo', 'wordnet']
RETURN count(n) as total_nodes;
"

# Edge count (relationships between concepts)
cypher-shell -u neo4j -p password "
MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
WHERE a.source IN ['swo', 'wordnet']
RETURN count(r) as total_edges;
"

# Verify both sources present
cypher-shell -u neo4j -p password "
MATCH (n:Concept)
WHERE n.source IN ['swo', 'wordnet']
RETURN n.source, count(n) as concepts
ORDER BY n.source;
"
```

### 3. FAISS Index
```bash
# Check FAISS metadata file
cat artifacts/faiss_meta.json | jq '.'

# Expected fields:
# - index_path: "artifacts/ontology_10k_*.index"
# - npz_path: "artifacts/ontology_10k_*.npz"
# - num_vectors: 10000
# - vector_dim: 784

# Verify NPZ file
python3 -c "
import numpy as np
data = np.load('$(cat artifacts/faiss_meta.json | jq -r .npz_path)')
print(f'Vectors: {data[\"vectors\"].shape[0]}')
print(f'Dimensions: {data[\"vectors\"].shape[1]}')
print(f'Expected: 10000 × 784')
"
```

### 4. Data Synchronization (CRITICAL!)
```bash
# All three stores must have same count
echo "PostgreSQL:" && psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'ontology-%';" -t
echo "Neo4j:" && cypher-shell -u neo4j -p password "MATCH (n:Concept) WHERE n.source IN ['swo', 'wordnet'] RETURN count(n);" --format plain
echo "FAISS:" && python3 -c "import numpy as np; print(np.load('$(cat artifacts/faiss_meta.json | jq -r .npz_path)')['vectors'].shape[0])"
```

---

## 🚀 READY FOR LVM TRAINING

### Training Data Metrics
- **Total concepts:** 10,000
- **Avg chain length:** ~5 positions
- **Training sequences:** ~50,000 (10K × 5 positions)
- **Train/Val/Test split:** 70/15/15 = 35K / 7.5K / 7.5K

### Quick Training Test
```bash
# Extract sequences from database
./.venv/bin/python -m src.lvm.extract_ocp_sequences \
  --output artifacts/lvm/ocp_training_10k.npz \
  --min-chain-length 3 \
  --max-chain-length 15

# Verify output
python3 -c "
import numpy as np
data = np.load('artifacts/lvm/ocp_training_10k.npz')
print(f'Training sequences: {data[\"train_context\"].shape[0]}')
print(f'Expected: ~35,000')
"
```

---

## 📊 EXPECTED TIMELINE

| Phase | Duration | Cumulative |
|-------|----------|------------|
| SWO ingestion (2K) | 2-3 hours | 3 hours |
| WordNet ingestion (8K) | 6-8 hours | 11 hours total |
| Verification | 5 minutes | Done |

**Recommended:** Run overnight, verify in morning

---

## ❌ TROUBLESHOOTING

### Issue: CPESH arrays empty
**Cause:** Ollama not running or wrong endpoint
**Fix:**
```bash
ollama serve &
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

### Issue: Neo4j connection refused
**Fix:**
```bash
brew services restart neo4j-community
```

### Issue: FAISS not saving
**Cause:** Missing `faiss_db.save()` call
**Check:** Look for "Saving FAISS index" in ingestion logs

### Issue: Vector dimension mismatch
**Cause:** GTR-T5 not loaded correctly
**Fix:** Verify embedder initialization in logs

---

## 🎯 SUCCESS CRITERIA

- [x] 10,000 concepts in PostgreSQL (`cpe_entry`)
- [x] 10,000 vectors in PostgreSQL (`cpe_vectors`) with 784D
- [x] 10,000 nodes in Neo4j (both `swo` and `wordnet`)
- [x] 10,000 vectors in FAISS index
- [x] CPESH completeness >95%
- [x] All three stores synchronized (same count)
- [x] Ready for LVM training extraction

**When all checked:** System ready for LVM training! 🚀
