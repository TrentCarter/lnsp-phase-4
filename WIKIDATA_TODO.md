# ⚠️ WIKIDATA INTEGRATION - DO NOT FORGET

## 🚨 CRITICAL REMINDER
**We chose Option A (4 datasets, 105K chains) to move faster.**
**We MUST return to add Wikidata (Option B) after initial training succeeds.**

---

## Why We Need Wikidata Later

1. **+20K chains** of CS/Programming content (19% more data)
2. **Better coverage** of algorithms, data structures, programming concepts
3. **Completes the 125K chain target** from original plan
4. **Quality**: 88% pass rate (still good, just slightly lower than others)

---

## When To Add Wikidata

### ✅ Add Wikidata After:
- [ ] Initial 4-dataset training completes successfully
- [ ] LVM shows good performance on 105K chain dataset
- [ ] P13 validation passes on trained model
- [ ] You want to improve CS/Programming domain coverage

### 🎯 Expected Timeline:
- Week 3-4 after initial training starts
- Or immediately if LVM needs more CS-specific data

---

## How To Add Wikidata (30-45 minutes)

### Step 1: Run SPARQL Queries (15-20 min)

Open the queries file:
```bash
cat data/datasets/ontology_datasets/wikidata/wikidata_sparql_queries.txt
```

For each query:
1. Go to https://query.wikidata.org/
2. Paste query from file
3. Click "Run" (▶️ button)
4. Wait 30-60 seconds for results
5. Click "Download" → "JSON" format
6. Save as:
   - `query1_programming_languages.json`
   - `query2_software_algorithms.json`
   - `query3_cs_concepts.json`
   - `query4_data_structures.json`
7. Move files to `data/datasets/ontology_datasets/wikidata/`

### Step 2: Create Wikidata Parser (10-15 min)

Run this command (parser will exist by then):
```bash
./.venv/bin/python src/parsers/parse_wikidata.py \
  --input-dir data/datasets/ontology_datasets/wikidata \
  --output artifacts/ontology_chains/wikidata_chains.jsonl
```

### Step 3: Test Ingestion (5 min)

```bash
# Test 1K samples
./.venv/bin/python src/test_ingest_ontology.py \
  --dataset wikidata \
  --num-samples 1000
```

### Step 4: Validate Quality (5 min)

```bash
# Run P13 validation
./.venv/bin/python src/pipeline/p13_ontology_validation.py \
  --input artifacts/ontology_samples/wikidata_1k.jsonl \
  --output artifacts/validation_reports/wikidata_1k_validation.json
```

### Step 5: Full Ingestion (1-2 days)

If validation passes (>80% pass rate):
```bash
make ontology-ingest-wikidata
```

---

## 📍 Current Status

- [x] Download scripts created for all 5 datasets
- [x] 4 datasets downloaded (SWO, GO, ConceptNet, DBpedia)
- [x] Wikidata SPARQL queries prepared
- [ ] **WIKIDATA DOWNLOAD PENDING** (deferred to later)
- [ ] Wikidata parser (will create when needed)
- [ ] Wikidata ingestion (will run when needed)

---

## 🔗 Related Files

- SPARQL Queries: `data/datasets/ontology_datasets/wikidata/wikidata_sparql_queries.txt`
- Download Script: `scripts/data_downloading/download_wikidata.py`
- Parser (future): `src/parsers/parse_wikidata.py` (to be created)
- Main Plan: `docs/PRDs/Ontology_Data_Pipeline_Complete_Plan.md`
- Dataset Map: `data/dataset_map.json`

---

## ✅ How To Remember

This file serves as a **permanent reminder** that Wikidata integration is:
1. **Planned** (not forgotten)
2. **Deferred** (intentionally postponed)
3. **Documented** (all steps ready)
4. **Quick to add** (30-45 min manual work + 1-2 days automated ingestion)

**Check this file after initial training completes!**

---

*Created: 2025-09-30*
*Decision: Proceed with 4 datasets first, add Wikidata later*
*Reason: Faster path to training while maintaining quality*
