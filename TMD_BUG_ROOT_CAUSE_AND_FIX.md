# TMD Vector Bug: Root Cause Analysis and Fix

## Problem Summary

The fused vectors (784D = 16D TMD + 768D concept) had **16 zeros at the beginning** instead of the correct TMD encoding. This affected **1,562 out of 4,484 vectors** (34.8%).

## Root Cause

The bug occurred in THREE STAGES during the ontology data ingestion on October 1, 2025:

### Stage 1: Initial Ingestion Failure (Oct 1, 14:36-18:47)

1. **LLM returned invalid/unmapped domain/task/modifier strings**
   - The Llama 3.1 LLM extracted strings that weren't in the mapping dictionaries
   - Example: LLM returned `domain="biochemistry"` but only `"science"` (0), `"medicine"` (4), etc. were mapped

2. **Fallback logic used (0,0,0) codes**
   ```python
   d = DOMAINS.get(domain, DOMAINS.get("technology", 2))  # But if "technology" also missing → 2
   t = TASKS.get(task, TASKS.get("fact_retrieval", 0))    # → 0
   m = MODIFIERS.get(modifier, MODIFIERS.get("descriptive", 27))  # → 27
   ```
   - However, in this case ALL lookups failed and returned 0,0,0

3. **TMD vector computed from (0,0,0)**
   ```python
   tmd16 = tmd16_deterministic(0, 0, 0)  # Returns all zeros!
   ```

4. **Fused vector created with zero TMD**
   ```python
   fused = np.concatenate([tmd16_zeros, concept_vec_768d])
   fused_normalized = fused / norm(fused)
   ```
   - Result: First 16 dimensions = all zeros ❌

### Stage 2: Incomplete Fix Attempt (Oct 1, later)

A fix script (`tools/fix_ontology_tmd*.py`) was run that:

1. ✅ Updated `cpe_entry` codes from (0,0,0) to (1,1,1)
2. ✅ Regenerated `tmd_dense` column with `tmd16_deterministic(1,1,1)`
3. ❌ **DID NOT update `fused_vec` column**

Result:
- Database had correct TMD codes (1,1,1) in `cpe_entry`
- Database had correct TMD vectors in `cpe_vectors.tmd_dense`
- Database STILL had broken fused vectors with 16 zeros in `cpe_vectors.fused_vec`

### Stage 3: Confusion

The NPZ export took data from the database, so:
- NPZ had correct `tmd_dense` (from regeneration)
- NPZ had broken `fused_vec` (never updated)
- FAISS index built from broken fused vectors

## The Actual Fix

**Script:** `tools/rebuild_fused_vectors.py`

For each vector in the database:
1. Read `tmd_dense` (16D, already correct from earlier fix)
2. Read `concept_vec` (768D, always was correct)
3. Rebuild fused: `fused = concat([tmd_dense, concept_vec])`
4. Normalize: `fused = fused / ||fused||`
5. Update database: `UPDATE cpe_vectors SET fused_vec = ...`

**Result:**
- ✅ All 4,484 fused vectors now have correct TMD prefixes
- ✅ All fused vectors are unit vectors (norm=1.0)
- ✅ TMD part of fused has norm ≈ 0.707 (1/√2, correct scaling)

## Why TMD != Fused[:16] (This is CORRECT!)

The fused vector first 16 dims are NOT identical to `tmd_dense` because:

1. **TMD alone:** unit vector, norm=1.0
2. **Concept alone:** unit vector, norm≈1.0
3. **Concatenated:** `[tmd, concept]` has norm ≈ √2
4. **After normalization:** `fused / √2` → each part scaled by 1/√2

So if `tmd_dense = [-0.163, -0.310, ...]`, then:
- `fused[:16] = [-0.115, -0.219, ...]` (scaled by 1/√2 ≈ 0.707)

This is mathematically correct! The direction is preserved (unit vectors point same way), but magnitude is scaled.

## Files Generated

### Database
- `cpe_entry`: TMD codes (1,1,1) for affected entries
- `cpe_vectors.tmd_dense`: Correct 16D unit vectors
- `cpe_vectors.fused_vec`: **NOW FIXED** - 784D unit vectors with correct TMD prefix

### Artifacts
- `artifacts/fw9k_vectors_tmd_fixed.npz` - Corrected NPZ export (4,484 vectors)
- `artifacts/fw9k_ivf_flat_ip_tmd_fixed.index` - Rebuilt FAISS index with corrected vectors
- `artifacts/index_meta.json` - Updated index metadata

### Tools
- `tools/rebuild_fused_vectors.py` - The actual fix script
- `tools/regenerate_all_tmd_vectors.py` - Earlier incomplete fix
- `tools/fix_ontology_tmd*.py` - Original fix attempts (incomplete)

## Verification

Run this to verify the fix:

```bash
# Check database
psql lnsp -c "
SELECT
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE substring(fused_vec::text, 1, 32) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0') as zero_tmd,
    COUNT(*) - COUNT(*) FILTER (WHERE substring(fused_vec::text, 1, 32) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0') as valid_tmd
FROM cpe_vectors;
"

# Expected output:
# total | zero_tmd | valid_tmd
# ------+----------+-----------
#  4484 |        0 |      4484
```

```python
# Check NPZ
import numpy as np
npz = np.load('artifacts/fw9k_vectors_tmd_fixed.npz')

# Check no zero-TMD vectors
vectors = npz['vectors']
zero_count = sum(1 for v in vectors if np.allclose(v[:16], 0.0))
print(f"Vectors with zero TMD: {zero_count}/{len(vectors)}")
# Expected: 0/4484

# Check all are unit vectors
norms = [np.linalg.norm(v) for v in vectors]
unit_count = sum(1 for n in norms if abs(n - 1.0) < 0.01)
print(f"Unit vectors: {unit_count}/{len(vectors)}")
# Expected: 4484/4484
```

## Lessons Learned

1. **Validate LLM outputs**: The LLM can return any string - always validate against enum mappings
2. **Fallback chains matter**: Nested dict.get() with another dict.get() as default is fragile
3. **Update ALL derived data**: When fixing codes, must also update TMD vectors AND fused vectors
4. **Vector fusion math**: After concatenating unit vectors and normalizing, components scale by 1/√n

## Next Steps

1. ✅ Database fused vectors fixed
2. ✅ NPZ export updated
3. ✅ FAISS index rebuilt
4. ⏳ Update LVM training to use new index
5. ⏳ Re-run evaluation with corrected vectors

## Code Changes Needed

### 1. Fix LLM extraction fallback (src/prompt_extractor.py)

```python
# BEFORE (BROKEN)
d = DOMAINS.get(domain, DOMAINS.get("technology", 2))

# AFTER (SAFER)
d = DOMAINS.get(domain)
if d is None:
    logger.warning(f"Unmapped domain: {domain}, using fallback")
    d = 1  # science as fallback
```

### 2. Add fused vector regeneration to fix scripts

Any script that updates TMD codes must:
1. Update `cpe_entry` codes
2. Regenerate `tmd_dense`
3. **Rebuild `fused_vec`** ← This was missing!

## Timeline

- **Oct 1, 14:36-18:47**: Ontology ingestion with broken TMD codes (0,0,0)
- **Oct 1, later**: Incomplete fix - updated codes and tmd_dense, missed fused_vec
- **Oct 2, 12:00+**: Root cause analysis
- **Oct 2, 13:00**: Complete fix - rebuilt all fused vectors
- **Oct 2, 13:30**: Verification complete ✅
