-- Migration 003: Add Sequential Ordering Metadata for LVM Training
-- Date: October 11, 2025
-- Purpose: Enable fast ordered retrieval of training sequences

-- ============================================================================
-- STEP 1: Add new columns
-- ============================================================================

ALTER TABLE cpe_entry
ADD COLUMN document_id TEXT,
ADD COLUMN sequence_index INTEGER,
ADD COLUMN episode_id TEXT,
ADD COLUMN parent_cpe_id UUID,
ADD COLUMN child_cpe_id UUID,
ADD COLUMN last_accessed TIMESTAMP;

-- ============================================================================
-- STEP 2: Backfill existing data (watercycle-mini, etc.)
-- ============================================================================

-- Strategy: Extract from existing chunk_position JSON
UPDATE cpe_entry
SET
    -- document_id = dataset_source + batch_id (groups chunks from same source)
    document_id = dataset_source || COALESCE('_' || batch_id, '_' || cpe_id::text),

    -- sequence_index = extract from chunk_position JSON (if available)
    sequence_index = COALESCE((chunk_position->>'index')::integer, 0),

    -- episode_id = batch_id (if available)
    episode_id = batch_id
WHERE chunk_position IS NOT NULL;

-- Fallback for data without chunk_position
UPDATE cpe_entry
SET
    document_id = 'legacy_' || dataset_source,
    sequence_index = 0
WHERE document_id IS NULL;

-- ============================================================================
-- STEP 3: Set NOT NULL constraints (after backfill)
-- ============================================================================

ALTER TABLE cpe_entry
ALTER COLUMN document_id SET NOT NULL,
ALTER COLUMN sequence_index SET NOT NULL;

-- ============================================================================
-- STEP 4: Create performance indexes
-- ============================================================================

-- Primary index: Fast ordered retrieval within documents
CREATE INDEX idx_document_sequence ON cpe_entry(document_id, sequence_index);

-- Episode-level index (optional, for coherence analysis)
CREATE INDEX idx_episode ON cpe_entry(episode_id, sequence_index) WHERE episode_id IS NOT NULL;

-- Parent-child index (for graph training, optional)
CREATE INDEX idx_parent_cpe ON cpe_entry(parent_cpe_id) WHERE parent_cpe_id IS NOT NULL;
CREATE INDEX idx_child_cpe ON cpe_entry(child_cpe_id) WHERE child_cpe_id IS NOT NULL;

-- ============================================================================
-- STEP 5: Add foreign key constraints (OPTIONAL - adds overhead)
-- ============================================================================

-- Uncomment these if you want referential integrity enforcement
-- WARNING: This adds ~10% overhead to inserts/updates

-- ALTER TABLE cpe_entry
-- ADD CONSTRAINT fk_parent_cpe
-- FOREIGN KEY (parent_cpe_id) REFERENCES cpe_entry(cpe_id) ON DELETE SET NULL;

-- ALTER TABLE cpe_entry
-- ADD CONSTRAINT fk_child_cpe
-- FOREIGN KEY (child_cpe_id) REFERENCES cpe_entry(cpe_id) ON DELETE SET NULL;

-- ============================================================================
-- STEP 6: Verification queries
-- ============================================================================

-- Check that all rows have document_id and sequence_index
DO $$
DECLARE
    null_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO null_count
    FROM cpe_entry
    WHERE document_id IS NULL OR sequence_index IS NULL;

    IF null_count > 0 THEN
        RAISE WARNING 'Migration incomplete: % rows with NULL document_id or sequence_index', null_count;
    ELSE
        RAISE NOTICE 'Migration successful: All rows have document_id and sequence_index';
    END IF;
END $$;

-- Show sample data
SELECT
    dataset_source,
    document_id,
    sequence_index,
    episode_id,
    LEFT(concept_text, 50) as concept_preview
FROM cpe_entry
ORDER BY document_id, sequence_index
LIMIT 20;

-- Show document counts
SELECT
    dataset_source,
    COUNT(DISTINCT document_id) as num_documents,
    COUNT(*) as num_chunks,
    ROUND(AVG(sequence_index), 2) as avg_sequence_index
FROM cpe_entry
GROUP BY dataset_source
ORDER BY num_chunks DESC;

-- ============================================================================
-- NOTES
-- ============================================================================

-- Performance expectations:
--   - Ordered retrieval: <10ms for 1000 chunks
--   - Index overhead: ~5% of table size
--   - Parent/child population: ~10ms per chunk (if enabled)

-- Training data extraction query (example):
--   SELECT cv.concept_vec, ce.sequence_index
--   FROM cpe_entry ce
--   JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
--   WHERE ce.document_id = 'wikipedia_12345'
--   ORDER BY ce.sequence_index;

-- Coherence validation query (example):
--   WITH consecutive AS (
--     SELECT
--       ce1.sequence_index as idx1,
--       ce2.sequence_index as idx2,
--       cv1.concept_vec as vec1,
--       cv2.concept_vec as vec2
--     FROM cpe_entry ce1
--     JOIN cpe_entry ce2 ON
--       ce1.document_id = ce2.document_id AND
--       ce2.sequence_index = ce1.sequence_index + 1
--     JOIN cpe_vectors cv1 ON ce1.cpe_id = cv1.cpe_id
--     JOIN cpe_vectors cv2 ON ce2.cpe_id = cv2.cpe_id
--     WHERE ce1.document_id = 'wikipedia_12345'
--   )
--   SELECT idx1, idx2, (vec1 <#> vec2) as cosine_similarity
--   FROM consecutive;
