-- Migration: Add usage tracking, quality metrics, and parent/child relationships
-- Date: 2025-10-09
-- Purpose: Enable training sequence generation and quality monitoring

-- Add usage tracking fields
ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;

-- Add quality tracking fields
ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS confidence_score REAL DEFAULT NULL;

ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS quality_metrics JSONB DEFAULT '{}'::jsonb;

-- Add parent/child relationship fields for training chains
ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS parent_cpe_ids JSONB DEFAULT '[]'::jsonb;

ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS child_cpe_ids JSONB DEFAULT '[]'::jsonb;

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_cpe_entry_confidence ON cpe_entry(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_cpe_entry_access_count ON cpe_entry(access_count DESC);
CREATE INDEX IF NOT EXISTS idx_cpe_entry_last_accessed ON cpe_entry(last_accessed_at DESC);

-- GIN index for parent/child relationship lookups
CREATE INDEX IF NOT EXISTS idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX IF NOT EXISTS idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);

-- Comments for documentation
COMMENT ON COLUMN cpe_entry.last_accessed_at IS 'Timestamp of last retrieval/access for usage tracking';
COMMENT ON COLUMN cpe_entry.access_count IS 'Number of times this concept has been retrieved';
COMMENT ON COLUMN cpe_entry.confidence_score IS 'Confidence score (0-1) based on CPESH completeness, vector quality, and negatives count';
COMMENT ON COLUMN cpe_entry.quality_metrics IS 'JSON object with detailed quality metrics: cpesh_completeness, vector_norm, text_length, negatives_count, etc.';
COMMENT ON COLUMN cpe_entry.parent_cpe_ids IS 'Array of parent concept UUIDs (for graph traversal and training sequence generation)';
COMMENT ON COLUMN cpe_entry.child_cpe_ids IS 'Array of child concept UUIDs (for graph traversal and training sequence generation)';
