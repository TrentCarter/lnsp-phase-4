-- PostgreSQL initialization script for LNSP database
-- Run automatically when PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enums
DO $$ BEGIN
  CREATE TYPE content_type AS ENUM ('factual','math','instruction','narrative');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE validation_status AS ENUM ('passed','failed','pending');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Create core CPE entry table
CREATE TABLE IF NOT EXISTS cpe_entry (
  cpe_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mission_text     TEXT NOT NULL,
  source_chunk     TEXT NOT NULL,
  concept_text     TEXT NOT NULL,
  probe_question   TEXT NOT NULL,
  expected_answer  TEXT NOT NULL,
  domain_code      SMALLINT NOT NULL,
  task_code        SMALLINT NOT NULL,
  modifier_code    SMALLINT NOT NULL,
  content_type     content_type NOT NULL,
  dataset_source   TEXT NOT NULL,
  chunk_position   JSONB NOT NULL,
  relations_text   JSONB,
  echo_score       REAL,
  validation_status validation_status NOT NULL DEFAULT 'pending',
  batch_id         UUID,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  tmd_bits         SMALLINT NOT NULL,
  tmd_lane         TEXT NOT NULL,
  lane_index       SMALLINT NOT NULL CHECK (lane_index BETWEEN 0 AND 32767)
);

-- Create vector table (optional pgvector support)
CREATE TABLE IF NOT EXISTS cpe_vectors (
  cpe_id      UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id) ON DELETE CASCADE,
  fused_vec   vector(784) NOT NULL,
  question_vec vector(768),
  concept_vec vector(768),
  tmd_dense   vector(16),
  fused_norm  REAL
);

-- Create fallback JSON vector table for development
CREATE TABLE IF NOT EXISTS cpe_vectors_json (
  cpe_id UUID PRIMARY KEY,
  fused_json TEXT,
  question_json TEXT,
  concept_json TEXT,
  tmd_json TEXT,
  fused_norm REAL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS cpe_lane_idx ON cpe_entry (lane_index);
CREATE INDEX IF NOT EXISTS cpe_ct_idx ON cpe_entry (content_type);
CREATE INDEX IF NOT EXISTS cpe_ds_idx ON cpe_entry (dataset_source);
CREATE INDEX IF NOT EXISTS cpe_created_idx ON cpe_entry (created_at DESC);

-- Create ANN indexes (build after initial load + ANALYZE)
-- Note: These will be built by the application after data loading
-- CREATE INDEX IF NOT EXISTS cpe_fused_ann ON cpe_vectors USING ivfflat (fused_vec vector_cosine_ops) WITH (lists = 1200);
-- CREATE INDEX IF NOT EXISTS cpe_question_ann ON cpe_vectors USING ivfflat (question_vec vector_cosine_ops) WITH (lists = 1200);
-- CREATE INDEX IF NOT EXISTS cpe_fused_norm_idx ON cpe_vectors (fused_norm);

-- Grant permissions (for development)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lnsp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lnsp;
