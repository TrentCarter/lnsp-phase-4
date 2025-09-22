-- FactoidWiki → LNSP PostgreSQL bootstrap (pgvector required)
-- Version: 0.1  (2025-09-21)
-- Idempotent: safe to re-run

-- Extensions ------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector

-- Enums -----------------------------------------------------------
DO $$ BEGIN
  CREATE TYPE content_type AS ENUM ('factual','math','instruction','narrative');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE validation_status AS ENUM ('passed','failed','pending');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Core table: text + metadata ------------------------------------
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

-- Helpful indexes
CREATE INDEX IF NOT EXISTS cpe_lane_idx    ON cpe_entry (lane_index);
CREATE INDEX IF NOT EXISTS cpe_ct_idx      ON cpe_entry (content_type);
CREATE INDEX IF NOT EXISTS cpe_ds_idx      ON cpe_entry (dataset_source);
CREATE INDEX IF NOT EXISTS cpe_created_idx ON cpe_entry (created_at DESC);

-- Vector table (LEAN mode stores fused + question; FULL can add concept + tmd_dense)
CREATE TABLE IF NOT EXISTS cpe_vectors (
  cpe_id       UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id) ON DELETE CASCADE,
  fused_vec    vector(784) NOT NULL,
  question_vec vector(768),
  concept_vec  vector(768),     -- optional (NULL in LEAN)
  tmd_dense    vector(16),      -- optional (NULL in LEAN)
  fused_norm   REAL
);

-- ANN indexes (build after some rows are present)
-- lists ≈ √N; start with 1200 for 10k and tune later
CREATE INDEX IF NOT EXISTS cpe_fused_ann
  ON cpe_vectors USING ivfflat (fused_vec vector_cosine_ops) WITH (lists = 1200);

CREATE INDEX IF NOT EXISTS cpe_question_ann
  ON cpe_vectors USING ivfflat (question_vec vector_cosine_ops) WITH (lists = 1200);

CREATE INDEX IF NOT EXISTS cpe_fused_norm_idx ON cpe_vectors (fused_norm);

-- Governance ------------------------------------------------------
-- Optional logical deletion
-- ALTER TABLE cpe_entry ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- Seed helpers ----------------------------------------------------
-- Minimal upsert functions to simplify loaders (optional)
CREATE OR REPLACE FUNCTION upsert_cpe_entry(p JSONB)
RETURNS UUID AS $$
DECLARE
  id UUID;
BEGIN
  INSERT INTO cpe_entry (
    cpe_id, mission_text, source_chunk, concept_text, probe_question, expected_answer,
    domain_code, task_code, modifier_code, content_type, dataset_source, chunk_position,
    relations_text, echo_score, validation_status, batch_id, tmd_bits, tmd_lane, lane_index
  ) VALUES (
    COALESCE((p->>'cpe_id')::uuid, uuid_generate_v4()),
    p->>'mission_text', p->>'source_chunk', p->>'concept_text', p->>'probe_question', p->>'expected_answer',
    (p->>'domain_code')::smallint, (p->>'task_code')::smallint, (p->>'modifier_code')::smallint,
    (p->>'content_type')::content_type, p->>'dataset_source', p->'chunk_position',
    p->'relations_text', (p->>'echo_score')::real, (p->>'validation_status')::validation_status,
    NULLIF(p->>'batch_id','')::uuid, (p->>'tmd_bits')::smallint, p->>'tmd_lane', (p->>'lane_index')::smallint
  ) ON CONFLICT (cpe_id) DO UPDATE SET
    mission_text     = EXCLUDED.mission_text,
    source_chunk     = EXCLUDED.source_chunk,
    concept_text     = EXCLUDED.concept_text,
    probe_question   = EXCLUDED.probe_question,
    expected_answer  = EXCLUDED.expected_answer,
    domain_code      = EXCLUDED.domain_code,
    task_code        = EXCLUDED.task_code,
    modifier_code    = EXCLUDED.modifier_code,
    content_type     = EXCLUDED.content_type,
    dataset_source   = EXCLUDED.dataset_source,
    chunk_position   = EXCLUDED.chunk_position,
    relations_text   = EXCLUDED.relations_text,
    echo_score       = EXCLUDED.echo_score,
    validation_status= EXCLUDED.validation_status,
    batch_id         = EXCLUDED.batch_id,
    tmd_bits         = EXCLUDED.tmd_bits,
    tmd_lane         = EXCLUDED.tmd_lane,
    lane_index       = EXCLUDED.lane_index
  RETURNING cpe_id INTO id;
  RETURN id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION upsert_cpe_vectors(
  p_cpe_id UUID,
  p_fused   vector,
  p_question vector DEFAULT NULL,
  p_concept  vector DEFAULT NULL,
  p_tmd      vector DEFAULT NULL,
  p_norm     REAL    DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
  INSERT INTO cpe_vectors (cpe_id, fused_vec, question_vec, concept_vec, tmd_dense, fused_norm)
  VALUES (p_cpe_id, p_fused, p_question, p_concept, p_tmd, p_norm)
  ON CONFLICT (cpe_id) DO UPDATE SET
    fused_vec = EXCLUDED.fused_vec,
    question_vec = EXCLUDED.question_vec,
    concept_vec = EXCLUDED.concept_vec,
    tmd_dense = EXCLUDED.tmd_dense,
    fused_norm = EXCLUDED.fused_norm;
END;
$$ LANGUAGE plpgsql;

-- Analyze for planner hints
ANALYZE cpe_entry;
ANALYZE cpe_vectors;

-- Tips -----------------------------------------------------------
-- psql -h localhost -U lnsp -d lnsp -f scripts/init_pg.sql
-- Verify indexes: \di+ cpe_*  and  SELECT * FROM pg_extension;
