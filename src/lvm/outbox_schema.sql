-- Outbox Pattern Schema for LNSP Phase 4
-- Purpose: Staged writes to ensure eventual consistency across PostgreSQL, Neo4j, FAISS
-- Reference: docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md (lines 209-294)

-- ============================================================================
-- Outbox Events Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS outbox_events (
    id BIGSERIAL PRIMARY KEY,
    aggregate_id UUID NOT NULL,           -- FK to cpe_entry.id
    event_type VARCHAR(50) NOT NULL,       -- 'CONCEPT_CREATED', 'CONCEPT_UPDATED', 'CONCEPT_DELETED'
    payload JSONB NOT NULL,                -- Event data (concept text, vectors, etc.)
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'processed', 'failed'
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    INDEX idx_outbox_status (status, created_at),
    INDEX idx_outbox_aggregate (aggregate_id)
);

-- ============================================================================
-- Update cpe_entry to include status field
-- ============================================================================

-- Add status column if not exists (for staged vs ready)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'cpe_entry' AND column_name = 'status'
    ) THEN
        ALTER TABLE cpe_entry ADD COLUMN status VARCHAR(20) DEFAULT 'ready';
        CREATE INDEX idx_cpe_status ON cpe_entry(status);
    END IF;
END $$;

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to insert concept with outbox event atomically
CREATE OR REPLACE FUNCTION create_concept_with_outbox(
    p_id UUID,
    p_concept_text TEXT,
    p_tmd_bits BYTEA,
    p_tmd_dense JSONB,
    p_vector_784d JSONB,
    p_parent_hint UUID,
    p_child_hint UUID
) RETURNS UUID AS $$
DECLARE
    v_event_id BIGINT;
BEGIN
    -- 1. Insert concept (staged)
    INSERT INTO cpe_entry (id, concept_text, tmd_bits, tmd_dense, status)
    VALUES (p_id, p_concept_text, p_tmd_bits, p_tmd_dense, 'staged')
    ON CONFLICT (id) DO UPDATE SET
        concept_text = EXCLUDED.concept_text,
        tmd_bits = EXCLUDED.tmd_bits,
        tmd_dense = EXCLUDED.tmd_dense,
        status = 'staged';

    -- 2. Insert outbox event
    INSERT INTO outbox_events (aggregate_id, event_type, payload, status)
    VALUES (
        p_id,
        'CONCEPT_CREATED',
        jsonb_build_object(
            'id', p_id,
            'text', p_concept_text,
            'vector_784d', p_vector_784d,
            'tmd_bits', encode(p_tmd_bits, 'base64'),
            'parent_hint', p_parent_hint,
            'child_hint', p_child_hint
        ),
        'pending'
    )
    RETURNING id INTO v_event_id;

    RETURN p_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark outbox event as processed
CREATE OR REPLACE FUNCTION mark_outbox_processed(
    p_event_id BIGINT,
    p_aggregate_id UUID
) RETURNS VOID AS $$
BEGIN
    -- Update outbox event
    UPDATE outbox_events
    SET status = 'processed', processed_at = NOW()
    WHERE id = p_event_id;

    -- Update concept status
    UPDATE cpe_entry
    SET status = 'ready'
    WHERE id = p_aggregate_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark outbox event as failed
CREATE OR REPLACE FUNCTION mark_outbox_failed(
    p_event_id BIGINT,
    p_error_msg TEXT
) RETURNS VOID AS $$
BEGIN
    UPDATE outbox_events
    SET
        status = 'failed',
        retry_count = retry_count + 1,
        last_error = p_error_msg
    WHERE id = p_event_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Monitoring Views
-- ============================================================================

-- View: Outbox lag metrics
CREATE OR REPLACE VIEW outbox_lag_metrics AS
SELECT
    status,
    COUNT(*) as count,
    MIN(EXTRACT(EPOCH FROM (NOW() - created_at))) as min_lag_sec,
    AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_lag_sec,
    MAX(EXTRACT(EPOCH FROM (NOW() - created_at))) as max_lag_sec,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (NOW() - created_at))) as p95_lag_sec
FROM outbox_events
WHERE status = 'pending'
GROUP BY status;

-- View: Failed events summary
CREATE OR REPLACE VIEW outbox_failed_summary AS
SELECT
    event_type,
    COUNT(*) as count,
    MAX(retry_count) as max_retries,
    AVG(retry_count) as avg_retries
FROM outbox_events
WHERE status = 'failed'
GROUP BY event_type;

-- ============================================================================
-- Cleanup Functions (for maintenance)
-- ============================================================================

-- Function to purge old processed events (keep last 7 days)
CREATE OR REPLACE FUNCTION purge_old_outbox_events(p_days INT DEFAULT 7) RETURNS INT AS $$
DECLARE
    v_deleted_count INT;
BEGIN
    DELETE FROM outbox_events
    WHERE status = 'processed' AND processed_at < NOW() - (p_days || ' days')::INTERVAL;

    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Composite index for worker polling
CREATE INDEX IF NOT EXISTS idx_outbox_worker_poll ON outbox_events(status, created_at) WHERE status = 'pending';

-- Index for retry logic
CREATE INDEX IF NOT EXISTS idx_outbox_retries ON outbox_events(status, retry_count) WHERE status = 'failed' AND retry_count < 3;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE outbox_events IS 'Transactional outbox pattern for eventual consistency across PostgreSQL, Neo4j, and FAISS';
COMMENT ON COLUMN outbox_events.aggregate_id IS 'UUID of the cpe_entry being synchronized';
COMMENT ON COLUMN outbox_events.payload IS 'JSONB containing all data needed to sync to Neo4j/FAISS';
COMMENT ON COLUMN outbox_events.status IS 'pending: awaiting processing, processed: synced successfully, failed: needs retry';
COMMENT ON FUNCTION create_concept_with_outbox IS 'Atomically creates concept and outbox event in single transaction';
