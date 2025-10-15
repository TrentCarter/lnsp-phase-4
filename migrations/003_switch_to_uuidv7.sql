-- migrations/003_switch_to_uuidv7.sql
-- Switch from random UUIDs to time-ordered UUIDv7
-- Custom implementation (no third-party extensions needed)

-- 0. Enable pgcrypto for gen_random_bytes()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Create UUIDv7 generator function
-- Format: [48-bit timestamp][4-bit version][12-bit random][2-bit variant][62-bit random]
CREATE OR REPLACE FUNCTION uuid_generate_v7()
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    unix_ts_ms BIGINT;
    uuid_bytes BYTEA;
BEGIN
    -- Get current timestamp in milliseconds
    unix_ts_ms := (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT;

    -- Build UUID bytes:
    -- 6 bytes: timestamp (48 bits)
    -- 2 bytes: version (4 bits) + random (12 bits)
    -- 8 bytes: variant (2 bits) + random (62 bits)
    uuid_bytes :=
        SUBSTRING(INT8SEND(unix_ts_ms), 3, 6) ||  -- 48-bit timestamp
        SET_BYTE(
            GEN_RANDOM_BYTES(2),
            0,
            (B'0111' || SUBSTRING(GET_BYTE(GEN_RANDOM_BYTES(1), 0)::BIT(8), 5, 4))::BIT(8)::INT
        ) ||  -- version 7 + 12-bit random
        SET_BYTE(
            GEN_RANDOM_BYTES(8),
            0,
            (B'10' || SUBSTRING(GET_BYTE(GEN_RANDOM_BYTES(1), 0)::BIT(8), 3, 6))::BIT(8)::INT
        );  -- variant 10 + 62-bit random

    RETURN ENCODE(uuid_bytes, 'hex')::UUID;
END;
$$;

-- 2. Create helper function to extract timestamp from UUIDv7
CREATE OR REPLACE FUNCTION uuid_v7_to_timestamptz(uuid_val UUID)
RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
AS $$
DECLARE
    uuid_bytes BYTEA;
    unix_ts_ms BIGINT;
BEGIN
    uuid_bytes := DECODE(REPLACE(uuid_val::TEXT, '-', ''), 'hex');
    unix_ts_ms := (GET_BYTE(uuid_bytes, 0)::BIGINT << 40) +
                  (GET_BYTE(uuid_bytes, 1)::BIGINT << 32) +
                  (GET_BYTE(uuid_bytes, 2)::BIGINT << 24) +
                  (GET_BYTE(uuid_bytes, 3)::BIGINT << 16) +
                  (GET_BYTE(uuid_bytes, 4)::BIGINT << 8) +
                   GET_BYTE(uuid_bytes, 5)::BIGINT;

    RETURN TO_TIMESTAMP(unix_ts_ms / 1000.0);
END;
$$;

-- 3. Update default for new inserts
ALTER TABLE cpe_entry
  ALTER COLUMN cpe_id SET DEFAULT uuid_generate_v7();

-- 4. Test the function
DO $$
DECLARE
    test_uuid UUID;
    test_ts TIMESTAMPTZ;
BEGIN
    test_uuid := uuid_generate_v7();
    test_ts := uuid_v7_to_timestamptz(test_uuid);

    RAISE NOTICE 'Sample UUIDv7: %', test_uuid;
    RAISE NOTICE 'Extracted timestamp: %', test_ts;
    RAISE NOTICE 'Current timestamp: %', clock_timestamp();
END $$;

-- 5. Create index for time-range queries
CREATE INDEX IF NOT EXISTS idx_cpe_entry_id_time ON cpe_entry(cpe_id);

-- Benefits:
-- ✅ Sortable by creation time (timestamp prefix)
-- ✅ Globally unique (random suffix)
-- ✅ Better B-tree index performance (sequential inserts)
-- ✅ No coordination needed (distributed-safe)
-- ✅ Standard RFC 9562 compliant
-- ✅ No third-party extensions required

-- Example queries:
--
-- 1. Get concepts from the last hour (using UUID time ordering):
-- SELECT cpe_id, concept_text,
--        uuid_v7_to_timestamptz(cpe_id) AS extracted_time
-- FROM cpe_entry
-- WHERE cpe_id >= uuid_generate_v7_from_timestamp(NOW() - INTERVAL '1 hour')
-- ORDER BY cpe_id DESC;
--
-- 2. Sequential training data (ordered by UUID):
-- SELECT cpe_id, concept_text
-- FROM cpe_entry
-- ORDER BY cpe_id
-- LIMIT 1000;

-- Notes:
-- - Existing UUIDs remain unchanged (random v4)
-- - New inserts will use UUIDv7 (time-ordered)
-- - UUIDs are sortable: ORDER BY cpe_id sorts by creation time
-- - Can extract creation time: uuid_v7_to_timestamptz(cpe_id)
