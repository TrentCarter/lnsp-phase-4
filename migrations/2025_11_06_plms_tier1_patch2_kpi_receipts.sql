-- PLMS Tier 1 Patch 2: KPI Receipts
-- Date: 2025-11-06
-- Description: Add KPI receipts table for per-task validation tracking

-- === KPI Receipts (per-task validation tracking) ===
CREATE TABLE IF NOT EXISTS kpi_receipts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  run_id TEXT NOT NULL,
  lane_id INTEGER,
  kpi_name TEXT NOT NULL,                    -- e.g., "test_pass_rate", "schema_diff", "bleu_score"
  kpi_value REAL,                            -- actual measured value
  kpi_threshold TEXT,                        -- JSON threshold (e.g., {">=": 0.9})
  pass BOOLEAN NOT NULL,                     -- did KPI meet threshold?
  logs_path TEXT,                            -- path to detailed logs
  measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_kpi_receipt_run
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kpi_receipts_run ON kpi_receipts(run_id);
CREATE INDEX IF NOT EXISTS idx_kpi_receipts_task ON kpi_receipts(task_id);
CREATE INDEX IF NOT EXISTS idx_kpi_receipts_kpi ON kpi_receipts(kpi_name, pass);

-- === PostgreSQL Version (Alternative) ===
/*
CREATE TABLE IF NOT EXISTS kpi_receipts (
  id BIGSERIAL PRIMARY KEY,
  task_id BIGINT NOT NULL,
  run_id UUID NOT NULL,
  lane_id INTEGER,
  kpi_name TEXT NOT NULL,
  kpi_value REAL,
  kpi_threshold JSONB,
  pass BOOLEAN NOT NULL,
  logs_path TEXT,
  measured_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_kpi_receipt_run
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kpi_receipts_run ON kpi_receipts(run_id);
CREATE INDEX IF NOT EXISTS idx_kpi_receipts_task ON kpi_receipts(task_id);
CREATE INDEX IF NOT EXISTS idx_kpi_receipts_kpi ON kpi_receipts(kpi_name, pass);
*/
