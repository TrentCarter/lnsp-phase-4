-- PLMS V1 Tier 1 Migration
-- Date: 2025-11-06
-- Description: Multi-run support, lane-specific KPIs, Bayesian calibration, active learning

-- SQLite-compatible version (use CURRENT_TIMESTAMP instead of NOW(), TEXT instead of JSONB)
-- For PostgreSQL, replace TEXT with JSONB and CURRENT_TIMESTAMP with NOW()

-- === Tier-1: Multi-run support, rehearsal, provider snapshot ===
ALTER TABLE project_runs
  ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'baseline';     -- 'baseline'|'rehearsal'|'replay'|'hotfix'
ALTER TABLE project_runs
  ADD COLUMN rehearsal_pct REAL;                            -- e.g., 0.01 for 1% canary
ALTER TABLE project_runs
  ADD COLUMN provider_matrix_json TEXT;                     -- deterministic replay snapshot (JSON)
ALTER TABLE project_runs
  ADD COLUMN capability_snapshot TEXT;                      -- portfolio/capabilities at start (JSON)
ALTER TABLE project_runs
  ADD COLUMN risk_score REAL;                               -- computed at /simulate

-- Index to group metrics by run kind and time
CREATE INDEX IF NOT EXISTS idx_project_runs_kind_time
  ON project_runs (project_id, run_kind, started_at);

-- === Tier-1: Lane-specific KPIs + active learning feedback ===
ALTER TABLE task_estimates
  ADD COLUMN kpi_formula TEXT;                              -- JSON: {"test_pass_rate":{">=":0.9},"schema_diff":{"==":0}}

CREATE TABLE IF NOT EXISTS lane_overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_estimate_id INTEGER NOT NULL,
  task_description TEXT,
  predicted_lane INTEGER,
  corrected_lane INTEGER,
  corrected_by TEXT,
  corrected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_lane_override_task
    FOREIGN KEY (task_estimate_id) REFERENCES task_estimates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lane_overrides_task ON lane_overrides(task_estimate_id);

-- Execution linking (if not already done)
-- Check if project_id column exists before adding
-- NOTE: For production, use proper migration tool that handles column existence checks
-- This is a simplified version for demonstration
ALTER TABLE action_logs ADD COLUMN project_id INTEGER; -- Will fail if already exists; handle gracefully

CREATE INDEX IF NOT EXISTS idx_action_logs_project ON action_logs(project_id);

-- === Tier-1.5 foundation: Bayesian estimate versions (safe to ship now) ===
CREATE TABLE IF NOT EXISTS estimate_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER,
  lane_id INTEGER,                  -- nullable for global
  provider_name TEXT,               -- 'gha/runner-large', 'pas/openai:gpt-4o', etc.
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  priors_hash TEXT,
  tokens_mean REAL,
  tokens_stddev REAL,
  duration_ms_mean REAL,
  duration_ms_stddev REAL,
  cost_usd_mean REAL,
  cost_usd_stddev REAL,
  n_observations INTEGER DEFAULT 0,
  mean_absolute_error_tokens REAL,
  mean_absolute_error_duration_ms REAL,
  CONSTRAINT fk_estimate_versions_project
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_estimate_versions_lane_provider
  ON estimate_versions(lane_id, provider_name, created_at);

-- Performance helpers
CREATE INDEX IF NOT EXISTS idx_task_estimates_project_status
  ON task_estimates(project_id, status, order_index);

-- === PostgreSQL Version (Alternative) ===
-- Uncomment below for PostgreSQL deployment
-- Replace TEXT with JSONB for JSON columns
-- Replace CURRENT_TIMESTAMP with NOW()
-- Replace INTEGER PRIMARY KEY AUTOINCREMENT with BIGSERIAL PRIMARY KEY

/*
-- PostgreSQL Version:

ALTER TABLE project_runs
  ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'baseline',
  ADD COLUMN rehearsal_pct REAL,
  ADD COLUMN provider_matrix_json JSONB,
  ADD COLUMN capability_snapshot JSONB,
  ADD COLUMN risk_score REAL;

CREATE INDEX IF NOT EXISTS idx_project_runs_kind_time
  ON project_runs (project_id, run_kind, started_at);

ALTER TABLE task_estimates
  ADD COLUMN kpi_formula JSONB;

CREATE TABLE IF NOT EXISTS lane_overrides (
  id BIGSERIAL PRIMARY KEY,
  task_estimate_id BIGINT NOT NULL,
  task_description TEXT,
  predicted_lane INTEGER,
  corrected_lane INTEGER,
  corrected_by TEXT,
  corrected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_lane_override_task
    FOREIGN KEY (task_estimate_id) REFERENCES task_estimates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lane_overrides_task ON lane_overrides(task_estimate_id);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='action_logs' AND column_name='project_id'
  )
  THEN
    ALTER TABLE action_logs ADD COLUMN project_id BIGINT;
    CREATE INDEX IF NOT EXISTS idx_action_logs_project ON action_logs(project_id);
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS estimate_versions (
  id BIGSERIAL PRIMARY KEY,
  project_id BIGINT,
  lane_id INTEGER,
  provider_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  priors_hash TEXT,
  tokens_mean REAL,
  tokens_stddev REAL,
  duration_ms_mean REAL,
  duration_ms_stddev REAL,
  cost_usd_mean REAL,
  cost_usd_stddev REAL,
  n_observations INTEGER DEFAULT 0,
  mean_absolute_error_tokens REAL,
  mean_absolute_error_duration_ms REAL,
  CONSTRAINT fk_estimate_versions_project
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_estimate_versions_lane_provider
  ON estimate_versions(lane_id, provider_name, created_at);

CREATE INDEX IF NOT EXISTS idx_task_estimates_project_status
  ON task_estimates(project_id, status, order_index);
*/
