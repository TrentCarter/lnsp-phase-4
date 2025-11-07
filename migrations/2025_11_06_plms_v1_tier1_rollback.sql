-- PLMS V1 Tier 1 Rollback Migration
-- Date: 2025-11-06
-- Description: Rollback PLMS Tier 1 changes (use with extreme caution!)
-- WARNING: This will PERMANENTLY DELETE all PLMS Tier 1 data

-- ⚠️  CAUTION: Only run if you understand data loss risk!
-- This script removes:
-- - Multi-run support columns (run_kind, rehearsal_pct, provider snapshots)
-- - KPI receipts and lane overrides
-- - Bayesian estimate versions
-- - All related indices

-- === Step 1: Drop KPI Receipts (Patch 2) ===
DROP TABLE IF EXISTS kpi_receipts;

-- === Step 2: Drop Lane Overrides ===
DROP TABLE IF EXISTS lane_overrides;

-- === Step 3: Drop Estimate Versions ===
DROP TABLE IF EXISTS estimate_versions;

-- === Step 4: Drop project_runs indices ===
DROP INDEX IF EXISTS idx_project_runs_kind_time;

-- === Step 5: Drop action_logs indices ===
DROP INDEX IF EXISTS idx_action_logs_project;

-- === Step 6: Drop task_estimates indices ===
DROP INDEX IF EXISTS idx_task_estimates_project_status;

-- === Step 7: Remove columns from project_runs ===
-- SQLite: Cannot directly drop columns in older versions
-- Workaround: Create new table without unwanted columns, copy data, swap tables

-- Check SQLite version first
-- For SQLite 3.35+, you can use:
-- ALTER TABLE project_runs DROP COLUMN run_kind;
-- ALTER TABLE project_runs DROP COLUMN rehearsal_pct;
-- etc.

-- For older SQLite versions, use this approach:
BEGIN TRANSACTION;

-- Backup existing data (exclude Tier 1 columns)
CREATE TABLE project_runs_rollback AS
  SELECT
    id,
    project_id,
    run_id,
    started_at,
    completed_at,
    status
    -- NOTE: Add any other pre-Tier-1 columns here
  FROM project_runs;

-- Drop original table
DROP TABLE project_runs;

-- Rename backup to original name
ALTER TABLE project_runs_rollback RENAME TO project_runs;

-- Recreate original indices (if any existed before Tier 1)
-- Example:
-- CREATE INDEX idx_project_runs_project ON project_runs(project_id);

COMMIT;

-- === Step 8: Remove columns from task_estimates ===
BEGIN TRANSACTION;

-- Backup existing data (exclude kpi_formula)
CREATE TABLE task_estimates_rollback AS
  SELECT
    id,
    project_id,
    task_name,
    status,
    order_index
    -- NOTE: Add any other pre-Tier-1 columns here
  FROM task_estimates;

-- Drop original table
DROP TABLE task_estimates;

-- Rename backup to original name
ALTER TABLE task_estimates_rollback RENAME TO task_estimates;

-- Recreate original indices (if any existed before Tier 1)
-- Example:
-- CREATE INDEX idx_task_estimates_project ON task_estimates(project_id);

COMMIT;

-- === Step 9: Remove project_id column from action_logs (if added by Tier 1) ===
-- NOTE: Only run if action_logs.project_id was added by Tier 1 migration
-- If it existed before, skip this step or restore pre-Tier-1 schema

BEGIN TRANSACTION;

-- Backup existing data (exclude project_id if added by Tier 1)
CREATE TABLE action_logs_rollback AS
  SELECT
    id,
    action_type,
    timestamp,
    metadata
    -- NOTE: Add any other pre-Tier-1 columns here
  FROM action_logs;

-- Drop original table
DROP TABLE action_logs;

-- Rename backup to original name
ALTER TABLE action_logs_rollback RENAME TO action_logs;

-- Recreate original indices (if any existed before Tier 1)
-- Example:
-- CREATE INDEX idx_action_logs_timestamp ON action_logs(timestamp);

COMMIT;

-- === Step 10: Verification ===
-- Check that tables are back to pre-Tier-1 schema
.schema project_runs
.schema task_estimates
.schema action_logs

-- === PostgreSQL Version (Alternative) ===
-- Uncomment below for PostgreSQL deployment

/*
-- PostgreSQL Version:

-- Step 1: Drop tables
DROP TABLE IF EXISTS kpi_receipts CASCADE;
DROP TABLE IF EXISTS lane_overrides CASCADE;
DROP TABLE IF EXISTS estimate_versions CASCADE;

-- Step 2: Drop indices
DROP INDEX IF EXISTS idx_project_runs_kind_time;
DROP INDEX IF EXISTS idx_action_logs_project;
DROP INDEX IF EXISTS idx_task_estimates_project_status;

-- Step 3: Drop columns from project_runs
ALTER TABLE project_runs
  DROP COLUMN IF EXISTS run_kind,
  DROP COLUMN IF EXISTS rehearsal_pct,
  DROP COLUMN IF EXISTS provider_matrix_json,
  DROP COLUMN IF EXISTS capability_snapshot,
  DROP COLUMN IF EXISTS risk_score,
  DROP COLUMN IF EXISTS validation_pass,
  DROP COLUMN IF EXISTS write_sandbox;

-- Step 4: Drop columns from task_estimates
ALTER TABLE task_estimates
  DROP COLUMN IF EXISTS kpi_formula;

-- Step 5: Drop column from action_logs (only if added by Tier 1)
ALTER TABLE action_logs
  DROP COLUMN IF EXISTS project_id;

-- Step 6: Verification
\d project_runs
\d task_estimates
\d action_logs
*/

-- === Notes ===
-- 1. This rollback is IRREVERSIBLE - all Tier 1 data will be lost
-- 2. Test on a backup database first
-- 3. For SQLite 3.35+, you can simplify Step 7-9 using DROP COLUMN
-- 4. Adjust column lists in backup tables to match your pre-Tier-1 schema
-- 5. If you have foreign key constraints, you may need to disable them first:
--    PRAGMA foreign_keys=OFF; (SQLite)
--    SET CONSTRAINTS ALL DEFERRED; (PostgreSQL)
