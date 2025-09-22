#!/usr/bin/env bash
set -euo pipefail
DB_NAME=${DB_NAME:-lnsp}
DB_USER=${DB_USER:-lnsp}
DB_PASS=${DB_PASS:-lnsp}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}

if command -v docker >/dev/null 2>&1; then
  echo "[init_pg] docker detected; please use bootstrap_all.sh docker path."
  exit 0
fi

echo "[init_pg] NO_DOCKER mode"
export PGPASSWORD=${DB_PASS}
psql -h "$DB_HOST" -U "$DB_USER" -tc "SELECT 1" postgres >/dev/null 2>&1 || {
  echo "[init_pg] Creating role/user $DB_USER"
  createuser -h "$DB_HOST" -s "$DB_USER" || true
}
createdb -h "$DB_HOST" -O "$DB_USER" "$DB_NAME" || true
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f scripts/init_pg.sql
echo "[init_pg] done"
