#!/bin/bash
# Replay a previous PAS run from its saved passport
# Disaster recovery utility - restores git state, env, artifacts, and resubmits to PAS

set -euo pipefail

# Configuration
REGISTRY_DB="${REGISTRY_DB:-artifacts/registry/registry.db}"
PAS_API="${PAS_API:-http://localhost:6100}"
BACKUP_DIR="${BACKUP_DIR:-artifacts/backups}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 <RUN_ID> [OPTIONS]

Replay a previous PAS run from its saved passport.

Arguments:
  RUN_ID              The run ID to replay

Options:
  --dry-run           Show what would be done without executing
  --skip-git          Don't restore git state
  --skip-artifacts    Don't restore artifacts from backup
  --skip-env          Don't restore environment variables
  --help              Show this help message

Environment Variables:
  REGISTRY_DB         Path to registry database (default: artifacts/registry/registry.db)
  PAS_API             PAS API base URL (default: http://localhost:6100)
  BACKUP_DIR          Backup directory (default: artifacts/backups)

Examples:
  # Replay run abc123
  $0 abc123

  # Dry run to see what would happen
  $0 abc123 --dry-run

  # Replay without restoring git state
  $0 abc123 --skip-git
EOF
    exit 1
}

log() {
    echo -e "${GREEN}[REPLAY]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

# Parse arguments
RUN_ID=""
DRY_RUN=false
SKIP_GIT=false
SKIP_ARTIFACTS=false
SKIP_ENV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-git)
            SKIP_GIT=true
            shift
            ;;
        --skip-artifacts)
            SKIP_ARTIFACTS=true
            shift
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [ -z "$RUN_ID" ]; then
                RUN_ID="$1"
            else
                error "Unknown argument: $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$RUN_ID" ]; then
    error "Missing required argument: RUN_ID"
fi

log "Replaying run: $RUN_ID"

# Check prerequisites
if ! command -v jq &> /dev/null; then
    error "jq is required but not installed. Install with: brew install jq"
fi

if ! command -v sqlite3 &> /dev/null; then
    error "sqlite3 is required but not installed."
fi

if [ ! -f "$REGISTRY_DB" ]; then
    error "Registry database not found: $REGISTRY_DB"
fi

# Fetch passport from registry
log "Fetching passport from registry..."
PASSPORT_JSON=$(sqlite3 "$REGISTRY_DB" \
    "SELECT passport_json FROM project_runs WHERE run_id='$RUN_ID';" 2>/dev/null || echo "")

if [ -z "$PASSPORT_JSON" ]; then
    error "Run ID not found in registry: $RUN_ID"
fi

# Parse passport fields
GIT_COMMIT=$(echo "$PASSPORT_JSON" | jq -r '.git_commit // empty')
PRD_SHA=$(echo "$PASSPORT_JSON" | jq -r '.prd_sha256 // empty')
ENV_SNAPSHOT=$(echo "$PASSPORT_JSON" | jq -r '.env_snapshot // {}')
POLICY_VERSION=$(echo "$PASSPORT_JSON" | jq -r '.allowlist_policy_version // empty')
ARTIFACT_MANIFEST=$(echo "$PASSPORT_JSON" | jq -r '.artifact_manifest[]? // empty')

log "Passport details:"
log "  Git Commit: ${GIT_COMMIT:-<missing>}"
log "  PRD SHA256: ${PRD_SHA:-<missing>}"
log "  Policy Version: ${POLICY_VERSION:-<missing>}"
log "  Artifacts: $(echo "$ARTIFACT_MANIFEST" | wc -l | tr -d ' ') files"

if [ "$DRY_RUN" = true ]; then
    warn "DRY RUN - no changes will be made"
    log "Would restore:"
    [ "$SKIP_GIT" = false ] && log "  - Git state to commit $GIT_COMMIT"
    [ "$SKIP_ENV" = false ] && log "  - Environment variables"
    [ "$SKIP_ARTIFACTS" = false ] && log "  - Artifacts from backup"
    log "  - Resubmit to PAS with idempotency key: replay-$RUN_ID"
    exit 0
fi

# Restore git state
if [ "$SKIP_GIT" = false ] && [ -n "$GIT_COMMIT" ]; then
    log "Restoring git state to commit $GIT_COMMIT..."

    # Check if we have uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        warn "You have uncommitted changes. Stashing them..."
        git stash push -m "Auto-stash before replay of $RUN_ID"
    fi

    # Checkout the commit
    if git checkout "$GIT_COMMIT" 2>/dev/null; then
        log "✓ Git state restored"
    else
        error "Failed to checkout commit $GIT_COMMIT"
    fi
else
    [ "$SKIP_GIT" = true ] && log "Skipping git restore (--skip-git)"
    [ -z "$GIT_COMMIT" ] && warn "No git commit in passport - skipping git restore"
fi

# Restore environment snapshot
if [ "$SKIP_ENV" = false ]; then
    log "Restoring environment variables..."
    ENV_FILE=".env.replay.$RUN_ID"

    echo "$ENV_SNAPSHOT" | jq -r 'to_entries[] | "\(.key)=\(.value)"' > "$ENV_FILE"

    if [ -s "$ENV_FILE" ]; then
        log "✓ Environment variables written to $ENV_FILE"
        log "  To use: source $ENV_FILE"
    else
        warn "No environment variables in passport"
        rm -f "$ENV_FILE"
    fi
else
    log "Skipping environment restore (--skip-env)"
fi

# Restore artifacts from backup
if [ "$SKIP_ARTIFACTS" = false ] && [ -n "$ARTIFACT_MANIFEST" ]; then
    log "Restoring artifacts from backup..."

    RUN_BACKUP_DIR="$BACKUP_DIR/$RUN_ID"
    if [ ! -d "$RUN_BACKUP_DIR" ]; then
        warn "Backup directory not found: $RUN_BACKUP_DIR"
        warn "Artifacts will not be restored"
    else
        RESTORED=0
        while IFS= read -r artifact; do
            [ -z "$artifact" ] && continue

            SRC="$RUN_BACKUP_DIR/$artifact"
            DST="$artifact"

            if [ -f "$SRC" ]; then
                mkdir -p "$(dirname "$DST")"
                cp "$SRC" "$DST"
                log "  ✓ Restored: $artifact"
                RESTORED=$((RESTORED + 1))
            else
                warn "  ✗ Not found in backup: $artifact"
            fi
        done <<< "$ARTIFACT_MANIFEST"

        log "✓ Restored $RESTORED artifacts"
    fi
else
    [ "$SKIP_ARTIFACTS" = true ] && log "Skipping artifact restore (--skip-artifacts)"
    [ -z "$ARTIFACT_MANIFEST" ] && log "No artifacts in passport - skipping restore"
fi

# Resubmit to PAS
log "Resubmitting to PAS..."

IDEMPOTENCY_KEY="replay-$RUN_ID"
RESPONSE=$(curl -s -X POST "$PAS_API/pas/v1/runs/start" \
    -H "Content-Type: application/json" \
    -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
    -H "Idempotent-Replay: true" \
    -d "$PASSPORT_JSON" \
    2>&1 || echo '{"error": "Request failed"}')

# Check response
if echo "$RESPONSE" | jq -e '.run_id' > /dev/null 2>&1; then
    NEW_RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')
    STATUS=$(echo "$RESPONSE" | jq -r '.status // "unknown"')

    log "✅ Replay submitted successfully"
    log "  Original Run ID: $RUN_ID"
    log "  New Run ID: $NEW_RUN_ID"
    log "  Status: $STATUS"
    log "  Idempotency Key: $IDEMPOTENCY_KEY"

    # Check if this was a cached response
    if echo "$RESPONSE" | jq -e '.cached' > /dev/null 2>&1; then
        warn "Response was cached (run already in progress or completed)"
    fi

    exit 0
else
    ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error // "Unknown error"')
    error "Failed to submit replay: $ERROR_MSG"
fi
