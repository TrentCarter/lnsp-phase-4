#!/usr/bin/env bash
# PLMS Invariants Checker Wrapper Script
# Runs nightly checks, sends alerts on failures

set -euo pipefail

# Detect script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load configuration
if [ -f "${PROJECT_ROOT}/config/plms_invariants.env" ]; then
    source "${PROJECT_ROOT}/config/plms_invariants.env"
else
    echo "Warning: config/plms_invariants.env not found, using defaults"
    DB_PATH="${DB_PATH:-${PROJECT_ROOT}/artifacts/registry/registry.db}"
    ENV_NAME="${ENV_NAME:-dev}"
fi

# Ensure log directory exists
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/plms}"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
OUT="$LOG_DIR/invariants_${TS}.log"

# Activate virtual environment if present
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

echo "=== PLMS Invariants Check @ $TS ===" | tee "$OUT"
echo "Environment: $ENV_NAME" | tee -a "$OUT"
echo "Database: $DB_PATH" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# Run checker with --strict mode
if python3 "${SCRIPT_DIR}/check_plms_invariants.py" --db "$DB_PATH" --strict >> "$OUT" 2>&1; then
    STATUS="OK"
    EXIT_CODE=0
else
    STATUS="FAIL"
    EXIT_CODE=1
fi

# Generate summary for alerts
SUMMARY=$(tail -n 50 "$OUT" | sed 's/"/\\"/g')
SUBJECT="[PLMS][$ENV_NAME] Invariants $STATUS @ $TS"

echo "" | tee -a "$OUT"
echo "Status: $STATUS" | tee -a "$OUT"
echo "Log: $OUT" | tee -a "$OUT"

# === Alert: Slack ===
if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    echo "Sending Slack alert..." | tee -a "$OUT"

    # Determine emoji and color based on status
    if [[ "$STATUS" == "OK" ]]; then
        EMOJI="✅"
        COLOR="good"
    else
        EMOJI="🔴"
        COLOR="danger"
    fi

    # Send Slack message with attachment
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"$EMOJI $SUBJECT\",
            \"attachments\": [{
                \"color\": \"$COLOR\",
                \"text\": \"\`\`\`\n$SUMMARY\n\`\`\`\",
                \"footer\": \"PLMS Invariants Checker\",
                \"ts\": $(date +%s)
            }]
        }" \
        "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || echo "Warning: Slack alert failed" | tee -a "$OUT"
fi

# === Alert: Email (only on FAIL) ===
if [[ "$STATUS" != "OK" ]] && [[ -n "${ALERT_EMAIL:-}" ]]; then
    echo "Sending email alert to $ALERT_EMAIL..." | tee -a "$OUT"

    # Use sendmail or mail command
    if command -v sendmail >/dev/null 2>&1; then
        {
            echo "To: $ALERT_EMAIL"
            echo "Subject: $SUBJECT"
            echo "Content-Type: text/plain; charset=UTF-8"
            echo ""
            cat "$OUT"
        } | /usr/sbin/sendmail -t || echo "Warning: Email alert failed" | tee -a "$OUT"
    elif command -v mail >/dev/null 2>&1; then
        cat "$OUT" | mail -s "$SUBJECT" "$ALERT_EMAIL" || echo "Warning: Email alert failed" | tee -a "$OUT"
    else
        echo "Warning: sendmail/mail not available, skipping email alert" | tee -a "$OUT"
    fi
fi

# === Alert: PagerDuty (only on FAIL) ===
if [[ "$STATUS" != "OK" ]] && [[ -n "${PD_ROUTING_KEY:-}" ]]; then
    echo "Triggering PagerDuty incident..." | tee -a "$OUT"

    if [ -f "${PROJECT_ROOT}/services/plms/alert_pd.py" ]; then
        python3 "${PROJECT_ROOT}/services/plms/alert_pd.py" "$SUBJECT" || echo "Warning: PagerDuty alert failed" | tee -a "$OUT"
    else
        echo "Warning: alert_pd.py not found, skipping PagerDuty alert" | tee -a "$OUT"
    fi
fi

# Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "invariants_*.log" -type f -mtime +30 -delete 2>/dev/null || true

echo ""
echo "$SUBJECT"
exit $EXIT_CODE
