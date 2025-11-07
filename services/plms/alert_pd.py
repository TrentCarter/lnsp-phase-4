"""
PagerDuty Alert Helper

Triggers PagerDuty incidents for critical PLMS failures.

Usage:
    python -m services.plms.alert_pd "Alert message here"

    Or as module:
    from services.plms.alert_pd import trigger
    trigger("PLMS invariants failed", severity="critical")
"""

import os
import json
import requests
import sys
from typing import Optional


def trigger(summary: str, severity: str = "error", source: str = "plms-invariants", custom_details: Optional[dict] = None):
    """
    Trigger a PagerDuty incident.

    Args:
        summary: Incident summary (e.g., "[PLMS][prod] Invariants FAIL @ 2025-11-06")
        severity: "critical" | "error" | "warning" | "info" (default: "error")
        source: Source identifier (default: "plms-invariants")
        custom_details: Additional context (optional)
    """
    routing_key = os.environ.get("PD_ROUTING_KEY", "")

    if not routing_key:
        print("Warning: PD_ROUTING_KEY not set, skipping PagerDuty alert", file=sys.stderr)
        return False

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "severity": severity,
            "source": source,
            "custom_details": custom_details or {}
        }
    }

    try:
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=5
        )

        if response.status_code == 202:
            print(f"✓ PagerDuty incident triggered: {summary}", file=sys.stderr)
            return True
        else:
            print(f"Warning: PagerDuty API returned {response.status_code}: {response.text}", file=sys.stderr)
            return False

    except requests.exceptions.RequestException as e:
        print(f"Warning: PagerDuty alert failed: {e}", file=sys.stderr)
        return False


def resolve(dedup_key: str):
    """
    Resolve a PagerDuty incident.

    Args:
        dedup_key: Deduplication key from previous trigger response
    """
    routing_key = os.environ.get("PD_ROUTING_KEY", "")

    if not routing_key:
        print("Warning: PD_ROUTING_KEY not set, cannot resolve incident", file=sys.stderr)
        return False

    payload = {
        "routing_key": routing_key,
        "event_action": "resolve",
        "dedup_key": dedup_key
    }

    try:
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=5
        )

        if response.status_code == 202:
            print(f"✓ PagerDuty incident resolved: {dedup_key}", file=sys.stderr)
            return True
        else:
            print(f"Warning: PagerDuty resolve API returned {response.status_code}", file=sys.stderr)
            return False

    except requests.exceptions.RequestException as e:
        print(f"Warning: PagerDuty resolve failed: {e}", file=sys.stderr)
        return False


def main():
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python -m services.plms.alert_pd \"Alert message\"", file=sys.stderr)
        print("       python -m services.plms.alert_pd \"Alert message\" critical", file=sys.stderr)
        sys.exit(1)

    summary = sys.argv[1]
    severity = sys.argv[2] if len(sys.argv) > 2 else "error"

    success = trigger(summary, severity=severity)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
