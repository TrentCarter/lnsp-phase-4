#!/usr/bin/env python3
"""
Create Test Action Logs for Agent Chat Visualization

This script creates minimal action log entries so that the HMI Sequencer
can find the test run created by test_agent_chat_visualization.py.

Usage:
    ./.venv/bin/python tools/create_test_action_logs.py
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import uuid

# Database path
DB_PATH = Path("artifacts/registry/registry.db")

def create_test_action_logs():
    """Create minimal action logs for test run"""

    test_run_id = "test-run-agent-chat-viz-001"
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Create Gateway → PAS Root action (root of the task tree)
        cursor.execute("""
            INSERT INTO action_logs
            (task_id, timestamp, from_agent, to_agent, action_type, action_name,
             tier_from, tier_to, status, parent_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_run_id,
            now,
            "Gateway",
            "PAS_ROOT",
            "delegate",
            "Submit Prime Directive: Test Agent Chat Visualization",
            0,
            1,
            "completed",
            None
        ))
        root_log_id = cursor.lastrowid

        # Create PAS Root → Architect action
        cursor.execute("""
            INSERT INTO action_logs
            (task_id, timestamp, from_agent, to_agent, action_type, action_name,
             tier_from, tier_to, status, parent_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_run_id,
            now,
            "PAS_ROOT",
            "Architect",
            "delegate",
            "Analyze requirements and delegate to directors",
            1,
            2,
            "completed",
            root_log_id
        ))
        architect_log_id = cursor.lastrowid

        # Create Architect → Dir-Code action
        cursor.execute("""
            INSERT INTO action_logs
            (task_id, timestamp, from_agent, to_agent, action_type, action_name,
             tier_from, tier_to, status, parent_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_run_id,
            now,
            "Architect",
            "Dir-Code",
            "delegate",
            "Refactor authentication to OAuth2",
            2,
            3,
            "completed",
            architect_log_id
        ))
        dir_code_log_id = cursor.lastrowid

        # Create Dir-Code → Architect report
        cursor.execute("""
            INSERT INTO action_logs
            (task_id, timestamp, from_agent, to_agent, action_type, action_name,
             tier_from, tier_to, status, parent_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_run_id,
            now,
            "Dir-Code",
            "Architect",
            "report",
            "OAuth2 authentication system completed",
            3,
            2,
            "completed",
            dir_code_log_id
        ))

        conn.commit()
        print(f"✅ Created action logs for test run: {test_run_id}")
        print(f"\n📊 Action log entries:")
        print(f"   - Gateway → PAS_ROOT")
        print(f"   - PAS_ROOT → Architect")
        print(f"   - Architect → Dir-Code")
        print(f"   - Dir-Code → Architect (report)")

        # Show count
        cursor.execute("SELECT COUNT(*) FROM action_logs WHERE task_id = ?", (test_run_id,))
        count = cursor.fetchone()[0]
        print(f"\n✓ Total action logs for {test_run_id}: {count}")

    finally:
        conn.close()

    print(f"\n🌐 Now you can:")
    print(f"   1. Open http://localhost:6101/sequencer")
    print(f"   2. Select task: 'Test Agent Chat Visualization' from dropdown")
    print(f"   3. You should see agent chat messages in the timeline!")


if __name__ == "__main__":
    create_test_action_logs()
