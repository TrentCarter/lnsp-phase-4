#!/usr/bin/env python3
"""
Test script for Tree View real-time updates via SSE.

This script simulates a live project by inserting action_log entries into the
Registry database at intervals, allowing you to observe real-time tree updates
in the browser.

Usage:
    python3 tests/test_tree_realtime_updates.py

Then open the Tree View in your browser:
    http://localhost:6101/tree?task_id=test-realtime-001
"""

import sqlite3
import time
import os
import json
from datetime import datetime

# Path to Registry database
DB_PATH = os.path.join(
    os.path.dirname(__file__),
    '../artifacts/registry/registry.db'
)

def insert_action_log(task_id, action_type, action_name, from_agent, to_agent, tier_from, tier_to, status='running'):
    """Insert a new action_log entry"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO action_logs
        (task_id, action_type, action_name, from_agent, to_agent, tier_from, tier_to, status, timestamp, action_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task_id,
        action_type,
        action_name,
        from_agent,
        to_agent,
        tier_from,
        tier_to,
        status,
        datetime.now().isoformat(),
        json.dumps({'test': True})
    ))

    conn.commit()
    log_id = cursor.lastrowid
    conn.close()

    print(f"[{datetime.now().isoformat()}] Inserted action_log #{log_id}: {action_name} ({from_agent} -> {to_agent})")
    return log_id

def main():
    """Run the test simulation"""
    print("=" * 80)
    print("Tree View Real-time Updates Test")
    print("=" * 80)
    print()
    print("This script will insert action_log entries every 3 seconds.")
    print("Open the Tree View in your browser to see real-time updates:")
    print()
    print("    http://localhost:6101/tree?task_id=test-realtime-001")
    print()
    print("Press Ctrl+C to stop the simulation.")
    print("=" * 80)
    print()

    task_id = "test-realtime-001"

    # Scenario: User delegates to VP, VP delegates to Director, Director delegates to Manager, Manager delegates to Programmer
    scenarios = [
        # Step 1: User -> VP
        {
            'action_type': 'delegate',
            'action_name': 'User requests feature implementation',
            'from_agent': 'user',
            'to_agent': 'vp_001',
            'tier_from': 0,
            'tier_to': 1,
            'status': 'running'
        },
        # Step 2: VP -> Director
        {
            'action_type': 'delegate',
            'action_name': 'VP delegates to Director',
            'from_agent': 'vp_001',
            'to_agent': 'director_code',
            'tier_from': 1,
            'tier_to': 2,
            'status': 'running'
        },
        # Step 3: Director -> Manager
        {
            'action_type': 'delegate',
            'action_name': 'Director assigns to Manager',
            'from_agent': 'director_code',
            'to_agent': 'manager_backend',
            'tier_from': 2,
            'tier_to': 3,
            'status': 'running'
        },
        # Step 4: Manager -> Programmer
        {
            'action_type': 'delegate',
            'action_name': 'Manager assigns task to Programmer',
            'from_agent': 'manager_backend',
            'to_agent': 'programmer_001',
            'tier_from': 3,
            'tier_to': 4,
            'status': 'running'
        },
        # Step 5: Programmer starts work
        {
            'action_type': 'code_generation',
            'action_name': 'Programmer implements feature',
            'from_agent': 'programmer_001',
            'to_agent': None,
            'tier_from': 4,
            'tier_to': None,
            'status': 'running'
        },
        # Step 6: Programmer completes work
        {
            'action_type': 'code_generation',
            'action_name': 'Programmer completes feature',
            'from_agent': 'programmer_001',
            'to_agent': 'manager_backend',
            'tier_from': 4,
            'tier_to': 3,
            'status': 'completed'
        },
        # Step 7: Manager reviews
        {
            'action_type': 'code_review',
            'action_name': 'Manager reviews code',
            'from_agent': 'manager_backend',
            'to_agent': 'director_code',
            'tier_from': 3,
            'tier_to': 2,
            'status': 'completed'
        },
        # Step 8: Director approves
        {
            'action_type': 'approval',
            'action_name': 'Director approves feature',
            'from_agent': 'director_code',
            'to_agent': 'vp_001',
            'tier_from': 2,
            'tier_to': 1,
            'status': 'completed'
        },
        # Step 9: VP reports to user
        {
            'action_type': 'report',
            'action_name': 'VP reports completion to user',
            'from_agent': 'vp_001',
            'to_agent': 'user',
            'tier_from': 1,
            'tier_to': 0,
            'status': 'completed'
        }
    ]

    try:
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n--- Step {i}/{len(scenarios)} ---")
            insert_action_log(
                task_id=task_id,
                action_type=scenario['action_type'],
                action_name=scenario['action_name'],
                from_agent=scenario['from_agent'],
                to_agent=scenario['to_agent'],
                tier_from=scenario['tier_from'],
                tier_to=scenario['tier_to'],
                status=scenario['status']
            )
            time.sleep(3)  # Wait 3 seconds between actions

        print("\n" + "=" * 80)
        print("Test complete! All actions inserted.")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
        print("=" * 80)

if __name__ == '__main__':
    main()
