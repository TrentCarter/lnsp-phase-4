#!/usr/bin/env python3
"""
Seed action_logs table with sample data for testing the Actions view
Demonstrates hierarchical task flow: VP_ENG → Dir_SW → SW-MGR_1 → Programmer_1 → work → responses back up
"""

import requests
import time
from datetime import datetime, timedelta

REGISTRY_URL = "http://localhost:6121"

def log_action(task_id, from_agent, to_agent, action_type, action_name, parent_log_id=None, status=None, tier_from=None, tier_to=None, action_data=None):
    """Log an action to the registry"""
    payload = {
        "task_id": task_id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "action_type": action_type,
        "action_name": action_name,
        "parent_log_id": parent_log_id,
        "status": status,
        "tier_from": tier_from,
        "tier_to": tier_to,
        "action_data": action_data
    }

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        response = requests.post(f"{REGISTRY_URL}/action_logs", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        print(f"✓ Logged: {from_agent} → {to_agent}: {action_name} (log_id: {result['log_id']})")
        return result['log_id']
    except Exception as e:
        print(f"✗ Error logging action: {e}")
        return None


def seed_sample_task_1():
    """
    Task 1: Implement Audio Playback Feature
    Flow: VP_ENG → Dir_SW → SW-MGR_1 → Programmer_1 → implementation → responses back up
    """
    task_id = "task_audio_playback_001"
    print(f"\n📋 Creating sample task: {task_id}")

    # 1. VP_ENG initiates task to Dir_SW
    log_1 = log_action(
        task_id=task_id,
        from_agent="VP_ENG",
        to_agent="Dir_SW",
        action_type="command",
        action_name="Implement audio playback feature",
        status="running",
        tier_from=0,
        tier_to=1,
        action_data={
            "priority": "high",
            "deadline": "2025-11-10",
            "description": "Add audio feedback when actions complete"
        }
    )

    time.sleep(0.1)

    # 2. Dir_SW delegates to SW-MGR_1
    log_2 = log_action(
        task_id=task_id,
        from_agent="Dir_SW",
        to_agent="SW-MGR_1",
        action_type="command",
        action_name="Assign audio feature implementation",
        parent_log_id=log_1,
        status="running",
        tier_from=1,
        tier_to=2,
        action_data={
            "team": "frontend",
            "estimated_tokens": 25000,
            "estimated_task_points": 8
        }
    )

    time.sleep(0.1)

    # 3. SW-MGR_1 assigns to Programmer_1
    log_3 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_1",
        to_agent="Programmer_1",
        action_type="command",
        action_name="Write function to play audio when action is complete",
        parent_log_id=log_2,
        status="running",
        tier_from=2,
        tier_to=3,
        action_data={
            "files": ["audio_service.py", "utils/audio.py"],
            "requirements": ["Support TTS", "Support MIDI notes", "Volume control"]
        }
    )

    time.sleep(0.1)

    # 4. Programmer_1 starts implementation
    log_4 = log_action(
        task_id=task_id,
        from_agent="Programmer_1",
        to_agent="Programmer_1",
        action_type="status_update",
        action_name="Started implementation - audio service setup",
        parent_log_id=log_3,
        status="running",
        tier_from=3,
        tier_to=3,
        action_data={
            "progress": 0.2,
            "current_step": "Setting up FastAPI service",
            "tokens_used": 3200
        }
    )

    time.sleep(0.1)

    # 5. Programmer_1 continues work
    log_5 = log_action(
        task_id=task_id,
        from_agent="Programmer_1",
        to_agent="Programmer_1",
        action_type="status_update",
        action_name="Implementing TTS and MIDI endpoints",
        parent_log_id=log_4,
        status="running",
        tier_from=3,
        tier_to=3,
        action_data={
            "progress": 0.6,
            "current_step": "Adding TTS integration with f5_tts_mlx",
            "tokens_used": 8500
        }
    )

    time.sleep(0.1)

    # 6. Programmer_1 completes and reports to SW-MGR_1
    log_6 = log_action(
        task_id=task_id,
        from_agent="Programmer_1",
        to_agent="SW-MGR_1",
        action_type="response",
        action_name="Implementation completed",
        parent_log_id=log_3,
        status="completed",
        tier_from=3,
        tier_to=2,
        action_data={
            "progress": 1.0,
            "files_modified": ["services/audio/audio_service.py", "scripts/start_audio_service.sh"],
            "tests_passed": True,
            "endpoints": ["/speak", "/play_note", "/play_tone"],
            "tokens_used": 18750,
            "task_duration": "2.5 hours"
        }
    )

    time.sleep(0.1)

    # 7. SW-MGR_1 reviews and reports to Dir_SW
    log_7 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_1",
        to_agent="Dir_SW",
        action_type="response",
        action_name="Status Update: Feature completed and tested",
        parent_log_id=log_2,
        status="completed",
        tier_from=2,
        tier_to=1,
        action_data={
            "code_review": "approved",
            "tests_status": "all passing",
            "deployment_ready": True,
            "tokens_used": 22400,
            "task_duration": "3.2 hours"
        }
    )

    time.sleep(0.1)

    # 8. Dir_SW reports to VP_ENG
    log_8 = log_action(
        task_id=task_id,
        from_agent="Dir_SW",
        to_agent="VP_ENG",
        action_type="response",
        action_name="Status Update: Task Complete",
        parent_log_id=log_1,
        status="completed",
        tier_from=1,
        tier_to=0,
        action_data={
            "completion_time": "2025-11-07T14:30:00Z",
            "quality": "high",
            "on_schedule": True,
            "tokens_used": 23150,
            "task_duration": "3.5 hours",
            "total_cost_usd": 0.347
        }
    )

    print(f"✅ Task {task_id} completed with 8 actions\n")


def seed_sample_task_2():
    """
    Task 2: Bug Fix - Tree View Auto-Refresh
    Simpler flow with error handling
    """
    task_id = "task_bugfix_tree_refresh_002"
    print(f"\n📋 Creating sample task: {task_id}")

    # 1. VP_ENG initiates bug fix
    log_1 = log_action(
        task_id=task_id,
        from_agent="VP_ENG",
        to_agent="Dir_SW",
        action_type="command",
        action_name="Fix tree view auto-refresh bug",
        status="running",
        tier_from=0,
        tier_to=1,
        action_data={
            "priority": "medium",
            "bug_report": "Tree view not respecting auto-refresh setting"
        }
    )

    time.sleep(0.1)

    # 2. Dir_SW assigns to SW-MGR_1
    log_2 = log_action(
        task_id=task_id,
        from_agent="Dir_SW",
        to_agent="SW-MGR_1",
        action_type="command",
        action_name="Debug and fix auto-refresh issue",
        parent_log_id=log_1,
        status="running",
        tier_from=1,
        tier_to=2,
        action_data={
            "estimated_tokens": 5000,
            "estimated_task_points": 2
        }
    )

    time.sleep(0.1)

    # 3. SW-MGR_1 investigates and finds root cause
    log_3 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_1",
        to_agent="SW-MGR_1",
        action_type="status_update",
        action_name="Root cause identified - WebSocket event handling",
        parent_log_id=log_2,
        status="running",
        tier_from=2,
        tier_to=2,
        action_data={
            "root_cause": "WebSocket events not checking settings before refreshing",
            "fix_approach": "Add settings check in event handler",
            "tokens_used": 2100
        }
    )

    time.sleep(0.1)

    # 4. SW-MGR_1 implements fix
    log_4 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_1",
        to_agent="SW-MGR_1",
        action_type="status_update",
        action_name="Fix implemented and tested",
        parent_log_id=log_3,
        status="completed",
        tier_from=2,
        tier_to=2,
        action_data={
            "files_modified": ["templates/tree.html"],
            "lines_changed": 3,
            "tests_passed": True,
            "tokens_used": 4200,
            "task_duration": "0.8 hours"
        }
    )

    time.sleep(0.1)

    # 5. SW-MGR_1 reports completion to Dir_SW
    log_5 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_1",
        to_agent="Dir_SW",
        action_type="response",
        action_name="Bug fix completed",
        parent_log_id=log_2,
        status="completed",
        tier_from=2,
        tier_to=1,
        action_data={
            "tokens_used": 4350,
            "task_duration": "0.9 hours"
        }
    )

    time.sleep(0.1)

    # 6. Dir_SW reports to VP_ENG
    log_6 = log_action(
        task_id=task_id,
        from_agent="Dir_SW",
        to_agent="VP_ENG",
        action_type="response",
        action_name="Bug fix deployed successfully",
        parent_log_id=log_1,
        status="completed",
        tier_from=1,
        tier_to=0,
        action_data={
            "tokens_used": 4500,
            "task_duration": "1.0 hours",
            "total_cost_usd": 0.068
        }
    )

    print(f"✅ Task {task_id} completed with 6 actions\n")


def seed_sample_task_3():
    """
    Task 3: In-Progress Task with Blocker
    """
    task_id = "task_cost_dashboard_003"
    print(f"\n📋 Creating sample task: {task_id}")

    # 1. VP_ENG initiates feature request
    log_1 = log_action(
        task_id=task_id,
        from_agent="VP_ENG",
        to_agent="Dir_SW",
        action_type="command",
        action_name="Enhance cost dashboard with detailed breakdown",
        status="running",
        tier_from=0,
        tier_to=1,
        action_data={
            "priority": "high",
            "requirements": ["Per-service costs", "Budget alerts", "Historical trends"]
        }
    )

    time.sleep(0.1)

    # 2. Dir_SW delegates to SW-MGR_2
    log_2 = log_action(
        task_id=task_id,
        from_agent="Dir_SW",
        to_agent="SW-MGR_2",
        action_type="command",
        action_name="Implement cost dashboard enhancements",
        parent_log_id=log_1,
        status="running",
        tier_from=1,
        tier_to=2,
        action_data={
            "estimated_tokens": 35000,
            "estimated_task_points": 13
        }
    )

    time.sleep(0.1)

    # 3. SW-MGR_2 assigns to Programmer_2
    log_3 = log_action(
        task_id=task_id,
        from_agent="SW-MGR_2",
        to_agent="Programmer_2",
        action_type="command",
        action_name="Design and implement cost breakdown API",
        parent_log_id=log_2,
        status="running",
        tier_from=2,
        tier_to=3,
        action_data={
            "estimated_tokens": 32000
        }
    )

    time.sleep(0.1)

    # 4. Programmer_2 starts work
    log_4 = log_action(
        task_id=task_id,
        from_agent="Programmer_2",
        to_agent="Programmer_2",
        action_type="status_update",
        action_name="Started API design - gathering requirements",
        parent_log_id=log_3,
        status="running",
        tier_from=3,
        tier_to=3,
        action_data={
            "progress": 0.3,
            "current_step": "Database schema design",
            "tokens_used": 9800
        }
    )

    time.sleep(0.1)

    # 5. Programmer_2 encounters blocker
    log_5 = log_action(
        task_id=task_id,
        from_agent="Programmer_2",
        to_agent="SW-MGR_2",
        action_type="status_update",
        action_name="Blocked: Need access to cost tracking database",
        parent_log_id=log_3,
        status="blocked",
        tier_from=3,
        tier_to=2,
        action_data={
            "blocker": "Missing database credentials",
            "required_access": "Read access to cost_receipts table",
            "tokens_used": 11200,
            "task_duration": "1.5 hours (incomplete)"
        }
    )

    print(f"⚠️  Task {task_id} blocked with 5 actions\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Seeding Action Logs Database")
    print("=" * 60)

    try:
        # Check if registry is running
        response = requests.get(f"{REGISTRY_URL}/health", timeout=2)
        response.raise_for_status()
        print("✓ Registry service is running\n")
    except Exception as e:
        print(f"✗ Error: Registry service not available at {REGISTRY_URL}")
        print(f"  Please start the registry service first:")
        print(f"  python services/registry/registry_service.py")
        exit(1)

    # Seed sample tasks
    seed_sample_task_1()  # Complete task: Audio feature
    seed_sample_task_2()  # Complete task: Bug fix
    seed_sample_task_3()  # In-progress task with blocker

    print("=" * 60)
    print("✅ Sample data seeded successfully!")
    print("=" * 60)
    print("\nYou can now view the Actions log at:")
    print("  http://localhost:6101/actions")
    print()
