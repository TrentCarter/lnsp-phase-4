#!/usr/bin/env python3
"""
Test Agent Chat Visualization - Create Sample Conversation Data

This script creates sample agent chat conversation data to test the HMI
Sequencer visualization. It creates a realistic conversation thread between
Architect and Dir-Code with questions, answers, and status updates.

Usage:
    ./.venv/bin/python tools/test_agent_chat_visualization.py
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for importing agent_chat
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'common'))
from agent_chat import AgentChatClient


async def create_sample_conversation():
    """Create a sample agent chat conversation for visualization testing"""

    client = AgentChatClient()

    # Create a test run ID (would normally come from Prime Directive)
    test_run_id = "test-run-agent-chat-viz-001"

    print(f"\n🧪 Creating sample agent chat conversation for run: {test_run_id}\n")

    # Step 1: Architect delegates task to Dir-Code
    thread = await client.create_thread(
        run_id=test_run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor the authentication system to use OAuth2 instead of JWT. Ensure backward compatibility.",
        metadata={
            "budget": 15000,
            "policy": "standard",
            "entry_files": ["src/auth.py", "src/oauth.py", "src/sessions.py"]
        }
    )

    print(f"✅ Created thread: {thread.thread_id}")
    print(f"   - Delegation: Architect → Dir-Code")

    await asyncio.sleep(0.2)  # Simulate processing time

    # Step 2: Dir-Code asks clarifying question (blocking urgency)
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Which OAuth2 library should I use - authlib or python-oauth2? I see both in requirements.txt.",
        metadata={
            "urgency": "blocking",
            "reasoning": "Cannot proceed with implementation without library decision",
            "tool_calls": [{"name": "ask_parent", "args": {"question": "Which OAuth2 library...", "urgency": "blocking"}}]
        }
    )

    print(f"❓ Question sent: Dir-Code → Architect (blocking)")

    await asyncio.sleep(0.3)  # Simulate Architect thinking

    # Step 3: Architect answers the question
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use authlib - it's better maintained and has more comprehensive OAuth2/OIDC support. Python-oauth2 is deprecated.",
        metadata={
            "sources": ["requirements.txt", "PRD"],
            "confidence": "high"
        }
    )

    print(f"💡 Answer sent: Architect → Dir-Code")

    await asyncio.sleep(0.4)  # Simulate Dir-Code processing

    # Step 4: Dir-Code sends status update (30% progress)
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Decomposing task into subtasks - delegating to Mgr-Code-Core",
        metadata={
            "progress": 30,
            "subtasks": 3,
            "managers_delegated": ["Mgr-Code-Core"]
        }
    )

    print(f"📊 Status update: Dir-Code → Architect (30% progress)")

    await asyncio.sleep(0.3)  # Simulate more work

    # Step 5: Dir-Code asks another question (important but not blocking)
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I migrate the session storage to Redis at the same time, or keep SQLite sessions for now?",
        metadata={
            "urgency": "important",
            "reasoning": "Session storage migration would improve performance but adds scope",
            "tool_calls": [{"name": "ask_parent", "args": {"question": "Should I migrate...", "urgency": "important"}}]
        }
    )

    print(f"❓ Question sent: Dir-Code → Architect (important)")

    await asyncio.sleep(0.2)  # Simulate Architect thinking

    # Step 6: Architect answers (keep scope focused)
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Keep SQLite sessions for now - let's focus on OAuth2 migration. Redis can be a separate task if needed.",
        metadata={
            "sources": ["PRD", "budget constraints"],
            "confidence": "high",
            "reasoning": "Avoid scope creep to stay within budget"
        }
    )

    print(f"💡 Answer sent: Architect → Dir-Code")

    await asyncio.sleep(0.5)  # Simulate final work

    # Step 7: Dir-Code sends completion message
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Successfully refactored authentication to OAuth2 using authlib. All tests passing. Backward compatibility maintained via adapter pattern.",
        metadata={
            "tests_passed": 47,
            "files_modified": 8,
            "budget_used": 12000,
            "managers_involved": ["Mgr-Code-Core", "Mgr-Test"]
        }
    )

    print(f"✅ Completion sent: Dir-Code → Architect")

    await asyncio.sleep(0.1)

    # Step 8: Close the thread successfully
    await client.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="OAuth2 authentication system successfully implemented with authlib. All tests passing, backward compatibility maintained."
    )

    print(f"🎉 Thread closed: status=completed\n")

    # Show stats
    stats = client.get_stats()
    print(f"📊 Database Stats:")
    print(f"   - Total threads: {stats['total_threads']}")
    print(f"   - Total messages: {stats['total_messages']}")
    print(f"   - Avg messages/thread: {stats['avg_messages_per_thread']}")
    print(f"   - Threads by status: {stats['threads_by_status']}\n")

    print(f"✨ Sample conversation created successfully!")
    print(f"\n🌐 To view in HMI:")
    print(f"   1. Ensure HMI is running: http://localhost:6101")
    print(f"   2. Navigate to Sequencer view")
    print(f"   3. Select task/run: {test_run_id}")
    print(f"   4. You should see:")
    print(f"      - 💬 Blue delegation message")
    print(f"      - 🔴 ❓ Amber blocking question")
    print(f"      - 💡 Green answers")
    print(f"      - 🟡 ❓ Yellow important question")
    print(f"      - 📊 Gray status update")
    print(f"      - ✅ Green completion message")
    print(f"\n   5. Hover over messages to see metadata tooltips")
    print(f"   6. Check urgency indicators (🔴 blocking, 🟡 important)")


if __name__ == "__main__":
    asyncio.run(create_sample_conversation())
