#!/usr/bin/env python3
"""
End-to-End test for Parent-Child Agent Chat

Tests the complete flow:
1. Architect creates thread for complex task
2. Architect delegates to Dir-Code via agent chat
3. Dir-Code receives message and processes
4. Dir-Code asks question to Architect
5. Architect answers question (via monitor loop)
6. Dir-Code continues execution
7. Dir-Code sends status updates
8. Dir-Code completes and closes thread

This test simulates the real production flow without running actual services.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.common.agent_chat import AgentChatClient


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield Path(path)
    Path(path).unlink()


@pytest.fixture
def agent_chat(temp_db):
    """Get agent chat client with temp database"""
    return AgentChatClient(db_path=temp_db)


async def simulate_architect_question_monitor(agent_chat, thread_id, max_iterations=10):
    """
    Simulate Architect's monitor_conversation_threads() loop

    Checks for pending questions and generates heuristic answers.
    This is what happens in production when Architect.monitor_directors() runs.
    """
    for _ in range(max_iterations):
        # Get pending questions for Architect
        questions = await agent_chat.get_pending_questions("Architect")

        for q in questions:
            if q["thread_id"] != thread_id:
                continue

            # Generate heuristic answer (simplified version of Architect's logic)
            question_text = q["content"].lower()

            if "library" in question_text or "authlib" in question_text:
                answer = "Use authlib - it's more actively maintained and supports OAuth2/OIDC"
            elif "file" in question_text or "focus" in question_text:
                answer = "Focus on src/auth.py and src/oauth.py first"
            elif "test" in question_text:
                answer = "Fix failing tests before adding new features"
            else:
                answer = "Please provide more context about your question"

            # Send answer
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Architect",
                to_agent="Dir-Code",
                message_type="answer",
                content=answer
            )

            return True  # Answer sent

        # Wait before next poll (simulates 10s poll interval)
        await asyncio.sleep(0.1)

    return False  # No questions found


async def simulate_dircode_processing(agent_chat, thread_id):
    """
    Simulate Dir-Code's process_agent_chat_thread() logic

    This is a simplified version that:
    1. Loads thread
    2. Sends status updates
    3. Asks a question (if needed)
    4. Waits for answer
    5. Continues processing
    6. Completes task
    """
    # Step 1: Load thread
    thread = await agent_chat.get_thread(thread_id)

    # Step 2: Send initial status
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Starting task analysis",
        metadata={"progress": 10}
    )

    # Step 3: Check if we need to ask a question
    delegation_msg = next(
        (m for m in thread.messages if m.message_type == "delegation"),
        None
    )

    task = delegation_msg.content if delegation_msg else ""
    entry_files = thread.metadata.get("entry_files", [])

    # Heuristic: Ask question if refactor task with no entry files
    if "refactor" in task.lower() and not entry_files:
        # Step 4: Ask question
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-Code",
            to_agent="Architect",
            message_type="question",
            content="Which files should I focus on for this refactoring task?"
        )

        # Step 5: Wait for answer (poll thread)
        answer_received = False
        for _ in range(30):  # 30 iterations * 0.1s = 3s timeout
            updated_thread = await agent_chat.get_thread(thread_id)
            latest_msg = updated_thread.messages[-1] if updated_thread.messages else None

            if latest_msg and latest_msg.message_type == "answer":
                answer_received = True
                break

            await asyncio.sleep(0.1)

        if not answer_received:
            # Timeout - proceed anyway
            pass

    # Step 6: Send progress update
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Decomposing task into Manager subtasks",
        metadata={"progress": 30}
    )

    await asyncio.sleep(0.05)

    # Step 7: Send more progress
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Delegating to Managers",
        metadata={"progress": 50}
    )

    await asyncio.sleep(0.05)

    # Step 8: Send final progress
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Running tests and validation",
        metadata={"progress": 90}
    )

    await asyncio.sleep(0.05)

    # Step 9: Complete task
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Task completed successfully. All tests pass.",
        metadata={"tests_passed": 15, "managers_completed": 3}
    )

    # Step 10: Close thread
    await agent_chat.close_thread(
        thread_id=thread_id,
        status="completed",
        result="Successfully refactored authentication to OAuth2"
    )


@pytest.mark.asyncio
async def test_e2e_simple_task_no_questions(agent_chat):
    """
    E2E test: Simple task (no questions needed)

    Flow:
    - Architect creates thread
    - Dir-Code processes without asking questions
    - Dir-Code completes
    """
    # Architect creates thread for simple task (has entry_files, clear scope)
    thread = await agent_chat.create_thread(
        run_id="e2e-run-001",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Add logging to authentication module",
        metadata={
            "entry_files": ["src/auth.py"],
            "budget_tokens": 5000
        }
    )

    # Simulate Dir-Code processing
    await simulate_dircode_processing(agent_chat, thread.thread_id)

    # Verify final state
    final_thread = await agent_chat.get_thread(thread.thread_id)

    assert final_thread.status == "completed"
    assert final_thread.result is not None

    # Verify message sequence: delegation → status → status → status → completion
    types = [m.message_type for m in final_thread.messages]
    assert types[0] == "delegation"
    assert types[-1] == "completion"
    assert types.count("status") >= 3

    # No questions asked (simple task)
    assert "question" not in types


@pytest.mark.asyncio
async def test_e2e_complex_task_with_questions(agent_chat):
    """
    E2E test: Complex task (requires Q&A)

    Flow:
    - Architect creates thread for refactor task (no entry_files)
    - Dir-Code asks question
    - Architect answers (via monitor loop)
    - Dir-Code continues processing
    - Dir-Code completes
    """
    # Architect creates thread for complex task (no entry_files = ambiguous)
    thread = await agent_chat.create_thread(
        run_id="e2e-run-002",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication to use OAuth2",
        metadata={
            "budget_tokens": 12000
        }
    )

    # Start both Architect monitor and Dir-Code processing concurrently
    architect_task = asyncio.create_task(
        simulate_architect_question_monitor(agent_chat, thread.thread_id)
    )

    dircode_task = asyncio.create_task(
        simulate_dircode_processing(agent_chat, thread.thread_id)
    )

    # Wait for both to complete
    await asyncio.gather(architect_task, dircode_task)

    # Verify final state
    final_thread = await agent_chat.get_thread(thread.thread_id)

    assert final_thread.status == "completed"
    assert final_thread.result is not None

    # Verify message sequence includes Q&A
    types = [m.message_type for m in final_thread.messages]

    assert "delegation" in types
    assert "question" in types
    assert "answer" in types
    assert "completion" in types
    assert types.count("status") >= 3

    # Verify Q&A order: question comes before answer
    question_idx = types.index("question")
    answer_idx = types.index("answer")
    assert question_idx < answer_idx

    # Verify answer content
    answer_msg = next(m for m in final_thread.messages if m.message_type == "answer")
    assert "auth.py" in answer_msg.content or "focus" in answer_msg.content.lower()


@pytest.mark.asyncio
async def test_e2e_full_conversation_flow(agent_chat):
    """
    E2E test: Full conversation with multiple questions

    This is the most comprehensive test - simulates a real production scenario
    with multiple rounds of Q&A.
    """
    # Architect creates thread
    thread = await agent_chat.create_thread(
        run_id="e2e-run-003",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication module to support OAuth2 and add comprehensive tests",
        metadata={
            "budget_tokens": 15000,
            "priority": "high"
        }
    )

    thread_id = thread.thread_id

    # === Simulated Conversation ===

    # Dir-Code: Send initial status
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Analyzing task requirements",
        metadata={"progress": 10}
    )

    # Dir-Code: Ask question #1 (files)
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Which files should I focus on for the OAuth2 refactoring?"
    )

    # Architect: Answer #1
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Focus on src/auth.py and src/oauth.py first. Then update tests/test_auth.py"
    )

    # Dir-Code: Status update
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Planning refactoring strategy",
        metadata={"progress": 30}
    )

    # Dir-Code: Ask question #2 (library choice)
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use authlib or python-oauth2 for OAuth2 implementation?"
    )

    # Architect: Answer #2
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use authlib - it's more actively maintained and has better OAuth2/OIDC support"
    )

    # Dir-Code: Status updates
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Installing authlib and updating dependencies",
        metadata={"progress": 50}
    )

    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Refactoring auth.py to use authlib",
        metadata={"progress": 70}
    )

    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Running tests and validating changes",
        metadata={"progress": 90}
    )

    # Dir-Code: Complete
    await agent_chat.send_message(
        thread_id=thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Successfully refactored to OAuth2 using authlib. All 18 tests pass.",
        metadata={
            "tests_passed": 18,
            "files_modified": ["src/auth.py", "src/oauth.py", "tests/test_auth.py"],
            "dependencies_added": ["authlib==1.2.0"]
        }
    )

    # Dir-Code: Close thread
    await agent_chat.close_thread(
        thread_id=thread_id,
        status="completed",
        result="Successfully refactored authentication to OAuth2 with authlib. All tests pass."
    )

    # === Verification ===

    final_thread = await agent_chat.get_thread(thread_id)

    # Verify completion
    assert final_thread.status == "completed"
    assert "OAuth2" in final_thread.result

    # Verify message count (1 delegation + 2 questions + 2 answers + 5 status + 1 completion = 11)
    assert len(final_thread.messages) == 11

    # Verify message sequence
    types = [m.message_type for m in final_thread.messages]
    expected_sequence = [
        "delegation",
        "status",
        "question",
        "answer",
        "status",
        "question",
        "answer",
        "status",
        "status",
        "status",
        "completion"
    ]
    assert types == expected_sequence

    # Verify metadata preserved
    assert final_thread.metadata.get("budget_tokens") == 15000
    assert final_thread.metadata.get("priority") == "high"

    # Verify completion metadata
    completion_msg = final_thread.messages[-1]
    assert completion_msg.metadata.get("tests_passed") == 18
    assert "authlib" in completion_msg.metadata.get("dependencies_added", [])[0]

    # Verify Q&A content
    questions = [m for m in final_thread.messages if m.message_type == "question"]
    answers = [m for m in final_thread.messages if m.message_type == "answer"]

    assert len(questions) == 2
    assert len(answers) == 2

    assert "files" in questions[0].content.lower()
    assert "authlib" in questions[1].content.lower()

    assert "auth.py" in answers[0].content
    assert "authlib" in answers[1].content


@pytest.mark.asyncio
async def test_e2e_stats_and_analytics(agent_chat):
    """
    E2E test: Verify statistics and analytics after multiple conversations
    """
    # Create multiple threads
    for i in range(3):
        thread = await agent_chat.create_thread(
            run_id=f"stats-run-{i:03d}",
            parent_agent_id="Architect",
            child_agent_id="Dir-Code",
            initial_message=f"Task {i}: Implement feature {i}",
            metadata={"task_num": i}
        )

        # Send some messages
        await agent_chat.send_message(
            thread_id=thread.thread_id,
            from_agent="Dir-Code",
            to_agent="Architect",
            message_type="status",
            content=f"Processing task {i}",
            metadata={"progress": 50}
        )

        # Complete
        await agent_chat.send_message(
            thread_id=thread.thread_id,
            from_agent="Dir-Code",
            to_agent="Architect",
            message_type="completion",
            content=f"Task {i} completed"
        )

        await agent_chat.close_thread(
            thread_id=thread.thread_id,
            status="completed",
            result=f"Task {i} done"
        )

    # Get stats
    stats = agent_chat.get_stats()

    assert stats["total_threads"] == 3
    assert stats["threads_by_status"]["completed"] == 3
    assert stats["total_messages"] >= 9  # At least 3 messages per thread
    assert stats["avg_messages_per_thread"] >= 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
