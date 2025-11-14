#!/usr/bin/env python3
"""
Unit tests for Agent Chat system (Parent ↔ Child communication)

Tests:
- Thread creation
- Message sending
- Thread retrieval with full history
- Thread closure
- Pending questions query
- Statistics
"""
import pytest

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)
import asyncio
import sqlite3
from pathlib import Path
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.common.agent_chat import AgentChatClient, AgentChatThread, AgentChatMessage


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield Path(path)
    Path(path).unlink()


@pytest.fixture
def client(temp_db):
    """Create agent chat client with temp database"""
    return AgentChatClient(db_path=temp_db)


@pytest.mark.asyncio
async def test_create_thread(client):
    """Test creating a new conversation thread"""
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication to OAuth2",
        metadata={"entry_files": ["src/auth.py"], "budget_tokens": 10000}
    )

    assert thread.thread_id is not None
    assert thread.run_id == "run-123"
    assert thread.parent_agent_id == "Architect"
    assert thread.child_agent_id == "Dir-Code"
    assert thread.status == "active"
    assert len(thread.messages) == 1
    assert thread.messages[0].message_type == "delegation"
    assert thread.messages[0].from_agent == "Architect"
    assert thread.messages[0].to_agent == "Dir-Code"
    assert thread.messages[0].content == "Refactor authentication to OAuth2"
    assert thread.metadata["entry_files"] == ["src/auth.py"]


@pytest.mark.asyncio
async def test_send_message(client):
    """Test sending messages on a thread"""
    # Create thread
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    # Child asks question
    msg_id = await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use authlib or python-oauth2?"
    )

    assert msg_id is not None

    # Parent answers
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use authlib - better maintained"
    )

    # Verify thread has 3 messages (delegation + question + answer)
    updated_thread = await client.get_thread(thread.thread_id)
    assert len(updated_thread.messages) == 3
    assert updated_thread.messages[1].message_type == "question"
    assert updated_thread.messages[1].content == "Should I use authlib or python-oauth2?"
    assert updated_thread.messages[2].message_type == "answer"
    assert updated_thread.messages[2].content == "Use authlib - better maintained"


@pytest.mark.asyncio
async def test_status_updates(client):
    """Test sending status updates"""
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    # Send multiple status updates
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Planning refactor...",
        metadata={"progress": 10}
    )

    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Installing authlib...",
        metadata={"progress": 20}
    )

    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Refactoring auth.py...",
        metadata={"progress": 50}
    )

    # Verify all status messages recorded
    updated_thread = await client.get_thread(thread.thread_id)
    status_messages = [m for m in updated_thread.messages if m.message_type == "status"]
    assert len(status_messages) == 3
    assert status_messages[0].metadata["progress"] == 10
    assert status_messages[1].metadata["progress"] == 20
    assert status_messages[2].metadata["progress"] == 50


@pytest.mark.asyncio
async def test_close_thread(client):
    """Test closing a thread with completion"""
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    # Send completion message
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Successfully refactored JWT auth to OAuth2. All tests pass."
    )

    # Close thread
    await client.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="Refactored JWT auth to OAuth2 using authlib"
    )

    # Verify thread closed
    closed_thread = await client.get_thread(thread.thread_id)
    assert closed_thread.status == "completed"
    assert closed_thread.completed_at is not None
    assert closed_thread.result == "Refactored JWT auth to OAuth2 using authlib"
    assert closed_thread.error is None


@pytest.mark.asyncio
async def test_close_thread_with_error(client):
    """Test closing a thread with error"""
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    # Send error message
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="error",
        content="Tests failing, cannot proceed"
    )

    # Close thread with failure
    await client.close_thread(
        thread_id=thread.thread_id,
        status="failed",
        error="Test failures in test_jwt_refresh - need guidance"
    )

    # Verify thread failed
    failed_thread = await client.get_thread(thread.thread_id)
    assert failed_thread.status == "failed"
    assert failed_thread.completed_at is not None
    assert failed_thread.error == "Test failures in test_jwt_refresh - need guidance"
    assert failed_thread.result is None


@pytest.mark.asyncio
async def test_get_threads_by_run(client):
    """Test retrieving all threads for a run"""
    run_id = "run-456"

    # Create multiple threads for same run
    thread1 = await client.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    thread2 = await client.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Models",
        initial_message="Train new model"
    )

    # Create thread for different run
    thread3 = await client.create_thread(
        run_id="run-789",
        parent_agent_id="Architect",
        child_agent_id="Dir-Data",
        initial_message="Ingest dataset"
    )

    # Get threads for run-456
    threads = await client.get_threads_by_run(run_id)
    assert len(threads) == 2
    assert all(t.run_id == run_id for t in threads)

    # Messages not loaded for list view
    assert all(len(t.messages) == 0 for t in threads)


@pytest.mark.asyncio
async def test_get_pending_questions(client):
    """Test retrieving pending questions for parent agent"""
    # Create thread 1: Architect → Dir-Code
    thread1 = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    await client.send_message(
        thread_id=thread1.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Which library should I use?"
    )

    # Create thread 2: Architect → Dir-Models
    thread2 = await client.create_thread(
        run_id="run-456",
        parent_agent_id="Architect",
        child_agent_id="Dir-Models",
        initial_message="Train model"
    )

    await client.send_message(
        thread_id=thread2.thread_id,
        from_agent="Dir-Models",
        to_agent="Architect",
        message_type="question",
        content="Should I use GPU or CPU?"
    )

    # Get pending questions for Architect
    questions = await client.get_pending_questions("Architect")
    assert len(questions) == 2
    assert questions[0]["content"] in ["Which library should I use?", "Should I use GPU or CPU?"]
    assert all(q["message_type"] == "question" for q in questions)


@pytest.mark.asyncio
async def test_message_count(client):
    """Test getting message count for a thread"""
    thread = await client.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    # Initial count (just delegation message)
    assert client.get_message_count(thread.thread_id) == 1

    # Add more messages
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Question?"
    )

    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Answer"
    )

    assert client.get_message_count(thread.thread_id) == 3


@pytest.mark.asyncio
async def test_stats(client):
    """Test getting overall statistics"""
    # Create some threads with different statuses
    thread1 = await client.create_thread(
        run_id="run-1",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Task 1"
    )

    thread2 = await client.create_thread(
        run_id="run-2",
        parent_agent_id="Architect",
        child_agent_id="Dir-Models",
        initial_message="Task 2"
    )

    # Add messages to thread1
    await client.send_message(
        thread_id=thread1.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Working..."
    )

    # Close thread1
    await client.close_thread(thread1.thread_id, status="completed", result="Done")

    # Get stats
    stats = client.get_stats()
    assert stats["total_threads"] == 2
    assert stats["threads_by_status"]["active"] == 1
    assert stats["threads_by_status"]["completed"] == 1
    assert stats["total_messages"] == 3  # 2 delegation + 1 status
    assert stats["avg_messages_per_thread"] == 1.5


@pytest.mark.asyncio
async def test_full_conversation_flow(client):
    """Test complete conversation flow: delegation → question → answer → status → completion"""
    # Step 1: Parent creates thread
    thread = await client.create_thread(
        run_id="run-789",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication to use OAuth2. Entry file: src/auth.py"
    )

    # Step 2: Child asks clarifying question
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="I found 3 auth implementations (basic, JWT, session). Should I refactor all or focus on one?"
    )

    # Step 3: Parent answers
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Focus on JWT first, keep others unchanged for backward compatibility."
    )

    # Step 4: Child sends status updates
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Planning refactor for JWTAuth...",
        metadata={"progress": 10}
    )

    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Installing authlib...",
        metadata={"progress": 20}
    )

    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Refactoring JWTAuth class...",
        metadata={"progress": 50}
    )

    # Step 5: Child asks another question
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use library 'authlib' (MIT, 5k stars) or 'python-oauth2' (Apache, 2k stars)?"
    )

    # Step 6: Parent answers
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use 'authlib' - more active maintenance and better docs."
    )

    # Step 7: Child continues with more status
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Running tests...",
        metadata={"progress": 80}
    )

    # Step 8: Child completes
    await client.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="✅ Refactored JWTAuth to OAuth2 using authlib. Tests: 15/15 passing. Git commit: a1b2c3d"
    )

    # Step 9: Close thread
    await client.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="Successfully refactored JWT auth to OAuth2"
    )

    # Verify full conversation
    final_thread = await client.get_thread(thread.thread_id)
    assert len(final_thread.messages) == 10
    assert final_thread.status == "completed"
    assert final_thread.result is not None

    # Verify message sequence
    expected_types = [
        "delegation",  # 1
        "question",    # 2
        "answer",      # 3
        "status",      # 4
        "status",      # 5
        "status",      # 6
        "question",    # 7
        "answer",      # 8
        "status",      # 9
        "completion"   # 10
    ]
    actual_types = [m.message_type for m in final_thread.messages]
    assert actual_types == expected_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
