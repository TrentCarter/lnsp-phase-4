#!/usr/bin/env python3
"""
Integration tests for Dir-Code agent chat (Child side)

Tests:
1. Dir-Code receives agent chat message
2. Dir-Code asks question to Architect
3. Dir-Code sends status updates
4. Dir-Code completes task and closes thread
5. Dir-Code handles errors and closes thread
6. Full conversation flow (delegation → question → answer → completion)
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.common.agent_chat import (
    AgentChatClient,
    AgentChatMessage,
    AgentChatThread
)


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


@pytest.mark.asyncio
async def test_dircode_receives_message(agent_chat):
    """Test Dir-Code can receive and process agent chat message"""

    # Simulate Architect creating thread with delegation
    thread = await agent_chat.create_thread(
        run_id="test-run-001",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Add logging to authentication module",
        metadata={
            "entry_files": ["src/auth.py"],
            "budget_tokens": 5000
        }
    )

    assert thread.thread_id is not None
    assert thread.status == "active"
    assert len(thread.messages) == 1
    assert thread.messages[0].message_type == "delegation"

    # Simulate Dir-Code loading thread (what /agent_chat/receive does)
    loaded_thread = await agent_chat.get_thread(thread.thread_id)

    assert loaded_thread.thread_id == thread.thread_id
    assert loaded_thread.run_id == "test-run-001"
    assert len(loaded_thread.messages) == 1

    # Extract delegation message
    delegation_msg = next(
        (m for m in loaded_thread.messages if m.message_type == "delegation"),
        None
    )

    assert delegation_msg is not None
    assert "logging" in delegation_msg.content


@pytest.mark.asyncio
async def test_dircode_asks_question(agent_chat):
    """Test Dir-Code can ask questions to Architect"""

    # Create thread
    thread = await agent_chat.create_thread(
        run_id="test-run-002",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication module",
        metadata={"budget_tokens": 8000}
    )

    # Dir-Code asks question
    question_msg = await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use authlib or python-oauth2?"
    )

    assert question_msg is not None

    # Verify question is in thread
    updated_thread = await agent_chat.get_thread(thread.thread_id)
    assert len(updated_thread.messages) == 2  # delegation + question

    question = updated_thread.messages[1]
    assert question.message_type == "question"
    assert question.from_agent == "Dir-Code"
    assert question.to_agent == "Architect"
    assert "authlib" in question.content


@pytest.mark.asyncio
async def test_dircode_sends_status_updates(agent_chat):
    """Test Dir-Code sends status updates during execution"""

    # Create thread
    thread = await agent_chat.create_thread(
        run_id="test-run-003",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Update user model schema",
        metadata={"entry_files": ["models/user.py"]}
    )

    # Dir-Code sends status updates
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Planning changes...",
        metadata={"progress": 10}
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Updating schema...",
        metadata={"progress": 50}
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Running tests...",
        metadata={"progress": 80}
    )

    # Verify status messages
    updated_thread = await agent_chat.get_thread(thread.thread_id)
    assert len(updated_thread.messages) == 4  # delegation + 3 status

    status_messages = [m for m in updated_thread.messages if m.message_type == "status"]
    assert len(status_messages) == 3
    assert status_messages[0].metadata.get("progress") == 10
    assert status_messages[1].metadata.get("progress") == 50
    assert status_messages[2].metadata.get("progress") == 80


@pytest.mark.asyncio
async def test_dircode_completes_task(agent_chat):
    """Test Dir-Code completes task and closes thread"""

    # Create thread
    thread = await agent_chat.create_thread(
        run_id="test-run-004",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Add validation to user input",
        metadata={"entry_files": ["handlers/user.py"]}
    )

    # Dir-Code sends completion
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Added input validation. All tests pass.",
        metadata={"tests_passed": 12}
    )

    # Dir-Code closes thread
    await agent_chat.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="Successfully added input validation"
    )

    # Verify thread is closed
    final_thread = await agent_chat.get_thread(thread.thread_id)
    assert final_thread.status == "completed"
    assert final_thread.result == "Successfully added input validation"
    assert final_thread.completed_at is not None


@pytest.mark.asyncio
async def test_dircode_handles_error(agent_chat):
    """Test Dir-Code handles errors and closes thread with failure"""

    # Create thread
    thread = await agent_chat.create_thread(
        run_id="test-run-005",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Migrate database schema",
        metadata={"entry_files": ["migrations/"]}
    )

    # Dir-Code encounters error
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="error",
        content="Migration failed: syntax error in SQL",
        metadata={"error_code": "SQL001"}
    )

    # Dir-Code closes thread with failure
    await agent_chat.close_thread(
        thread_id=thread.thread_id,
        status="failed",
        result="Migration failed",
        error="Syntax error in SQL migration script"
    )

    # Verify thread is closed with failure
    final_thread = await agent_chat.get_thread(thread.thread_id)
    assert final_thread.status == "failed"
    assert final_thread.error is not None
    assert "SQL" in final_thread.error


@pytest.mark.asyncio
async def test_full_conversation_flow(agent_chat):
    """Test full conversation: delegation → question → answer → status → completion"""

    # Step 1: Architect delegates task
    thread = await agent_chat.create_thread(
        run_id="test-run-006",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication to use OAuth2",
        metadata={
            "entry_files": ["src/auth.py"],
            "budget_tokens": 12000
        }
    )

    assert thread.status == "active"
    assert len(thread.messages) == 1

    # Step 2: Dir-Code asks question
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use authlib or python-oauth2?"
    )

    # Step 3: Architect answers (simulated - in real system, monitor_conversation_threads does this)
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use authlib - it's more actively maintained and supports OAuth2/OIDC"
    )

    # Step 4: Dir-Code sends status updates
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Installing authlib...",
        metadata={"progress": 20}
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Refactoring auth.py...",
        metadata={"progress": 60}
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Running tests...",
        metadata={"progress": 90}
    )

    # Step 5: Dir-Code completes
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Successfully refactored to OAuth2. All 15 tests pass.",
        metadata={"tests_passed": 15, "git_commit": "abc123def"}
    )

    # Step 6: Dir-Code closes thread
    await agent_chat.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="Successfully refactored JWT auth to OAuth2 using authlib"
    )

    # Verify final state
    final_thread = await agent_chat.get_thread(thread.thread_id)

    assert final_thread.status == "completed"
    assert len(final_thread.messages) == 7  # delegation + question + answer + 3 status + completion

    # Verify message sequence
    types = [m.message_type for m in final_thread.messages]
    assert types == ["delegation", "question", "answer", "status", "status", "status", "completion"]

    # Verify metadata preserved
    assert final_thread.metadata.get("entry_files") == ["src/auth.py"]
    assert final_thread.metadata.get("budget_tokens") == 12000

    # Verify completion
    completion_msg = final_thread.messages[-1]
    assert completion_msg.metadata.get("tests_passed") == 15
    assert completion_msg.metadata.get("git_commit") == "abc123def"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
