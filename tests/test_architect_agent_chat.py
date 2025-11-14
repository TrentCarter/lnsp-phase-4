#!/usr/bin/env python3
"""
Integration tests for Architect agent chat (Parent-Child communication)

Tests the full flow:
1. Architect creates thread when delegating complex tasks
2. Architect monitors threads for pending questions
3. Architect generates answers using heuristics/LLM
"""
import pytest
import sys
from pathlib import Path
import tempfile
import os

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add parent directory to path for imports
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
    """Create agent chat client with temp database"""
    return AgentChatClient(db_path=temp_db)


@pytest.mark.asyncio
async def test_architect_creates_thread_for_complex_task(agent_chat):
    """Test that Architect creates thread for refactoring task"""
    # Simulate Architect delegating complex task
    task = "Refactor authentication to use OAuth2"  # Contains "refactor" keyword

    # Architect creates thread
    thread = await agent_chat.create_thread(
        run_id="run-test-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message=task,
        metadata={
            "entry_files": ["src/auth.py"],
            "budget_tokens": 10000,
            "lane": "Code"
        }
    )

    # Verify thread created
    assert thread.thread_id is not None
    assert thread.parent_agent_id == "Architect"
    assert thread.child_agent_id == "Dir-Code"
    assert thread.status == "active"
    assert len(thread.messages) == 1
    assert thread.messages[0].message_type == "delegation"
    assert thread.metadata["lane"] == "Code"


@pytest.mark.asyncio
async def test_director_asks_question_architect_answers(agent_chat):
    """Test full Q&A flow: Dir-Code asks question, Architect answers"""
    # Step 1: Create thread (Architect delegates)
    thread = await agent_chat.create_thread(
        run_id="run-qa-456",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor authentication to OAuth2",
        metadata={"entry_files": ["src/auth.py"]}
    )

    # Step 2: Director asks question
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use library 'authlib' or 'python-oauth2'?"
    )

    # Step 3: Get pending questions (simulates Architect monitoring)
    questions = await agent_chat.get_pending_questions("Architect")
    assert len(questions) == 1
    assert questions[0]["content"] == "Should I use library 'authlib' or 'python-oauth2'?"
    assert questions[0]["from_agent"] == "Dir-Code"

    # Step 4: Architect answers (simulates heuristic answer generation)
    answer = "Use authlib - it's more actively maintained, has better documentation, and supports OAuth2/OIDC out of the box."
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content=answer
    )

    # Verify thread has full conversation
    updated_thread = await agent_chat.get_thread(thread.thread_id)
    assert len(updated_thread.messages) == 3
    assert updated_thread.messages[0].message_type == "delegation"
    assert updated_thread.messages[1].message_type == "question"
    assert updated_thread.messages[2].message_type == "answer"
    assert "authlib" in updated_thread.messages[2].content


@pytest.mark.asyncio
async def test_multiple_questions_in_conversation(agent_chat):
    """Test Director asking multiple questions over time"""
    # Create thread
    thread = await agent_chat.create_thread(
        run_id="run-multi-789",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth to OAuth2",
        metadata={"entry_files": ["src/auth.py"]}
    )

    # Question 1: Library choice
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Which library should I use?"
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Use authlib"
    )

    # Status update
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="status",
        content="Installing authlib...",
        metadata={"progress": 20}
    )

    # Question 2: Test failures
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Tests are failing in test_jwt_refresh. Should I fix them now?"
    )

    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content="Yes, fix test failures before proceeding"
    )

    # Completion
    await agent_chat.send_message(
        thread_id=thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="completion",
        content="Done! All tests passing."
    )

    # Close thread
    await agent_chat.close_thread(
        thread_id=thread.thread_id,
        status="completed",
        result="Successfully refactored to OAuth2"
    )

    # Verify full conversation
    final_thread = await agent_chat.get_thread(thread.thread_id)
    assert len(final_thread.messages) == 7  # delegation + Q1 + A1 + status + Q2 + A2 + completion
    assert final_thread.status == "completed"

    # Verify message types
    message_types = [m.message_type for m in final_thread.messages]
    assert message_types == [
        "delegation",
        "question",
        "answer",
        "status",
        "question",
        "answer",
        "completion"
    ]


@pytest.mark.asyncio
async def test_thread_metadata_includes_budget(agent_chat):
    """Test that thread metadata includes budget information"""
    thread = await agent_chat.create_thread(
        run_id="run-budget-001",
        parent_agent_id="Architect",
        child_agent_id="Dir-Models",
        initial_message="Train sentiment classifier",
        metadata={
            "entry_files": ["data/sentiment.csv"],
            "budget_tokens": 50000,
            "expected_artifacts": ["model.pkl", "metrics.json"]
        }
    )

    assert thread.metadata["budget_tokens"] == 50000
    assert thread.metadata["expected_artifacts"] == ["model.pkl", "metrics.json"]


@pytest.mark.asyncio
async def test_get_threads_by_run(agent_chat):
    """Test retrieving all threads for a specific run"""
    run_id = "run-multi-thread-001"

    # Create multiple threads for same run (Architect → different Directors)
    thread1 = await agent_chat.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth"
    )

    thread2 = await agent_chat.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Models",
        initial_message="Train model"
    )

    thread3 = await agent_chat.create_thread(
        run_id=run_id,
        parent_agent_id="Architect",
        child_agent_id="Dir-Docs",
        initial_message="Update README"
    )

    # Get all threads for run
    threads = await agent_chat.get_threads_by_run(run_id)
    assert len(threads) == 3
    assert all(t.run_id == run_id for t in threads)
    assert all(t.parent_agent_id == "Architect" for t in threads)

    # Verify different children
    child_ids = {t.child_agent_id for t in threads}
    assert child_ids == {"Dir-Code", "Dir-Models", "Dir-Docs"}


@pytest.mark.asyncio
async def test_thread_links_to_job_card(agent_chat):
    """Test that thread metadata can link back to job card"""
    job_card_id = "jc-run-123-code-001"

    thread = await agent_chat.create_thread(
        run_id="run-123",
        parent_agent_id="Architect",
        child_agent_id="Dir-Code",
        initial_message="Refactor auth",
        metadata={
            "job_card_id": job_card_id,  # Link to traditional job card
            "lane": "Code"
        }
    )

    assert thread.metadata["job_card_id"] == job_card_id
    assert thread.metadata["lane"] == "Code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
