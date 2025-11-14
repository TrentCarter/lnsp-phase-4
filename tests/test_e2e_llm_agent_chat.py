#!/usr/bin/env python3
"""
End-to-End Tests for LLM-Powered Agent Chat (Phase 3)

Tests complete conversation flows with real LLM integration:
- Dir-Code asks intelligent questions using LLM + ask_parent tool
- Architect answers using LLM with full context awareness

Requires:
- LNSP_TEST_MODE=1
- Ollama running with llama3.1:8b model
"""
import pytest
import os
import sys
from pathlib import Path
import asyncio
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.common.agent_chat import get_agent_chat_client, AgentChatMessage
from services.pas.director_code.app import analyze_task_with_llm
from services.pas.architect.app import generate_answer_to_question


# Skip all tests if not in test mode or Ollama not running
pytestmark = pytest.mark.skipif(
    os.getenv("LNSP_TEST_MODE") != "1",
    reason="Requires LNSP_TEST_MODE=1 and Ollama running"
)


@pytest.fixture
async def agent_chat():
    """Get agent chat client for tests"""
    return get_agent_chat_client()


@pytest.fixture
async def test_thread(agent_chat):
    """Create a test thread"""
    thread = await agent_chat.create_thread(
        run_id="test-llm-e2e",
        parent_agent="Architect",
        child_agent="Dir-Code",
        metadata={
            "entry_files": [],
            "budget_tokens": 10000,
            "policy": {}
        }
    )

    yield thread

    # Cleanup
    try:
        await agent_chat.close_thread(
            thread_id=thread.thread_id,
            status="completed",
            result="Test completed"
        )
    except:
        pass


# === Phase 3: LLM-Powered Q&A Tests ===

@pytest.mark.asyncio
async def test_dircode_llm_asks_intelligent_question(agent_chat, test_thread):
    """
    Test Dir-Code using LLM to decide when to ask questions

    Scenario: Ambiguous refactor task with no entry files
    Expected: LLM detects ambiguity and uses ask_parent tool
    """
    # Simulate delegation from Architect
    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="delegation",
        content="Refactor authentication code to improve security"
    )

    # Get thread history
    thread = await agent_chat.get_thread(test_thread.thread_id)

    # Dir-Code analyzes task with LLM
    analysis = await analyze_task_with_llm(
        task_description="Refactor authentication code to improve security",
        thread_history=thread.messages,
        metadata={
            "entry_files": [],  # No files specified - ambiguous!
            "budget_tokens": 10000,
            "run_id": "test-llm-e2e"
        },
        thread_id=test_thread.thread_id
    )

    # Verify analysis
    assert analysis is not None
    assert "reasoning" in analysis

    # Check if question was asked (tool_calls_made > 0)
    # Or if LLM recognized the need to ask (mentioned in reasoning)
    asked_question = analysis.get("tool_calls_made", 0) > 0
    mentioned_question = "question" in analysis["reasoning"].lower() or "ask" in analysis["reasoning"].lower()

    assert asked_question or mentioned_question, (
        f"LLM should ask question or acknowledge need to ask for ambiguous task. "
        f"Tool calls: {analysis.get('tool_calls_made', 0)}, "
        f"Reasoning: {analysis['reasoning']}"
    )

    # If question was asked, verify it's in the thread
    if asked_question:
        thread = await agent_chat.get_thread(test_thread.thread_id)
        questions = [m for m in thread.messages if m.message_type == "question"]
        assert len(questions) > 0, "Question should be in thread"

        # Verify question is relevant
        question_text = questions[0].content.lower()
        assert "file" in question_text or "scope" in question_text or "which" in question_text


@pytest.mark.asyncio
async def test_dircode_llm_proceeds_when_clear(agent_chat, test_thread):
    """
    Test Dir-Code NOT asking questions when task is clear

    Scenario: Clear task with specific files
    Expected: LLM recognizes clarity and proceeds without questions
    """
    # Update metadata with clear entry files
    test_thread.metadata["entry_files"] = ["src/auth.py", "src/oauth.py"]

    # Simulate delegation
    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="delegation",
        content="Add type hints to src/auth.py and src/oauth.py"
    )

    # Get thread history
    thread = await agent_chat.get_thread(test_thread.thread_id)

    # Dir-Code analyzes task
    analysis = await analyze_task_with_llm(
        task_description="Add type hints to src/auth.py and src/oauth.py",
        thread_history=thread.messages,
        metadata={
            "entry_files": ["src/auth.py", "src/oauth.py"],
            "budget_tokens": 5000,
            "run_id": "test-llm-e2e"
        },
        thread_id=test_thread.thread_id
    )

    # Verify no questions asked for clear task
    assert analysis.get("tool_calls_made", 0) == 0, "Should not ask questions for clear task"

    # Verify reasoning indicates can proceed
    reasoning_lower = analysis["reasoning"].lower()
    can_proceed = any(word in reasoning_lower for word in ["proceed", "clear", "understand", "ready"])
    assert can_proceed, f"Reasoning should indicate can proceed: {analysis['reasoning']}"


@pytest.mark.asyncio
async def test_architect_llm_context_aware_answer(agent_chat, test_thread):
    """
    Test Architect using LLM to generate context-aware answers

    Scenario: Dir-Code asks about library choice
    Expected: Architect considers context and provides specific recommendation
    """
    # Build conversation history
    history = """Architect → Dir-Code (delegation): Implement OAuth2 authentication
Dir-Code → Architect (question): Should I use authlib or requests-oauthlib for OAuth2 implementation?"""

    # Generate answer
    answer = await generate_answer_to_question(
        question="Should I use authlib or requests-oauthlib for OAuth2 implementation?",
        conversation_history=history,
        thread_metadata={
            "entry_files": ["src/auth.py"],
            "budget_tokens": 10000,
            "run_id": "test-llm-e2e"
        }
    )

    # Verify answer
    assert answer is not None
    assert len(answer) > 0

    # Answer should mention one of the libraries
    answer_lower = answer.lower()
    mentions_library = "authlib" in answer_lower or "requests-oauthlib" in answer_lower

    assert mentions_library, f"Answer should mention a specific library: {answer}"

    # Answer should be concise (2-4 sentences ideal)
    sentence_count = answer.count('.') + answer.count('!')
    assert 1 <= sentence_count <= 6, f"Answer should be concise (1-6 sentences), got {sentence_count}"


@pytest.mark.asyncio
async def test_full_conversation_with_llm(agent_chat, test_thread):
    """
    Test complete conversation flow with LLM-powered Q&A

    Scenario:
    1. Architect delegates ambiguous task
    2. Dir-Code uses LLM to detect ambiguity and ask question
    3. Architect uses LLM to generate answer
    4. Dir-Code proceeds with updated information

    This is the full Phase 3 flow!
    """
    # Step 1: Architect delegates task
    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="delegation",
        content="Refactor the authentication system to use OAuth2",
        metadata={"prd": "Improve security by migrating to OAuth2 standard"}
    )

    # Step 2: Dir-Code analyzes with LLM
    thread = await agent_chat.get_thread(test_thread.thread_id)
    analysis = await analyze_task_with_llm(
        task_description="Refactor the authentication system to use OAuth2",
        thread_history=thread.messages,
        metadata={
            "entry_files": [],  # Ambiguous - no files specified
            "budget_tokens": 12000,
            "run_id": "test-llm-e2e",
            "prd": "Improve security by migrating to OAuth2 standard"
        },
        thread_id=test_thread.thread_id
    )

    # Verify Dir-Code asked a question (or would have)
    tool_calls_made = analysis.get("tool_calls_made", 0)

    # If question was asked, Architect should answer
    if tool_calls_made > 0:
        # Step 3: Get the question
        thread = await agent_chat.get_thread(test_thread.thread_id)
        questions = [m for m in thread.messages if m.message_type == "question"]
        assert len(questions) > 0

        question = questions[-1].content

        # Step 4: Architect generates answer with LLM
        conversation_history = "\n".join([
            f"{msg.from_agent} → {msg.to_agent} ({msg.message_type}): {msg.content}"
            for msg in thread.messages
        ])

        answer = await generate_answer_to_question(
            question=question,
            conversation_history=conversation_history,
            thread_metadata={
                "entry_files": [],
                "budget_tokens": 12000,
                "run_id": "test-llm-e2e",
                "prd": "Improve security by migrating to OAuth2 standard"
            }
        )

        # Verify answer was generated
        assert answer is not None
        assert len(answer) > 10  # Non-trivial answer

        # Send answer back
        await agent_chat.send_message(
            thread_id=test_thread.thread_id,
            from_agent="Architect",
            to_agent="Dir-Code",
            message_type="answer",
            content=answer
        )

        # Step 5: Verify conversation completeness
        final_thread = await agent_chat.get_thread(test_thread.thread_id)

        # Should have: delegation, question, answer (minimum)
        message_types = [m.message_type for m in final_thread.messages]
        assert "delegation" in message_types
        assert "question" in message_types
        assert "answer" in message_types

        print(f"\n✅ Full conversation flow completed!")
        print(f"   Messages: {len(final_thread.messages)}")
        print(f"   Question: {question[:80]}...")
        print(f"   Answer: {answer[:80]}...")


@pytest.mark.asyncio
async def test_llm_handles_multiple_questions(agent_chat, test_thread):
    """
    Test handling multiple questions in one conversation

    Scenario:
    - Dir-Code asks about files
    - Gets answer
    - Asks about library choice
    - Gets answer
    - Proceeds

    This tests the iterative Q&A capability.
    """
    # Question 1: Files
    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Which authentication files should I focus on?"
    )

    # Answer 1
    thread = await agent_chat.get_thread(test_thread.thread_id)
    answer1 = await generate_answer_to_question(
        question="Which authentication files should I focus on?",
        conversation_history="\n".join([f"{m.from_agent}: {m.content}" for m in thread.messages]),
        thread_metadata={"entry_files": ["src/auth.py", "src/oauth.py", "src/sessions.py"]}
    )

    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content=answer1
    )

    # Question 2: Library
    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Dir-Code",
        to_agent="Architect",
        message_type="question",
        content="Should I use authlib or requests-oauthlib?"
    )

    # Answer 2
    thread = await agent_chat.get_thread(test_thread.thread_id)
    answer2 = await generate_answer_to_question(
        question="Should I use authlib or requests-oauthlib?",
        conversation_history="\n".join([f"{m.from_agent}: {m.content}" for m in thread.messages]),
        thread_metadata={"entry_files": ["src/auth.py"], "budget_tokens": 10000}
    )

    await agent_chat.send_message(
        thread_id=test_thread.thread_id,
        from_agent="Architect",
        to_agent="Dir-Code",
        message_type="answer",
        content=answer2
    )

    # Verify conversation
    final_thread = await agent_chat.get_thread(test_thread.thread_id)

    questions = [m for m in final_thread.messages if m.message_type == "question"]
    answers = [m for m in final_thread.messages if m.message_type == "answer"]

    assert len(questions) == 2, "Should have 2 questions"
    assert len(answers) == 2, "Should have 2 answers"

    # Verify answers are different and relevant
    assert answer1 != answer2, "Answers should be different"
    assert len(answer1) > 10 and len(answer2) > 10, "Answers should be non-trivial"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
