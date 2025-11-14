#!/usr/bin/env python3
"""
Agent Chat Mixin - Universal Agent Chat Integration

This mixin provides instant Agent Chat integration for any FastAPI agent.

Usage:
    1. Import the mixin and helper functions
    2. Add the mixin routes to your FastAPI app
    3. Start the background message poller on app startup

Example (Director agent):
    ```python
    from services.common.agent_chat_mixin import (
        add_agent_chat_routes,
        start_message_poller,
        send_message_to_parent,
        send_message_to_child
    )
    from services.common.agent_chat import get_agent_chat_client

    app = FastAPI(title="Dir-Models")
    agent_chat = get_agent_chat_client()

    # Add Agent Chat routes (receive messages, SSE events)
    add_agent_chat_routes(
        app=app,
        agent_id="Dir-Models",
        agent_chat=agent_chat,
        on_message_received=handle_incoming_message  # Your handler
    )

    @app.on_event("startup")
    async def startup():
        # Start polling for incoming messages
        await start_message_poller(
            agent_id="Dir-Models",
            agent_chat=agent_chat,
            poll_interval=2.0
        )

    async def handle_incoming_message(message: AgentChatMessage):
        '''Handle incoming messages from parent/children'''
        if message.message_type == "delegation":
            # Handle delegation from parent
            await execute_task(message.content)
        elif message.message_type == "question":
            # Handle question from child
            await answer_question(message)
    ```

Example (Manager agent):
    Same pattern - just change agent_id and parent relationship.

Example (Programmer agent):
    Same pattern - works for any tier (Architect → Directors → Managers → Programmers).

This mixin provides:
- ✅ Agent Chat integration (bidirectional messaging)
- ✅ Send messages (to parent or children)
- ✅ Receive messages (background polling)
- ✅ SSE events (real-time updates to HMI)
- ✅ Thread management (create, close, get status)

All agents use the SAME code - no duplication!
"""
import asyncio
import json
import uuid
from typing import Callable, Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.common.agent_chat import get_agent_chat_client, AgentChatMessage, AgentChatThread


# === Pydantic Models ===

class SendMessageRequest(BaseModel):
    """Request to send message to another agent"""
    thread_id: str
    to_agent: str
    message_type: str  # question, answer, status, completion, error, escalation, abort
    content: str
    metadata: Dict[str, Any] = {}


class CreateThreadRequest(BaseModel):
    """Request to create new conversation thread"""
    run_id: str
    parent_agent_id: str
    child_agent_id: str
    initial_message: str
    metadata: Dict[str, Any] = {}


# === Message Poller (Background Task) ===

_poller_tasks: Dict[str, asyncio.Task] = {}
_message_handlers: Dict[str, Callable] = {}
_last_poll_time: Dict[str, datetime] = {}


async def _poll_for_messages(
    agent_id: str,
    agent_chat,
    poll_interval: float = 2.0
):
    """
    Background task to poll for incoming messages.

    This runs continuously in the background and calls the registered
    message handler when new messages arrive.

    Args:
        agent_id: This agent's ID
        agent_chat: AgentChatClient instance
        poll_interval: How often to poll (seconds)
    """
    print(f"[{agent_id}] Started message poller (interval={poll_interval}s)")

    # Initialize last poll time
    _last_poll_time[agent_id] = datetime.now()

    while True:
        try:
            # Get pending questions if this agent is a parent
            questions = await agent_chat.get_pending_questions(agent_id)

            # Process each question
            for q in questions:
                # Check if we've already processed this message
                message_time = datetime.fromisoformat(q["created_at"])
                if message_time > _last_poll_time[agent_id]:
                    # New message - call handler
                    if agent_id in _message_handlers:
                        message = AgentChatMessage(
                            message_id=q["message_id"],
                            thread_id=q["thread_id"],
                            from_agent=q["from_agent"],
                            to_agent=q["to_agent"],
                            message_type=q["message_type"],
                            content=q["content"],
                            created_at=q["created_at"],
                            metadata={}
                        )
                        await _message_handlers[agent_id](message)

            # Update last poll time
            _last_poll_time[agent_id] = datetime.now()

            # Wait before next poll
            await asyncio.sleep(poll_interval)

        except Exception as e:
            print(f"[{agent_id}] Error in message poller: {e}")
            await asyncio.sleep(poll_interval)


async def start_message_poller(
    agent_id: str,
    agent_chat,
    poll_interval: float = 2.0
):
    """
    Start background message poller for this agent.

    Call this in your app's startup event handler.

    Args:
        agent_id: This agent's ID
        agent_chat: AgentChatClient instance
        poll_interval: How often to poll (seconds)
    """
    if agent_id in _poller_tasks:
        print(f"[{agent_id}] Message poller already running")
        return

    task = asyncio.create_task(_poll_for_messages(agent_id, agent_chat, poll_interval))
    _poller_tasks[agent_id] = task
    print(f"[{agent_id}] Message poller started")


async def stop_message_poller(agent_id: str):
    """Stop background message poller for this agent"""
    if agent_id in _poller_tasks:
        _poller_tasks[agent_id].cancel()
        del _poller_tasks[agent_id]
        print(f"[{agent_id}] Message poller stopped")


# === Helper Functions ===

async def send_message_to_parent(
    thread_id: str,
    from_agent: str,
    parent_agent: str,
    message_type: str,
    content: str,
    agent_chat,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Send message to parent agent.

    Args:
        thread_id: Thread ID
        from_agent: This agent's ID
        parent_agent: Parent agent ID
        message_type: Message type (question, status, completion, error)
        content: Message content
        agent_chat: AgentChatClient instance
        metadata: Optional metadata

    Returns:
        Message ID
    """
    return await agent_chat.send_message(
        thread_id=thread_id,
        from_agent=from_agent,
        to_agent=parent_agent,
        message_type=message_type,
        content=content,
        metadata=metadata or {}
    )


async def send_message_to_child(
    thread_id: str,
    from_agent: str,
    child_agent: str,
    message_type: str,
    content: str,
    agent_chat,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Send message to child agent.

    Args:
        thread_id: Thread ID
        from_agent: This agent's ID
        child_agent: Child agent ID
        message_type: Message type (delegation, answer, escalation, abort)
        content: Message content
        agent_chat: AgentChatClient instance
        metadata: Optional metadata

    Returns:
        Message ID
    """
    return await agent_chat.send_message(
        thread_id=thread_id,
        from_agent=from_agent,
        to_agent=child_agent,
        message_type=message_type,
        content=content,
        metadata=metadata or {}
    )


# === FastAPI Route Injection ===

def add_agent_chat_routes(
    app: FastAPI,
    agent_id: str,
    agent_chat,
    on_message_received: Callable[[AgentChatMessage], None]
):
    """
    Add Agent Chat routes to a FastAPI app.

    This adds the following endpoints:
    - POST /agent-chat/send - Send message to another agent
    - POST /agent-chat/create-thread - Create new conversation thread
    - GET /agent-chat/threads/{run_id} - Get all threads for a run
    - GET /agent-chat/thread/{thread_id} - Get thread with message history
    - POST /agent-chat/close-thread - Close a thread
    - GET /agent-chat/events - SSE endpoint for real-time events

    Args:
        app: FastAPI app instance
        agent_id: This agent's ID
        agent_chat: AgentChatClient instance
        on_message_received: Callback function for incoming messages
    """
    # Register message handler for poller
    _message_handlers[agent_id] = on_message_received

    @app.post("/agent-chat/send")
    async def send_message(request: SendMessageRequest):
        """Send message on existing thread"""
        try:
            message_id = await agent_chat.send_message(
                thread_id=request.thread_id,
                from_agent=agent_id,
                to_agent=request.to_agent,
                message_type=request.message_type,
                content=request.content,
                metadata=request.metadata
            )
            return {"message_id": message_id, "status": "sent"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agent-chat/create-thread")
    async def create_thread(request: CreateThreadRequest):
        """Create new conversation thread"""
        try:
            thread = await agent_chat.create_thread(
                run_id=request.run_id,
                parent_agent_id=request.parent_agent_id,
                child_agent_id=request.child_agent_id,
                initial_message=request.initial_message,
                metadata=request.metadata
            )
            return thread.dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent-chat/threads/{run_id}")
    async def get_threads_by_run(run_id: str):
        """Get all threads for a run"""
        try:
            threads = await agent_chat.get_threads_by_run(run_id)
            return [t.dict() for t in threads]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent-chat/thread/{thread_id}")
    async def get_thread(thread_id: str):
        """Get thread with full message history"""
        try:
            thread = await agent_chat.get_thread(thread_id)
            return thread.dict()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agent-chat/close-thread")
    async def close_thread(
        thread_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Close conversation thread"""
        try:
            await agent_chat.close_thread(
                thread_id=thread_id,
                status=status,
                result=result,
                error=error
            )
            return {"status": "closed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent-chat/events")
    async def events():
        """
        SSE endpoint for real-time agent chat events.

        This provides a Server-Sent Events stream that HMI can subscribe to
        for real-time visualization of agent conversations.
        """
        async def event_generator():
            """Generate SSE events from agent chat activity"""
            # For now, just send heartbeat events
            # In the future, this could stream from event_stream service
            while True:
                yield f"data: {json.dumps({'agent': agent_id, 'status': 'active'})}\n\n"
                await asyncio.sleep(5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    print(f"[{agent_id}] Added Agent Chat routes: /agent-chat/send, /agent-chat/create-thread, /agent-chat/events")
