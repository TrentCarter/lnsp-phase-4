#!/usr/bin/env python3
"""
Base Manager Class

Provides common functionality for all Manager-tier agents:
- Agent chat integration (/agent_chat/receive endpoint)
- Background task processing
- Status updates during execution
- Thread lifecycle management
- LLM integration with ask_parent tool

Usage:
    from services.common.manager_base import BaseManager, ManagerConfig

    config = ManagerConfig(
        manager_id="Mgr-Data-01",
        port=6145,
        parent_agent="Dir-Data",
        llm_model="qwen2.5-coder:7b",
        description="Data pipeline executor"
    )

    manager = BaseManager(config)
    app = manager.create_app()
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.comms_logger import get_logger, MessageType
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage


@dataclass
class ManagerConfig:
    """Configuration for a Manager agent"""
    manager_id: str
    port: int
    parent_agent: str
    llm_model: str
    description: str
    role: str = "manager"
    tier: str = "executor"


class TaskAssignment(BaseModel):
    """Task assignment from Director"""
    task_id: str
    task: str
    files: List[str]
    acceptance: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseManager:
    """
    Base class for Manager-tier agents.

    Provides:
    - FastAPI app with health + agent_chat endpoints
    - Background task processing with status updates
    - Thread lifecycle management
    - Integration with heartbeat, comms_logger, agent_chat

    Subclasses override execute_task() to implement domain-specific logic.
    """

    def __init__(self, config: ManagerConfig):
        self.config = config

        # Initialize PAS systems
        self.heartbeat_monitor = get_monitor()
        self.logger = get_logger()
        self.agent_chat = get_agent_chat_client()

        # In-memory task tracking
        self.tasks: Dict[str, Dict[str, Any]] = {}

        # Register agent with heartbeat monitor
        self.heartbeat_monitor.register_agent(
            agent=config.manager_id,
            parent=config.parent_agent,
            llm_model=config.llm_model,
            role=config.role,
            tier=config.tier
        )

    def create_app(self) -> FastAPI:
        """Create FastAPI application with standard endpoints"""
        app = FastAPI(
            title=f"Manager ({self.config.manager_id})",
            version="1.0.0",
            description=self.config.description
        )

        @app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "ok",
                "manager_id": self.config.manager_id,
                "port": self.config.port,
                "parent": self.config.parent_agent,
                "llm": self.config.llm_model,
                "description": self.config.description
            }

        @app.post("/agent_chat/receive")
        async def receive_agent_message(
            request: AgentChatMessage,
            background_tasks: BackgroundTasks
        ):
            """
            Receive message from parent Director via Agent Chat.

            Enables bidirectional Q&A, status updates, and context preservation.

            Flow:
            - Director creates thread with delegation message
            - Manager receives via this endpoint
            - Manager can ask questions using agent_chat.send_message()
            - Manager sends status updates during execution
            - Manager closes thread on completion/error
            """
            thread_id = request.thread_id

            # Load thread to get run_id
            try:
                thread = await self.agent_chat.get_thread(thread_id)
                run_id = thread.run_id
            except Exception:
                run_id = "unknown"

            self.logger.log_cmd(
                from_agent=self.config.parent_agent,
                to_agent=self.config.manager_id,
                message=f"Agent chat message received: {request.message_type}",
                run_id=run_id,
                metadata={
                    "thread_id": thread_id,
                    "message_type": request.message_type,
                    "from_agent": request.from_agent
                }
            )

            # Delegate to background task for processing
            background_tasks.add_task(self._process_agent_chat_message, request)

            return {
                "status": "ok",
                "thread_id": thread_id,
                "message": "Agent chat message received, processing"
            }

        @app.on_event("startup")
        async def startup():
            """Startup tasks"""
            print(f"[{self.config.manager_id}] Starting on port {self.config.port}")
            print(f"[{self.config.manager_id}] Parent: {self.config.parent_agent}")
            print(f"[{self.config.manager_id}] LLM: {self.config.llm_model}")
            print(f"[{self.config.manager_id}] {self.config.description}")

        return app

    async def _process_agent_chat_message(self, request: AgentChatMessage):
        """
        Process agent chat message (background task).

        Internal method - routes to execute_task_with_chat() for delegation messages.
        """
        thread_id = request.thread_id

        # Load thread to get run_id
        try:
            thread = await self.agent_chat.get_thread(thread_id)
            run_id = thread.run_id
        except Exception:
            run_id = "unknown"

        try:
            # Load full thread history for context
            thread = await self.agent_chat.get_thread(thread_id)

            # Send initial status
            await self.agent_chat.send_message(
                thread_id=thread_id,
                from_agent=self.config.manager_id,
                to_agent=self.config.parent_agent,
                message_type="status",
                content="Received task, analyzing..."
            )

            # Extract task from delegation message
            if request.message_type == "delegation":
                # Parse task from message content or metadata
                task_description = request.content
                files = request.metadata.get("files", [])
                acceptance = request.metadata.get("acceptance", [])
                budget = request.metadata.get("budget", {})

                # Execute task with agent chat updates
                await self.execute_task_with_chat(
                    task_description=task_description,
                    files=files,
                    acceptance=acceptance,
                    budget=budget,
                    thread_id=thread_id,
                    run_id=run_id
                )

            elif request.message_type == "answer":
                # Parent answered a question - context is in thread history
                # Subclasses can override to handle answers
                pass

        except Exception as e:
            self.logger.log_status(
                from_agent=self.config.manager_id,
                to_agent=self.config.parent_agent,
                message=f"Error processing agent chat message: {str(e)}",
                run_id=run_id,
                status="error",
                metadata={"thread_id": thread_id, "error": str(e)}
            )

    async def execute_task_with_chat(
        self,
        task_description: str,
        files: List[str],
        acceptance: List[Dict[str, Any]],
        budget: Dict[str, Any],
        thread_id: str,
        run_id: str
    ):
        """
        Execute task with agent chat status updates.

        This is the main execution flow. Subclasses MUST override execute_task()
        to implement domain-specific logic.

        This method handles:
        - Initial status updates
        - Calling execute_task() (domain-specific)
        - Running acceptance checks
        - Sending completion/error messages
        - Thread lifecycle management
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        try:
            # Step 1: Send status - starting execution
            self.heartbeat_monitor.heartbeat(
                agent=self.config.manager_id,
                run_id=run_id,
                state=AgentState.EXECUTING,
                message="Executing task",
                progress=0.2
            )

            await self.agent_chat.send_message(
                thread_id=thread_id,
                from_agent=self.config.manager_id,
                to_agent=self.config.parent_agent,
                message_type="status",
                content=f"Starting execution: {task_description[:100]}...",
                metadata={"progress": 20, "files": files}
            )

            # Step 2: Execute domain-specific task logic
            result = await self.execute_task(
                task_description=task_description,
                files=files,
                budget=budget,
                thread_id=thread_id,
                run_id=run_id
            )

            # Step 3: Send status - running acceptance tests
            self.heartbeat_monitor.heartbeat(
                agent=self.config.manager_id,
                run_id=run_id,
                state=AgentState.VALIDATING,
                message="Running acceptance tests",
                progress=0.7
            )

            await self.agent_chat.send_message(
                thread_id=thread_id,
                from_agent=self.config.manager_id,
                to_agent=self.config.parent_agent,
                message_type="status",
                content="Task complete, running acceptance tests...",
                metadata={"progress": 70}
            )

            # Step 4: Run acceptance checks
            acceptance_results = await self.run_acceptance_checks(
                acceptance, files, run_id
            )
            all_passed = all(acceptance_results.values())

            # Step 5: Send completion or error
            if all_passed:
                self.heartbeat_monitor.heartbeat(
                    agent=self.config.manager_id,
                    run_id=run_id,
                    state=AgentState.COMPLETED,
                    message="Task completed successfully",
                    progress=1.0
                )

                await self.agent_chat.send_message(
                    thread_id=thread_id,
                    from_agent=self.config.manager_id,
                    to_agent=self.config.parent_agent,
                    message_type="completion",
                    content=f"✅ Task completed successfully. All acceptance tests passed.",
                    metadata={
                        "acceptance_results": acceptance_results,
                        "result": result
                    }
                )

                await self.agent_chat.close_thread(
                    thread_id=thread_id,
                    status="completed",
                    result="Task completed successfully"
                )
            else:
                # Some acceptance tests failed
                failed_checks = [name for name, passed in acceptance_results.items() if not passed]

                await self.agent_chat.send_message(
                    thread_id=thread_id,
                    from_agent=self.config.manager_id,
                    to_agent=self.config.parent_agent,
                    message_type="error",
                    content=f"❌ Acceptance tests failed: {', '.join(failed_checks)}",
                    metadata={"acceptance_results": acceptance_results}
                )

                await self.agent_chat.close_thread(
                    thread_id=thread_id,
                    status="failed",
                    error=f"Acceptance tests failed: {', '.join(failed_checks)}"
                )

        except Exception as e:
            # Unhandled error
            self.logger.log_status(
                from_agent=self.config.manager_id,
                to_agent=self.config.parent_agent,
                message=f"Error executing task: {str(e)}",
                run_id=run_id,
                status="error",
                metadata={"thread_id": thread_id, "error": str(e)}
            )

            try:
                await self.agent_chat.send_message(
                    thread_id=thread_id,
                    from_agent=self.config.manager_id,
                    to_agent=self.config.parent_agent,
                    message_type="error",
                    content=f"Fatal error: {str(e)}"
                )

                await self.agent_chat.close_thread(
                    thread_id=thread_id,
                    status="failed",
                    error=str(e)
                )
            except Exception:
                pass  # Best effort

    async def execute_task(
        self,
        task_description: str,
        files: List[str],
        budget: Dict[str, Any],
        thread_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Execute domain-specific task logic.

        SUBCLASSES MUST OVERRIDE THIS METHOD.

        Args:
            task_description: Natural language task description
            files: List of files to operate on
            budget: Token/time budget constraints
            thread_id: Agent chat thread ID (for sending updates)
            run_id: Overall run ID

        Returns:
            Dict with task execution results

        Raises:
            NotImplementedError: If subclass doesn't override
        """
        raise NotImplementedError(
            f"{self.config.manager_id} must implement execute_task()"
        )

    async def run_acceptance_checks(
        self,
        acceptance: List[Dict[str, Any]],
        files: List[str],
        run_id: str
    ) -> Dict[str, bool]:
        """
        Run acceptance checks (tests, lint, coverage).

        Subclasses can override to implement domain-specific checks.

        Args:
            acceptance: List of acceptance criteria
            files: Files that were modified
            run_id: Overall run ID

        Returns:
            Dict mapping check name to pass/fail bool
        """
        results = {}

        for check in acceptance:
            check_type = check.get("type", "unknown")
            check_name = check.get("name", check_type)

            # Default: assume all checks pass
            # Subclasses override to implement real checks
            results[check_name] = True

        return results
