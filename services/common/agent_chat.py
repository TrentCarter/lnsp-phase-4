#!/usr/bin/env python3
"""
Agent Chat Client - Parent ↔ Child Communication System

Provides stateful conversation threads between Parent and Child agents with:
- Full message history preservation
- Bidirectional messaging (Parent ↔ Child)
- Status updates and progress tracking
- Thread lifecycle management (create → active → completed/failed)

Database: artifacts/registry/registry.db
Tables: agent_conversation_threads, agent_conversation_messages

See: docs/PRDs/PRD_Parent_Child_Chat_Communications.md
"""
import sqlite3
import uuid
import json
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime, timezone
from pydantic import BaseModel


# Database location (shared with registry)
DB_PATH = Path("artifacts/registry/registry.db")

# Event Stream URL for HMI visualization
EVENT_STREAM_URL = "http://localhost:6102"


async def _emit_agent_chat_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Emit agent chat event to Event Stream for HMI visualization.

    Non-blocking: Failures don't affect agent chat operations.

    Args:
        event_type: Event type (e.g., 'thread_created', 'message_sent', 'thread_closed')
        data: Event data payload
    """
    try:
        payload = {
            "event_type": f"agent_chat_{event_type}",
            "data": data
        }

        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.post(f"{EVENT_STREAM_URL}/broadcast", json=payload)
    except Exception:
        # Don't fail operations if event broadcast fails
        # HMI visualization is non-critical
        pass


class AgentChatMessage(BaseModel):
    """Single message in an agent conversation thread"""
    message_id: str
    thread_id: str
    from_agent: str
    to_agent: str
    message_type: str  # delegation, question, answer, status, completion, error, escalation, abort
    content: str
    created_at: str
    metadata: Dict[str, Any] = {}


class AgentChatThread(BaseModel):
    """Agent conversation thread with full message history"""
    thread_id: str
    run_id: str
    parent_agent_id: str
    child_agent_id: str
    status: str  # active, completed, failed, timeout, abandoned
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    messages: List[AgentChatMessage] = []


class AgentChatClient:
    """
    Client for agent-to-agent conversation threads.

    Usage:
        # Parent creates thread
        client = AgentChatClient()
        thread = await client.create_thread(
            run_id="run-123",
            parent_agent_id="Architect",
            child_agent_id="Dir-Code",
            initial_message="Refactor auth to OAuth2"
        )

        # Child loads thread and asks question
        thread = await client.get_thread(thread_id)
        await client.send_message(
            thread_id=thread.thread_id,
            from_agent="Dir-Code",
            to_agent="Architect",
            message_type="question",
            content="Should I use authlib or python-oauth2?"
        )

        # Parent answers
        await client.send_message(
            thread_id=thread.thread_id,
            from_agent="Architect",
            to_agent="Dir-Code",
            message_type="answer",
            content="Use authlib - better maintained"
        )

        # Child completes task
        await client.close_thread(
            thread_id=thread.thread_id,
            status="completed",
            result="Successfully refactored JWT auth to OAuth2"
        )
    """

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize agent chat client"""
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create agent_conversation_threads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_conversation_threads (
                thread_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                parent_agent_id TEXT NOT NULL,
                child_agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                result TEXT,
                error TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Create agent_conversation_messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_conversation_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (thread_id) REFERENCES agent_conversation_threads(thread_id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_run
            ON agent_conversation_threads(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_status
            ON agent_conversation_threads(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_parent
            ON agent_conversation_threads(parent_agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_child
            ON agent_conversation_threads(child_agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_thread
            ON agent_conversation_messages(thread_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_created
            ON agent_conversation_messages(created_at)
        """)

        conn.commit()
        conn.close()

    async def create_thread(
        self,
        run_id: str,
        parent_agent_id: str,
        child_agent_id: str,
        initial_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentChatThread:
        """
        Create new conversation thread with initial delegation message.

        Args:
            run_id: Prime Directive run ID
            parent_agent_id: Parent agent (e.g., "Architect")
            child_agent_id: Child agent (e.g., "Dir-Code")
            initial_message: Initial delegation message
            metadata: Optional metadata (entry_files, budget, etc.)

        Returns:
            AgentChatThread with initial message
        """
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Create thread
            cursor.execute("""
                INSERT INTO agent_conversation_threads
                (thread_id, run_id, parent_agent_id, child_agent_id, created_at, updated_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
            """, (thread_id, run_id, parent_agent_id, child_agent_id, now, now, json.dumps(metadata or {})))

            # Add initial delegation message
            message_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO agent_conversation_messages
                (message_id, thread_id, from_agent, to_agent, message_type, content, created_at, metadata)
                VALUES (?, ?, ?, ?, 'delegation', ?, ?, '{}')
            """, (message_id, thread_id, parent_agent_id, child_agent_id, initial_message, now))

            conn.commit()
        finally:
            conn.close()

        # Broadcast thread creation event to HMI
        await _emit_agent_chat_event('thread_created', {
            'thread_id': thread_id,
            'run_id': run_id,
            'parent_agent': parent_agent_id,
            'child_agent': child_agent_id,
            'metadata': metadata or {},
            'timestamp': now
        })

        return await self.get_thread(thread_id)

    async def send_message(
        self,
        thread_id: str,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send message on existing thread.

        Args:
            thread_id: Thread ID
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            message_type: One of: question, answer, status, completion, error, escalation, abort
            content: Message content
            metadata: Optional metadata (progress, tool_calls, etc.)

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO agent_conversation_messages
                (message_id, thread_id, from_agent, to_agent, message_type, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (message_id, thread_id, from_agent, to_agent, message_type, content, now, json.dumps(metadata or {})))

            # Update thread timestamp
            cursor.execute("""
                UPDATE agent_conversation_threads SET updated_at = ? WHERE thread_id = ?
            """, (now, thread_id))

            conn.commit()
        finally:
            conn.close()

        # Broadcast message sent event to HMI
        await _emit_agent_chat_event('message_sent', {
            'message_id': message_id,
            'thread_id': thread_id,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'message_type': message_type,
            'content': content[:100] if content else '',  # Truncate for event stream
            'metadata': metadata or {},
            'timestamp': now
        })

        return message_id

    async def get_thread(self, thread_id: str) -> AgentChatThread:
        """
        Get thread with full message history.

        Args:
            thread_id: Thread ID

        Returns:
            AgentChatThread with all messages

        Raises:
            ValueError: If thread not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get thread
            cursor.execute("""
                SELECT thread_id, run_id, parent_agent_id, child_agent_id,
                       created_at, updated_at, completed_at, status, result, error, metadata
                FROM agent_conversation_threads
                WHERE thread_id = ?
            """, (thread_id,))

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Thread {thread_id} not found")

            # Get messages
            cursor.execute("""
                SELECT message_id, thread_id, from_agent, to_agent, message_type, content, created_at, metadata
                FROM agent_conversation_messages
                WHERE thread_id = ?
                ORDER BY created_at ASC
            """, (thread_id,))

            messages = [
                AgentChatMessage(
                    message_id=r[0],
                    thread_id=r[1],
                    from_agent=r[2],
                    to_agent=r[3],
                    message_type=r[4],
                    content=r[5],
                    created_at=r[6],
                    metadata=json.loads(r[7]) if r[7] else {}
                )
                for r in cursor.fetchall()
            ]
        finally:
            conn.close()

        return AgentChatThread(
            thread_id=row[0],
            run_id=row[1],
            parent_agent_id=row[2],
            child_agent_id=row[3],
            created_at=row[4],
            updated_at=row[5],
            completed_at=row[6],
            status=row[7],
            result=row[8],
            error=row[9],
            metadata=json.loads(row[10]) if row[10] else {},
            messages=messages
        )

    async def get_threads_by_run(self, run_id: str) -> List[AgentChatThread]:
        """
        Get all threads for a run.

        Args:
            run_id: Prime Directive run ID

        Returns:
            List of AgentChatThread (without full message history - use get_thread for that)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT thread_id, run_id, parent_agent_id, child_agent_id,
                       created_at, updated_at, completed_at, status, result, error, metadata
                FROM agent_conversation_threads
                WHERE run_id = ?
                ORDER BY created_at DESC
            """, (run_id,))

            threads = [
                AgentChatThread(
                    thread_id=r[0],
                    run_id=r[1],
                    parent_agent_id=r[2],
                    child_agent_id=r[3],
                    created_at=r[4],
                    updated_at=r[5],
                    completed_at=r[6],
                    status=r[7],
                    result=r[8],
                    error=r[9],
                    metadata=json.loads(r[10]) if r[10] else {},
                    messages=[]  # Don't load messages for list view
                )
                for r in cursor.fetchall()
            ]
        finally:
            conn.close()

        return threads

    async def close_thread(
        self,
        thread_id: str,
        status: str,  # "completed", "failed", "timeout", "abandoned"
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Close conversation thread.

        Args:
            thread_id: Thread ID
            status: Final status (completed, failed, timeout, abandoned)
            result: Success message (if completed)
            error: Error message (if failed)
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE agent_conversation_threads
                SET status = ?, completed_at = ?, result = ?, error = ?, updated_at = ?
                WHERE thread_id = ?
            """, (status, now, result, error, now, thread_id))

            conn.commit()
        finally:
            conn.close()

        # Broadcast thread closure event to HMI
        await _emit_agent_chat_event('thread_closed', {
            'thread_id': thread_id,
            'status': status,
            'result': result,
            'error': error,
            'timestamp': now
        })

    async def get_pending_questions(self, parent_agent_id: str) -> List[Dict[str, Any]]:
        """
        Get pending questions for a parent agent (messages awaiting response).

        Args:
            parent_agent_id: Parent agent ID (e.g., "Architect")

        Returns:
            List of questions with thread context
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get latest message for each active thread where parent is recipient
            cursor.execute("""
                SELECT
                    m.message_id, m.thread_id, m.from_agent, m.to_agent,
                    m.message_type, m.content, m.created_at,
                    t.run_id, t.child_agent_id
                FROM agent_conversation_messages m
                JOIN agent_conversation_threads t ON m.thread_id = t.thread_id
                WHERE t.parent_agent_id = ?
                  AND t.status = 'active'
                  AND m.message_type = 'question'
                  AND m.to_agent = ?
                ORDER BY m.created_at DESC
            """, (parent_agent_id, parent_agent_id))

            questions = [
                {
                    "message_id": r[0],
                    "thread_id": r[1],
                    "from_agent": r[2],
                    "to_agent": r[3],
                    "message_type": r[4],
                    "content": r[5],
                    "created_at": r[6],
                    "run_id": r[7],
                    "child_agent_id": r[8]
                }
                for r in cursor.fetchall()
            ]
        finally:
            conn.close()

        return questions

    def get_message_count(self, thread_id: str) -> int:
        """Get total message count for a thread"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT COUNT(*) FROM agent_conversation_messages WHERE thread_id = ?
            """, (thread_id,))
            count = cursor.fetchone()[0]
        finally:
            conn.close()

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get overall agent chat statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Total threads by status
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM agent_conversation_threads
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Average messages per thread
            cursor.execute("""
                SELECT AVG(msg_count) FROM (
                    SELECT COUNT(*) as msg_count
                    FROM agent_conversation_messages
                    GROUP BY thread_id
                )
            """)
            avg_messages = cursor.fetchone()[0] or 0

            # Total messages
            cursor.execute("SELECT COUNT(*) FROM agent_conversation_messages")
            total_messages = cursor.fetchone()[0]

        finally:
            conn.close()

        return {
            "total_threads": sum(status_counts.values()),
            "threads_by_status": status_counts,
            "total_messages": total_messages,
            "avg_messages_per_thread": round(avg_messages, 2)
        }


# Global singleton instance
_client = None

def get_agent_chat_client() -> AgentChatClient:
    """Get singleton agent chat client"""
    global _client
    if _client is None:
        _client = AgentChatClient()
    return _client
