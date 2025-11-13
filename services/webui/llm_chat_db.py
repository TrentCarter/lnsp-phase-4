"""
SQLAlchemy ORM Models for LLM Chat Interface (V1 - SQLite)

Database: services/webui/data/llm_chat.db

Models:
- ConversationSession: Chat sessions with agent/model binding
- Message: Individual messages in conversations

Following PRD v1.2 Persistence Strategy:
- V1: SQLite + SQLAlchemy ORM exclusively
- JSON fields: TEXT columns + json.dumps()/json.loads()
- UUIDs: Client-side uuid.uuid4()
"""

from sqlalchemy import create_engine, Column, String, Text, Integer, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from datetime import datetime
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

# Database location
DB_DIR = Path(__file__).parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "llm_chat.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


class ConversationSession(Base):
    """
    Chat session with a specific agent and model.

    Key constraint: One agent per session (enforces context isolation).
    Switching agents requires creating a new session.
    """
    __tablename__ = 'conversation_sessions'

    # Primary key
    session_id = Column(String(36), primary_key=True)  # UUID4 string

    # User identification
    user_id = Column(String(255), nullable=False, index=True)

    # Agent binding (enforces context isolation)
    agent_id = Column(String(50), nullable=False, index=True)  # e.g., "architect", "dir-code"
    agent_name = Column(String(100), nullable=False)  # e.g., "Architect", "Dir-Code"
    parent_role = Column(String(50), nullable=False)  # User's assumed role (e.g., "PAS Root")

    # Model binding
    model_id = Column(String(50), nullable=True)  # Optional model ID (e.g., "claude-sonnet-4")
    model_name = Column(String(100), nullable=False)  # e.g., "Claude Sonnet 4"

    # Timestamps (stored as ISO 8601 strings for SQLite compatibility)
    created_at = Column(String(32), nullable=False)  # ISO format: "2025-11-12T10:30:00Z"
    updated_at = Column(String(32), nullable=False)

    # Status
    status = Column(String(20), nullable=False, default='active', index=True)  # active, archived, deleted
    archived_at = Column(String(32), nullable=True)  # ISO format when archived

    # Optional title (user-provided or auto-generated)
    title = Column(String(255), nullable=True)

    # Metadata (TEXT column with JSON serialization)
    metadata_str = Column('metadata', Text, default='{}')

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def __init__(self, user_id: str, agent_id: str, agent_name: str, parent_role: str,
                 model_name: str, model_id: Optional[str] = None, title: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.parent_role = parent_role
        self.model_id = model_id
        self.model_name = model_name
        self.created_at = datetime.utcnow().isoformat() + 'Z'
        self.updated_at = self.created_at
        self.status = 'active'
        self.title = title
        self.metadata_str = json.dumps(metadata or {})

    def get_metadata(self) -> Dict[str, Any]:
        """Deserialize metadata from JSON string"""
        return json.loads(self.metadata_str) if self.metadata_str else {}

    def set_metadata(self, value: Dict[str, Any]):
        """Serialize metadata to JSON string"""
        self.metadata_str = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'parent_role': self.parent_role,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status': self.status,
            'archived_at': self.archived_at,
            'title': self.title,
            'metadata': self.get_metadata()
        }


class Message(Base):
    """
    Individual message in a conversation.

    Supports user messages, assistant responses, system messages, and status updates.
    Includes token usage tracking and task status.
    """
    __tablename__ = 'messages'

    # Primary key
    message_id = Column(String(36), primary_key=True)  # UUID4 string

    # Foreign key to session
    session_id = Column(String(36), ForeignKey('conversation_sessions.session_id', ondelete='CASCADE'),
                       nullable=False, index=True)

    # Message type (user, assistant, system, status)
    message_type = Column(String(20), nullable=False, index=True)

    # Legacy role field (retained for compatibility)
    role = Column(String(20), nullable=True)

    # Content
    content = Column(Text, nullable=False)

    # Agent that generated this message (for assistant messages)
    agent_id = Column(String(50), nullable=True)

    # Model used for this message
    model_name = Column(String(100), nullable=True)

    # Timestamp (ISO 8601 string)
    timestamp = Column(String(32), nullable=False, index=True)

    # Task status (for status-type messages)
    status = Column(String(50), nullable=True)  # planning, executing, complete, error, awaiting_approval

    # Token usage (TEXT column with JSON serialization)
    usage_json = Column('usage', Text, nullable=True)

    # Additional metadata (TEXT column with JSON serialization)
    metadata_str = Column('metadata', Text, default='{}')

    # Relationships
    session = relationship("ConversationSession", back_populates="messages")

    def __init__(self, session_id: str, message_type: str, content: str,
                 agent_id: Optional[str] = None, model_name: Optional[str] = None,
                 status: Optional[str] = None, usage: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.message_id = str(uuid.uuid4())
        self.session_id = session_id
        self.message_type = message_type
        self.role = message_type if message_type in ['user', 'assistant', 'system'] else None
        self.content = content
        self.agent_id = agent_id
        self.model_name = model_name
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.status = status
        self.usage_json = json.dumps(usage) if usage else None
        self.metadata_str = json.dumps(metadata or {})

    def get_usage(self) -> Optional[Dict[str, Any]]:
        """Deserialize usage from JSON string"""
        return json.loads(self.usage_json) if self.usage_json else None

    def set_usage(self, value: Optional[Dict[str, Any]]):
        """Serialize usage to JSON string"""
        self.usage_json = json.dumps(value) if value else None

    def get_metadata(self) -> Dict[str, Any]:
        """Deserialize metadata from JSON string"""
        return json.loads(self.metadata_str) if self.metadata_str else {}

    def set_metadata(self, value: Dict[str, Any]):
        """Serialize metadata to JSON string"""
        self.metadata_str = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'message_id': self.message_id,
            'session_id': self.session_id,
            'message_type': self.message_type,
            'role': self.role,
            'content': self.content,
            'agent_id': self.agent_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'status': self.status,
            'usage': self.get_usage(),
            'metadata': self.get_metadata()
        }


# Create indices for performance
Index('idx_messages_session_timestamp', Message.session_id, Message.timestamp)


def init_db():
    """Initialize database (create tables if they don't exist)"""
    Base.metadata.create_all(bind=engine)
    print(f"✓ Database initialized at {DB_PATH}")


def get_session():
    """Get database session (for use in routes)"""
    return SessionLocal()


# Initialize database on module import
init_db()
