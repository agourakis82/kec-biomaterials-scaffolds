"""
Darwin Platform Memory Router

Advanced conversation memory and context management.
Supports multiple projects, knowledge areas, and conversation threads.
Provides persistent context for GPT interactions.
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from kec_biomat_api.security import rate_limit, require_api_key
from kec_biomat_api.services.rag_vertex import get_rag

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memory",
    tags=["Memory"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""

    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ConversationEntry(BaseModel):
    """Conversation entry model."""

    conversation_id: str = Field(..., description="Unique conversation ID")
    project_id: str = Field(..., description="Project identifier")
    knowledge_area: str = Field(..., description="Knowledge area/domain")
    title: str = Field(..., description="Conversation title/summary")
    messages: List[ConversationMessage] = Field(default_factory=list, description="Conversation messages")
    tags: List[str] = Field(default_factory=list, description="Conversation tags")
    is_active: bool = Field(default=True, description="Whether conversation is active")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class ConversationResponse(BaseModel):
    """Conversation response model."""

    conversation_id: str = Field(..., description="Unique conversation ID")
    project_id: str = Field(..., description="Project identifier")
    knowledge_area: str = Field(..., description="Knowledge area/domain")
    title: str = Field(..., description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    last_updated: datetime = Field(..., description="Last update timestamp")
    tags: List[str] = Field(..., description="Conversation tags")
    is_active: bool = Field(..., description="Whether conversation is active")


class ProjectInfo(BaseModel):
    """Project information model."""

    project_id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    knowledge_area: str = Field(..., description="Primary knowledge area")
    description: str = Field(..., description="Project description")
    conversation_count: int = Field(..., description="Number of conversations")
    last_activity: datetime = Field(..., description="Last activity timestamp")


class ContextRetrievalRequest(BaseModel):
    """Context retrieval request model."""

    conversation_id: Optional[str] = Field(None, description="Specific conversation ID")
    project_id: Optional[str] = Field(None, description="Project identifier")
    knowledge_area: Optional[str] = Field(None, description="Knowledge area filter")
    query: Optional[str] = Field(None, description="Search query")
    max_messages: int = Field(50, description="Maximum messages to retrieve")
    include_metadata: bool = Field(True, description="Include message metadata")


class ContextRetrievalResponse(BaseModel):
    """Context retrieval response model."""

    conversation_id: str = Field(..., description="Conversation ID")
    project_id: str = Field(..., description="Project identifier")
    knowledge_area: str = Field(..., description="Knowledge area")
    messages: List[ConversationMessage] = Field(..., description="Retrieved messages")
    total_messages: int = Field(..., description="Total messages in conversation")
    last_updated: datetime = Field(..., description="Last update timestamp")


class SessionLogEntry(BaseModel):
    """Session log entry model."""

    summary: str = Field(..., description="Session summary")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    actor: Optional[str] = Field(None, description="Session actor/user")


class SessionLogResponse(BaseModel):
    """Session log response model."""

    request_id: str = Field(..., description="Unique request ID")
    timestamp_utc: datetime = Field(..., description="UTC timestamp")
    summary: str = Field(..., description="Session summary")
    tags: List[str] = Field(..., description="Session tags")
    actor: Optional[str] = Field(None, description="Session actor/user")


class MemorySearchResponse(BaseModel):
    """Memory search response model."""

    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total results found")


class AdvancedMemoryStorage:
    """Advanced memory storage manager with conversation support."""

    def __init__(self, storage_path: str = "data/memory"):
        """
        Initialize advanced memory storage.

        Args:
            storage_path: Base path for storage files
        """
        self.storage_path = storage_path
        self.conversations_file = os.path.join(storage_path, "conversations.jsonl")
        self.projects_file = os.path.join(storage_path, "projects.jsonl")
        self.db_file = os.path.join(storage_path, "memory.db")

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for conversations and projects."""
        with sqlite3.connect(self.db_file) as conn:
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    knowledge_area TEXT NOT NULL,
                    title TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Projects table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    knowledge_area TEXT NOT NULL,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Sessions table (legacy)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    actor TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_project 
                ON conversations(project_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_area 
                ON conversations(knowledge_area)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_active 
                ON conversations(is_active)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_area 
                ON projects(knowledge_area)
            """)

    async def create_or_update_conversation(self, entry: ConversationEntry) -> ConversationResponse:
        """
        Create or update a conversation.

        Args:
            entry: Conversation entry

        Returns:
            Conversation response
        """
        conversation_id = entry.conversation_id or str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Ensure project exists
        await self._ensure_project_exists(entry.project_id, entry.knowledge_area)

        # Prepare messages for storage
        messages_data = []
        for msg in entry.messages:
            messages_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else timestamp.isoformat(),
                "metadata": msg.metadata or {}
            })

        try:
            with sqlite3.connect(self.db_file) as conn:
                # Insert or replace conversation
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, project_id, knowledge_area, title, messages, tags, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        conversation_id,
                        entry.project_id,
                        entry.knowledge_area,
                        entry.title,
                        json.dumps(messages_data),
                        json.dumps(entry.tags),
                        entry.is_active,
                        timestamp.isoformat(),
                    ),
                )

                # Update project last activity
                conn.execute(
                    """
                    UPDATE projects 
                    SET last_activity = ? 
                    WHERE project_id = ?
                """,
                    (timestamp.isoformat(), entry.project_id),
                )

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise HTTPException(status_code=500, detail="Failed to save conversation")

        logger.info(f"Saved conversation: {conversation_id}")
        return ConversationResponse(
            conversation_id=conversation_id,
            project_id=entry.project_id,
            knowledge_area=entry.knowledge_area,
            title=entry.title,
            message_count=len(entry.messages),
            last_updated=timestamp,
            tags=entry.tags,
            is_active=entry.is_active,
        )

    async def _ensure_project_exists(self, project_id: str, knowledge_area: str):
        """Ensure project exists in database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.execute(
                    "SELECT project_id FROM projects WHERE project_id = ?",
                    (project_id,),
                )
                if not cursor.fetchone():
                    conn.execute(
                        """
                        INSERT INTO projects (project_id, name, knowledge_area, description)
                        VALUES (?, ?, ?, ?)
                    """,
                        (project_id, project_id, knowledge_area, f"Project {project_id}"),
                    )
        except Exception as e:
            logger.error(f"Failed to ensure project exists: {e}")

    async def get_conversation_context(self, request: ContextRetrievalRequest) -> List[ContextRetrievalResponse]:
        """
        Retrieve conversation context.

        Args:
            request: Context retrieval request

        Returns:
            List of conversation contexts
        """
        results = []

        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row

                # Build query based on filters
                query_parts = ["SELECT * FROM conversations WHERE 1=1"]
                params = []

                if request.conversation_id:
                    query_parts.append("AND conversation_id = ?")
                    params.append(request.conversation_id)

                if request.project_id:
                    query_parts.append("AND project_id = ?")
                    params.append(request.project_id)

                if request.knowledge_area:
                    query_parts.append("AND knowledge_area = ?")
                    params.append(request.knowledge_area)

                query_parts.append("ORDER BY last_updated DESC")

                cursor = conn.execute(" ".join(query_parts), params)

                for row in cursor:
                    messages_data = json.loads(row["messages"])
                    messages = []

                    # Convert stored messages back to objects
                    for msg_data in messages_data[-request.max_messages:]:  # Get last N messages
                        messages.append(ConversationMessage(
                            role=msg_data["role"],
                            content=msg_data["content"],
                            timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                            metadata=msg_data.get("metadata", {})
                        ))

                    results.append(ContextRetrievalResponse(
                        conversation_id=row["conversation_id"],
                        project_id=row["project_id"],
                        knowledge_area=row["knowledge_area"],
                        messages=messages,
                        total_messages=len(messages_data),
                        last_updated=datetime.fromisoformat(row["last_updated"]),
                    ))

                    # If specific conversation requested, return only that one
                    if request.conversation_id:
                        break

        except Exception as e:
            logger.error(f"Failed to retrieve conversation context: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve context")

        return results

    async def list_projects(self) -> List[ProjectInfo]:
        """List all projects with conversation counts."""
        projects = []

        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute("""
                    SELECT p.*,
                           COUNT(c.conversation_id) as conversation_count
                    FROM projects p
                    LEFT JOIN conversations c ON p.project_id = c.project_id
                    GROUP BY p.project_id
                    ORDER BY p.last_activity DESC
                """)

                for row in cursor:
                    projects.append(ProjectInfo(
                        project_id=row["project_id"],
                        name=row["name"],
                        knowledge_area=row["knowledge_area"],
                        description=row["description"],
                        conversation_count=row["conversation_count"],
                        last_activity=datetime.fromisoformat(row["last_activity"]),
                    ))

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")

        return projects

    async def search_conversations(self, query: str, project_id: Optional[str] = None,
                                 knowledge_area: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations by content."""
        results = []

        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row

                query_parts = ["""
                    SELECT conversation_id, project_id, knowledge_area, title, messages, tags, last_updated
                    FROM conversations
                    WHERE (title LIKE ? OR messages LIKE ? OR tags LIKE ?)
                """]

                params = [f"%{query}%", f"%{query}%", f"%{query}%"]

                if project_id:
                    query_parts.append("AND project_id = ?")
                    params.append(project_id)

                if knowledge_area:
                    query_parts.append("AND knowledge_area = ?")
                    params.append(knowledge_area)

                query_parts.append("ORDER BY last_updated DESC LIMIT ?")
                params.append(limit)

                cursor = conn.execute(" ".join(query_parts), params)

                for row in cursor:
                    messages_data = json.loads(row["messages"])
                    results.append({
                        "conversation_id": row["conversation_id"],
                        "project_id": row["project_id"],
                        "knowledge_area": row["knowledge_area"],
                        "title": row["title"],
                        "message_count": len(messages_data),
                        "tags": json.loads(row["tags"]),
                        "last_updated": row["last_updated"],
                        "preview": messages_data[-1]["content"][:200] if messages_data else "",
                    })

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")

        return results

    # Legacy methods for backward compatibility
    async def log_session(self, entry: SessionLogEntry) -> SessionLogResponse:
        """Legacy session logging."""
        request_id = str(uuid.uuid4())
        timestamp_utc = datetime.now(timezone.utc)

        response = SessionLogResponse(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            summary=entry.summary,
            tags=entry.tags,
            actor=entry.actor,
        )

        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute(
                    """
                    INSERT INTO sessions 
                    (request_id, timestamp_utc, summary, tags, actor)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        request_id,
                        timestamp_utc.isoformat(),
                        entry.summary,
                        json.dumps(entry.tags),
                        entry.actor,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to log session: {e}")

        return response

    async def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Legacy session search."""
        results = []

        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    """
                    SELECT request_id, timestamp_utc, summary, tags, actor
                    FROM sessions
                    WHERE summary LIKE ? OR tags LIKE ?
                    ORDER BY timestamp_utc DESC
                    LIMIT ?
                """,
                    (f"%{query}%", f"%{query}%", limit),
                )

                for row in cursor:
                    results.append({
                        "request_id": row["request_id"],
                        "timestamp_utc": row["timestamp_utc"],
                        "summary": row["summary"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "actor": row["actor"],
                    })

        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")

        return results


# Global memory storage instance
memory_storage = AdvancedMemoryStorage()


# Conversation Management Endpoints

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(entry: ConversationEntry) -> ConversationResponse:
    """
    Create or update a conversation with messages.

    Args:
        entry: Conversation entry with messages

    Returns:
        Conversation response
    """
    return await memory_storage.create_or_update_conversation(entry)


@router.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(
    conversation_id: str,
    message: ConversationMessage,
    project_id: str = Query(..., description="Project identifier"),
    knowledge_area: str = Query(..., description="Knowledge area"),
) -> Dict[str, Any]:
    """
    Add a message to an existing conversation.

    Args:
        conversation_id: Conversation ID
        message: Message to add
        project_id: Project identifier
        knowledge_area: Knowledge area

    Returns:
        Success response
    """
    # First, get existing conversation
    context_request = ContextRetrievalRequest(
        conversation_id=conversation_id,
        max_messages=1000  # Get all messages
    )

    existing_contexts = await memory_storage.get_conversation_context(context_request)

    if not existing_contexts:
        # Create new conversation
        new_entry = ConversationEntry(
            conversation_id=conversation_id,
            project_id=project_id,
            knowledge_area=knowledge_area,
            title=f"Conversation {conversation_id}",
            messages=[message],
        )
    else:
        # Update existing conversation
        existing = existing_contexts[0]
        updated_messages = existing.messages + [message]
        new_entry = ConversationEntry(
            conversation_id=conversation_id,
            project_id=existing.project_id,
            knowledge_area=existing.knowledge_area,
            title=f"Conversation {conversation_id}",
            messages=updated_messages,
        )

    result = await memory_storage.create_or_update_conversation(new_entry)

    return {
        "success": True,
        "conversation_id": result.conversation_id,
        "message_added": True,
        "total_messages": result.message_count,
    }


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    project_id: Optional[str] = Query(None, description="Filter by project"),
    knowledge_area: Optional[str] = Query(None, description="Filter by knowledge area"),
    active_only: bool = Query(True, description="Show only active conversations"),
) -> List[ConversationResponse]:
    """
    List conversations with optional filters.

    Args:
        project_id: Filter by project
        knowledge_area: Filter by knowledge area
        active_only: Show only active conversations

    Returns:
        List of conversations
    """
    try:
        with sqlite3.connect(memory_storage.db_file) as conn:
            conn.row_factory = sqlite3.Row

            query_parts = ["SELECT * FROM conversations WHERE 1=1"]
            params = []

            if project_id:
                query_parts.append("AND project_id = ?")
                params.append(project_id)

            if knowledge_area:
                query_parts.append("AND knowledge_area = ?")
                params.append(knowledge_area)

            if active_only:
                query_parts.append("AND is_active = 1")

            query_parts.append("ORDER BY last_updated DESC")

            cursor = conn.execute(" ".join(query_parts), params)

            results = []
            for row in cursor:
                messages_data = json.loads(row["messages"])
                results.append(ConversationResponse(
                    conversation_id=row["conversation_id"],
                    project_id=row["project_id"],
                    knowledge_area=row["knowledge_area"],
                    title=row["title"],
                    message_count=len(messages_data),
                    last_updated=datetime.fromisoformat(row["last_updated"]),
                    tags=json.loads(row["tags"]),
                    is_active=bool(row["is_active"]),
                ))

            return results

    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@router.get("/conversations/{conversation_id}/context", response_model=ContextRetrievalResponse)
async def get_conversation_context(
    conversation_id: str,
    max_messages: int = Query(50, description="Maximum messages to retrieve"),
) -> ContextRetrievalResponse:
    """
    Get conversation context for GPT.

    Args:
        conversation_id: Conversation ID
        max_messages: Maximum messages to retrieve

    Returns:
        Conversation context
    """
    request = ContextRetrievalRequest(
        conversation_id=conversation_id,
        max_messages=max_messages,
    )

    contexts = await memory_storage.get_conversation_context(request)

    if not contexts:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return contexts[0]


@router.post("/context/retrieve", response_model=List[ContextRetrievalResponse])
async def retrieve_context(request: ContextRetrievalRequest) -> List[ContextRetrievalResponse]:
    """
    Retrieve conversation context with advanced filtering.

    Args:
        request: Context retrieval request

    Returns:
        List of conversation contexts
    """
    return await memory_storage.get_conversation_context(request)


@router.get("/projects", response_model=List[ProjectInfo])
async def list_projects() -> List[ProjectInfo]:
    """
    List all projects with conversation counts.

    Returns:
        List of projects
    """
    return await memory_storage.list_projects()


@router.get("/conversations/search")
async def search_conversations(
    q: str = Query(..., description="Search query"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    knowledge_area: Optional[str] = Query(None, description="Filter by knowledge area"),
    limit: int = Query(10, description="Maximum results"),
) -> Dict[str, Any]:
    """
    Search conversations by content.

    Args:
        q: Search query
        project_id: Filter by project
        knowledge_area: Filter by knowledge area
        limit: Maximum results

    Returns:
        Search results
    """
    results = await memory_storage.search_conversations(q, project_id, knowledge_area, limit)

    return {
        "query": q,
        "results": results,
        "total_found": len(results),
        "filters": {
            "project_id": project_id,
            "knowledge_area": knowledge_area,
        }
    }


# Legacy Endpoints (for backward compatibility)

@router.post("/session/log", response_model=SessionLogResponse)
async def log_session(entry: SessionLogEntry) -> SessionLogResponse:
    """
    Log a session entry (legacy endpoint).

    Args:
        entry: Session log entry

    Returns:
        Session log response
    """
    return await memory_storage.log_session(entry)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    q: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=50, description="Number of results"),
) -> MemorySearchResponse:
    """
    Search project memory using RAG (legacy endpoint).

    Args:
        q: Search query
        k: Number of results to return

    Returns:
        Search results with citations
    """
    try:
        # Use RAG to search for relevant information
        rag_backend = get_rag()
        rag_response = await rag_backend.retrieve(q, k)

        # Also search local session logs
        session_results = await memory_storage.search_sessions(q, k)

        # Combine results
        all_results = []

        # Add RAG sources
        for source in rag_response.sources:
            all_results.append(
                {
                    "type": "rag_source",
                    "title": source.title,
                    "snippet": source.snippet,
                    "url_or_doi": source.url_or_doi,
                    "score": source.score,
                    "backend": rag_response.backend_type,
                }
            )

        # Add session logs
        for session in session_results:
            all_results.append(
                {
                    "type": "session_log",
                    "request_id": session["request_id"],
                    "timestamp": session["timestamp_utc"],
                    "summary": session["summary"],
                    "tags": session["tags"],
                    "actor": session.get("actor"),
                }
            )

        return MemorySearchResponse(
            query=q, results=all_results, total_found=len(all_results)
        )

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")


def get_memory_storage() -> AdvancedMemoryStorage:
    """Get memory storage instance."""
    return memory_storage
