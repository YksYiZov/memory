from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from api_specs.memory_types import BaseMemory
from api_specs.memory_models import MemoryType, Metadata, MemoryModel, RetrieveMethod
from common_utils.datetime_utils import get_timezone
from core.oxm.constants import MAGIC_ALL, MAX_FETCH_LIMIT, MAX_RETRIEVE_LIMIT


@dataclass
class FetchMemRequest:
    """Memory retrieval request

    Note:
    - user_id and group_id support special value MAGIC_ALL ("__all__") to skip filtering
    - Empty string or None for user_id/group_id filters for null/empty values
    - user_id and group_id cannot both be MAGIC_ALL
    - limit is capped at MAX_FETCH_LIMIT (500)
    """

    user_id: Optional[str] = None
    group_id: Optional[str] = None
    limit: Optional[int] = 40
    offset: Optional[int] = 0
    memory_type: Optional[MemoryType] = MemoryType.EPISODIC_MEMORY
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"
    version_range: Optional[tuple[Optional[str], Optional[str]]] = (
        None  # Version range (start, end), closed interval [start, end]
    )
    start_time: Optional[str] = None  # Start time for time range filtering
    end_time: Optional[str] = None  # End time for time range filtering

    def __post_init__(self):
        """Validate request parameters"""
        if self.user_id == MAGIC_ALL and self.group_id == MAGIC_ALL:
            raise ValueError("user_id and group_id cannot both be MAGIC_ALL")

        # Cap limit at MAX_FETCH_LIMIT
        if self.limit and self.limit > MAX_FETCH_LIMIT:
            self.limit = MAX_FETCH_LIMIT

    def get_memory_types(self) -> List[MemoryType]:
        """Get the list of memory types to query"""
        return [self.memory_type]


@dataclass
class FetchMemResponse:
    """Memory retrieval response"""

    memories: List[MemoryModel]
    total_count: int
    has_more: bool = False
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class RetrieveMemRequest:
    """Memory retrieval request"""

    user_id: Optional[str] = None
    group_id: Optional[str] = None
    memory_types: List[MemoryType] = field(default_factory=list)
    top_k: int = 40
    include_metadata: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    query: Optional[str] = None  # when retrieving
    retrieve_method: RetrieveMethod = field(default=RetrieveMethod.KEYWORD)
    current_time: Optional[str] = (
        None  # Current time, used to filter forward-looking events within validity period happened_at current_time
    )
    radius: Optional[float] = (
        None  # COSINE similarity threshold (use default 0.6 if None)
    )

    def __post_init__(self):
        """Validate request parameters"""
        if self.user_id == MAGIC_ALL and self.group_id == MAGIC_ALL:
            raise ValueError("user_id and group_id cannot both be MAGIC_ALL")

        if self.top_k and self.top_k > MAX_RETRIEVE_LIMIT:
            self.top_k = MAX_RETRIEVE_LIMIT


@dataclass
class PendingMessage:
    """Pending message that has not yet been extracted into memory.

    Represents a cached message waiting for boundary detection or memory extraction.
    """

    id: str  # MongoDB ObjectId as string
    request_id: str  # Request ID
    message_id: Optional[str] = None  # Message ID
    group_id: Optional[str] = None  # Group ID
    user_id: Optional[str] = None  # User ID
    sender: Optional[str] = None  # Sender ID
    sender_name: Optional[str] = None  # Sender name
    group_name: Optional[str] = None  # Group name
    content: Optional[str] = None  # Message content
    refer_list: Optional[List[str]] = None  # List of referenced message IDs
    message_create_time: Optional[str] = None  # Message creation time (ISO 8601 format)
    created_at: Optional[str] = None  # Record creation time (ISO 8601 format)
    updated_at: Optional[str] = None  # Record update time (ISO 8601 format)


@dataclass
class RetrieveMemResponse:
    """Memory retrieval response"""

    memories: List[Dict[str, List[BaseMemory]]]
    scores: List[Dict[str, List[float]]]
    importance_scores: List[float] = field(
        default_factory=list
    )  # New: group importance scores
    original_data: List[Dict[str, List[Dict[str, Any]]]] = field(
        default_factory=list
    )  # New: original data
    total_count: int = 0
    has_more: bool = False
    query_metadata: Metadata = field(default_factory=Metadata)
    metadata: Metadata = field(default_factory=Metadata)
    # Pending messages: unconsumed cached memory data (sync_status=-1 or 0)
    pending_messages: List[PendingMessage] = field(default_factory=list)


@dataclass
class UserDetail:
    """User details

    Structure for the value of ConversationMetaRequest.user_details
    """

    full_name: str  # User's full name
    role: Optional[str] = None  # User role
    extra: Optional[Dict[str, Any]] = None  # Additional information, schema is dynamic


@dataclass
class ConversationMetaRequest:
    """Conversation metadata request"""

    version: str  # Version number
    scene: str  # Scene identifier
    scene_desc: Dict[
        str, Any
    ]  # Scene description, usually contains fields like bot_ids
    name: str  # Conversation name
    group_id: str  # Group ID
    created_at: str  # Creation time, ISO format string
    description: Optional[str] = None  # Conversation description
    default_timezone: Optional[str] = get_timezone().key  # Default timezone
    user_details: Dict[str, UserDetail] = field(
        default_factory=dict
    )  # User details, key is dynamic (e.g., user_001, robot_001), value structure is fixed
    tags: List[str] = field(default_factory=list)  # List of tags
