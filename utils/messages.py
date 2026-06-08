"""
Message Utilities for the BigTapp Agentic Chatbot.

This module provides enterprise-grade message handling following LangChain best practices:

1. **Message ID Generation**: Unique IDs for all messages enabling precise RemoveMessage operations
2. **Metadata Tracking**: Timestamps, turn counts, channels, and session context on every message
3. **Content Block Support**: Structured content for multimodal and reasoning blocks
4. **Usage Metadata Extraction**: Token counts and cost tracking from AIMessage responses

Key LangChain concepts used:
- `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` with `id` and `metadata` fields
- `RemoveMessage` for targeted message deletion
- `REMOVE_ALL_MESSAGES` for clearing conversation history
- `add_messages` reducer for proper state management
- `content_blocks` for structured multimodal content

References:
- https://docs.langchain.com/oss/python/langchain/messages
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langchain/short-term-memory
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)

try:
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
except ImportError:
    # Fallback for older versions
    REMOVE_ALL_MESSAGES = "__remove_all__"

logger = logging.getLogger(__name__)

# ============================================================================
# Message ID Generation
# ============================================================================

def generate_message_id(prefix: str = "msg") -> str:
    """
    Generate a unique message ID.
    
    IDs are essential for:
    - Targeted message removal via RemoveMessage
    - Tracing and debugging
    - Correlation in logs and metrics
    
    Format: {prefix}_{uuid4_short}_{timestamp_ms}
    
    Args:
        prefix: Prefix to identify message type (msg, human, ai, tool, sys)
        
    Returns:
        Unique message ID string
    """
    short_uuid = str(uuid.uuid4())[:8]
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000) % 1000000
    return f"{prefix}_{short_uuid}_{timestamp_ms}"


def generate_human_message_id() -> str:
    """Generate ID for HumanMessage."""
    return generate_message_id("human")


def generate_ai_message_id() -> str:
    """Generate ID for AIMessage."""
    return generate_message_id("ai")


def generate_system_message_id() -> str:
    """Generate ID for SystemMessage."""
    return generate_message_id("sys")


def generate_tool_message_id() -> str:
    """Generate ID for ToolMessage."""
    return generate_message_id("tool")


# ============================================================================
# Metadata Construction
# ============================================================================

def build_message_metadata(
    session_id: Optional[str] = None,
    turn_count: Optional[int] = None,
    channel: str = "api",
    product: Optional[str] = None,
    intent: Optional[str] = None,
    **extra_metadata: Any,
) -> Dict[str, Any]:
    """
    Build standardized metadata for messages.
    
    Metadata enables:
    - Turn-based logic and conversation phase tracking
    - Channel-specific behavior (WhatsApp vs Web)
    - Debugging and audit trails
    - Analytics and reporting
    
    Args:
        session_id: Unique session/thread identifier
        turn_count: Current turn number in conversation
        channel: Message channel (api, whatsapp, web)
        product: Current product context
        intent: Current intent classification
        **extra_metadata: Additional custom metadata
        
    Returns:
        Standardized metadata dictionary
    """
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
    }
    
    if session_id:
        metadata["session_id"] = session_id
    if turn_count is not None:
        metadata["turn_count"] = turn_count
    if product:
        metadata["product"] = product
    if intent:
        metadata["intent"] = intent
    
    # Add any extra metadata
    metadata.update(extra_metadata)
    
    return metadata


# ============================================================================
# Message Factory Functions
# ============================================================================

def create_human_message(
    content: str,
    session_id: Optional[str] = None,
    turn_count: Optional[int] = None,
    channel: str = "api",
    name: Optional[str] = None,
    **extra_metadata: Any,
) -> HumanMessage:
    """
    Create a HumanMessage with proper ID and metadata.
    
    Args:
        content: Message content text
        session_id: Session identifier for tracking
        turn_count: Current turn number
        channel: Message channel (api, whatsapp, web)
        name: Optional user name/identifier
        **extra_metadata: Additional metadata fields
        
    Returns:
        HumanMessage with ID and metadata
    """
    msg_id = generate_human_message_id()
    metadata = build_message_metadata(
        session_id=session_id,
        turn_count=turn_count,
        channel=channel,
        **extra_metadata,
    )
    
    logger.debug(
        "Message.create_human: id=%s session=%s turn=%s content_len=%d",
        msg_id, session_id, turn_count, len(content)
    )
    
    kwargs = {
        "content": content,
        "id": msg_id,
        "metadata": metadata,
    }
    if name:
        kwargs["name"] = name
    
    return HumanMessage(**kwargs)


def create_ai_message(
    content: str,
    session_id: Optional[str] = None,
    turn_count: Optional[int] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    **extra_metadata: Any,
) -> AIMessage:
    """
    Create an AIMessage with proper ID and metadata.
    
    Args:
        content: Message content text
        session_id: Session identifier for tracking
        turn_count: Current turn number
        tool_calls: Optional list of tool calls
        **extra_metadata: Additional metadata fields
        
    Returns:
        AIMessage with ID and metadata
    """
    msg_id = generate_ai_message_id()
    metadata = build_message_metadata(
        session_id=session_id,
        turn_count=turn_count,
        **extra_metadata,
    )
    
    logger.debug(
        "Message.create_ai: id=%s session=%s turn=%s content_len=%d tool_calls=%s",
        msg_id, session_id, turn_count, len(content), len(tool_calls) if tool_calls else 0
    )
    
    kwargs = {
        "content": content,
        "id": msg_id,
        "metadata": metadata,
    }
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    
    return AIMessage(**kwargs)


def create_system_message(
    content: str,
    session_id: Optional[str] = None,
    purpose: str = "context",
    **extra_metadata: Any,
) -> SystemMessage:
    """
    Create a SystemMessage with proper ID and metadata.
    
    Args:
        content: System message content
        session_id: Session identifier for tracking
        purpose: Purpose of system message (context, instruction, summary)
        **extra_metadata: Additional metadata fields
        
    Returns:
        SystemMessage with ID and metadata
    """
    msg_id = generate_system_message_id()
    metadata = build_message_metadata(
        session_id=session_id,
        purpose=purpose,
        **extra_metadata,
    )
    
    logger.debug(
        "Message.create_system: id=%s purpose=%s content_len=%d",
        msg_id, purpose, len(content)
    )
    
    return SystemMessage(
        content=content,
        id=msg_id,
        metadata=metadata,
    )


def create_tool_message(
    content: str,
    tool_call_id: str,
    name: str,
    status: str = "success",
    artifact: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    **extra_metadata: Any,
) -> ToolMessage:
    """
    Create a ToolMessage with proper ID, status, and optional artifact.
    
    The artifact field stores supplementary data (like document IDs, sources)
    that is NOT sent to the model but is available programmatically.
    
    Args:
        content: Tool result content (sent to model)
        tool_call_id: ID of the tool call this responds to
        name: Name of the tool
        status: Execution status ('success' or 'error')
        artifact: Optional supplementary data (not sent to model)
        session_id: Session identifier for tracking
        **extra_metadata: Additional metadata fields
        
    Returns:
        ToolMessage with ID, status, and metadata
    """
    msg_id = generate_tool_message_id()
    metadata = build_message_metadata(
        session_id=session_id,
        tool_name=name,
        tool_status=status,
        **extra_metadata,
    )
    
    logger.debug(
        "Message.create_tool: id=%s tool=%s status=%s call_id=%s content_len=%d",
        msg_id, name, status, tool_call_id, len(content)
    )
    
    kwargs = {
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
        "id": msg_id,
        "status": status,
    }
    
    # Add artifact if provided (for RAG sources, etc.)
    if artifact:
        kwargs["artifact"] = artifact
    
    # Note: ToolMessage doesn't have a metadata field in the same way
    # We store it in additional_kwargs instead
    kwargs["additional_kwargs"] = {"metadata": metadata}
    
    return ToolMessage(**kwargs)


# ============================================================================
# Message Removal Utilities
# ============================================================================

def create_remove_message(message_id: str) -> RemoveMessage:
    """
    Create a RemoveMessage to delete a specific message by ID.
    
    IMPORTANT: The target message must have an ID set for this to work.
    Messages without IDs cannot be targeted for removal.
    
    Args:
        message_id: ID of the message to remove
        
    Returns:
        RemoveMessage marker
    """
    logger.debug("Message.create_remove: target_id=%s", message_id)
    return RemoveMessage(id=message_id)


def create_remove_all_messages() -> RemoveMessage:
    """
    Create a RemoveMessage to delete ALL messages.
    
    Use with caution - this clears the entire conversation history.
    Typically used for session resets or "hi" greetings.
    
    Returns:
        RemoveMessage marker for all messages
    """
    logger.info("Message.create_remove_all: clearing all messages")
    return RemoveMessage(id=REMOVE_ALL_MESSAGES)


def create_remove_messages_by_ids(message_ids: List[str]) -> List[RemoveMessage]:
    """
    Create RemoveMessage markers for multiple message IDs.
    
    Args:
        message_ids: List of message IDs to remove
        
    Returns:
        List of RemoveMessage markers
    """
    logger.debug("Message.create_remove_batch: count=%d", len(message_ids))
    return [RemoveMessage(id=mid) for mid in message_ids if mid]


def get_removable_message_ids(
    messages: List[BaseMessage],
    keep_recent: int = 4,
    keep_system: bool = True,
) -> List[str]:
    """
    Get IDs of messages that can be safely removed.
    
    Preserves:
    - Recent N messages
    - System messages (optionally)
    - Messages without IDs (can't be removed anyway)
    
    Args:
        messages: List of messages to analyze
        keep_recent: Number of recent messages to preserve
        keep_system: Whether to preserve system messages
        
    Returns:
        List of message IDs that can be removed
    """
    if len(messages) <= keep_recent:
        return []
    
    # Determine which messages to keep
    recent_indices = set(range(len(messages) - keep_recent, len(messages)))
    
    removable_ids = []
    for i, msg in enumerate(messages):
        # Skip recent messages
        if i in recent_indices:
            continue
        
        # Skip system messages if configured
        if keep_system and isinstance(msg, SystemMessage):
            continue
        
        # Only include messages with IDs
        msg_id = getattr(msg, "id", None)
        if msg_id:
            removable_ids.append(msg_id)
    
    logger.debug(
        "Message.get_removable: total=%d keep_recent=%d removable=%d",
        len(messages), keep_recent, len(removable_ids)
    )
    
    return removable_ids


# ============================================================================
# Usage Metadata Extraction
# ============================================================================

def extract_usage_metadata(ai_message: AIMessage) -> Optional[Dict[str, Any]]:
    """
    Extract usage metadata (token counts, cost info) from an AIMessage.
    
    Usage metadata is valuable for:
    - Cost tracking and budgeting
    - Performance analysis
    - Context window monitoring
    
    Args:
        ai_message: AIMessage returned from model invocation
        
    Returns:
        Usage metadata dict or None if not available
    """
    usage = getattr(ai_message, "usage_metadata", None)
    
    if usage is None:
        return None
    
    # Convert to dict if it's a Pydantic model or similar
    if hasattr(usage, "model_dump"):
        usage_dict = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage_dict = usage.dict()
    elif isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = {"raw": str(usage)}
    
    logger.debug(
        "Message.extract_usage: input_tokens=%s output_tokens=%s total=%s",
        usage_dict.get("input_tokens"),
        usage_dict.get("output_tokens"),
        usage_dict.get("total_tokens"),
    )
    
    return usage_dict


def log_usage_metadata(ai_message: AIMessage, session_id: Optional[str] = None) -> None:
    """
    Log usage metadata from an AIMessage for monitoring.
    
    Args:
        ai_message: AIMessage to extract usage from
        session_id: Optional session ID for correlation
    """
    usage = extract_usage_metadata(ai_message)
    
    if usage:
        logger.info(
            "Message.usage: session=%s input_tokens=%s output_tokens=%s total=%s",
            session_id,
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
            usage.get("total_tokens", "?"),
        )


# ============================================================================
# Message Inspection Utilities
# ============================================================================

def get_message_id(message: BaseMessage) -> Optional[str]:
    """
    Safely get the ID from a message.
    
    Args:
        message: Any LangChain message
        
    Returns:
        Message ID or None
    """
    return getattr(message, "id", None)


def get_message_metadata(message: BaseMessage) -> Dict[str, Any]:
    """
    Safely get metadata from a message.
    
    Args:
        message: Any LangChain message
        
    Returns:
        Metadata dict (empty if none)
    """
    metadata = getattr(message, "metadata", None)
    if metadata is None:
        # Check additional_kwargs for ToolMessage
        additional = getattr(message, "additional_kwargs", {})
        metadata = additional.get("metadata", {})
    return metadata or {}


def get_message_turn_count(message: BaseMessage) -> Optional[int]:
    """
    Get the turn count from message metadata.
    
    Args:
        message: Any LangChain message
        
    Returns:
        Turn count or None
    """
    metadata = get_message_metadata(message)
    return metadata.get("turn_count")


def get_message_timestamp(message: BaseMessage) -> Optional[str]:
    """
    Get the timestamp from message metadata.
    
    Args:
        message: Any LangChain message
        
    Returns:
        ISO timestamp string or None
    """
    metadata = get_message_metadata(message)
    return metadata.get("timestamp")


def describe_message(message: BaseMessage) -> str:
    """
    Create a human-readable description of a message for logging.
    
    Args:
        message: Any LangChain message
        
    Returns:
        Description string
    """
    msg_type = type(message).__name__
    msg_id = get_message_id(message) or "no-id"
    content = str(getattr(message, "content", "") or "")
    content_preview = content[:50] + "..." if len(content) > 50 else content
    
    return f"{msg_type}[{msg_id}]: {content_preview}"


def log_messages_summary(messages: List[BaseMessage], context: str = "") -> None:
    """
    Log a summary of a message list for debugging.
    
    Args:
        messages: List of messages to summarize
        context: Optional context string for the log
    """
    if not messages:
        logger.debug("Message.summary[%s]: empty list", context)
        return
    
    type_counts = {}
    with_ids = 0
    with_metadata = 0
    
    for msg in messages:
        msg_type = type(msg).__name__
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        if get_message_id(msg):
            with_ids += 1
        if get_message_metadata(msg):
            with_metadata += 1
    
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    
    logger.debug(
        "Message.summary[%s]: total=%d with_ids=%d with_metadata=%d types={%s}",
        context, len(messages), with_ids, with_metadata, type_str
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # ID Generation
    "generate_message_id",
    "generate_human_message_id",
    "generate_ai_message_id",
    "generate_system_message_id",
    "generate_tool_message_id",
    # Metadata
    "build_message_metadata",
    # Message Creation
    "create_human_message",
    "create_ai_message",
    "create_system_message",
    "create_tool_message",
    # Message Removal
    "create_remove_message",
    "create_remove_all_messages",
    "create_remove_messages_by_ids",
    "get_removable_message_ids",
    "REMOVE_ALL_MESSAGES",
    # Usage Metadata
    "extract_usage_metadata",
    "log_usage_metadata",
    # Inspection
    "get_message_id",
    "get_message_metadata",
    "get_message_turn_count",
    "get_message_timestamp",
    "describe_message",
    "log_messages_summary",
]

