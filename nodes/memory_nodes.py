"""
Token-aware Memory Management for the Agentic Chatbot.

This module implements enterprise-grade memory management following LangChain/LangGraph
best practices:

1. Token-aware message trimming using `trim_messages` and `count_tokens_approximately`
2. Rolling summarization with structured RunningSummary tracking
3. Safe tool-call chain preservation (never split AIMessage + ToolMessage pairs)
4. Product-aware context management to prevent slot bleeding
5. Proper message ID handling for targeted RemoveMessage operations

Key concepts from LangChain docs:
- `trim_messages`: Truncates message history to fit within token limits
- `count_tokens_approximately`: Fast token estimation for any model
- `RemoveMessage`: Marks messages for deletion from state (requires message IDs)
- `REMOVE_ALL_MESSAGES`: Special constant to clear all messages
- `add_messages` reducer: Handles message list updates including deletions

Message ID Requirements:
- Messages MUST have IDs for RemoveMessage to work
- Use create_system_message() from utils.messages for proper ID generation
- Messages without IDs will be logged as warnings and skipped during removal

References:
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langchain/short-term-memory
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    RemoveMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from ..state import AgentState, RunningSummaryData
from ..infrastructure import get_router_llm
from ..utils.memory import _build_history_context_from_messages
from ..utils.messages import (
    create_system_message,
    get_message_id,
    create_remove_message,
    get_removable_message_ids,
    log_messages_summary,
    describe_message,
)
from ..infrastructure.metrics import (
    MEMORY_SUMMARIZATION_TOTAL,
    MEMORY_SUMMARIZATION_LATENCY,
    MEMORY_TOKENS_BEFORE_TRIM,
    MEMORY_TOKENS_AFTER_TRIM,
    MEMORY_MESSAGES_PRUNED,
    MESSAGE_REMOVED_TOTAL,
    MESSAGE_HISTORY_SIZE,
    MESSAGE_HISTORY_TOKENS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Token budget for the LLM context window (conservative estimate for Azure GPT-4)
MAX_CONTEXT_TOKENS = 8000

# Trigger summarization when message history exceeds this token count
MAX_TOKENS_BEFORE_SUMMARY = 6000

# Maximum tokens to keep after trimming (leaves room for system prompt + response)
MAX_TOKENS_AFTER_TRIM = 2500

# Maximum tokens for the summary itself
MAX_SUMMARY_TOKENS = 500

# Minimum messages to keep in recent history (never trim below this)
MIN_MESSAGES_TO_KEEP = 4

# Maximum messages to keep without summarization (message count threshold)
# Increased to avoid premature summarization - rely more on token count
MAX_MESSAGES_BEFORE_SUMMARY = 25


# ============================================================================
# Token Counting Utilities
# ============================================================================

def count_message_tokens(messages: List[BaseMessage]) -> int:
    """
    Count approximate tokens in a list of messages.
    
    Uses LangChain's built-in count_tokens_approximately for fast estimation.
    This is more accurate than character-based heuristics and works across models.
    
    Args:
        messages: List of LangChain message objects
        
    Returns:
        Approximate token count
    """
    if not messages:
        return 0
    
    try:
        return count_tokens_approximately(messages)
    except Exception as e:
        logger.warning("Memory.count_tokens: count_tokens_approximately failed, using fallback: %s", e)
        # Fallback: rough estimate of 4 chars per token
        total_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
        return total_chars // 4


def count_string_tokens(text: str) -> int:
    """
    Count approximate tokens in a string.
    
    Args:
        text: Plain text string
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    
    try:
        # Wrap in a message for the token counter
        return count_tokens_approximately([HumanMessage(content=text)])
    except Exception:
        # Fallback: rough estimate
        return len(text) // 4


# ============================================================================
# Tool Chain Safety
# ============================================================================

def _find_safe_prune_boundary(messages: List[BaseMessage], keep_recent: int = 4) -> int:
    """
    Find a safe index to split messages for pruning, ensuring we don't break tool call chains.
    
    Tool call chains must be kept together:
    - AIMessage with tool_calls must stay with its following ToolMessage(s)
    - ToolMessage must stay with its preceding AIMessage
    
    This is CRITICAL for LangGraph state consistency. Breaking a tool chain causes:
    - Invalid state that can crash the graph
    - Lost context about what tools were called
    - Potential infinite loops in tool execution
    
    Args:
        messages: Full list of messages
        keep_recent: Minimum number of recent messages to preserve
        
    Returns:
        Index up to which we can safely prune (exclusive)
    """
    if len(messages) <= keep_recent:
        logger.debug("Memory.prune_boundary: messages=%d <= keep_recent=%d, no pruning", len(messages), keep_recent)
        return 0
    
    # Start from the desired boundary
    boundary = len(messages) - keep_recent
    
    # Walk backward to find a safe split point (not in the middle of a tool chain)
    iterations = 0
    max_iterations = len(messages)  # Prevent infinite loop
    
    while boundary > 0 and iterations < max_iterations:
        iterations += 1
        msg_at_boundary = messages[boundary]
        
        # If the message at boundary is a ToolMessage, we can't split here
        # (it needs its preceding AIMessage with tool_calls)
        if isinstance(msg_at_boundary, ToolMessage):
            logger.debug("Memory.prune_boundary: ToolMessage at boundary %d, moving back", boundary)
            boundary -= 1
            continue
        
        # If message at boundary is an AIMessage with tool_calls,
        # we need to keep it with its ToolMessage responses (which come after)
        if isinstance(msg_at_boundary, AIMessage):
            tool_calls = getattr(msg_at_boundary, "tool_calls", None)
            if tool_calls:
                logger.debug("Memory.prune_boundary: AIMessage with tool_calls at boundary %d, moving back", boundary)
                boundary -= 1
                continue
        
        # Check if the message BEFORE boundary has tool_calls that we'd be splitting
        if boundary > 0:
            msg_before = messages[boundary - 1]
            if isinstance(msg_before, AIMessage):
                tool_calls = getattr(msg_before, "tool_calls", None)
                if tool_calls:
                    # The AI message we'd be pruning has tool_calls
                    # Check if any of its tool responses are in the "keep" section
                    has_tool_response_after = False
                    for i in range(boundary, min(boundary + 5, len(messages))):
                        if isinstance(messages[i], ToolMessage):
                            has_tool_response_after = True
                            break
                    
                    if has_tool_response_after:
                        logger.debug(
                            "Memory.prune_boundary: AIMessage before boundary %d has tool_calls with responses after, moving back",
                            boundary
                        )
                        boundary -= 1
                        continue
        
        # Safe to split here
        break
    
    logger.debug(
        "Memory.prune_boundary: found safe boundary at %d (total messages=%d, keep_recent=%d)",
        boundary, len(messages), keep_recent
    )
    return boundary


# ============================================================================
# Summarization Logic
# ============================================================================

def _build_summarization_prompt(
    messages_to_summarize: List[BaseMessage],
    existing_summary: str,
    current_product: Optional[str],
    current_slots: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build the system and user prompts for summarization.
    
    The prompt is designed to:
    1. Preserve slot values EXACTLY (critical for recommendations)
    2. Tag information by product to prevent context bleeding
    3. Mark old product info as ARCHIVED when user switches
    4. Keep summaries concise but information-dense
    
    Args:
        messages_to_summarize: Messages to be summarized
        existing_summary: Any existing summary to extend
        current_product: Current product context
        current_slots: Currently collected slot values
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Build history text from messages
    history_text = _build_history_context_from_messages(messages_to_summarize)
    
    # Format current slots for preservation
    slots_str = ""
    if current_slots:
        slot_items = [f"{k}={v}" for k, v in current_slots.items() if v and not k.startswith("_")]
        if slot_items:
            slots_str = f"\n[CURRENT SLOTS TO PRESERVE]: {', '.join(slot_items)}"

    system_prompt = """You are an expert conversation summarizer for an insurance chatbot.
Your summaries must preserve SLOT VALUES precisely for the recommendation system.

CRITICAL RULES:
1. ALWAYS preserve exact slot values in key:value format (e.g., 'duration: 26 months', 'destination: Japan', 'maid_country: Russia')
2. Tag information with the product it belongs to (e.g., '[Travel] destination: Japan')
3. When user switches products, mark old product info as ARCHIVED
4. Never lose slot values - they are critical for recommendations
5. Format slots clearly: [SLOTS] duration=26months, maid_country=Russia, coverage_above_mom=yes
6. Keep the summary CONCISE - aim for under 200 words
7. Focus on FACTS and DECISIONS, not pleasantries or filler"""

    if existing_summary:
        user_prompt = f"""Current Product Context: {current_product or 'unknown'}{slots_str}

Current Summary:
{existing_summary}

New Conversation Lines:
{history_text}

Task: Update the summary. Structure it as:
[ACTIVE - {current_product or 'unknown'}] [SLOTS: key=value pairs] Current product details
[ARCHIVED] Previous product discussions (if any, keep brief)

IMPORTANT: Preserve ALL slot values exactly. Keep it concise."""
    else:
        user_prompt = f"""Current Product Context: {current_product or 'unknown'}{slots_str}

Conversation History:
{history_text}

Task: Create a summary with slot values preserved.
Format: [ACTIVE - {current_product or 'unknown'}] [SLOTS: key=value] followed by key facts."""

    return system_prompt, user_prompt


async def _generate_summary(
    messages_to_summarize: List[BaseMessage],
    existing_summary: str,
    current_product: Optional[str],
    current_slots: Dict[str, Any],
) -> Optional[str]:
    """
    Generate a summary of messages using the LLM.
    
    Args:
        messages_to_summarize: Messages to summarize
        existing_summary: Existing summary to extend
        current_product: Current product context
        current_slots: Current slot values
        
    Returns:
        New summary text, or None if summarization failed
    """
    start_time = time.perf_counter()
    status = "error"
    
    try:
        system_prompt, user_prompt = _build_summarization_prompt(
            messages_to_summarize,
            existing_summary,
            current_product,
            current_slots,
        )
        
        logger.debug(
            "Memory.summarize: generating summary for %d messages, existing_summary_len=%d",
            len(messages_to_summarize),
            len(existing_summary or ""),
        )
        
        response = await get_router_llm().ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        
        new_summary = str(getattr(response, "content", "") or "").strip()
        
        if new_summary:
            status = "success"
            logger.info(
                "Memory.summarize.success: summarized %d messages, summary_len=%d, product=%s",
                len(messages_to_summarize),
                len(new_summary),
                current_product,
            )
            return new_summary
        else:
            logger.warning("Memory.summarize: LLM returned empty summary")
            return None
            
    except Exception as e:
        logger.error("Memory.summarize.error: summarization failed: %s", e, exc_info=True)
        return None
    finally:
        duration = time.perf_counter() - start_time
        MEMORY_SUMMARIZATION_LATENCY.observe(duration)
        MEMORY_SUMMARIZATION_TOTAL.labels(status=status).inc()


# ============================================================================
# Main Memory Management Node
# ============================================================================

async def compress_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Token-aware rolling memory compression node.
    
    This node implements the LangChain/LangGraph memory management best practices:
    
    1. **Token Counting**: Uses `count_tokens_approximately` to accurately measure
       message history size, not just message count.
       
    2. **Threshold-based Summarization**: Only summarizes when token count exceeds
       `MAX_TOKENS_BEFORE_SUMMARY`, avoiding unnecessary LLM calls.
       
    3. **Safe Pruning**: Uses `_find_safe_prune_boundary` to ensure tool call chains
       (AIMessage + ToolMessage) are never split.
       
    4. **Product-aware Summarization**: Tags summary sections by product to prevent
       context bleeding when users switch products.
       
    5. **Structured Memory Context**: Stores summarization metadata in `memory_context`
       for debugging and analytics.
    
    Args:
        state: Current agent state
        
    Returns:
        State updates including:
        - summary: Updated summary text
        - memory_context: Structured metadata about the summary
        - messages: RemoveMessage markers for pruned messages
        - total_message_tokens: Updated token count
    """
    messages = list(state.get("messages", []) or [])
    current_product = state.get("product")
    current_slots = state.get("slots") or {}
    existing_summary = state.get("summary", "")
    memory_context = state.get("memory_context") or {}
    
    # Count current tokens
    current_tokens = count_message_tokens(messages)
    summary_tokens = count_string_tokens(existing_summary)
    total_tokens = current_tokens + summary_tokens
    
    logger.info(
        "Memory.compress.start: messages=%d, message_tokens=%d, summary_tokens=%d, total=%d, product=%s",
        len(messages),
        current_tokens,
        summary_tokens,
        total_tokens,
        current_product,
    )
    
    # Log message summary for debugging
    log_messages_summary(messages, context="compress_start")
    
    # Record metrics
    MEMORY_TOKENS_BEFORE_TRIM.observe(current_tokens)
    MESSAGE_HISTORY_SIZE.observe(len(messages))
    MESSAGE_HISTORY_TOKENS.observe(current_tokens)
    
    # Check if we need to summarize based on token count OR message count
    needs_summary = (
        current_tokens > MAX_TOKENS_BEFORE_SUMMARY or
        len(messages) > MAX_MESSAGES_BEFORE_SUMMARY
    )
    
    # NON-BLOCKING OPTIMIZATION: Only summarize on even turn counts to reduce blocking
    # This spreads the summarization cost across turns instead of blocking every time
    turn_count = state.get("turn_count", 0)
    if needs_summary and turn_count % 2 != 0:
        logger.info(
            "Memory.compress.deferred: skipping summarization on turn %d (will run on next turn)",
            turn_count
        )
        needs_summary = False
    
    if not needs_summary:
        logger.debug(
            "Memory.compress: no summarization needed (tokens=%d < %d, messages=%d < %d)",
            current_tokens, MAX_TOKENS_BEFORE_SUMMARY,
            len(messages), MAX_MESSAGES_BEFORE_SUMMARY,
        )
        # Just update token count tracking
        return {
            "total_message_tokens": current_tokens,
            "last_token_count_at": state.get("turn_count", 0),
        }
    
    # Find safe boundary for pruning
    prune_boundary = _find_safe_prune_boundary(messages, keep_recent=MIN_MESSAGES_TO_KEEP)
    
    if prune_boundary <= 0:
        logger.debug("Memory.compress: no safe prune boundary found, skipping summarization")
        return {
            "total_message_tokens": current_tokens,
            "last_token_count_at": state.get("turn_count", 0),
        }
    
    # Get messages to summarize
    to_summarize = messages[:prune_boundary]
    to_keep = messages[prune_boundary:]
    
    logger.info(
        "Memory.compress.pruning: summarizing %d messages, keeping %d recent",
        len(to_summarize),
        len(to_keep),
    )
    
    # Generate new summary
    new_summary = await _generate_summary(
        to_summarize,
        existing_summary,
        current_product,
        current_slots,
    )
    
    if not new_summary:
        logger.warning("Memory.compress: summarization failed, keeping messages intact")
        return {
            "total_message_tokens": current_tokens,
            "last_token_count_at": state.get("turn_count", 0),
        }
    
    # Calculate new token counts
    new_message_tokens = count_message_tokens(to_keep)
    new_summary_tokens = count_string_tokens(new_summary)
    
    # Record metrics
    MEMORY_TOKENS_AFTER_TRIM.observe(new_message_tokens)
    MEMORY_MESSAGES_PRUNED.inc(len(to_summarize))
    
    # Update memory context with structured metadata
    running_summary_data = RunningSummaryData(
        summary_text=new_summary,
        token_count=new_summary_tokens,
        last_summarized_at=datetime.now(timezone.utc).isoformat(),
        messages_summarized=memory_context.get("running_summary", {}).get("messages_summarized", 0) + len(to_summarize),
        product_context=current_product,
    )
    
    new_memory_context = {
        **memory_context,
        "running_summary": running_summary_data.model_dump(),
        "last_prune_count": len(to_summarize),
        "last_prune_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Build RemoveMessage markers for messages with IDs
    # CRITICAL: Messages without IDs cannot be removed - log warnings for these
    remove_markers = []
    messages_without_ids = 0
    
    for msg in to_summarize:
        msg_id = get_message_id(msg)
        if msg_id:
            remove_markers.append(create_remove_message(msg_id))
            logger.debug(
                "Memory.compress.remove_marker: id=%s type=%s",
                msg_id, type(msg).__name__
            )
        else:
            messages_without_ids += 1
            logger.warning(
                "Memory.compress.no_id: cannot remove message without ID: %s",
                describe_message(msg)
            )
    
    if messages_without_ids > 0:
        logger.warning(
            "Memory.compress.id_warning: %d of %d messages have no ID and cannot be removed",
            messages_without_ids, len(to_summarize)
        )
    
    # Record removal metrics
    MESSAGE_REMOVED_TOTAL.labels(removal_type="specific").inc(len(remove_markers))
    
    logger.info(
        "Memory.compress.success: pruned=%d (removed=%d, no_id=%d), kept=%d, old_tokens=%d, new_tokens=%d, summary_tokens=%d",
        len(to_summarize),
        len(remove_markers),
        messages_without_ids,
        len(to_keep),
        current_tokens,
        new_message_tokens,
        new_summary_tokens,
    )
    
    # Return state updates
    # Use RemoveMessage markers for deletion (handled by add_messages reducer)
    return {
        "summary": new_summary,
        "has_summary": True,
        "memory_context": new_memory_context,
        "messages": remove_markers,
        "total_message_tokens": new_message_tokens,
        "last_token_count_at": state.get("turn_count", 0),
    }


# ============================================================================
# Pre-Model Message Trimming
# ============================================================================

def trim_messages_for_context(
    messages: List[BaseMessage],
    max_tokens: int = MAX_TOKENS_AFTER_TRIM,
    summary: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Trim messages to fit within the LLM context window.
    
    This function uses LangChain's `trim_messages` utility to intelligently
    truncate message history while respecting message boundaries.
    
    Key features:
    - Uses `count_tokens_approximately` for accurate token counting
    - Preserves recent messages (strategy="last")
    - Ensures we start on a human message and end on human/tool
    - Injects summary as system message if available
    
    Args:
        messages: Full message list
        max_tokens: Maximum tokens to keep
        summary: Optional summary to prepend as context
        
    Returns:
        Trimmed message list ready for LLM invocation
    """
    if not messages:
        return []
    
    start_time = time.perf_counter()
    original_count = len(messages)
    original_tokens = count_message_tokens(messages)
    
    try:
        # Use LangChain's trim_messages for intelligent truncation
        trimmed = trim_messages(
            messages,
            strategy="last",  # Keep most recent messages
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens,
            start_on="human",  # Ensure we start on a user message
            end_on=("human", "tool"),  # Valid ending points
            include_system=True,  # Keep system messages
            allow_partial=False,  # Don't split messages
        )
        
        # If we have a summary, prepend it as a system message with proper ID
        if summary:
            summary_msg = create_system_message(
                content=f"CONVERSATION SUMMARY:\n{summary}",
                purpose="summary_context",
            )
            trimmed = [summary_msg] + list(trimmed)
        
        trimmed_tokens = count_message_tokens(trimmed)
        
        logger.debug(
            "Memory.trim: original=%d msgs (%d tokens) -> trimmed=%d msgs (%d tokens)",
            original_count,
            original_tokens,
            len(trimmed),
            trimmed_tokens,
        )
        
        return list(trimmed)
        
    except Exception as e:
        logger.error("Memory.trim.error: trim_messages failed: %s", e, exc_info=True)
        # Fallback: return last N messages
        fallback_count = max(MIN_MESSAGES_TO_KEEP, len(messages) // 2)
        logger.warning("Memory.trim: using fallback, keeping last %d messages", fallback_count)
        return messages[-fallback_count:]
    finally:
        duration = time.perf_counter() - start_time
        logger.debug("Memory.trim: completed in %.4fs", duration)


def get_trimmed_messages_for_llm(state: AgentState, max_tokens: int = MAX_TOKENS_AFTER_TRIM) -> List[BaseMessage]:
    """
    Get trimmed messages ready for LLM invocation.
    
    This is a convenience function that extracts messages from state,
    applies trimming, and injects the summary if available.
    
    Args:
        state: Current agent state
        max_tokens: Maximum tokens for the trimmed output
        
    Returns:
        List of messages ready for LLM invocation
    """
    messages = list(state.get("messages", []) or [])
    summary = state.get("summary", "")
    
    return trim_messages_for_context(messages, max_tokens, summary if summary else None)


# ============================================================================
# Legacy Compatibility
# ============================================================================

# Alias for backward compatibility with existing graph.py imports
_compress_memory_node = compress_memory_node


__all__ = [
    "compress_memory_node",
    "_compress_memory_node",  # Legacy alias
    "trim_messages_for_context",
    "get_trimmed_messages_for_llm",
    "count_message_tokens",
    "count_string_tokens",
    "_find_safe_prune_boundary",
    "MAX_CONTEXT_TOKENS",
    "MAX_TOKENS_BEFORE_SUMMARY",
    "MAX_TOKENS_AFTER_TRIM",
    "MAX_SUMMARY_TOKENS",
    "MIN_MESSAGES_TO_KEEP",
]
