"""
Master Agent Node - The main entry point for the autonomous ReAct agent.

This module implements the core agent logic with:
- Middleware-based context engineering (dynamic prompts, tool filtering)
- Token-aware message trimming before LLM calls
- Tool execution via Command-based tools
- State management without manual slot extraction (tools handle their own state updates)
- Comprehensive logging for debugging

Context Engineering (via middleware.py):
- @dynamic_prompt: State-aware system prompts that adapt to conversation phase
- @wrap_model_call: Tool filtering based on slots collected and rec_given status
- LoggingMiddleware: Before/after model call logging
- RetryMiddleware: Automatic retry with exponential backoff

The tools use Command objects to update state directly, eliminating the need
for manual slot extraction from tool call arguments.

Memory Management:
- Uses trim_messages_for_context() to ensure messages fit in context window
- Summary is injected via dynamic_prompt middleware
- Preserves recent messages for immediate context
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from typing import Dict, Any, Optional, List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from ..infrastructure import get_router_llm
from ..tools.unified import TOOLS
from ..state import AgentState
from ..utils.slots import _get_slot_value
from ..utils.messages import (
    create_ai_message,
    log_usage_metadata,
    get_message_id,
    log_messages_summary,
)
from ..infrastructure.metrics import AGENTIC_LATENCY
from .memory_nodes import (
    trim_messages_for_context,
    count_message_tokens,
    MAX_TOKENS_AFTER_TRIM,
)
from ..middleware import (
    AgentContext,
    get_default_middleware,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT CREATION WITH MIDDLEWARE
# =============================================================================

# Create the ReAct Agent with middleware-based context engineering
# The middleware stack handles:
# - Dynamic system prompts (state_aware_system_prompt)
# - Tool filtering by conversation phase (filter_tools_by_phase)
# - Logging (LoggingMiddleware)
# - Retry logic (RetryMiddleware)
# - Output validation (validate_response_content)
#
# Tools return Command objects that update state directly
_react_agent = None
_react_agent_lock = threading.Lock()


def _get_react_agent():
    """Get or lazily create the ReAct agent with the shared router LLM."""
    global _react_agent
    if _react_agent is not None:
        return _react_agent
    with _react_agent_lock:
        if _react_agent is None:
            _react_agent = create_agent(
                get_router_llm(),
                tools=TOOLS,
                middleware=get_default_middleware(),
                state_schema=AgentState,
                context_schema=AgentContext,
            )
    return _react_agent


def _inject_travel_advisory(
    last_ai_msg: AIMessage,
    product: str,
    slots: Dict[str, Any],
) -> None:
    """
    Inject travel advisory into recommendation response if needed.
    
    This modifies the AIMessage content in place to ensure the medical
    cost advisory appears in the right location.
    """
    if not product or str(product).lower() != "travel":
        return
    
    if last_ai_msg is None:
        return
    
    try:
        content_text = str(getattr(last_ai_msg, "content", "") or "")
        is_recommendation = (
            "Most people find the *" in content_text
            or "Would you like to proceed with purchasing this plan?" in content_text
        )

        if not is_recommendation:
            return

        destination = _get_slot_value(slots, "destination").strip()

        if destination:
            advisory = (
                f"Medical treatment in {destination} is very good, but can be very expensive. "
                "Some foreign visitors who cannot cover their medical costs may face restrictions in the future."
            )
            medical_prefix = f"Medical treatment in {destination}"
        else:
            advisory = (
                "Medical treatment abroad is very good, but can be very expensive. "
                "Some foreign visitors who cannot cover their medical costs may face restrictions in the future."
            )
            medical_prefix = "Medical treatment abroad"

        # Remove the canonical advisory once (we'll reinsert in the right place)
        content_wo_adv = content_text.strip()
        if advisory in content_wo_adv:
            content_wo_adv = content_wo_adv.replace(advisory, "", 1).strip()

        # Drop any paragraphs that already talk about medical treatment
        paragraphs = [p.strip() for p in content_wo_adv.split("\n\n") if p.strip()]
        cleaned_paragraphs = []
        for p in paragraphs:
            if medical_prefix and medical_prefix in p:
                continue
            cleaned_paragraphs.append(p)

        paragraphs = cleaned_paragraphs

        if paragraphs:
            first = paragraphs[0]
            rest = "\n\n".join(paragraphs[1:]).strip()
        else:
            first, rest = "", content_wo_adv

        if first:
            # Desired order: acknowledgement (first) → advisory → rest
            new_content = first
            new_content += f"\n\n{advisory}"
            if rest:
                new_content += f"\n\n{rest}"
        else:
            # Fallback: advisory first, then whatever response we had
            new_content = advisory
            if content_wo_adv:
                new_content += f"\n\n{content_wo_adv}"

        last_ai_msg.content = new_content
        
        logger.debug(
            "MasterAgent.advisory_injected: destination=%s",
            destination
        )
        
    except Exception as e:
        # Advisory injection is best-effort; never break the main flow
        logger.warning(
            "MasterAgent.advisory_injection_failed: product=%s error=%s",
            product, str(e)
        )


async def master_agent_node(state: AgentState, session_id: str = None, channel: str = "web") -> Dict[str, Any]:
    """
    Main entry point for the autonomous ReAct agent.

    This node:
    1. Applies token-aware message trimming
    2. Invokes the ReAct agent with middleware-based context engineering
    3. Processes the response (advisory injection for travel)
    4. Returns updated state (tools handle their own state updates via Command)
    
    Context Engineering:
    - Dynamic system prompts are generated by @dynamic_prompt middleware
    - Tool filtering is handled by @wrap_model_call middleware
    - Logging is handled by LoggingMiddleware
    - Retry logic is handled by RetryMiddleware
    
    Note: Tools return Command objects that directly update the state,
    eliminating the need for manual slot extraction.
    
    Args:
        state: Current agent state
        session_id: Optional session ID for context (passed to middleware)
        channel: Communication channel ("web", "whatsapp", "api")
    """
    start_time = time.time()
    
    messages = list(state.get("messages", []) or [])
    last_user_full = None
    last_user = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_full = str(getattr(m, "content", "") or "")
            last_user = last_user_full.replace("\n", " ")[:160]
            break

    product = state.get("product")
    current_slots = state.get("slots") or {}
    turn_count = state.get("turn_count", 0)

    logger.info(
        "MasterAgent.start: turn=%d intent=%s product=%s msgs=%d slots=%s channel=%s last_user='%s'",
        turn_count,
        state.get("intent"),
        product,
        len(messages),
        list(current_slots.keys()) if current_slots else [],
        channel,
        last_user or "",
    )

    # Token-aware message trimming BEFORE sending to LLM
    # This ensures we stay within context window limits
    summary = state.get("summary", "")
    original_token_count = count_message_tokens(messages)
    
    # Apply trimming if we have many messages or tokens exceed limit
    if original_token_count > MAX_TOKENS_AFTER_TRIM or len(messages) > 10:
        trimmed_messages = trim_messages_for_context(
            messages,
            max_tokens=MAX_TOKENS_AFTER_TRIM,
            summary=summary if summary else None,
        )
        trimmed_token_count = count_message_tokens(trimmed_messages)
        
        logger.info(
            "MasterAgent.trim_messages: original=%d msgs (%d tokens) -> trimmed=%d msgs (%d tokens)",
            len(messages),
            original_token_count,
            len(trimmed_messages),
            trimmed_token_count,
        )
        
        messages_for_agent = trimmed_messages
    else:
        messages_for_agent = messages
        logger.debug(
            "MasterAgent.no_trim_needed: %d msgs (%d tokens)",
            len(messages),
            original_token_count,
        )

    # Prepare agent inputs
    # Context engineering is now handled by middleware (dynamic_prompt, tool filtering)
    # The middleware reads state directly from request.state
    agent_inputs = {**state, "messages": messages_for_agent}
    
    # Create runtime context for middleware access
    # This provides user-level context (session_id, channel) to middleware and tools
    runtime_context = AgentContext(
        session_id=session_id or "",
        channel=channel,
    )
    
    logger.debug(
        "MasterAgent.invoking: msgs=%d context=%s",
        len(messages_for_agent),
        f"session={runtime_context.session_id}, channel={runtime_context.channel}",
    )

    try:
        # Invoke agent with runtime context
        # Middleware handles:
        # - @dynamic_prompt: State-aware system prompt generation
        # - @wrap_model_call: Tool filtering by conversation phase
        # - LoggingMiddleware: Before/after model logging
        # - RetryMiddleware: Automatic retry with backoff
        # - validate_response_content: Output validation
        agent = _get_react_agent()
        if hasattr(agent, "ainvoke"):
            result = await agent.ainvoke(agent_inputs, context=runtime_context)
        else:
            result = await asyncio.to_thread(
                agent.invoke,
                agent_inputs,
                context=runtime_context,
            )
        out_messages = list(result.get("messages", []) or [])
        
        # Log output message summary for debugging
        log_messages_summary(out_messages, context=f"master_output_turn={turn_count}")
        
        # Find the last AI message for post-processing
        last_ai_msg: Optional[AIMessage] = None
        last_ai = None
        for m in reversed(out_messages):
            if isinstance(m, AIMessage):
                last_ai_msg = m
                last_ai = str(getattr(m, "content", "") or "").replace("\n", " ")[:160]
                break
        
        # Extract and log usage metadata from AI response
        if last_ai_msg:
            try:
                log_usage_metadata(last_ai_msg, session_id=None)
            except Exception as e:
                logger.debug("MasterAgent.usage_metadata_error: %s", e)

        # Get the updated state from the result (tools may have updated it via Command)
        # The result dict may contain state updates from Command objects
        updated_slots = result.get("slots", current_slots)
        updated_product = result.get("product", product)
        
        # Travel-specific advisory injection (robust, tool-independent)
        if updated_product and last_ai_msg is not None:
            _inject_travel_advisory(last_ai_msg, updated_product, updated_slots)
        
        duration = time.time() - start_time
        
        logger.info(
            "MasterAgent.completed: turn=%d intent=%s product=%s out_msgs=%d duration=%.3fs last_ai='%s'",
            turn_count,
            state.get("intent"),
            updated_product,
            len(out_messages),
            duration,
            last_ai or "",
        )
        
        # Record metrics
        try:
            AGENTIC_LATENCY.labels(endpoint="master_agent").observe(duration)
        except Exception:
            pass

        # Build result - include turn count increment
        final_result = {
            "messages": out_messages,
            "turn_count": turn_count + 1,
        }
        
        # Propagate any state updates from Command objects
        # These may have been set by tools via Command(update={...})
        for key in ["slots", "product", "tiers", "rec_ready", "rec_given", "purchase_offered",
                    "last_tool_called", "last_tool_status", "tool_call_count", "tool_errors", "sources"]:
            if key in result and result[key] is not None:
                final_result[key] = result[key]
        
        return final_result
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            "MasterAgent.failed: turn=%d intent=%s product=%s duration=%.3fs error=%s\n%s",
            turn_count,
            state.get("intent"),
            product,
            duration,
            str(e),
            traceback.format_exc()
        )
        
        # Return error message - do NOT swallow the error silently
        # The error should be visible in the response for debugging
        error_message = (
            "I apologize, but I encountered a technical issue while processing your request. "
            f"Error: {type(e).__name__}: {str(e)[:100]}. "
            "Please try again or rephrase your question."
        )
        
        # Create error AIMessage with proper ID for tracking
        error_ai_msg = create_ai_message(
            content=error_message,
            session_id=None,
            turn_count=turn_count,
            error=True,
            error_type=type(e).__name__,
        )
        
        return {
            "messages": [error_ai_msg],
            "turn_count": turn_count + 1,
            "last_tool_status": "error",
            "tool_errors": (state.get("tool_errors") or []) + [str(e)],
        }
