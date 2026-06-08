"""
Custom ToolNode implementation with enterprise-grade error handling.

This module provides:
- Custom error handler that produces context-aware error messages
- ToolNode wrapper with fallback support
- Comprehensive logging for debugging tool execution
- Metrics tracking for tool calls

The error handler follows LangGraph best practices for returning structured
ToolMessage objects with status='error' so the agent can reason about failures.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from ..state import AgentState
from ..infrastructure.metrics import (
    TOOL_LATENCY,
    TOOL_CALLS_TOTAL,
    TOOL_ERRORS_TOTAL,
    TOOL_RETRIES_TOTAL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

def _classify_error(error: Exception, tool_name: str) -> tuple[str, str]:
    """
    Classify an error and return (error_category, user_friendly_message).
    
    Categories:
    - validation: Input validation failed
    - service: External service (Weaviate, LLM) failed
    - timeout: Operation timed out
    - permission: Access denied
    - unknown: Unclassified error
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Validation errors
    if "validation" in error_str or "invalid" in error_str or "required" in error_str:
        return "validation", f"Invalid input for {tool_name}: {str(error)}"
    
    # Service errors
    if any(svc in error_str for svc in ["weaviate", "vector", "embedding", "openai", "azure", "llm"]):
        return "service", f"I'm having trouble accessing my knowledge base. Please try again in a moment."
    
    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout", f"That's taking longer than expected. Please try a simpler question."
    
    # Connection errors
    if any(conn in error_str for conn in ["connection", "connect", "network", "dns"]):
        return "service", f"I'm experiencing connectivity issues. Please try again shortly."
    
    # Rate limit errors
    if "rate" in error_str and "limit" in error_str:
        return "rate_limit", f"I'm receiving too many requests right now. Please wait a moment and try again."
    
    # Tool-specific fallback messages
    tool_messages = {
        "save_progress": "I had trouble saving your information. Please try again.",
        "search_product_knowledge": "I couldn't search the knowledge base. Let me try to help with what I know.",
        "compare_plans": "I had trouble comparing the plans. Please try rephrasing your question.",
        "get_product_recommendation": "I couldn't generate a recommendation. Could you confirm the details you've provided?",
        "generate_purchase_link": "I had trouble generating the purchase link. Please try again.",
    }
    
    return "unknown", tool_messages.get(tool_name, f"Tool '{tool_name}' encountered an error. Please try a different approach.")


# =============================================================================
# CUSTOM ERROR HANDLER
# =============================================================================

def handle_tool_error(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """
    Custom error handler for ToolNode.
    
    This handler:
    1. Logs the error with full context for debugging
    2. Classifies the error type
    3. Returns a user-friendly ToolMessage with status='error'
    4. Updates state with error tracking information
    
    The returned ToolMessage allows the agent to:
    - See that the tool failed
    - Understand why (in user-friendly terms)
    - Decide whether to retry, try a different approach, or escalate
    
    Args:
        state: Current agent state containing messages
        error: The exception that was raised
        
    Returns:
        Dict with 'messages' containing error ToolMessage(s)
    """
    messages = state.get("messages", [])
    
    # Find the tool calls that failed
    tool_calls = []
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
            tool_calls = last_message.tool_calls or []
    
    if not tool_calls:
        # Fallback if we can't find tool calls
        logger.error(
            "ToolNode.error_handler.no_tool_calls: error=%s\n%s",
            str(error),
            traceback.format_exc()
        )
        return {
            "messages": [
                ToolMessage(
                    content=f"Tool execution failed: {str(error)}. Please try a different approach.",
                    tool_call_id="unknown",
                    status="error",
                )
            ]
        }
    
    # Generate error messages for each tool call
    error_messages = []
    for tc in tool_calls:
        tool_name = tc.get("name", "unknown")
        tool_call_id = tc.get("id", "unknown")
        tool_args = tc.get("args", {})
        
        # Classify the error
        error_category, user_message = _classify_error(error, tool_name)
        
        # Log with full context
        logger.error(
            "ToolNode.error_handler: tool=%s tool_call_id=%s category=%s args=%s error=%s\n%s",
            tool_name,
            tool_call_id,
            error_category,
            tool_args,
            str(error),
            traceback.format_exc()
        )
        
        # Record metric
        try:
            TOOL_CALLS_TOTAL.labels(tool=tool_name, status="error").inc()
            TOOL_ERRORS_TOTAL.labels(tool=tool_name, error_category=error_category).inc()
        except Exception:
            pass  # Don't fail on metrics
        
        # Create error message
        error_messages.append(
            ToolMessage(
                content=user_message,
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                additional_kwargs={
                    "error_type": type(error).__name__,
                    "error_category": error_category,
                    "error_details": str(error),
                },
            )
        )
    
    return {"messages": error_messages}


# =============================================================================
# TOOL NODE FACTORY
# =============================================================================

def create_tool_node_with_error_handling(
    tools: List[Any],
    name: str = "tools",
) -> ToolNode:
    """
    Create a ToolNode with custom error handling.
    
    This factory function creates a ToolNode that:
    1. Uses our custom error handler for graceful error recovery
    2. Logs all tool executions for debugging
    3. Records metrics for monitoring
    
    Args:
        tools: List of tool functions to include
        name: Name for the node (for logging)
        
    Returns:
        Configured ToolNode instance
    """
    logger.info(
        "ToolNode.create: name=%s tools=%s",
        name,
        [t.name for t in tools]
    )
    
    return ToolNode(
        tools=tools,
        handle_tool_errors=handle_tool_error,
    )


def create_tool_node_with_fallback(
    tools: List[Any],
    name: str = "tools",
) -> Any:
    """
    Create a ToolNode with fallback error handling using RunnableLambda.
    
    This is an alternative approach that wraps the ToolNode with a fallback
    that catches any unhandled exceptions and converts them to error messages.
    
    Args:
        tools: List of tool functions to include
        name: Name for the node (for logging)
        
    Returns:
        ToolNode with fallback error handling
    """
    
    def fallback_handler(state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback handler that extracts error from state."""
        error = state.get("error")
        if error:
            return handle_tool_error(state, error)
        
        # If no error in state, return a generic error message
        messages = state.get("messages", [])
        tool_calls = []
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls or []
        
        return {
            "messages": [
                ToolMessage(
                    content="An unexpected error occurred. Please try again.",
                    tool_call_id=tc.get("id", "unknown"),
                    name=tc.get("name", "unknown"),
                    status="error",
                )
                for tc in tool_calls
            ] if tool_calls else [
                ToolMessage(
                    content="An unexpected error occurred. Please try again.",
                    tool_call_id="unknown",
                    status="error",
                )
            ]
        }
    
    logger.info(
        "ToolNode.create_with_fallback: name=%s tools=%s",
        name,
        [t.name for t in tools]
    )
    
    tool_node = ToolNode(tools=tools)
    
    return tool_node.with_fallbacks(
        [RunnableLambda(fallback_handler)],
        exception_key="error",
    )


# =============================================================================
# INSTRUMENTED TOOL EXECUTION
# =============================================================================

class InstrumentedToolNode:
    """
    A wrapper around ToolNode that adds instrumentation:
    - Timing for each tool call
    - Success/failure metrics
    - Detailed logging
    
    This can be used as a drop-in replacement for ToolNode.
    """
    
    def __init__(self, tools: List[Any], name: str = "tools"):
        self.tools = tools
        self.name = name
        self._tool_node = create_tool_node_with_error_handling(tools, name)
        self._tools_by_name = {t.name: t for t in tools}
        
        logger.info(
            "InstrumentedToolNode.init: name=%s tools=%s",
            name,
            list(self._tools_by_name.keys())
        )
    
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the tool node with instrumentation.
        
        Args:
            state: Current agent state
            config: Optional configuration
            
        Returns:
            Updated state with tool results
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "InstrumentedToolNode.invoke called in async context; use await ainvoke() instead."
            )

        messages = state.get("messages", [])
        
        # Extract tool calls for logging
        tool_calls = []
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls or []
        
        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        last_tool_called = state.get("last_tool_called")
        last_tool_status = state.get("last_tool_status")
        
        logger.info(
            "InstrumentedToolNode.invoke.start: name=%s tools=%s",
            self.name,
            tool_names
        )

        if last_tool_status == "error" and last_tool_called:
            for tool_name in tool_names:
                if tool_name == last_tool_called:
                    try:
                        TOOL_RETRIES_TOTAL.labels(tool=tool_name).inc()
                    except Exception:
                        pass
        
        start_time = time.time()
        
        try:
            result = self._tool_node.invoke(state, config)
            
            duration = time.time() - start_time
            
            # Log success
            logger.info(
                "InstrumentedToolNode.invoke.completed: name=%s tools=%s duration=%.3fs",
                self.name,
                tool_names,
                duration
            )
            
            # Record metrics
            for tool_name in tool_names:
                try:
                    TOOL_LATENCY.labels(tool_name=tool_name, status="success").observe(duration)
                    TOOL_CALLS_TOTAL.labels(tool=tool_name, status="success").inc()
                except Exception:
                    pass  # Don't fail on metrics
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "InstrumentedToolNode.invoke.error: name=%s tools=%s duration=%.3fs error=%s\n%s",
                self.name,
                tool_names,
                duration,
                str(e),
                traceback.format_exc()
            )
            
            # Record metrics
            for tool_name in tool_names:
                try:
                    TOOL_LATENCY.labels(tool_name=tool_name, status="error").observe(duration)
                    TOOL_CALLS_TOTAL.labels(tool=tool_name, status="error").inc()
                except Exception:
                    pass  # Don't fail on metrics
            
            # Use our error handler
            return handle_tool_error(state, e)
    
    async def ainvoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async invoke the tool node with instrumentation.
        
        Args:
            state: Current agent state
            config: Optional configuration
            
        Returns:
            Updated state with tool results
        """
        messages = state.get("messages", [])
        
        # Extract tool calls for logging
        tool_calls = []
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls or []
        
        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        last_tool_called = state.get("last_tool_called")
        last_tool_status = state.get("last_tool_status")
        
        logger.info(
            "InstrumentedToolNode.ainvoke.start: name=%s tools=%s",
            self.name,
            tool_names
        )

        if last_tool_status == "error" and last_tool_called:
            for tool_name in tool_names:
                if tool_name == last_tool_called:
                    try:
                        TOOL_RETRIES_TOTAL.labels(tool=tool_name).inc()
                    except Exception:
                        pass
        
        start_time = time.time()
        
        try:
            result = await self._tool_node.ainvoke(state, config)
            
            duration = time.time() - start_time
            
            logger.info(
                "InstrumentedToolNode.ainvoke.completed: name=%s tools=%s duration=%.3fs",
                self.name,
                tool_names,
                duration
            )
            
            # Record metrics
            for tool_name in tool_names:
                try:
                    TOOL_LATENCY.labels(tool_name=tool_name, status="success").observe(duration)
                    TOOL_CALLS_TOTAL.labels(tool=tool_name, status="success").inc()
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "InstrumentedToolNode.ainvoke.error: name=%s tools=%s duration=%.3fs error=%s\n%s",
                self.name,
                tool_names,
                duration,
                str(e),
                traceback.format_exc()
            )
            
            # Record metrics
            for tool_name in tool_names:
                try:
                    TOOL_LATENCY.labels(tool_name=tool_name, status="error").observe(duration)
                    TOOL_CALLS_TOTAL.labels(tool=tool_name, status="error").inc()
                except Exception:
                    pass
            
            return handle_tool_error(state, e)

