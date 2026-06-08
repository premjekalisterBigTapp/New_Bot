"""
Tools module for the BigTapp Agentic Chatbot.

This module provides:
- Enterprise-grade tool implementations with Command-based state updates
- Custom ToolNode with comprehensive error handling
- Tool metrics and logging

All tools follow LangGraph v1 patterns:
- InjectedToolCallId for proper ToolMessage generation
- InjectedState for accessing agent state
- Command objects for state updates
- Structured error handling with status='error'
"""

from .unified import (
    TOOLS,
    TOOLS_BY_NAME,
    save_progress,
    search_product_knowledge,
    compare_plans,
    get_product_recommendation,
    generate_purchase_link,
    ToolExecutionError,
    ValidationError,
    ExternalServiceError,
)

from .tool_node import (
    handle_tool_error,
    create_tool_node_with_error_handling,
    create_tool_node_with_fallback,
    InstrumentedToolNode,
)

__all__ = [
    # Tools
    "TOOLS",
    "TOOLS_BY_NAME",
    "save_progress",
    "search_product_knowledge",
    "compare_plans",
    "get_product_recommendation",
    "generate_purchase_link",
    # Error types
    "ToolExecutionError",
    "ValidationError",
    "ExternalServiceError",
    # Tool node utilities
    "handle_tool_error",
    "create_tool_node_with_error_handling",
    "create_tool_node_with_fallback",
    "InstrumentedToolNode",
]

