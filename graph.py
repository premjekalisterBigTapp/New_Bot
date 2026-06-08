"""
LangGraph State Machine for the BigTapp Agentic Chatbot.

This module defines the main conversation graph with:
- Token-aware memory compression node for rolling summarization
- Autonomous supervisor with Command-based routing
- Specialist agent nodes for different intents
- Self-correction node for tool error recovery
- Live agent handoff node for human escalation
- Styler node for response polishing
- Redis checkpointer for conversation persistence
- Context schema for runtime context (session_id, channel)

Agent Autonomy Features:
- Command-based routing: Supervisor uses Command(goto=...) for dynamic routing
- Self-correction: Automatic recovery from repeated tool errors
- Live agent handoff: Seamless escalation to human agents
- Flexible flow: Agents can influence routing via state updates

Memory Management:
- Uses token-aware summarization (count_tokens_approximately)
- Preserves tool call chains (AIMessage + ToolMessage pairs)
- Product-aware context to prevent slot bleeding
- Structured RunningSummary tracking in memory_context

Context Engineering:
- AgentContext schema provides user-level context to nodes
- Nodes can access context via runtime.context
- Context is passed when invoking the graph

The graph supports:
- Parallel execution of memory compression and main flow
- Command-based routing from supervisor (no conditional edges needed)
- State persistence across server restarts
"""
from __future__ import annotations

import os
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .middleware import AgentContext  # Context schema for runtime context
from .nodes.memory_nodes import compress_memory_node  # Token-aware memory management
from .nodes.supervisor import _supervisor_node
from .nodes.autonomous_routing import (
    self_correction_node,
    live_agent_handoff_node,
)
from .nodes.agents import (
    _greet_agent_node,
    _capabilities_agent_node,
    _chat_agent_node,
    _info_agent_node,
    _summary_agent_node,
    _compare_agent_node,
    _purchase_agent_node,
)
from .nodes.life_insurance import _life_insurance_agent_node
from .nodes.sales_agent import _sales_agent_node
from .nodes.rec_subgraph import recommendation_subgraph
from .nodes.service_subgraph import service_subgraph
from .nodes.styler import _style_reply_node

logger = logging.getLogger(__name__)

# =============================================================================
# GRAPH BUILDER
# =============================================================================

# Create graph with state schema and context schema
# The context_schema allows passing runtime context (session_id, channel) when invoking
_graph_builder: StateGraph[AgentState] = StateGraph(
    AgentState,
    context_schema=AgentContext,
)

_registered_nodes = []


def _register_node(name: str, node) -> None:
    _graph_builder.add_node(name, node)
    _registered_nodes.append(name)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

# Core nodes
# Memory compression uses token-aware summarization with safe tool chain handling
_register_node("compress", compress_memory_node)

# Autonomous supervisor with Command-based routing
# The supervisor now returns Command(update={...}, goto="target_node")
# This eliminates the need for conditional edges - routing is done via Command
_register_node("supervisor", _supervisor_node)

# Autonomous routing nodes
# Self-correction node: handles tool error recovery
_register_node("self_correction", self_correction_node)
# Live agent handoff node: handles human escalation
_register_node("live_agent_handoff", live_agent_handoff_node)

# Specialist agent nodes
_register_node("greet_agent", _greet_agent_node)
_register_node("capabilities_agent", _capabilities_agent_node)
_register_node("chat_agent", _chat_agent_node)
_register_node("info_agent", _info_agent_node)
_register_node("summary_agent", _summary_agent_node)
_register_node("compare_agent", _compare_agent_node)
_register_node("purchase_agent", _purchase_agent_node)
# Life insurance agent: answers grounded on the brochure KB (life_insurance intent)
_register_node("life_insurance_agent", _life_insurance_agent_node)
# LLM-driven end-to-end sales journey agent (discovery → upsell → payment → cross-sell)
_register_node("sales_agent", _sales_agent_node)

# Recommendation subgraph (handles multi-step slot collection and recommendation)
_register_node("recommendation", recommendation_subgraph)

# Service flow subgraph (handles policy/claim status and customer updates)
_register_node("service_flow", service_subgraph)

# Response styling node
_register_node("styler", _style_reply_node)

logger.info("Graph.nodes_registered: %d nodes", len(_registered_nodes))

# =============================================================================
# EDGE CONFIGURATION
# =============================================================================

edge_count = 0

# Parallel execution: run rolling memory compression alongside the main
# multi-agent flow. The main flow is: supervisor → (Command-based routing) → styler.
_graph_builder.add_edge(START, "compress")
edge_count += 1
_graph_builder.add_edge(START, "supervisor")
edge_count += 1

# NOTE: No conditional edges from supervisor!
# The supervisor now uses Command(goto="node_name") for routing.
# LangGraph automatically follows the goto target from Command returns.
# This enables autonomous routing to self_correction, live_agent_handoff, etc.

# All specialist agents route to styler for response polishing
_styler_sources = [
    "greet_agent",
    "capabilities_agent",
    "chat_agent",
    "info_agent",
    "summary_agent",
    "compare_agent",
    "purchase_agent",
    "life_insurance_agent",
    "sales_agent",
    "recommendation",
    # "service_flow",  # Handling service_flow separately below with conditional routing
]
for node_name in _styler_sources:
    _graph_builder.add_edge(node_name, "styler")
edge_count += len(_styler_sources)

# Conditional routing for service_flow
# If exiting with intent, go back to supervisor. Otherwise, go to styler.
def _route_service_flow(state: AgentState) -> str:
    if state.get("service_exit_intent"):
        logger.info("Graph.route_service_flow: exit_intent=%s -> supervisor", state.get("service_exit_intent"))
        return "supervisor"
    return "styler"

_service_flow_targets = ["supervisor", "styler"]
_graph_builder.add_conditional_edges(
    "service_flow",
    _route_service_flow,
    _service_flow_targets,
)
edge_count += len(_service_flow_targets)

# Self-correction routes back to supervisor for re-routing
# This enables the correction → retry cycle
_graph_builder.add_edge("self_correction", "supervisor")
edge_count += 1

# Live agent handoff routes to styler (the handoff flag is in state)
# Actual handoff is handled by the API/WhatsApp layer based on state
_graph_builder.add_edge("live_agent_handoff", "styler")
edge_count += 1

# Final edges
_graph_builder.add_edge("styler", END)
edge_count += 1
_graph_builder.add_edge("compress", END)
edge_count += 1

logger.info(
    "Graph.edges_configured: edges=%d autonomous_routing_enabled=True",
    edge_count,
)

# =============================================================================
# CHECKPOINTER CONFIGURATION
# =============================================================================

_checkpointer = None


def _get_checkpointer():
    """Get the appropriate checkpointer based on environment.
    
    Uses Redis in production for distributed state persistence,
    falls back to MemorySaver for local development.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    
    use_redis = os.getenv("AGENTIC_USE_REDIS_CHECKPOINTER", "true").lower() in ("true", "1", "yes")
    
    if use_redis:
        try:
            from .infrastructure.redis_checkpointer import RedisCheckpointer
            _checkpointer = RedisCheckpointer()
            logger.info("Graph.checkpointer: using Redis for conversation persistence")
            return _checkpointer
        except Exception as e:
            logger.warning(
                "Graph.checkpointer: failed to initialize Redis, falling back to MemorySaver: %s",
                str(e)
            )
    
    _checkpointer = MemorySaver()
    logger.info("Graph.checkpointer: using in-memory MemorySaver")
    return _checkpointer


# =============================================================================
# GRAPH COMPILATION
# =============================================================================

# Lazy initialization - compile graph when first accessed
_agent_graph = None
_memory_saver = None  # Keep for backwards compatibility


def get_agent_graph():
    """Get the compiled agent graph with appropriate checkpointer.
    
    The graph is compiled lazily on first access to allow for
    configuration to be loaded first.
    
    Returns:
        Compiled StateGraph instance with checkpointer
    """
    global _agent_graph, _memory_saver
    if _agent_graph is None:
        logger.info("Graph.compile: initializing agent graph")
        _memory_saver = _get_checkpointer()
        _agent_graph = _graph_builder.compile(checkpointer=_memory_saver)
        logger.info("Graph.compile: agent graph ready")
    return _agent_graph


def reset_graph():
    """Reset the graph to force recompilation.
    
    This is useful for testing or when configuration changes.
    """
    global _agent_graph, _memory_saver, _checkpointer
    _agent_graph = None
    _memory_saver = None
    _checkpointer = None
    logger.info("Graph.reset: graph state cleared")
