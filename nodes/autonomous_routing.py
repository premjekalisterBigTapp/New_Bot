"""
Autonomous Routing Module - Command-based flow control for agent autonomy.

This module addresses the "Agent Autonomy Limitations" identified in IMPROVEMENT_ANALYSIS.md:
1. Rigid Routing Logic → Command-based dynamic routing
2. No Self-Correction/Reflection → Tool error recovery and self-critique
3. Agents Can't Control Flow → Command with goto for node navigation
4. No Interrupt Support → Live agent handoff with interrupt

Key concepts from LangGraph:
- Command(update={...}, goto="node_name"): Update state AND route to next node
- Command(goto="node", graph=Command.PARENT): Navigate to parent graph node
- interrupt({...}): Pause execution for human approval

This enables:
- Specialist agents to delegate back to supervisor
- Self-correction when tool calls fail
- Dynamic routing based on conversation state
- Human-in-the-loop for sensitive operations
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from ..state import AgentState
from ..infrastructure.metrics import (
    TOOL_ERRORS_TOTAL,
    LIVE_AGENT_HANDOFFS,
    INTENT_CLASSIFICATION_TOTAL,
    AUTONOMOUS_ROUTING_TOTAL,
    SELF_CORRECTION_TOTAL,
    REFLECTION_LATENCY,
    INTERRUPT_REQUESTS_TOTAL,
    ROUTING_DECISION_LATENCY,
    COMMAND_RETURNS_TOTAL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING DECISION TYPES
# =============================================================================

class RoutingDecision(str, Enum):
    """Possible routing decisions for autonomous agents."""
    CONTINUE = "continue"           # Continue to next node in normal flow
    DELEGATE_SUPERVISOR = "delegate_supervisor"  # Go back to supervisor for re-routing
    SELF_CORRECT = "self_correct"   # Trigger self-correction
    ESCALATE_HUMAN = "escalate_human"  # Escalate to live agent
    RECLASSIFY = "reclassify"       # Reclassify intent
    END = "end"                     # End the conversation turn


class RoutingContext(BaseModel):
    """Context for making routing decisions."""
    last_tool_status: Optional[str] = None
    tool_error_count: int = 0
    turn_count: int = 0
    rec_given: bool = False
    purchase_offered: bool = False
    live_agent_requested: bool = False
    confidence: float = 1.0
    
    model_config = ConfigDict(extra="allow")


# =============================================================================
# ROUTING DECISION LOGIC
# =============================================================================

def analyze_routing_context(state: AgentState) -> RoutingContext:
    """
    Analyze current state to build routing context.
    
    This extracts relevant signals for routing decisions:
    - Tool errors and their count
    - Conversation phase (rec_given, purchase_offered)
    - Live agent request detection
    - Turn count for long conversation handling
    """
    messages = state.get("messages", [])
    
    # Count recent tool errors
    tool_error_count = 0
    for msg in messages[-10:]:  # Check last 10 messages
        if isinstance(msg, ToolMessage):
            status = getattr(msg, "status", None) or msg.additional_kwargs.get("status")
            if status == "error":
                tool_error_count += 1
    
    # Check for live agent request in recent AI messages
    live_agent_requested = False
    for msg in reversed(messages[-5:]):
        if isinstance(msg, AIMessage):
            content = str(getattr(msg, "content", "") or "").lower()
            if "connect you with a live agent" in content:
                live_agent_requested = True
                break
    
    context = RoutingContext(
        last_tool_status=state.get("last_tool_status"),
        tool_error_count=tool_error_count,
        turn_count=state.get("turn_count", 0),
        rec_given=state.get("rec_given", False),
        purchase_offered=state.get("purchase_offered", False),
        live_agent_requested=live_agent_requested,
    )
    
    logger.debug(
        "AutonomousRouting.context: tool_errors=%d turn=%d rec_given=%s live_agent=%s",
        context.tool_error_count,
        context.turn_count,
        context.rec_given,
        context.live_agent_requested,
    )
    
    return context


def decide_routing(
    state: AgentState,
    current_node: str,
    context: Optional[RoutingContext] = None,
) -> RoutingDecision:
    """
    Make autonomous routing decision based on state and context.
    
    Decision logic:
    1. If live agent requested → ESCALATE_HUMAN
    2. If multiple tool errors → SELF_CORRECT
    3. If recommendation given and user says "yes" → Continue to purchase
    4. If conversation stuck → DELEGATE_SUPERVISOR for re-routing
    5. Otherwise → CONTINUE
    """
    if context is None:
        context = analyze_routing_context(state)
    
    decision = RoutingDecision.CONTINUE
    reason = "default_flow"
    
    # Priority 1: Live agent escalation
    if context.live_agent_requested:
        decision = RoutingDecision.ESCALATE_HUMAN
        reason = "live_agent_requested"
        logger.info(
            "AutonomousRouting.decision: %s from %s reason=%s",
            decision.value, current_node, reason
        )
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node=current_node,
            target_node="live_agent_handoff",
        ).inc()
        return decision
    
    # Priority 2: Self-correction on repeated tool errors
    if context.tool_error_count >= 2:
        decision = RoutingDecision.SELF_CORRECT
        reason = "repeated_tool_errors"
        logger.warning(
            "AutonomousRouting.decision: %s from %s reason=%s error_count=%d",
            decision.value, current_node, reason, context.tool_error_count
        )
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node=current_node,
            target_node="self_correction",
        ).inc()
        return decision
    
    # Priority 3: Long conversation without progress
    if context.turn_count > 20 and not context.rec_given:
        decision = RoutingDecision.DELEGATE_SUPERVISOR
        reason = "long_conversation_no_progress"
        logger.info(
            "AutonomousRouting.decision: %s from %s reason=%s turn=%d",
            decision.value, current_node, reason, context.turn_count
        )
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node=current_node,
            target_node="supervisor",
        ).inc()
        return decision
    
    # Default: continue normal flow
    AUTONOMOUS_ROUTING_TOTAL.labels(
        source_node=current_node,
        target_node="continue",
    ).inc()
    
    return decision


# =============================================================================
# SELF-CORRECTION NODE
# =============================================================================

def self_correction_node(state: AgentState) -> Dict[str, Any]:
    """
    Self-correction node for recovering from tool errors.
    
    This node:
    1. Analyzes recent tool errors
    2. Generates a corrective system message
    3. Clears error state for retry
    4. Optionally delegates to supervisor for re-routing
    
    Returns state updates (not Command) to allow normal flow continuation.
    """
    start_time = time.perf_counter()
    
    messages = state.get("messages", [])
    tool_errors = state.get("tool_errors", [])
    turn_count = state.get("turn_count", 0)
    
    logger.info(
        "SelfCorrection.start: turn=%d errors=%d",
        turn_count, len(tool_errors)
    )
    
    # Analyze recent tool errors
    recent_errors = []
    for msg in messages[-10:]:
        if isinstance(msg, ToolMessage):
            status = getattr(msg, "status", None) or msg.additional_kwargs.get("status")
            if status == "error":
                error_info = {
                    "tool": getattr(msg, "name", "unknown"),
                    "content": str(getattr(msg, "content", ""))[:200],
                }
                recent_errors.append(error_info)
    
    # Build corrective guidance
    if recent_errors:
        error_summary = "\n".join([
            f"- {e['tool']}: {e['content']}" for e in recent_errors[-3:]
        ])
        corrective_message = (
            f"SELF-CORRECTION TRIGGERED:\n"
            f"Recent tool errors:\n{error_summary}\n\n"
            f"Please try a different approach:\n"
            f"1. If validation failed, check your input format\n"
            f"2. If service failed, try a simpler query\n"
            f"3. If stuck, ask the user for clarification\n"
            f"Do NOT repeat the same failing tool call."
        )
    else:
        corrective_message = (
            "SELF-CORRECTION TRIGGERED:\n"
            "The previous approach didn't work well. "
            "Please try a different strategy or ask for clarification."
        )
    
    duration = time.perf_counter() - start_time
    
    logger.info(
        "SelfCorrection.completed: turn=%d errors_analyzed=%d duration=%.3fs",
        turn_count, len(recent_errors), duration
    )
    
    # Record metrics
    SELF_CORRECTION_TOTAL.labels(
        trigger="tool_error",
        outcome="guidance_injected"
    ).inc()
    REFLECTION_LATENCY.observe(duration)
    
    # Return state updates
    # The corrective message will be injected as a system message
    # Tool errors are cleared to allow retry
    return {
        "messages": [SystemMessage(content=corrective_message)],
        "tool_errors": [],  # Clear errors for retry
        "last_tool_status": "self_corrected",
    }


# =============================================================================
# LIVE AGENT HANDOFF NODE
# =============================================================================

# Valid target nodes for live agent handoff
# Only routes to styler - actual handoff is handled by API/WhatsApp layer
LiveAgentTargets = Literal["styler"]


def live_agent_handoff_node(state: AgentState) -> Union[Command[LiveAgentTargets], Dict[str, Any]]:
    """
    Live agent handoff node with interrupt support.
    
    This node:
    1. Detects live agent request from conversation
    2. Optionally interrupts for confirmation (if configured)
    3. Sets live_agent_requested flag
    4. Routes to appropriate next node
    
    Uses Command for flow control when routing is needed.
    """
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 0)
    
    logger.info(
        "LiveAgentHandoff.start: turn=%d msgs=%d",
        turn_count, len(messages)
    )
    
    # Check if live agent was already requested
    already_requested = state.get("live_agent_requested", False)
    
    # Detect live agent request from recent messages
    live_agent_detected = False
    for msg in reversed(messages[-5:]):
        if isinstance(msg, AIMessage):
            content = str(getattr(msg, "content", "") or "").lower()
            if "connect you with a live agent" in content:
                live_agent_detected = True
                break
        elif isinstance(msg, HumanMessage):
            content = str(getattr(msg, "content", "") or "").lower()
            # User explicitly asking for human
            human_keywords = ["human", "agent", "person", "customer service", "speak to someone"]
            if any(kw in content for kw in human_keywords):
                live_agent_detected = True
                break
    
    if live_agent_detected or already_requested:
        logger.info(
            "LiveAgentHandoff.detected: turn=%d already_requested=%s",
            turn_count, already_requested
        )
        
        # Record metric
        LIVE_AGENT_HANDOFFS.inc()
        INTERRUPT_REQUESTS_TOTAL.labels(interrupt_type="live_agent").inc()
        
        # Set the flag and continue to styler
        # The actual handoff is handled by the WhatsApp handler or API layer
        # based on the live_agent_requested flag in the response
        return Command(
            update={
                "live_agent_requested": True,
                "last_tool_status": "live_agent_handoff",
            },
            goto="styler",
        )
    
    # No handoff needed, continue normal flow
    logger.debug("LiveAgentHandoff.not_needed: turn=%d", turn_count)
    return {}


# =============================================================================
# AUTONOMOUS SUPERVISOR WITH COMMAND ROUTING
# =============================================================================

# Valid target nodes for supervisor routing
SupervisorTargets = Literal[
    "greet_agent",
    "capabilities_agent", 
    "chat_agent",
    "info_agent",
    "summary_agent",
    "compare_agent",
    "purchase_agent",
    "recommendation",
    "self_correction",
    "live_agent_handoff",
    "styler",
]


async def autonomous_supervisor_node(state: AgentState) -> Command[SupervisorTargets]:
    """
    Autonomous supervisor with Command-based routing.
    
    This replaces the original supervisor + conditional edge pattern with
    a single node that uses Command for both state updates AND routing.
    
    Benefits:
    - State update and routing in one place
    - Can route to self_correction or live_agent_handoff
    - Cleaner code without separate edge functions
    
    Flow control via Command:
    - Command(update={...}, goto="node_name")
    """
    from .feedback import (
        _classify_feedback_from_messages_async,
        _self_critique_and_rewrite_from_messages_async,
    )
    from .intent import _classify_intent_from_messages_async
    
    start_time = time.perf_counter()
    
    messages = list(state.get("messages", []) or [])
    if not messages:
        logger.warning("AutonomousSupervisor: no messages, routing to chat_agent")
        return Command(update={}, goto="chat_agent")
    
    known_product = state.get("product")
    turn_count = state.get("turn_count", 0)
    
    # Build routing context
    routing_context = analyze_routing_context(state)
    
    # Priority 1: Check for live agent escalation
    if routing_context.live_agent_requested:
        logger.info(
            "AutonomousSupervisor.live_agent: turn=%d",
            turn_count
        )
        return Command(
            update={"intent": "live_agent"},
            goto="live_agent_handoff",
        )
    
    # Priority 2: Check for self-correction need
    if routing_context.tool_error_count >= 2:
        logger.warning(
            "AutonomousSupervisor.self_correction: turn=%d errors=%d",
            turn_count, routing_context.tool_error_count
        )
        return Command(
            update={"intent": "self_correct"},
            goto="self_correction",
        )
    
    # Priority 3: Negative feedback handling / reflection
    feedback = await _classify_feedback_from_messages_async(messages)
    if feedback and feedback.category == "negative_feedback":
        revised = await _self_critique_and_rewrite_from_messages_async(messages)
        if revised:
            logger.info(
                "AutonomousSupervisor.negative_feedback: turn=%d",
                turn_count
            )
            return Command(
                update={
                    "messages": [AIMessage(content=revised)],
                    "feedback": "negative_feedback",
                    "sources": [],
                    "intent": "reflect_done",
                },
                goto="styler",
            )
    
    # Priority 4: Intent classification
    pending_slot = state.get("pending_slot")
    intent_pred = await _classify_intent_from_messages_async(
        messages,
        known_product,
        active_slot=pending_slot,
        rec_paused=state.get("rec_paused", False),
    )
    
    raw_intent = (intent_pred.intent or "").strip().lower()
    
    # Map intent to target node
    intent_to_node = {
        "info": "info_agent",
        "summary": "summary_agent",
        "compare": "compare_agent",
        "recommend": "recommendation",
        "purchase": "purchase_agent",
        "capabilities": "capabilities_agent",
        "greet": "greet_agent",
        "chat": "chat_agent",
        "other": "chat_agent",
    }
    
    # Normalize intent
    if raw_intent not in intent_to_node:
        normalized_intent = "chat"
        target_node = "chat_agent"
    else:
        normalized_intent = raw_intent
        target_node = intent_to_node[raw_intent]
    
    product = intent_pred.product or known_product
    
    # Build state updates
    updates: Dict[str, Any] = {
        "intent": normalized_intent,
        "product": product,
    }
    
    # Handle product switch or reset
    is_reset = getattr(intent_pred, "reset", False)
    is_product_switch = (product and known_product and product != known_product)
    
    if is_product_switch or is_reset:
        reason = "product switch" if is_product_switch else "explicit reset request"
        logger.info(
            "AutonomousSupervisor.clearing_slots: reason=%s old=%s new=%s",
            reason, known_product, product
        )
        updates["slots"] = {}
        updates["rec_ready"] = False
    
    duration = time.perf_counter() - start_time
    
    logger.info(
        "AutonomousSupervisor.routing: turn=%d intent=%s product=%s -> %s duration=%.3fs",
        turn_count, normalized_intent, product, target_node, duration
    )
    
    # Record metrics
    INTENT_CLASSIFICATION_TOTAL.labels(
        intent=normalized_intent,
        product=product or "unknown"
    ).inc()
    AUTONOMOUS_ROUTING_TOTAL.labels(
        source_node="supervisor",
        target_node=target_node,
    ).inc()
    
    return Command(update=updates, goto=target_node)


# =============================================================================
# AGENT NODE WRAPPER FOR COMMAND-BASED ROUTING
# =============================================================================

def wrap_agent_with_autonomous_routing(
    agent_node_func,
    node_name: str,
) -> callable:
    """
    Wrap an agent node function to add autonomous routing capabilities.
    
    This wrapper:
    1. Calls the original agent node
    2. Analyzes the result for routing decisions
    3. Returns Command if routing is needed, or original result otherwise
    
    This allows existing agent nodes to participate in autonomous routing
    without modifying their core logic.
    """
    def wrapped_node(state: AgentState) -> Union[Command, Dict[str, Any]]:
        # Call original agent
        result = agent_node_func(state)
        
        # Merge result into state for routing analysis
        merged_state = {**state, **result}
        
        # Analyze for routing decision
        context = analyze_routing_context(merged_state)
        decision = decide_routing(merged_state, node_name, context)
        
        if decision == RoutingDecision.ESCALATE_HUMAN:
            logger.info(
                "WrappedAgent.%s: escalating to live agent",
                node_name
            )
            return Command(
                update={**result, "live_agent_requested": True},
                goto="live_agent_handoff",
            )
        
        if decision == RoutingDecision.SELF_CORRECT:
            logger.info(
                "WrappedAgent.%s: triggering self-correction",
                node_name
            )
            return Command(
                update=result,
                goto="self_correction",
            )
        
        if decision == RoutingDecision.DELEGATE_SUPERVISOR:
            logger.info(
                "WrappedAgent.%s: delegating to supervisor",
                node_name
            )
            return Command(
                update=result,
                goto="supervisor",
            )
        
        # Normal flow - return result and let graph edges handle routing
        return result
    
    wrapped_node.__name__ = f"autonomous_{node_name}"
    wrapped_node.__doc__ = f"Autonomous wrapper for {node_name}"
    
    return wrapped_node


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Routing types
    "RoutingDecision",
    "RoutingContext",
    
    # Analysis functions
    "analyze_routing_context",
    "decide_routing",
    
    # Nodes
    "self_correction_node",
    "live_agent_handoff_node",
    "autonomous_supervisor_node",
    
    # Wrapper
    "wrap_agent_with_autonomous_routing",
    
    # Metrics
    "AUTONOMOUS_ROUTING_TOTAL",
    "SELF_CORRECTION_TOTAL",
    "REFLECTION_LATENCY",
    "INTERRUPT_REQUESTS_TOTAL",
]

