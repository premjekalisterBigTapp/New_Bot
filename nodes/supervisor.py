"""
Supervisor Node - Intent classification, phase tracking, and autonomous routing.

This module implements the supervisor with:
- Explicit conversation phase tracking (ConversationPhase enum)
- Phase history for debugging and analytics
- Pronoun/reference context extraction and passing
- Summary-aware intent classification
- Autonomous Command-based routing

Multi-Turn Conversation Fixes:
1. Intent classification now uses summary for full history context
2. Pronoun resolution via ReferenceContext
3. Explicit phase tracking with phase_history
4. Phase-aware intent classification

The supervisor uses Command(update={...}, goto="node") to:
- Update state AND route in a single operation
- Track conversation phase transitions
- Route to self_correction on repeated tool errors
- Route to live_agent_handoff when detected
- Handle negative feedback with reflection
"""
from __future__ import annotations

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, Union, Literal, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ..utils.pii_masker import get_pii_masker
from ..utils.slots import _normalize_product_key

from ..state import AgentState, ConversationPhase, ReferenceContext
from .feedback import (
    _classify_feedback_from_messages_async,
    _self_critique_and_rewrite_from_messages_async,
)
from .intent import _classify_intent_from_messages_async, _extract_reference_context
from .autonomous_routing import (
    analyze_routing_context,
    AUTONOMOUS_ROUTING_TOTAL,
    SELF_CORRECTION_TOTAL,
)
from ..infrastructure.metrics import INTENT_CLASSIFICATION_TOTAL

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE TRACKING METRICS (imported from centralized metrics)
# =============================================================================

from ..infrastructure.metrics import PHASE_TRANSITION_TOTAL, PHASE_DURATION_TURNS as PHASE_DURATION


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
    "service_flow",
    "self_correction",
    "live_agent_handoff",
    "styler",
]

def _normalize_user_text(text: str) -> str:
    """Normalize user text for lightweight loop detection."""
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t.strip()


def _is_capabilities_or_services_query(text: str) -> bool:
    """Heuristic detection for 'services' / 'capabilities' questions (no extra LLM call)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    triggers = [
        "what can you do",
        "what do you do",
        "what can u do",
        "what do u do",
        "what are the services",
        "what are your services",
        "what services do you offer",
        "services you offer",
        "how can you help",
        "how do you help",
        "what help can you provide",
        "supported products",
        "which products do you support",
        "what products do you have",
        "capabilities",
        "features",
    ]
    return any(k in t for k in triggers)


def _last_user_text(messages: List[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return str(getattr(m, "content", "") or "").strip()
    return ""


def _last_ai_text(messages: List[Any]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            return str(getattr(m, "content", "") or "").strip()
    return ""


def _user_repeated_question(messages: List[Any]) -> bool:
    """True if the last two user messages are effectively the same."""
    user_msgs = [
        str(getattr(m, "content", "") or "")
        for m in (messages or [])
        if isinstance(m, HumanMessage)
    ]
    if len(user_msgs) < 2:
        return False
    a = _normalize_user_text(user_msgs[-1])
    b = _normalize_user_text(user_msgs[-2])
    return bool(a) and a == b


def _compute_new_phase(
    intent: str,
    product: Optional[str],
    rec_given: bool,
    purchase_offered: bool,
    live_agent_requested: bool,
) -> ConversationPhase:
    """
    Compute the new conversation phase based on intent and state.
    
    This provides a deterministic mapping from intent + state to phase,
    ensuring consistent phase tracking across the system.
    """
    if live_agent_requested:
        return ConversationPhase.ESCALATION
    
    return ConversationPhase.from_intent(
        intent=intent,
        has_product=bool(product),
        rec_given=rec_given,
        purchase_offered=purchase_offered,
    )


def _update_phase_history(
    current_history: List[str],
    new_phase: ConversationPhase,
    max_history: int = 20,
) -> List[str]:
    """
    Update phase history with new phase, maintaining max length.
    
    Args:
        current_history: Existing phase history
        new_phase: New phase to add
        max_history: Maximum history length to maintain
        
    Returns:
        Updated phase history
    """
    history = list(current_history) if current_history else []
    history.append(new_phase.value)
    
    # Trim to max length
    if len(history) > max_history:
        history = history[-max_history:]
    
    return history


async def _supervisor_node(state: AgentState, config: RunnableConfig) -> Union[Command[SupervisorTargets], Dict[str, Any]]:
    """
    Supervisor node with phase tracking, pronoun resolution, and autonomous routing.
    
    This node addresses Multi-Turn Conversation Failures:
    1. Uses summary for intent classification (full history context)
    2. Extracts and passes reference context for pronoun resolution
    3. Tracks explicit conversation phase with phase_history
    4. Routes autonomously based on state and intent
    
    Autonomous routing features:
    - Routes to self_correction on repeated tool errors
    - Routes to live_agent_handoff when detected
    - Clears slots on product switch or reset
    - Updates conversation phase on every turn
    
    Performance optimization:
    - Runs feedback and intent classification in PARALLEL using asyncio.gather
    - Saves ~1-2s per turn by eliminating sequential LLM calls
    
    Returns:
        Command with update and goto
    """
    start_time = time.perf_counter()
    
    messages = list(state.get("messages", []) or [])
    if not messages:
        logger.warning("Supervisor.no_messages: routing to chat_agent")
        return Command(
            update={
                "phase": ConversationPhase.GREETING.value,
                "phase_history": [ConversationPhase.GREETING.value],
            },
            goto="chat_agent"
        )

    known_product = state.get("product")
    turn_count = state.get("turn_count", 0)
    current_phase = state.get("phase")
    current_slots = state.get("slots") or {}
    summary = state.get("summary", "")
    rec_given = state.get("rec_given", False)
    purchase_offered = state.get("purchase_offered", False)
    phase_history = state.get("phase_history") or []
    
    # Build routing context for autonomous decisions
    routing_context = analyze_routing_context(state)
    
    # Extract reference context for pronoun resolution
    reference_context = _extract_reference_context(messages, known_product, current_slots)
    
    logger.debug(
        "Supervisor.context: turn=%d phase=%s tool_errors=%d live_agent=%s product=%s",
        turn_count,
        current_phase,
        routing_context.tool_error_count,
        routing_context.live_agent_requested,
        known_product,
    )
    
    # Priority 1: Check for live agent escalation
    if routing_context.live_agent_requested:
        new_phase = ConversationPhase.ESCALATION
        
        logger.info(
            "Supervisor.live_agent_detected: turn=%d phase=%s->%s routing to live_agent_handoff",
            turn_count, current_phase, new_phase.value
        )
        
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node="supervisor",
            target_node="live_agent_handoff",
        ).inc()
        PHASE_TRANSITION_TOTAL.labels(
            from_phase=current_phase or "unknown",
            to_phase=new_phase.value,
        ).inc()
        
        return Command(
            update={
                "intent": "live_agent",
                "phase": new_phase.value,
                "phase_history": _update_phase_history(phase_history, new_phase),
            },
            goto="live_agent_handoff",
        )
    
    # Priority 2: Check for self-correction need (repeated tool errors)
    if routing_context.tool_error_count >= 2:
        logger.warning(
            "Supervisor.tool_errors_detected: turn=%d errors=%d routing to self_correction",
            turn_count, routing_context.tool_error_count
        )
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node="supervisor",
            target_node="self_correction",
        ).inc()
        SELF_CORRECTION_TOTAL.labels(
            trigger="tool_error",
            outcome="routing_to_correction"
        ).inc()
        return Command(
            update={"intent": "self_correct"},
            goto="self_correction",
        )

    # ======================================================================
    # SERVICE EXIT INTENT: Handle direct routing from Policy Service Orchestrator
    # 
    # When the service orchestrator detects the user wants to exit to a different
    # agent (info, recommend, compare), it sets service_exit_intent. We route
    # directly to the target without re-classification, preventing loops.
    # ======================================================================
    service_exit_intent = state.get("service_exit_intent")
    if service_exit_intent:
        service_exit_query = state.get("service_exit_query", "")
        
        logger.info(
            "Supervisor.service_exit: turn=%d exit_intent=%s query='%s'",
            turn_count, service_exit_intent, (service_exit_query or "")[:50]
        )
        
        # Clear the exit intent to prevent loops
        base_update = {
            "service_exit_intent": None,
            "service_exit_query": None,
        }
        
        if service_exit_intent == "info":
            new_phase = ConversationPhase.INFO_QUERY
            return Command(
                update={
                    **base_update,
                    "intent": "info",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                    "info_skip_filter": True,  # Skip product filter for service questions
                },
                goto="info_agent",
            )
        
        elif service_exit_intent == "recommend":
            # Compute a consistent phase for recommendation entry.
            # If product is known, we're typically slot-filling; otherwise product selection.
            has_product = bool(state.get("product"))
            new_phase = ConversationPhase.from_intent(
                intent="recommend",
                has_product=has_product,
                rec_given=state.get("rec_given", False),
                purchase_offered=state.get("purchase_offered", False),
            )
            return Command(
                update={
                    **base_update,
                    "intent": "recommend",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                },
                goto="recommendation",
            )
        
        elif service_exit_intent == "compare":
            new_phase = ConversationPhase.COMPARISON
            return Command(
                update={
                    **base_update,
                    "intent": "compare",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                },
                goto="compare_agent",
            )
        
        elif service_exit_intent == "summary":
            # Summary is treated as an info-style phase in this state machine.
            new_phase = ConversationPhase.INFO_QUERY
            return Command(
                update={
                    **base_update,
                    "intent": "summary",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                },
                goto="summary_agent",
            )
        
        else:
            # Unknown exit intent - treat as info
            logger.warning("Supervisor.unknown_exit_intent: %s", service_exit_intent)
            new_phase = ConversationPhase.INFO_QUERY
            return Command(
                update={
                    **base_update,
                    "intent": "info",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                    "info_skip_filter": True,  # Skip product filter for service questions
                },
                goto="info_agent",
            )

    # ======================================================================
    # SERVICE FLOW GUARD: while in policy/claim service flow, do not
    # re-interpret user messages as new top-level intents.
    #
    # This prevents short replies like NRIC fragments or initials from being
    # classified as 'other' and routed to the generic chat agent, which the
    # user experiences as hallucination. As long as we are mid service flow
    # (not yet validated, collecting credentials, or executing an action),
    # always route back to the service_flow subgraph.
    # ======================================================================
    if current_phase == ConversationPhase.SERVICE_FLOW.value:
        customer_validated = state.get("customer_validated", False)
        service_action = state.get("service_action")
        service_pending_slot = state.get("service_pending_slot")

        in_active_service_flow = (
            not customer_validated
            or bool(service_pending_slot)
            or bool(service_action)
        )

        if in_active_service_flow:
            new_phase = ConversationPhase.SERVICE_FLOW
            logger.info(
                "Supervisor.service_flow_guard: turn=%d phase=%s validated=%s action=%s pending_slot=%s -> service_flow",
                turn_count,
                current_phase,
                customer_validated,
                service_action,
                service_pending_slot,
            )

            return Command(
                update={
                    "intent": "policy_service",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                },
                goto="service_flow",
            )
    
    # ======================================================================
    # FAST-PATH: Capabilities / services queries (no extra LLM call)
    #
    # Prevents UX loops where service/capabilities questions get routed into
    # product-discovery or product-filtered info flows.
    # ======================================================================
    last_user = _last_user_text(messages)
    if _is_capabilities_or_services_query(last_user):
        new_phase = ConversationPhase.INFO_QUERY
        logger.info(
            "Supervisor.fastpath.capabilities: turn=%d query='%s'",
            turn_count,
            (last_user or "")[:80],
        )
        return Command(
            update={
                "intent": "capabilities",
                "phase": new_phase.value,
                "phase_history": _update_phase_history(phase_history, new_phase),
                # Clear discovery/pending info state to avoid repeated prompts
                "info_pending_question": None,
                "product_discovery_step": None,
                "choice_info_step": None,
                "info_skip_filter": False,
            },
            goto="capabilities_agent",
        )

    # ======================================================================
    # LOOP BREAKER: repeated product discovery prompt + repeated user question
    # ======================================================================
    # Only apply this loop breaker in info/product-exploration context.
    # Recommendation now uses product_discovery_step too, and we should not reroute
    # those turns to capabilities.
    if (
        current_phase == ConversationPhase.INFO_QUERY.value
        and state.get("product_discovery_step")
        and _user_repeated_question(messages)
    ):
        last_ai = _last_ai_text(messages).lower()
        if "specific product" in last_ai and "customize" in last_ai:
            new_phase = ConversationPhase.INFO_QUERY
            logger.info(
                "Supervisor.loop_breaker.product_discovery: turn=%d step=%s",
                turn_count,
                state.get("product_discovery_step"),
            )
            return Command(
                update={
                    "intent": "capabilities",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                    "info_pending_question": None,
                    "product_discovery_step": None,
                    "choice_info_step": None,
                    "info_skip_filter": False,
                },
                goto="capabilities_agent",
            )

    # ======================================================================
    # EMOJI / SYMBOL-ONLY INPUT GUARD
    # If the latest user message contains NO alphanumeric characters (emojis,
    # symbols, whitespace only), skip the expensive LLM classification and
    # return a gentle rejection immediately.
    # ======================================================================
    _last_msg_raw = _last_user_text(messages)
    if _last_msg_raw and not any(c.isalnum() for c in _last_msg_raw):
        logger.info(
            "Supervisor.emoji_guard: emoji/symbol-only input detected, returning rejection"
        )
        rejection = (
            "⚠️ That doesn't look like a valid input. "
            "Please type your request in words so I can assist you.\n\n"
            "How may I help you today?"
        )
        return Command(
            update={
                "messages": [AIMessage(content=rejection)],
                "intent": "chat",
                "phase": (current_phase or ConversationPhase.GREETING.value),
                "phase_history": phase_history,
            },
            goto="styler",
        )

    # NOTE: We intentionally do NOT short-circuit routing based on pending_slot anymore.
    # The intent classifier is now responsible for deciding whether the user is:
    # - answering the pending slot (continue recommendation), OR
    # - explicitly switching to a different task (policy service / compare / purchase / etc.)
    #
    # This avoids UX dead-ends where users cannot interrupt slot-filling.
    
    # pending_slot is used by the intent classifier to interpret short replies.
    # It MUST be defined before the async classify_intent closure runs.
    pending_slot = state.get("pending_slot")

    # ==========================================================================
    # PARALLEL CLASSIFICATION: Run feedback and intent classification concurrently
    # This saves ~1-2s per turn by eliminating sequential LLM calls
    # ==========================================================================
    
    # Define the classification tasks
    async def classify_feedback():
        return await _classify_feedback_from_messages_async(messages)
    
    async def classify_intent():
        return await _classify_intent_from_messages_async(
            messages=messages,
            known_product=known_product,
            active_slot=pending_slot,
            summary=summary,
            current_phase=current_phase,
            current_slots=current_slots,
            reference_context=reference_context,
            rec_given=rec_given,  # Pass rec_given to prevent re-recommending after upsell
            rec_paused=state.get("rec_paused", False),
            product_discovery_step=state.get("product_discovery_step"),
            choice_info_step=state.get("choice_info_step"),
            last_intent=state.get("intent"),
        )
    
    # Run both classifiers in parallel
    parallel_start = time.perf_counter()
    feedback, intent_pred = await asyncio.gather(
        classify_feedback(),
        classify_intent(),
    )
    parallel_duration = time.perf_counter() - parallel_start
    logger.debug(
        "Supervisor.parallel_classification: duration=%.3fs",
        parallel_duration
    )
    
    # Priority 3: Negative feedback handling / reflection
    # (Moved to parallel execution block below)
    pending_slot = state.get("pending_slot")
    if feedback and feedback.category == "negative_feedback":
        revised = await _self_critique_and_rewrite_from_messages_async(
            messages,
            pending_slot=pending_slot,
            product=known_product,
        )
        if revised:
            logger.info(
                "Supervisor.negative_feedback: turn=%d triggering self-critique (slot_mode=%s)",
                turn_count, bool(pending_slot)
            )
            AUTONOMOUS_ROUTING_TOTAL.labels(
                source_node="supervisor",
                target_node="styler",
            ).inc()
            
            # If we're in slot collection mode, keep pending_slot so flow continues
            # Otherwise clear it as the context has changed
            slot_updates = {}
            if pending_slot:
                # Keep the slot context - user should answer the re-asked question
                slot_updates["pending_slot"] = pending_slot
                slot_updates["is_slot_reask"] = True
            else:
                slot_updates["pending_slot"] = None
                slot_updates["is_slot_reask"] = None
            
            return Command(
                update={
                    "messages": [AIMessage(content=revised)],
                    "feedback": "negative_feedback",
                    "sources": [],
                    "intent": "reflect_done",
                    **slot_updates,
                },
                goto="styler",
            )

    # Priority 4: Use intent classification result (already computed in parallel)
    
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
        "policy_service": "service_flow",
        "life_insurance": "life_insurance_agent",
        "other": "chat_agent",
    }
    
    # Normalize intent and determine target
    if raw_intent not in intent_to_node:
        normalized_intent = "chat"
        target_node = "chat_agent"
        logger.debug(
            "Supervisor.unknown_intent: raw=%s normalized to chat",
            raw_intent
        )
    else:
        normalized_intent = raw_intent
        target_node = intent_to_node[raw_intent]

    # -----------------------------------------------------------------------
    # SALES JOURNEY OVERRIDE
    # Route all product-related intents to the LLM-driven sales agent.
    # The sales agent handles the full journey in one prompt: discovery,
    # explanation, upsell, payment link, confirmation, and cross-sell.
    # Unchanged: greet, capabilities, policy_service, life_insurance.
    # -----------------------------------------------------------------------
    _SALES_INTENTS = {"info", "summary", "compare", "recommend", "purchase", "other"}
    if normalized_intent in _SALES_INTENTS or (
        normalized_intent == "chat" and (known_product or intent_pred.product)
    ):
        logger.info(
            "Supervisor.sales_journey: overriding %s -> sales_agent (product=%s)",
            normalized_intent,
            intent_pred.product or known_product,
        )
        target_node = "sales_agent"

    # Normalize product key to keep state consistent (e.g., "ChoiceProtect360" -> "choice").
    product_raw = intent_pred.product or known_product
    product = _normalize_product_key(product_raw) or product_raw
    
    # Compute new conversation phase
    new_phase = _compute_new_phase(
        intent=normalized_intent,
        product=product,
        rec_given=rec_given,
        purchase_offered=purchase_offered,
        live_agent_requested=False,
    )
    
    # Track phase transition
    if current_phase and current_phase != new_phase.value:
        PHASE_TRANSITION_TOTAL.labels(
                from_phase=current_phase,
                to_phase=new_phase.value,
            ).inc()
        logger.info(
            "Supervisor.phase_transition: %s -> %s (intent=%s)",
            current_phase, new_phase.value, normalized_intent
        )
    
    # Build state updates
    updates: Dict[str, Any] = {
        "intent": normalized_intent,
        "product": product,
        "phase": new_phase.value,
        "phase_history": _update_phase_history(phase_history, new_phase),
        # Store reference context for downstream use
        "reference_context": {
            "last_mentioned_product": reference_context.last_mentioned_product,
            "last_mentioned_tier": reference_context.last_mentioned_tier,
            "last_mentioned_destination": reference_context.last_mentioned_destination,
            "compared_items": reference_context.compared_items,
            "last_bot_question": reference_context.last_bot_question,
            "last_updated_turn": turn_count,
        },
    }

    # ----------------------------------------------------------------------
    # Recommendation flow interruption handling
    #
    # If the user was mid slot-filling (pending_slot set) but explicitly switched
    # to another task (policy service / compare / purchase / etc.), pause the
    # recommendation so we don't keep dragging them back.
    # ----------------------------------------------------------------------
    pending_slot = state.get("pending_slot")
    rec_paused = state.get("rec_paused", False)
    # Only consider recommendation "incomplete" if we actually started it
    # (slot collection began), not merely because a product is known.
    has_incomplete_rec = (bool(pending_slot) or bool(state.get("slots") or {})) and not rec_given

    if normalized_intent != "recommend" and (pending_slot or has_incomplete_rec):
        # Only pause if the user is *actually switching away* from recommendation work.
        # We treat info/summary/compare/purchase/policy_service/chat as switch-away intents.
        updates["rec_paused"] = True
        # Clear the active slot question so the user isn't forced back automatically.
        updates["pending_slot"] = None
        updates["is_slot_reask"] = None
    elif normalized_intent == "recommend":
        # Resuming/continuing recommendation flow
        if rec_paused:
            updates["rec_paused"] = False

    # Clear recommendation discovery flags when user switches flows.
    # These flags influence intent classification for short replies, so leaving them
    # set outside recommendation can cause misrouting.
    if normalized_intent != "recommend":
        updates["product_discovery_step"] = None
        updates["choice_info_step"] = None
    
    # Intelligent State Management:
    # Clear slots on product switch, explicit reset, OR greeting (fresh start)
    is_reset = getattr(intent_pred, "reset", False)
    
    # logical product switch check (case-insensitive)
    is_product_switch = False
    if product and known_product:
        is_product_switch = (product.lower() != known_product.lower())
    
    is_greeting = (normalized_intent == "greet")  # User says hi/hello = fresh start
    
    if is_product_switch or is_reset or is_greeting:
        reason = "greeting" if is_greeting else ("product_switch" if is_product_switch else "explicit_reset")
        logger.info(
            "Supervisor.clearing_state: reason=%s old_product=%s new_product=%s",
            reason, known_product, product
        )
        
        # Clear slot/flow state for all reset types
        updates["slots"] = {}
        updates["rec_ready"] = False
        updates["rec_given"] = False
        updates["pending_slot"] = None
        updates["info_pending_question"] = None
        updates["product_discovery_step"] = None
        updates["choice_info_step"] = None
        updates["slot_validation_errors"] = {}
        updates["side_info"] = None
        updates["pending_side_question"] = None
        updates["product_switch_pending"] = None
        updates["is_slot_reask"] = None
        updates["rec_paused"] = False
        
        # Clear service flow state
        updates["service_action"] = None
        updates["service_slots"] = {}
        updates["service_pending_slot"] = None
        updates["customer_validated"] = False
        updates["customer_nric"] = None
        updates["customer_data"] = None
        
        # Specific handling for greeting/reset vs product switch
        if is_greeting or is_reset:
            # CLEAR PII MAPPING ON RESET/GREETING
            try:
                session_id = config["configurable"].get("thread_id")
                if session_id:
                    get_pii_masker().clear_session(session_id)
            except Exception as e:
                logger.warning("Supervisor.reset: failed to clear PII session: %s", e)

            updates["product"] = None
            updates["pii_mapping"] = {}  # Clear persisted PII mapping
            # Clear long-term memory so "reset" actually feels like a fresh start.
            # (We cannot reliably clear `messages` due to add_messages reducer, so we must
            # at least clear summaries + memory metadata that affect routing/prompting.)
            updates["summary"] = ""
            updates["has_summary"] = False
            updates["memory_context"] = {}
            updates["reference_context"] = {}
            updates["phase"] = ConversationPhase.GREETING.value
            updates["phase_history"] = _update_phase_history(
                updates["phase_history"],
                ConversationPhase.GREETING
            )
        # For product switch, we KEEP the new product and the computed phase (already in updates)
    
    duration = time.perf_counter() - start_time
    
    logger.info(
        "Supervisor.routing: turn=%d intent=%s product=%s phase=%s -> %s duration=%.3fs",
        turn_count, normalized_intent, product, new_phase.value, target_node, duration
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
