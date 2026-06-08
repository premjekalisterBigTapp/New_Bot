"""
Enterprise-grade tool implementations with:
- Structured error handling via ToolMessage with status='error'
- State updates via Command objects
- InjectedToolCallId for proper tool message generation
- InjectedState for accessing agent state
- Comprehensive logging for debugging

All tools follow the LangGraph v1 patterns for state management and error handling.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional, Annotated, Union

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .info import _info_tool_async
from .compare import _compare_tool_async, _compare_tool_any_async
from .recommendation import _generate_recommendation_text_async
from .purchase import _purchase_tool
from ..utils.slots import _normalize_product_key, _required_slots_for_product
from ..state import AgentState

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL ERROR TYPES
# =============================================================================

class ToolExecutionError(Exception):
    """Base exception for tool execution errors."""
    
    def __init__(self, message: str, tool_name: str, recoverable: bool = True):
        super().__init__(message)
        self.tool_name = tool_name
        self.recoverable = recoverable


class ValidationError(ToolExecutionError):
    """Raised when tool input validation fails."""
    pass


class ExternalServiceError(ToolExecutionError):
    """Raised when an external service (Weaviate, LLM) fails."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_success_message(
    content: str,
    tool_call_id: str,
    tool_name: str,
) -> ToolMessage:
    """Create a successful ToolMessage with proper status."""
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="success",
    )


def _create_error_message(
    error: Exception,
    tool_call_id: str,
    tool_name: str,
    user_friendly_message: Optional[str] = None,
) -> ToolMessage:
    """Create an error ToolMessage with proper status and debugging info."""
    if user_friendly_message:
        content = user_friendly_message
    elif isinstance(error, ValidationError):
        content = f"Validation error: {str(error)}. Please check your input."
    elif isinstance(error, ExternalServiceError):
        content = f"Service temporarily unavailable: {str(error)}. Please try again."
    else:
        content = f"Error: {str(error)}. Please try a different approach."
    
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error",
        additional_kwargs={
            "error_type": type(error).__name__,
            "error_details": str(error),
        },
    )


def _merge_slots(current_slots: Dict[str, Any], new_slots: Dict[str, Any], product: str) -> Dict[str, Any]:
    """
    Intelligently merge new slots into existing slots.
    
    - Preserves existing values unless explicitly overwritten
    - Tags slots with the product they belong to
    - Filters out None values
    """
    # Start with current slots
    merged = {k: v for k, v in current_slots.items() if v is not None}
    
    # Add new slots (overwriting existing)
    for k, v in new_slots.items():
        if v is not None and not k.startswith("_"):
            merged[k] = v
    
    # Tag with product
    if product:
        merged["_product"] = product
    
    return merged


# =============================================================================
# TOOLS WITH COMMAND-BASED STATE UPDATES
# =============================================================================

@tool
def save_progress(
    product: str,
    slots: Dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """
    Save collected user information (slots) to the conversation memory.
    Call this tool IMMEDIATELY when you extract new info (like destination, age, etc.), 
    even if you still need more details.
    
    This tool directly updates the agent's state with the collected slots.
    
    Args:
        product: The product name (e.g., "travel", "maid").
        slots: A dictionary of collected information (e.g., {"destination": "Japan"}).
    """
    logger.info(
        "Tool.save_progress.start: product=%s slots=%s tool_call_id=%s",
        product, slots, tool_call_id
    )
    
    try:
        # Validate product
        prod = _normalize_product_key(product)
        if not prod:
            logger.warning(
                "Tool.save_progress.invalid_product: product=%s",
                product
            )
            raise ValidationError(
                f"Invalid product '{product}'. Valid products are: choice, travel, maid, car, personalaccident, home, early, fraud, hospital.",
                tool_name="save_progress",
            )
        
        # Get current slots from state
        current_slots = state.get("slots") or {}
        current_product = current_slots.get("_product")
        
        # Check for product switch
        if current_product and current_product != prod:
            logger.info(
                "Tool.save_progress.product_switch: %s -> %s, clearing old slots",
                current_product, prod
            )
            current_slots = {}  # Clear old slots on product switch
        
        # Merge slots
        updated_slots = _merge_slots(current_slots, slots, prod)
        
        # Calculate required vs collected for messaging
        req = _required_slots_for_product(prod)
        collected = [k for k in req if updated_slots.get(k)]
        missing = [k for k in req if not updated_slots.get(k)]
        
        # Build response message
        slots_saved = {k: v for k, v in slots.items() if v is not None and not k.startswith("_")}
        if missing:
            response_content = (
                f"Saved: {slots_saved}. "
                f"Still needed for {prod.title()}: {', '.join(missing)}. "
                "Please ask for the missing information one question at a time."
            )
        else:
            response_content = (
                f"Saved: {slots_saved}. "
                f"All required information for {prod.title()} has been collected! "
                "You can now call get_product_recommendation."
            )
        
        logger.info(
            "Tool.save_progress.completed: product=%s collected=%s missing=%s",
            prod, collected, missing
        )
        
        # Return Command with state update
        return Command(
            update={
                "slots": updated_slots,
                "product": prod,
                "rec_ready": len(missing) == 0,
                "last_tool_called": "save_progress",
                "last_tool_status": "success",
                "tool_call_count": (state.get("tool_call_count") or 0) + 1,
                "messages": [
                    _create_success_message(response_content, tool_call_id, "save_progress")
                ],
            }
        )
        
    except ValidationError as e:
        logger.error(
            "Tool.save_progress.validation_error: %s",
            str(e),
            exc_info=True
        )
        return Command(
            update={
                "last_tool_called": "save_progress",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(e, tool_call_id, "save_progress")
                ],
            }
        )
    except Exception as e:
        logger.error(
            "Tool.save_progress.unexpected_error: %s\n%s",
            str(e),
            traceback.format_exc()
        )
        return Command(
            update={
                "last_tool_called": "save_progress",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(
                        e, tool_call_id, "save_progress",
                        user_friendly_message="Failed to save progress. Please try again."
                    )
                ],
            }
        )


@tool
async def search_product_knowledge(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
    product: Optional[str] = None,
) -> Command:
    """Search the BigTapp knowledge base (RAG) for specific coverage details, benefits, exclusions, or claims info.

    LLM NOTE: When the user asks what is covered, not covered, excluded, or how a policy applies in a
    specific scenario (e.g., whether a destination or situation is covered), you MUST call this tool
    instead of answering from your own general knowledge.

    Args:
        query: The user's question or keywords (e.g., "Does travel insurance cover covid?").
        product: The product to filter by (e.g., "travel", "maid"). If unknown, leave null.
    """
    query_preview = (query or "").replace("\n", " ")[:160]
    logger.info(
        "Tool.search_product_knowledge.start: product=%s query='%s' tool_call_id=%s",
        product, query_preview, tool_call_id
    )
    
    try:
        # Use product from state if not provided
        effective_product = product or state.get("product")
        
        # Call the underlying RAG tool
        ans, sources = await _info_tool_async(effective_product, query)
        
        logger.info(
            "Tool.search_product_knowledge.completed: product=%s sources=%d answer_len=%d",
            effective_product, len(sources or []), len(ans or "")
        )
        
        # Build response with sources
        if sources:
            response_content = f"{ans}\n\n(Sources: {', '.join(sources)})"
        else:
            response_content = ans
        
        # Update sources in state
        current_sources = state.get("sources") or []
        updated_sources = list(set(current_sources + (sources or [])))
        
        return Command(
            update={
                "sources": updated_sources,
                "last_tool_called": "search_product_knowledge",
                "last_tool_status": "success",
                "tool_call_count": (state.get("tool_call_count") or 0) + 1,
                "messages": [
                    _create_success_message(response_content, tool_call_id, "search_product_knowledge")
                ],
            }
        )
        
    except Exception as e:
        logger.error(
            "Tool.search_product_knowledge.error: product=%s query='%s' error=%s\n%s",
            product, query_preview, str(e), traceback.format_exc()
        )
        
        # Provide context-aware error message
        if "weaviate" in str(e).lower() or "vector" in str(e).lower():
            user_message = "I couldn't search the knowledge base right now. Let me try to help with what I know."
        elif "timeout" in str(e).lower():
            user_message = "The search is taking longer than expected. Please try a simpler question."
        else:
            user_message = f"I had trouble searching for information about {product or 'that topic'}. Please try rephrasing your question."
        
        return Command(
            update={
                "last_tool_called": "search_product_knowledge",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(
                        e, tool_call_id, "search_product_knowledge",
                        user_friendly_message=user_message
                    )
                ],
            }
        )


@tool
async def compare_plans(
    product: str,
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """
    Compare different tiers or plans for a specific product.
    
    IMPORTANT: The output contains coverage amounts. You MUST include these dollar amounts in your response.
    
    Args:
        product: The product name (e.g., "travel", "maid").
        question: The user's comparison question (e.g., "What is the difference between Basic and Premier?").
    """
    question_preview = (question or "").replace("\n", " ")[:160]
    logger.info(
        "Tool.compare_plans.start: product=%s question='%s' tool_call_id=%s",
        product, question_preview, tool_call_id
    )
    
    try:
        # Validate product
        prod = _normalize_product_key(product)
        if not prod:
            logger.warning(
                "Tool.compare_plans.invalid_product: product=%s",
                product
            )
            raise ValidationError(
                f"Invalid product '{product}'. Please specify a valid product to compare plans.",
                tool_name="compare_plans",
            )
        
        # Call the underlying comparison tool (supports cross-product comparisons)
        ans, _ = await _compare_tool_any_async(prod, [], question)
        
        logger.info(
            "Tool.compare_plans.completed: product=%s answer_len=%d",
            prod, len(ans or "")
        )
        
        return Command(
            update={
                "product": prod,  # Ensure product is set
                "last_tool_called": "compare_plans",
                "last_tool_status": "success",
                "tool_call_count": (state.get("tool_call_count") or 0) + 1,
                "messages": [
                    _create_success_message(ans, tool_call_id, "compare_plans")
                ],
            }
        )
        
    except ValidationError as e:
        logger.error(
            "Tool.compare_plans.validation_error: %s",
            str(e)
        )
        return Command(
            update={
                "last_tool_called": "compare_plans",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(e, tool_call_id, "compare_plans")
                ],
            }
        )
    except Exception as e:
        logger.error(
            "Tool.compare_plans.error: product=%s error=%s\n%s",
            product, str(e), traceback.format_exc()
        )
        return Command(
            update={
                "last_tool_called": "compare_plans",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(
                        e, tool_call_id, "compare_plans",
                        user_friendly_message=f"I had trouble comparing {product or 'the'} plans. Please try again."
                    )
                ],
            }
        )


@tool
async def get_product_recommendation(
    product: str,
    slots: Dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """
    Generate a specific plan recommendation based on collected user information.
    ONLY use this tool when you have collected the REQUIRED slots for the product.
    
    IMPORTANT: The output contains coverage amounts (e.g., "$500,000", "$300,000"). 
    You MUST include these dollar amounts in your response to the user.
    
    Args:
        product: The product name (e.g., "travel", "maid").
        slots: A dictionary of collected information. 
               Keys must match the required slots (e.g., {"destination": "Japan", "coverage_scope": "Individual"}).
    """
    logger.info(
        "Tool.get_product_recommendation.start: product=%s slots=%s tool_call_id=%s",
        product, slots, tool_call_id
    )
    
    try:
        # Validate product
        prod = _normalize_product_key(product)
        if not prod:
            logger.warning(
                "Tool.get_product_recommendation.invalid_product: product=%s",
                product
            )
            raise ValidationError(
                f"Invalid product '{product}'. Cannot generate recommendation.",
                tool_name="get_product_recommendation",
            )
        
        # Merge with state slots (prefer provided slots over state)
        current_slots = state.get("slots") or {}
        merged_slots = _merge_slots(current_slots, slots, prod)
        
        # Check required slots
        req = _required_slots_for_product(prod)
        missing = [s for s in req if not merged_slots.get(s)]
        
        if missing:
            logger.warning(
                "Tool.get_product_recommendation.missing_slots: product=%s missing=%s",
                prod, missing
            )
            raise ValidationError(
                f"Cannot generate recommendation. Missing required information: {', '.join(missing)}. "
                "Please collect this information first using save_progress.",
                tool_name="get_product_recommendation",
            )
        
        # Generate recommendation
        tier, rec_text = await _generate_recommendation_text_async(prod, merged_slots)
        
        if not rec_text:
            logger.error(
                "Tool.get_product_recommendation.empty_response: product=%s tier=%s",
                prod, tier
            )
            raise ExternalServiceError(
                "Failed to generate recommendation text.",
                tool_name="get_product_recommendation",
            )
        
        logger.info(
            "Tool.get_product_recommendation.completed: product=%s tier=%s rec_len=%d",
            prod, tier, len(rec_text)
        )
        
        # Update state with recommendation given flag
        return Command(
            update={
                "slots": merged_slots,
                "product": prod,
                "tiers": [tier] if tier else [],
                "rec_ready": True,
                "rec_given": True,  # Mark that recommendation was provided
                "last_tool_called": "get_product_recommendation",
                "last_tool_status": "success",
                "tool_call_count": (state.get("tool_call_count") or 0) + 1,
                "messages": [
                    _create_success_message(rec_text, tool_call_id, "get_product_recommendation")
                ],
            }
        )
        
    except ValidationError as e:
        logger.error(
            "Tool.get_product_recommendation.validation_error: %s",
            str(e)
        )
        return Command(
            update={
                "last_tool_called": "get_product_recommendation",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(e, tool_call_id, "get_product_recommendation")
                ],
            }
        )
    except ExternalServiceError as e:
        logger.error(
            "Tool.get_product_recommendation.service_error: %s",
            str(e)
        )
        return Command(
            update={
                "last_tool_called": "get_product_recommendation",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(e, tool_call_id, "get_product_recommendation")
                ],
            }
        )
    except Exception as e:
        logger.error(
            "Tool.get_product_recommendation.error: product=%s error=%s\n%s",
            product, str(e), traceback.format_exc()
        )
        return Command(
            update={
                "last_tool_called": "get_product_recommendation",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(
                        e, tool_call_id, "get_product_recommendation",
                        user_friendly_message="I had trouble generating a recommendation. Could you confirm the details you've provided?"
                    )
                ],
            }
        )


@tool
def generate_purchase_link(
    product: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """
    Generate a direct purchase link for the user to buy the product.
    Use this when the user expresses intent to buy or asks for a quote/link.
    
    Args:
        product: The product name.
    """
    logger.info(
        "Tool.generate_purchase_link.start: product=%s tool_call_id=%s",
        product, tool_call_id
    )
    
    try:
        # Validate product
        prod = _normalize_product_key(product)
        if not prod:
            # Try to use product from state
            prod = _normalize_product_key(state.get("product"))
        
        if not prod:
            logger.warning(
                "Tool.generate_purchase_link.no_product: product=%s state_product=%s",
                product, state.get("product")
            )
            raise ValidationError(
                "Please specify which product you'd like to purchase: Choice Protect360, Travel, Maid, Car, Personal Accident, Home, Fraud, Early Critical Illness, or Hospital.",
                tool_name="generate_purchase_link",
            )
        
        # Generate purchase link
        link_response = _purchase_tool(prod)
        
        logger.info(
            "Tool.generate_purchase_link.completed: product=%s",
            prod
        )
        
        return Command(
            update={
                "product": prod,
                "purchase_offered": True,  # Mark that purchase link was offered
                "last_tool_called": "generate_purchase_link",
                "last_tool_status": "success",
                "tool_call_count": (state.get("tool_call_count") or 0) + 1,
                "messages": [
                    _create_success_message(link_response, tool_call_id, "generate_purchase_link")
                ],
            }
        )
        
    except ValidationError as e:
        logger.error(
            "Tool.generate_purchase_link.validation_error: %s",
            str(e)
        )
        return Command(
            update={
                "last_tool_called": "generate_purchase_link",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(e, tool_call_id, "generate_purchase_link")
                ],
            }
        )
    except Exception as e:
        logger.error(
            "Tool.generate_purchase_link.error: product=%s error=%s\n%s",
            product, str(e), traceback.format_exc()
        )
        return Command(
            update={
                "last_tool_called": "generate_purchase_link",
                "last_tool_status": "error",
                "tool_errors": (state.get("tool_errors") or []) + [str(e)],
                "messages": [
                    _create_error_message(
                        e, tool_call_id, "generate_purchase_link",
                        user_friendly_message="I had trouble generating the purchase link. Please try again."
                    )
                ],
            }
        )


# =============================================================================
# LIVE AGENT ESCALATION TOOL
# =============================================================================

@tool
def escalate_to_live_agent(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """
    Escalate the conversation to a live human agent.
    
    Use this tool when:
    - The user explicitly asks to speak to a human, agent, or customer service
    - The query is too complex for the bot to handle
    - The user is frustrated and needs human assistance
    - You've tried multiple approaches and can't resolve the issue
    
    Args:
        reason: Brief explanation of why escalation is needed
    """
    logger.info(
        "Tool.escalate_to_live_agent.start: reason='%s' tool_call_id=%s",
        reason, tool_call_id
    )
    
    turn_count = state.get("turn_count", 0)
    product = state.get("product")
    
    # Build handoff message
    handoff_message = (
        "I'll connect you with a live agent who can assist you further. "
        "Please hold on while I transfer you."
    )
    
    logger.info(
        "Tool.escalate_to_live_agent.completed: reason='%s' turn=%d product=%s",
        reason, turn_count, product
    )
    
    # Return Command that:
    # 1. Sets live_agent_requested flag (triggers handoff in API layer)
    # 2. Updates tool tracking
    # 3. Provides handoff message
    return Command(
        update={
            "live_agent_requested": True,
            "last_tool_called": "escalate_to_live_agent",
            "last_tool_status": "success",
            "last_routing_decision": "live_agent_handoff",
            "tool_call_count": (state.get("tool_call_count") or 0) + 1,
            "messages": [
                _create_success_message(
                    handoff_message,
                    tool_call_id,
                    "escalate_to_live_agent"
                )
            ],
        },
        # Note: goto is handled by graph routing, not tool
        # The live_agent_requested flag is checked by the graph
    )


# =============================================================================
# TOOLS LIST FOR EXPORT
# =============================================================================

TOOLS = [
    save_progress,
    search_product_knowledge,
    compare_plans,
    get_product_recommendation,
    generate_purchase_link,
    escalate_to_live_agent,
]

# Mapping for custom tool node if needed
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
