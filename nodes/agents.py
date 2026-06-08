from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any, List, Literal

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..state import AgentState
from ..infrastructure import get_router_llm
from ..utils.memory import _get_last_user_message
from ..tools.info import _info_tool_async
from ..tools.summary import _summary_tool_async
from ..tools.compare import _compare_tool_async, _compare_tool_any_async
from ..tools.purchase import _purchase_tool
from ..tools.capabilities import _capabilities_tool

logger = logging.getLogger(__name__)

def _is_services_or_capabilities_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    triggers = [
        "what can you do",
        "what do you do",
        "what are the services",
        "what are your services",
        "what services do you offer",
        "services you offer",
        "how can you help",
        "supported products",
        "which products do you support",
        "what products do you have",
        "capabilities",
        "features",
    ]
    return any(k in t for k in triggers)


# =============================================================================
# SMALL CLASSIFIERS (cached structured outputs)
# =============================================================================

class ProductDiscoverySelection(BaseModel):
    selection: Literal["specific", "customizable", "unclear"] = Field(
        description="User wants a specific product, a customizable plan (Choice Protect360), or unclear."
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class YesNoClassification(BaseModel):
    intent: Literal["yes", "no", "unclear"] = Field(
        description="User's intent: 'yes' for affirmative, 'no' for negative, 'unclear' if neither"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


_product_discovery_classifier = None
_yes_no_classifier = None


def _get_product_discovery_classifier():
    global _product_discovery_classifier
    if _product_discovery_classifier is None:
        _product_discovery_classifier = get_router_llm().with_structured_output(ProductDiscoverySelection)
    return _product_discovery_classifier


def _get_yes_no_classifier():
    global _yes_no_classifier
    if _yes_no_classifier is None:
        _yes_no_classifier = get_router_llm().with_structured_output(YesNoClassification)
    return _yes_no_classifier


async def _classify_product_discovery_reply(user_message: str) -> str:
    """Classify whether the user wants a specific product or a customizable plan."""
    try:
        classifier = _get_product_discovery_classifier()
        sys_msg = (
            "You are classifying a user's reply in a product discovery flow.\n"
            "The bot asked: 'Would you like to find out about a specific product or something you can customize?'\n\n"
            "Return:\n"
            "- customizable: user wants a tailored/flexible/customizable plan\n"
            "- specific: user wants a particular product\n"
            "- unclear: anything else\n\n"
            "Examples:\n"
            "- 'customizable', 'tailored', 'flexible', 'can customize', 'something I can customise' -> customizable\n"
            "- 'specific product', 'a specific plan', 'travel insurance', 'maid insurance' -> specific\n"
        )
        result = await classifier.ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"User reply: {user_message}"),
            ]
        )
        return (result.selection or "unclear").strip().lower()
    except Exception as e:
        logger.warning("ProductDiscovery.classify failed: %s", e)
        return "unclear"


async def _classify_yes_no(user_message: str, question: str) -> str:
    """LLM-based yes/no/unclear classification for short follow-ups."""
    try:
        classifier = _get_yes_no_classifier()
        sys_msg = (
            "You are classifying a user's response to a yes/no question.\n"
            "Return intent='yes' if the user is affirming/proceeding, intent='no' if declining, otherwise 'unclear'.\n"
            "Handle slang like 'yup', 'ok', 'sure', 'nah', 'not now'."
        )
        result = await classifier.ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"Question asked: {question}\nUser reply: {user_message}"),
            ]
        )
        return (result.intent or "unclear").strip().lower()
    except Exception as e:
        logger.warning("YesNo.classify failed: %s", e)
        return "unclear"


def _get_continuation_for_incomplete_rec(state: AgentState) -> Tuple[Optional[str], Optional[str]]:
    """
    Check if we're in an incomplete recommendation flow and return continuation prompt.
    
    Returns:
        Tuple of (continuation_text, missing_slot_name) or (None, None) if not applicable
    """
    product = state.get("product")
    rec_given = state.get("rec_given", False)
    rec_paused = state.get("rec_paused", False)
    pending_slot = state.get("pending_slot")
    
    # If the user intentionally switched away from recommendation, do not auto-nudge them back.
    if rec_paused:
        return None, None

    # Only nudge if a recommendation slot is ACTUALLY pending.
    # This prevents accidentally starting recommendation just because a product is known.
    if not pending_slot:
        return None, None

    if not product or rec_given:
        return None, None
    
    # Import here to avoid circular imports
    from ..utils.products import PRODUCT_DEFINITIONS
    from ..utils.slots import _normalize_product_key
    
    prod_key = _normalize_product_key(product)
    if not prod_key or prod_key not in PRODUCT_DEFINITIONS:
        return None, None
    
    prod_def = PRODUCT_DEFINITIONS[prod_key]
    required_slots = prod_def.required_slots
    
    # Use the existing pending slot; do not compute a new one here.
    slot_config = prod_def.slot_config.get(pending_slot)
    if slot_config and slot_config.question:
        continuation = f"\n\nNow, back to your recommendation — {slot_config.question}"
    else:
        # Build a friendly prompt based on slot name
        slot_label = str(pending_slot).replace("_", " ")
        continuation = f"\n\nNow, back to your recommendation — could you please share your {slot_label}?"
    
    return continuation, str(pending_slot)


def _greet_agent_node(state: AgentState) -> AgentState:
    logger.info("Agentic.agents: executing greet_agent")
    reply = (
        "Hi. I'm your assistant for today. "
        "Here to guide you through BigTapp insurance products and services, "
        "answer your questions instantly, and make things easier for you. How can I help you today?"
    )
    return {"messages": [AIMessage(content=reply)], "sources": []}


async def _capabilities_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    logger.info("Agentic.agents: executing capabilities_agent for query: %s", user_text)
    reply = await _capabilities_tool(user_text)
    return {"messages": [AIMessage(content=reply)], "sources": []}


async def _chat_agent_node(state: AgentState) -> AgentState:
    """General conversation agent with guardrails.

    Handles greetings and farewells naturally, but politely redirects
    out-of-scope questions back to insurance topics.
    """

    user_text = _get_last_user_message(state.get("messages", []) or [])
    logger.info("Agentic.agents: executing chat_agent for query: %s", user_text)

    system_prompt = """You are BigTapp's friendly digital insurance assistant.

IMPORTANT GUARDRAILS - You must follow these strictly:

1. OUT-OF-SCOPE QUESTIONS: If the user asks about topics unrelated to insurance 
   (weather, sports, news, coding, recipes, general knowledge, etc.), politely 
   decline and redirect:
   - "I'm your insurance assistant, so I can't help with that. But I'd love to 
     help you with travel insurance, motor insurance, or any other coverage needs!"
   - "That's outside my expertise! I specialize in insurance - would you like 
     help finding the right coverage for you?"

2. GREETINGS & FAREWELLS: Respond naturally to "hi", "hello", "how are you", 
   "thanks", "bye" etc. Keep it brief and warm.

3. INSURANCE-ADJACENT TOPICS: If the user mentions something that COULD relate 
   to insurance (travel plans, new car, health concerns), acknowledge it warmly 
   and offer relevant insurance help.

4. NEVER engage with or answer:
   - Weather questions
   - General knowledge questions
   - Requests for information outside insurance
   - Controversial topics (politics, religion, etc.)
   - Personal advice unrelated to insurance

Keep replies concise (1-2 sentences max for redirects).
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_text}"),
        ]
    )

    # Reuse the router model for chat-style responses to keep configuration
    # simple and fully LLM-driven.
    chain = prompt | get_router_llm()
    ai_msg = await chain.ainvoke({"user_text": user_text or ""})

    content = getattr(ai_msg, "content", None) or str(ai_msg)
    return {"messages": [AIMessage(content=content)], "sources": []}


async def _info_agent_node(state: AgentState) -> AgentState:
    messages = state.get("messages", []) or []
    user_text = _get_last_user_message(messages)
    product = state.get("product")
    skip_filter = state.get("info_skip_filter", False)
    pending_info_q = state.get("info_pending_question")
    # NOTE: product_discovery_step / choice_info_step are reserved for the
    # recommendation flow UX (specific vs customizable). Info flows should not
    # enter that loop.
    discovery_step = state.get("product_discovery_step")
    choice_step = state.get("choice_info_step")
    
    # Extract last bot message for context (helps reformulate vague queries like "yes please")
    last_bot_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            last_bot_message = (getattr(msg, "content", "") or "").strip()
            break

    # If the user is asking about what the bot can do / services offered,
    # answer via the capabilities tool instead of product discovery.
    # This avoids the UX loop of repeatedly asking them to choose a product.
    if _is_services_or_capabilities_query(user_text):
        reply = await _capabilities_tool(user_text)
        return {
            "messages": [AIMessage(content=reply)],
            "sources": [],
            "info_pending_question": None,
            "product_discovery_step": None,
            "choice_info_step": None,
            "info_skip_filter": False,
        }

    # Info flow should NOT use the recommendation-style discovery prompts.
    # If these flags are set (from previous turns), we simply ignore them and clear
    # them on return so the user doesn't get stuck in a loop across flows.
    
    # If we previously asked the user to specify a product, treat this message as the
    # product selection and answer the ORIGINAL question.
    if pending_info_q:
        logger.info(
            "Agentic.agents: info_agent resolving pending_info_question with product=%s selection='%s'",
            product or user_text,
            (user_text or "")[:40],
        )
        answer, srcs = await _info_tool_async(
            product or user_text,
            pending_info_q,
            skip_product_filter=skip_filter,
            conversation_context=last_bot_message,
        )
        pending_info_update: Dict[str, Any] = {"info_pending_question": None}
    else:
        # If no product and not skipping filter, ask the product BUT remember the original question
        # so the next turn answers it (instead of treating the product word as a new query).
        if not product and not skip_filter:
            # Info flow: ask for product directly (no customizable discovery prompts).
            prompt = (
                "Which product would you like to ask about: Choice Protect360, Travel Protect360, Maid Protect360, "
                "Car Protect360, Personal Accident (Family Protect360), Home Protect360, "
                "Early Critical Illness Protect360, Fraud Protect360, or Hospital Cash Protect360?"
            )
            return {
                "messages": [AIMessage(content=prompt)],
                "sources": [],
                "info_pending_question": user_text,
                "info_skip_filter": False,
                "product_discovery_step": None,
                "choice_info_step": None,
            }

        logger.info("Agentic.agents: executing info_agent for product=%s query=%s skip_filter=%s", product, user_text, skip_filter)
        answer, srcs = await _info_tool_async(
            product,
            user_text,
            skip_product_filter=skip_filter,
            conversation_context=last_bot_message,
        )
        pending_info_update = {}
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: info_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            **pending_info_update,
            "info_skip_filter": False,  # Reset flag
            "product_discovery_step": None,
            "choice_info_step": None,
        }
    
    return {
        "messages": [AIMessage(content=answer)], 
        "sources": srcs,
        **pending_info_update,
        "info_skip_filter": False,  # Reset flag
        "product_discovery_step": None,
        "choice_info_step": None,
    }


async def _summary_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    product = state.get("product")
    logger.info("Agentic.agents: executing summary_agent for product=%s", product)
    answer, srcs = await _summary_tool_async(product, state.get("tiers") or [], user_text)
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: summary_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            "pending_slot": missing_slot,
        }
    
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


async def _compare_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    product = state.get("product")
    logger.info("Agentic.agents: executing compare_agent for product=%s", product)
    answer, srcs = await _compare_tool_any_async(product, state.get("tiers") or [], user_text)
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: compare_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            "pending_slot": missing_slot,
        }
    
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _purchase_agent_node(state: AgentState) -> AgentState:
    product = state.get("product")
    logger.info("Agentic.agents: executing purchase_agent for product=%s", product)
    reply = _purchase_tool(product)
    return {"messages": [AIMessage(content=reply)], "sources": []}
