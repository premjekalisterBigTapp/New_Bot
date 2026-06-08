from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from ..state import AgentState, IntentPrediction, ConversationPhase, ReferenceContext
from ..infrastructure import get_router_llm
from ..infrastructure.metrics import (
    PRONOUN_RESOLUTION_TOTAL,
    REFERENCE_CONTEXT_UPDATES,
    INTENT_WITH_SUMMARY_TOTAL,
)
from ..utils.slots import _detect_product_llm_async
from ..utils.products import get_product_names_str, get_product_aliases_prompt
from ..utils.memory import _build_history_context_from_messages, _get_last_user_message

logger = logging.getLogger(__name__)


# =============================================================================
# CACHED STRUCTURED OUTPUT WRAPPERS
# Cache these to avoid rebuilding on each call (~50-100ms saved per call)
# =============================================================================
_intent_classifier_cached = None

def _get_intent_classifier():
    """Get cached intent classifier with structured output."""
    global _intent_classifier_cached
    if _intent_classifier_cached is None:
        _intent_classifier_cached = get_router_llm().with_structured_output(IntentPrediction)
    return _intent_classifier_cached


# =============================================================================
# PRONOUN RESOLUTION
# =============================================================================

def _extract_reference_context(
    messages: List[BaseMessage],
    current_product: Optional[str] = None,
    current_slots: Optional[Dict[str, Any]] = None,
) -> ReferenceContext:
    """
    Extract reference context from recent messages for pronoun resolution.
    
    This addresses the "No Pronoun Resolution" issue by tracking:
    - Last mentioned product, tier, destination
    - Recently compared items
    - Last question asked by bot
    
    Args:
        messages: Recent conversation messages
        current_product: Current product from state
        current_slots: Current slots from state
        
    Returns:
        ReferenceContext with extracted references
    """
    context = ReferenceContext()
    
    if not messages:
        return context
    
    # Track last mentioned product
    context.last_mentioned_product = current_product
    
    # Extract destination from slots if available
    if current_slots:
        dest = current_slots.get("destination") or current_slots.get("travel_destination")
        if dest:
            context.last_mentioned_destination = str(dest)
    
    # Scan recent messages for references
    recent_messages = messages[-10:]  # Look at last 10 messages
    
    for msg in reversed(recent_messages):
        content = str(getattr(msg, "content", "") or "").lower()
        
        # Extract last bot question
        if isinstance(msg, AIMessage) and context.last_bot_question is None:
            if "?" in content:
                # Extract the question part
                question_start = content.rfind("?")
                # Find sentence start (look for period, newline, or start)
                sentence_start = max(
                    content.rfind(".", 0, question_start),
                    content.rfind("\n", 0, question_start),
                    0
                )
                context.last_bot_question = content[sentence_start:question_start + 1].strip()
        
        # Extract tier mentions from AI messages (for "it", "that plan")
        if isinstance(msg, AIMessage) and context.last_mentioned_tier is None:
            tier_keywords = ["gold", "silver", "platinum", "basic", "bronze", "essential", "premium"]
            for tier in tier_keywords:
                if tier in content:
                    context.last_mentioned_tier = tier.capitalize()
                    break
        
        # Extract compared items from comparison responses
        if isinstance(msg, AIMessage) and not context.compared_items:
            if "compare" in content or "vs" in content or "difference" in content:
                # Look for tier names in the comparison
                tier_keywords = ["gold", "silver", "platinum", "basic", "bronze", "essential", "premium"]
                found_tiers = [t.capitalize() for t in tier_keywords if t in content]
                if len(found_tiers) >= 2:
                    context.compared_items = found_tiers[:3]  # Max 3
    
    logger.debug(
        "Intent.reference_context: product=%s tier=%s dest=%s compared=%s question=%s",
        context.last_mentioned_product,
        context.last_mentioned_tier,
        context.last_mentioned_destination,
        context.compared_items,
        context.last_bot_question[:50] if context.last_bot_question else None,
    )

    try:
        if context.last_mentioned_product:
            REFERENCE_CONTEXT_UPDATES.labels(context_field="last_mentioned_product").inc()
        if context.last_mentioned_tier:
            REFERENCE_CONTEXT_UPDATES.labels(context_field="last_mentioned_tier").inc()
        if context.last_mentioned_destination:
            REFERENCE_CONTEXT_UPDATES.labels(context_field="last_mentioned_destination").inc()
        if context.compared_items:
            REFERENCE_CONTEXT_UPDATES.labels(context_field="compared_items").inc()
        if context.last_bot_question:
            REFERENCE_CONTEXT_UPDATES.labels(context_field="last_bot_question").inc()
    except Exception:
        pass

    return context


def _build_pronoun_resolution_prompt(
    reference_context: ReferenceContext,
    last_user_message: str,
) -> str:
    """
    Build pronoun resolution guidance for the intent classifier.
    
    This helps the classifier understand what pronouns refer to.
    """
    # Check if user message contains pronouns that need resolution
    lower_message = last_user_message.lower()
    tokens = lower_message.split()
    pronoun_hits = set()
    if any(token in tokens for token in ["it", "that", "this"]) or "the one" in lower_message:
        pronoun_hits.add("it")
    if any(token in tokens for token in ["them", "those"]) or "the first" in lower_message or "the second" in lower_message:
        pronoun_hits.add("them")
    if "there" in tokens:
        pronoun_hits.add("there")

    if not pronoun_hits:
        return ""

    try:
        for pronoun_type in pronoun_hits:
            if pronoun_type == "it":
                resolved = bool(reference_context.last_mentioned_tier or reference_context.last_mentioned_product)
            elif pronoun_type == "them":
                resolved = bool(reference_context.compared_items)
            else:
                resolved = bool(reference_context.last_mentioned_destination)
            PRONOUN_RESOLUTION_TOTAL.labels(
                pronoun_type=pronoun_type,
                resolved="true" if resolved else "false",
            ).inc()
    except Exception:
        pass
    
    parts = ["PRONOUN RESOLUTION (user message contains references):"]
    
    ref_context_str = reference_context.to_prompt_context()
    if ref_context_str:
        parts.append(ref_context_str)
    
    parts.append(
        "RESOLUTION RULES:\n"
        "  - 'it', 'that', 'this' → likely refers to last_mentioned_tier or last_mentioned_product\n"
        "  - 'them', 'those' → likely refers to compared_items\n"
        "  - 'there' → likely refers to last_mentioned_destination\n"
        "  - 'the first one', 'the second' → refers to compared_items in order\n"
        "  - Short answers (yes, no, numbers, locations) → likely answers to last_bot_question"
    )
    
    return "\n".join(parts)


# Phrases that unambiguously mean the user is asking about life insurance.
# The live product catalog has no life product, so without this guard the LLM
# maps "life insurance" onto the nearest product (e.g. Early Protect360 Plus).
_LIFE_INSURANCE_KEYS = (
    "life insurance",
    "life insurence",   # common misspelling
    "life insaurance",  # common misspelling
    "life assurance",
    "life cover",
    "term life",
    "term insurance",
    "life protect360",
    "life protect 360",
    "life protect",
)


def _mentions_life_insurance(text: str) -> bool:
    """Phrasing-tolerant detection of a life-insurance question."""
    if not text:
        return False
    norm = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text.lower())
    norm = " ".join(norm.split())
    return any(key in norm for key in _LIFE_INSURANCE_KEYS)


async def _classify_intent_from_messages_async(
    messages: List[BaseMessage], 
    known_product: Optional[str] = None,
    active_slot: Optional[str] = None,
    summary: Optional[str] = None,
    current_phase: Optional[str] = None,
    current_slots: Optional[Dict[str, Any]] = None,
    reference_context: Optional[ReferenceContext] = None,
    rec_given: bool = False,
    rec_paused: bool = False,
    product_discovery_step: Optional[str] = None,
    choice_info_step: Optional[str] = None,
    last_intent: Optional[str] = None,
) -> IntentPrediction:
    """
    Async intent classifier with full context support.
    """
    start_time = time.perf_counter()

    if not messages:
        logger.debug("Intent.classify.async: no messages, returning info intent")
        return IntentPrediction(intent="info", product=known_product, reason="no_messages")

    # Build history context - use fewer messages since we have summary for long-term context
    history_window = 3 if summary else 5
    history_ctx = _build_history_context_from_messages(messages[-history_window:])
    last_user = _get_last_user_message(messages) or ""
    
    product_list = get_product_names_str()
    product_aliases = get_product_aliases_prompt()
    
    # Build reference context if not provided
    if reference_context is None:
        reference_context = _extract_reference_context(messages, known_product, current_slots)
    
    # Build context sections
    context_parts = []
    
    # 1. Summary context (addresses "Intent Classification Doesn't Consider Full History")
    if summary:
        context_parts.append(
            f"CONVERSATION SUMMARY (long-term context):\n{summary}\n"
            "Use this summary to understand the full conversation context, "
            "but prioritize recent messages for current intent."
        )
    
    # 2. Active slot context
    if active_slot:
        context_parts.append(
            f"[IMPORTANT CONTEXT]: The bot explicitly asked the user for the '{active_slot}' slot. "
            f"If the user's message '{last_user}' looks like an answer to this (e.g. a location, a number, a yes/no), "
            "you MUST classify this as 'recommend' to continue the form-filling flow. "
            "Only choose a different intent if they explicitly change the topic to a DIFFERENT task "
            "(for example: policy/claim service, plan comparison, buying, or resetting the chat)."
        )

    # 2.5. Product discovery context (recommendation UX: specific vs customizable)
    if product_discovery_step:
        context_parts.append(
            "PRODUCT DISCOVERY IN PROGRESS:\n"
            "The bot is in a multi-turn product discovery flow and is waiting for the user to choose:\n"
            "- a specific product, OR\n"
            "- a customizable plan (Choice Protect360).\n\n"
            "CRITICAL:\n"
            "- Treat the user's latest message as an answer to this product discovery question.\n"
            "- Classify as 'recommend' (not 'chat') unless they clearly ask for purchase or policy service.\n"
            "- If the user indicates customizable/tailored/flexible coverage, set product='choice'."
        )

    # 2.6. Choice Protect360 intro follow-up context
    if choice_info_step:
        context_parts.append(
            "CHOICE PROTECT360 EDUCATIONAL FLOW:\n"
            "The bot asked the user if they'd like to find out more about Choice Protect360.\n"
            "Treat short replies like 'yes', 'ok', 'sure' as continuing this info flow.\n"
            "Classify as 'recommend' and keep product='choice' unless user explicitly changes topic."
        )
    
    # 3. Current phase context
    if current_phase:
        phase_guidance = {
            ConversationPhase.GREETING.value: "User is in greeting phase. Look for product interest or general questions.",
            ConversationPhase.PRODUCT_SELECTION.value: "User is exploring products. Look for product mentions or comparison requests.",
            ConversationPhase.SLOT_FILLING.value: "User is providing information for a recommendation. Short answers likely relate to pending questions.",
            ConversationPhase.RECOMMENDATION.value: "User received a recommendation. Look for purchase intent, comparison, or new questions.",
            ConversationPhase.COMPARISON.value: "User is comparing plans. Look for selection, more comparisons, or purchase intent.",
            ConversationPhase.PURCHASE.value: "User is in purchase flow. Look for confirmation or additional questions.",
            ConversationPhase.INFO_QUERY.value: "User is asking information questions. Follow-up questions about the same topic should keep the same product and intent='info'. Only change product if user explicitly names a different product. EXCEPTION: if the conversation is about life insurance / Life Protect360, keep intent='life_insurance' (NOT 'info') for follow-ups.",
        }
        guidance = phase_guidance.get(current_phase, "")
        if guidance:
            context_parts.append(f"CURRENT PHASE: {current_phase}\n{guidance}")
    
    # 3.2. Life-insurance stickiness: keep follow-ups on the life_insurance intent.
    if (last_intent or "").lower() == "life_insurance":
        context_parts.append(
            "LIFE INSURANCE CONTEXT:\n"
            "The previous turn was about life insurance (Life Protect360). Unless the user "
            "clearly switches to a DIFFERENT named product or a greeting/reset, classify "
            "follow-ups (including premium/cost, riders, payout, eligibility, free look, "
            "'how much at 35', 'what about Life Plus') as intent='life_insurance' with product=None. "
            "Do NOT classify these as 'info' or 'purchase'."
        )

    # 3.5. Incomplete recommendation flow context (when user might be resuming)
    # This helps classify short answers correctly even when pending_slot is None
    if known_product and not rec_given and not active_slot and not rec_paused:
        context_parts.append(
            f"INCOMPLETE RECOMMENDATION FLOW:\n"
            f"The user has been working on a {known_product} recommendation but hasn't received one yet.\n"
            "If the user provides a short answer that looks like slot data (e.g., a number like '14' or '26', "
            "a duration like '14 months', a location, or 'yes'/'no'), classify as 'recommend' to continue the flow.\n"
            "This takes priority over 'chat' or 'other' for ambiguous short answers."
        )
    
    # 4. Post-recommendation context (prevents re-recommending after upsell offer)
    if rec_given:
        context_parts.append(
            "CRITICAL - POST-RECOMMENDATION STATE:\n"
            "A recommendation has ALREADY been given to this user. Do NOT classify as 'recommend' "
            "unless they explicitly ask for a NEW or DIFFERENT recommendation.\n"
            "- If user says 'yes', 'sure', 'tell me more', 'details' → classify as 'info' (they want more details about the mentioned tier/plan)\n"
            "- If user wants to buy, get quote, or proceed → classify as 'purchase'\n"
            "- If user asks to compare plans → classify as 'compare'\n"
            "- Only use 'recommend' if they say 'different plan', 'new recommendation', or switch products"
        )
    
    # 5. Pronoun resolution context (addresses "No Pronoun Resolution")
    pronoun_prompt = _build_pronoun_resolution_prompt(reference_context, last_user)
    if pronoun_prompt:
        context_parts.append(pronoun_prompt)
    
    # Build system message
    sys_msg = (
        "You are an intent classifier for BTBot (BigTapp insurance assistant). "
        "Your job is to decide what the user is trying to do and which "
        "insurance product (if any) they are talking about.\n\n"
        "You MUST choose one of these intents exactly: "
        "'info', 'summary', 'compare', 'recommend', 'purchase', "
        "'capabilities', 'greet', 'chat', 'policy_service', 'life_insurance', 'other'.\n\n"
        "Guidelines (be strict about 'summary'):\n"
        "- info: asking about coverage, benefits, exclusions, scenarios, product catalogs, or 'tell me about X'. "
        "Includes 'what tiers are available?'. "
        "Also includes product exploration questions like 'what products do you have?', 'what insurance do you offer?'. "
        "ALSO includes company/contact questions like 'hotline', 'phone number', 'contact number', "
        "'email address', 'operating hours', 'office address', 'how to reach you', 'customer service'. "
        "Use 'info' for ANY informational question that is NOT about managing existing policies.\n"
        "- summary: ONLY when the user explicitly asks for a brief/short overview or to 'summarize' a plan that was already discussed. Do NOT use for coverage, tiers, or benefit questions.\n"
        "- compare: asking for differences between plans/tiers OR comparing two products/offerings.\n"
        "- recommend: wants a personalised plan suggestion, 'best plan', OR says 'I want X insurance'. "
        "INCLUDES answering slot-filling questions like 'Where are you traveling?'. "
        "Use 'recommend' when user expresses intent to GET a specific product (e.g., 'I want Fraud Protect360').\n"
        "- purchase: when user wants to buy, get a quote, asks for price/cost/premium, or wants a purchase link. "
        "Also use this if they ask 'how much is it', 'how much does it cost', 'what is the price', 'can I buy', "
        "'get a quote', 'give me a quote', 'I want to purchase', 'pricing for X', 'cost for X days', or any "
        "request for specific pricing or premium amounts. If the user asks about cost with specific details "
        "(e.g., 'how much for France 10 days'), classify as 'purchase'.\n"
        "  EXCEPTION: If 'can I buy' is followed by a CONDITION or SCENARIO (e.g., 'can I buy after departure', "
        "'can I buy if I'm already overseas', 'can I buy without NRIC', 'can I still buy after traveling'), "
        "this is an ELIGIBILITY question — classify as 'info', NOT 'purchase'.\n"
        "- policy_service: when the user is ACTIVELY requesting to manage THEIR OWN existing policies or claims. "
        "Examples: 'what is my policy status', 'check my claim', 'update my email', 'change my phone number', "
        "'where is my claim', 'list my policies', 'update my address', 'change payment info', "
        "'I want to cancel my policy', 'I want to make a claim'. "
        "Use ONLY when the user is requesting action on their own account.\n"
        "  CRITICAL EXCEPTION: Questions that ask HYPOTHETICALLY about processes, rules, or what happens in certain "
        "situations are 'info', NOT 'policy_service'. Examples of 'info' (NOT policy_service):\n"
        "  - 'What should a policyholder do if they want to change details?' (hypothetical/educational)\n"
        "  - 'What happens if the application form info is inaccurate?' (policy rule question)\n"
        "  - 'How do I make a claim for home contents?' (process question)\n"
        "  - 'Can I cancel the policy?' (eligibility/process question)\n"
        "  - 'Is premium refunded on cancellation?' (policy rule question)\n"
        "  - 'How are claims calculated?' (informational)\n"
        "  Only use 'policy_service' when the user says 'I want to...', 'please update my...', 'check MY claim/policy'.\n"
        "- capabilities: asks what the bot can do / what services it offers / how it can help.\n"
        "Examples: 'what can you do', 'what are your services', 'what services do you offer', 'how can you help me'.\n"
        "- greet: very short greetings like 'hi', 'hello', 'hey'.\n"
        "- chat: small-talk or open conversation without a clear insurance task yet.\n"
        "- life_insurance: ANY question about life insurance / term life / life cover / 'Life Protect360'. "
        "This covers plans (Life, Life Plus, Life Secure), death-benefit payout options, premiums/cost, "
        "the Step-Up option, riders (Critical Illness Plus, Permanent Disability), wellness benefits, "
        "eligibility/entry age, free look period, grace/revival, maturity, and tax. "
        "Examples: 'tell me about life insurance', 'do you have life cover', 'how much is term life at 35', "
        "'what riders are available', 'what is the free look period for the life plan'. "
        "IMPORTANT: For life insurance, cost/premium questions stay 'life_insurance' (NOT 'purchase'), "
        "since there is no life purchase link. Once the conversation is about life insurance, keep "
        "classifying related follow-ups as 'life_insurance'.\n"
        "- other: ONLY use this if none of the above intents fit. Avoid using 'other' for questions that could be answered with information.\n\n"
        f"Products available: {product_list}\n"
        "PRODUCT ALIAS MAPPING (Use these strict mappings):\n"
        f"{product_aliases}\n\n"
        "IMPORTANT: If the user says phrases like 'start over', 'fresh recommendation', 'reset', or 'new quote', "
        "set the 'reset' field to True.\n\n"
        "PRODUCT SWITCHING vs STICKINESS RULES:\n"
        "STEP 1 — CHECK FOR PRODUCT MENTION FIRST: Before applying stickiness, check if the user's "
        "latest message mentions ANY product name or alias from the PRODUCT ALIAS MAPPING above. "
        "This includes full names (e.g., 'Fraud Protect360'), short names (e.g., 'fraud protection', "
        "'home insurance', 'travel plan', 'car insurance', 'maid insurance'), or any alias listed above. "
        "If the user mentions a DIFFERENT product than the current one, you MUST switch to the new product. "
        "Do NOT apply stickiness when the user names a different product.\n"
        "STEP 2 — APPLY STICKINESS ONLY FOR VAGUE FOLLOW-UPS: If (and ONLY if) the user's message "
        "contains NO product name or alias at all, and is a follow-up question about the same topic "
        "(e.g., 'what about kidney failure?', 'is that covered?', 'what are the exclusions?', 'tell me more', "
        "'how much is it?'), THEN keep the same product from history. "
        "Do NOT set product to None for these follow-up questions.\n\n"
    )
    
    # Add context sections
    if context_parts:
        sys_msg += "ADDITIONAL CONTEXT:\n" + "\n\n".join(context_parts)

    user_ctx = (
        f"Known product from history: {known_product or 'None'}\n\n"
        f"Recent conversation (most recent last):\n{history_ctx}\n\n"
        f"Latest user message:\n{last_user}"
    )

    try:
        structured = _get_intent_classifier()  # Use cached classifier
        result = await structured.ainvoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_ctx),
        ])

        duration = time.perf_counter() - start_time

        intent_prediction = (
            result
            if isinstance(result, IntentPrediction)
            else IntentPrediction.model_validate(result)
        )

        # Deterministic life-insurance guard. The catalog has no life product,
        # so the LLM otherwise mis-routes life questions to 'info' on the nearest
        # product (e.g. Early Protect360). If the latest message clearly asks
        # about life insurance, force the life_insurance intent and drop the
        # mis-detected product. We do NOT override 'greet' or 'policy_service'
        # (e.g. "check my life insurance policy status" stays a service request).
        if _mentions_life_insurance(last_user) and intent_prediction.intent in (
            "info", "summary", "compare", "recommend", "purchase", "chat", "other",
        ):
            if intent_prediction.intent != "life_insurance" or intent_prediction.product:
                logger.info(
                    "Intent.classify.async: life_insurance keyword override (was intent=%s product=%s)",
                    intent_prediction.intent, intent_prediction.product,
                )
            intent_prediction.intent = "life_insurance"
            intent_prediction.product = None

        # Explicitly prioritize strong product signals from detection
        if intent_prediction.product and intent_prediction.product.lower() != (known_product or "").lower():
            logger.info(
                "Intent.classify.async: product switch detected: %s -> %s",
                known_product, intent_prediction.product,
            )

        logger.info(
            "Intent.classify.async: intent=%s product=%s reason=%s phase=%s duration=%.3fs",
            intent_prediction.intent,
            intent_prediction.product,
            intent_prediction.reason,
            current_phase,
            duration,
        )

        try:
            INTENT_WITH_SUMMARY_TOTAL.labels(
                intent=intent_prediction.intent,
                had_summary="true" if summary else "false",
            ).inc()
        except Exception:
            pass

        return intent_prediction
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(
            "Intent.classify.async.FAILED: error=%s duration=%.3fs",
            str(e), duration,
            exc_info=True
        )
        return IntentPrediction(
            intent="info", product=known_product, reason="classification_failed"
        )

async def detect_product_node(state: AgentState) -> AgentState:
    """Parallel Product Detection Node.

    Runs concurrently with Master Agent.
    Detects if the user has switched product context and updates the state.
    This ensures the NEXT turn has the correct product context.
    """
    messages = list(state.get("messages", []) or [])
    if not messages:
        return {}

    # Get the latest user message
    last_user_full = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_full = str(getattr(m, "content", "") or "")
            break
            
    if not last_user_full:
        return {}

    current_product = state.get("product")
    
    try:
        # Use pure LLM detection with context awareness
        # This runs in parallel so latency is hidden
        detected_product = await _detect_product_llm_async(
            last_user_full,
            current_product=current_product,
        )
        
        if detected_product and detected_product != current_product:
            logger.info(
                "Agentic.detect_node: product switch detected %s -> %s",
                current_product,
                detected_product,
            )
            # Only update product - slots will be cleared by master_agent when it sees the product change
            return {"product": detected_product}
        
        # Add debug log for detection result even if no change
        logger.debug(
            "Agentic.detect_node: detection run. current=%s detected=%s",
            current_product, detected_product
        )
            
    except Exception as e:
        logger.warning("Agentic.detect_node: detection failed: %s", e)

    return {}
