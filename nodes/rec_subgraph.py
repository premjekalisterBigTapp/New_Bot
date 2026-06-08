from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Literal

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..state import AgentState
from ..config import _load_slot_rules
from ..infrastructure import get_router_llm
from ..utils.slots import (
    _required_slots_for_product,
    _slot_descriptions,
    _slot_config,
    _get_slot_value,
    _detect_product_llm_async,
    _normalize_product_key,
)
from ..utils.products import get_product_names_str
from ..utils.memory import _get_last_user_message, _build_history_context_from_messages
from ..tools.recommendation import _generate_recommendation_text_async

logger = logging.getLogger(__name__)

class SlotUpdate(BaseModel):
    slot_name: str
    value: str
    confidence: float

class SlotExtraction(BaseModel):
    updates: List[SlotUpdate]
    side_question: Optional[str] = Field(
        default=None,
        description="If user asks a question instead of answering."
    )
    product_switch_request: Optional[str] = Field(
        default=None,
        description=(
            "If user explicitly wants to ABANDON current flow and switch to a different "
            "insurance product, return that product name (e.g., 'car', 'maid', 'home'). "
            "Only set if they clearly want to switch, not if they just mention a product word."
        )
    )


class YesNoClassification(BaseModel):
    """LLM-based classification for yes/no responses in conversational flows."""
    intent: Literal["yes", "no", "unclear"] = Field(
        description="User's intent: 'yes' for affirmative, 'no' for negative, 'unclear' if neither"
    )
    confidence: float = Field(
        description="Confidence score 0.0-1.0",
        ge=0.0,
        le=1.0
    )


# Cache the structured output models to avoid rebuilding on each call (~50-100ms saved per call)
_yes_no_classifier = None
_slot_extractor_cached = None

def _get_yes_no_classifier():
    """Get cached yes/no classifier model."""
    global _yes_no_classifier
    if _yes_no_classifier is None:
        _yes_no_classifier = get_router_llm().with_structured_output(YesNoClassification)
    return _yes_no_classifier

def _get_slot_extractor():
    """Get cached slot extraction model."""
    global _slot_extractor_cached
    if _slot_extractor_cached is None:
        _slot_extractor_cached = get_router_llm().with_structured_output(SlotExtraction)
    return _slot_extractor_cached


async def _classify_yes_no(user_message: str, context_question: str) -> Literal["yes", "no", "unclear"]:
    """
    Use LLM to classify if user's response is affirmative, negative, or unclear.
    
    This is more robust than hardcoded string matching as it handles:
    - Variations like "yup", "sure thing", "go ahead", "sounds good"
    - Negative variations like "not now", "maybe later", "skip"
    - Context-aware classification
    
    Args:
        user_message: The user's response
        context_question: The question that was asked (for context)
        
    Returns:
        "yes", "no", or "unclear"
    """
    try:
        classifier = _get_yes_no_classifier()
        result = await classifier.ainvoke(
            [
                SystemMessage(content=(
                    "You are classifying a user's response to a yes/no question.\n"
                    "Determine if the user is saying YES (affirmative, agreeing, wanting to proceed) "
                    "or NO (negative, declining, wanting to skip).\n"
                    "Common affirmatives: yes, yeah, yep, yup, sure, ok, okay, alright, go ahead, sounds good, definitely, absolutely, please, let's do it\n"
                    "Common negatives: no, nope, nah, skip, pass, not now, maybe later, no thanks, not interested\n"
                    "If the response is unrelated or ambiguous, classify as 'unclear'."
                )),
                HumanMessage(content=f"Question asked: {context_question}\nUser's response: {user_message}")
            ],
        )
        logger.debug("YesNo.classify: message='%s' -> intent=%s confidence=%.2f", 
                    user_message[:50], result.intent, result.confidence)
        return result.intent
    except Exception as e:
        logger.warning("YesNo.classify failed: %s, defaulting to unclear", e)
        return "unclear"

async def _rec_ensure_product(state: AgentState) -> AgentState:
    """Ensure product is known before proceeding."""
    msg = _get_last_user_message(state["messages"])
    prod = _normalize_product_key(state.get("product")) or state.get("product")
    rec_discovery_step = state.get("product_discovery_step")
    
    # ------------------------------------------------------------------
    # Recommendation-only product discovery:
    # Ask "specific product vs customizable" ONLY for recommendation intent.
    # ------------------------------------------------------------------
    if not prod and not rec_discovery_step:
        logger.info("RecSubgraph.ensure_product: product unknown -> asking rec discovery (specific vs customizable)")
        return {
            "messages": [AIMessage(content="Nice! Would you like to find out about a specific product or something you can customize?")],
            "rec_ready": False,
            "pending_slot": None,
            "product_discovery_step": "rec_mode",
            "sources": [],
        }

    if rec_discovery_step == "rec_mode":
        # Interpret the user's reply (even if supervisor already set a product):
        # - If they indicate customizable/tailored -> route to Choice Protect360.
        # - If they mention a specific product -> proceed with that product.
        if prod:
            prod_key = _normalize_product_key(prod) or _normalize_product_key(msg)
        else:
            detected = await _detect_product_llm_async(msg)
            prod_key = _normalize_product_key(detected) or _normalize_product_key(msg)

        # If the LLM detects ChoiceProtect360 (customizable language), go to choice.
        if prod_key == "choice":
            return {
                "product": "choice",
                "product_discovery_step": None,
                # Start the Choice educational sequence as part of recommendation UX.
                "choice_info_step": "more_info",
                "messages": [AIMessage(content=(
                    "Great! Our *Choice Protect360* is a great fit for you.\n\n"
                    "Why this works for you:\n"
                    "• All-in-one protection under a single plan\n"
                    "• Flexible coverage – pay only for what you need\n"
                    "• Covers accidental death, disability & income support\n"
                    "• Option to cover family members\n\n"
                    "Would you like to find out more?"
                ))],
                "rec_ready": False,
                "pending_slot": None,
                "sources": [],
            }

        # If we detected a specific product, proceed.
        if prod_key:
            return {
                "product": prod_key,
                "product_discovery_step": None,
                "choice_info_step": None,
            }

        # If unclear, ask again.
        return {
            "messages": [AIMessage(content="Just to check — would you like a *specific product*, or something you can *customize*?")],
            "rec_ready": False,
            "pending_slot": None,
            "product_discovery_step": "rec_mode",
            "sources": [],
        }

    # If we’re mid Choice educational follow-up, accept yes/no before generating a rec.
    if state.get("choice_info_step") == "more_info" and (prod == "choice" or (prod or "").lower() == "choice"):
        yn = await _classify_yes_no(msg or "", "Would you like to find out more about Choice Protect360?")
        if yn == "yes":
            # Match the intended scripted UX for the Choice Protect360 follow-up.
            details = (
                "The base plan provides comprehensive Personal Accident (PA) coverage, which can be enhanced with "
                "dynamic add-ons such as Home Contents, Hospital Income, or even an upgraded Personal Accident (PA) coverage.\n\n"
                "Customize your coverage to match your individual needs and take advantage of stacked discounts—"
                "5% off if you purchase one add-on on top of the base PA plan, and 10% off when you opt for two or more."
            )
            return {
                "product": "choice",
                "choice_info_step": "details_sent",
                "product_discovery_step": None,
                "messages": [AIMessage(content=details)],
                "rec_ready": False,
                "pending_slot": None,
                "sources": [],
            }
        if yn == "no":
            return {
                "product": "choice",
                "choice_info_step": None,
                "product_discovery_step": None,
                "messages": [AIMessage(content="No problem — if you tell me what you want to add on (home contents, hospital income, annual travel, or upgraded PA), I’ll guide you.")],
                "rec_ready": False,
                "pending_slot": None,
                "sources": [],
            }
        return {
            "product": "choice",
            "choice_info_step": "more_info",
            "product_discovery_step": None,
            "messages": [AIMessage(content="Just reply *Yes* or *No* — would you like to find out more about *Choice Protect360*?")],
            "rec_ready": False,
            "pending_slot": None,
            "sources": [],
        }

    if not prod:
        detected = await _detect_product_llm_async(msg)
        prod = _normalize_product_key(detected)
    
    logger.info("RecSubgraph.ensure_product: detected product=%s", prod)

    if not prod:
        # If product is still unknown here, ask a concrete list (rec-only).
        logger.info("RecSubgraph.ensure_product: product unknown after detection, asking product list.")
        return {
            "messages": [AIMessage(content=(
                "Which product would you like a recommendation for?\n\n"
                "• Choice Protect360 (customizable)\n"
                "• Travel Protect360\n"
                "• Maid Protect360\n"
                "• Car Protect360\n"
                "• Personal Accident (Family Protect360)\n"
                "• Home Protect360\n"
                "• Early Critical Illness Protect360\n"
                "• Fraud Protect360\n"
                "• Hospital Cash Protect360"
            ))],
            "rec_ready": False,
            "pending_slot": None,
            "product_discovery_step": None,
            "choice_info_step": None,
            "sources": []
        }
    
    # IMPORTANT: Do NOT clear pending_slot here!
    # pending_slot tracks what question we asked last turn, and extract_slots needs
    # to read it to properly handle the user's response (especially for Fraud flow).
    # extract_slots and ask_next_slot will manage pending_slot appropriately.
    return {"product": prod, "product_discovery_step": None, "choice_info_step": None}

from ..tools.info import _info_tool_async

async def _rec_extract_slots(state: AgentState) -> AgentState:
    """Extract slots from the latest message."""
    prod = state.get("product")
    if not prod:
        return {} # Should be handled by ensure_product
        
    messages = list(state.get("messages", []) or [])
    msg = _get_last_user_message(messages)
    msg_lower = (msg or "").lower().strip()
    current_slots = state.get("slots") or {}
    required = _required_slots_for_product(prod)
    pending_slot = state.get("pending_slot")
    prod_lower = (prod or "").lower()
    
    # ==========================================================================
    # PRODUCT SWITCH CONFIRMATION HANDLING
    # ==========================================================================
    product_switch_pending = state.get("product_switch_pending")
    if product_switch_pending:
        # User was asked if they want to switch products - check their response
        switch_intent = await _classify_yes_no(
            msg,
            f"Does the user want to switch to {product_switch_pending} insurance?"
        )
        
        if switch_intent == "yes":
            # User confirmed switch - clear current state and switch product
            logger.info(
                "RecSubgraph.extract_slots: product_switch confirmed from %s to %s",
                prod, product_switch_pending
            )
            
            # Build friendly product names
            product_display_names = {
                "choice": "Choice Protect360",
                "travel": "Travel Protect360",
                "maid": "Maid Protect360",
                "car": "Car Protect360",
                "personalaccident": "Personal Accident Protect360",
                "home": "Home Protect360",
                "early": "Early Critical Illness Protect360",
                "fraud": "Fraud Protect360",
                "hospital": "Hospital Cash Protect360",
            }
            new_display = product_display_names.get(product_switch_pending, product_switch_pending)
            
            return {
                "product": product_switch_pending,
                "slots": {},  # Clear slots for new product
                "pending_slot": None,
                "rec_ready": False,
                "rec_given": False,
                "product_switch_pending": None,  # Clear the pending switch
                "slot_validation_errors": {},  # Clear validation errors for new product
                "is_slot_reask": None,  # Clear reask flag
                "side_info": f"Great! Let's start fresh with *{new_display}*. 🎉",
            }
        elif switch_intent == "no":
            # User wants to continue with current product
            logger.info(
                "RecSubgraph.extract_slots: product_switch declined, continuing with %s",
                prod
            )
            
            product_display_names = {
                "choice": "Choice Protect360",
                "travel": "Travel Protect360",
                "maid": "Maid Protect360",
                "car": "Car Protect360",
                "personalaccident": "Personal Accident Protect360",
                "home": "Home Protect360",
                "early": "Early Critical Illness Protect360",
                "fraud": "Fraud Protect360",
                "hospital": "Hospital Cash Protect360",
            }
            current_display = product_display_names.get(prod_lower, prod)
            
            return {
                "product_switch_pending": None,  # Clear the pending switch
                "side_info": f"No problem! Let's continue with your *{current_display}* recommendation. 👍",
            }
        else:
            # Unclear response - ask again
            product_display_names = {
                "choice": "Choice Protect360",
                "travel": "Travel Protect360",
                "maid": "Maid Protect360",
                "car": "Car Protect360",
                "personalaccident": "Personal Accident Protect360",
                "home": "Home Protect360",
                "early": "Early Critical Illness Protect360",
                "fraud": "Fraud Protect360",
                "hospital": "Hospital Cash Protect360",
            }
            current_display = product_display_names.get(prod_lower, prod)
            new_display = product_display_names.get(product_switch_pending, product_switch_pending)
            
            return {
                "side_info": (
                    f"I didn't quite catch that. Would you like to:\n"
                    f"• *Continue* with {current_display}, or\n"
                    f"• *Start fresh* with {new_display}?"
                ),
            }
    
    # ==========================================================================
    # FRAUD PROTECT360 EDUCATIONAL FLOW - LLM-based yes/no classification
    # ==========================================================================
    if prod_lower == "fraud" and pending_slot in ["fraud_intro_shown", "fraud_example_shown", "fraud_ready_for_rec"]:
        # Define context questions for each pending slot
        context_questions = {
            "fraud_intro_shown": "Would you like to learn more about our Fraud Protect360 product?",
            "fraud_example_shown": "Want to see how it protects you in real life situations?",
            "fraud_ready_for_rec": "Would you like me to recommend a personalized coverage for you?"
        }
        
        # Use LLM to classify yes/no intent (robust to variations like "yup", "sure thing", etc.)
        user_intent = await _classify_yes_no(msg, context_questions[pending_slot])
        logger.info("RecSubgraph.extract_slots: Fraud flow - pending=%s intent=%s", pending_slot, user_intent)
        
        # User responding to intro question
        if pending_slot == "fraud_intro_shown":
            if user_intent == "yes":
                new_slots = dict(current_slots)
                new_slots["fraud_intro_shown"] = "yes"
                logger.info("RecSubgraph.extract_slots: Fraud intro acknowledged -> showing content")
                return {"slots": new_slots, "side_info": None}
            elif user_intent == "no":
                # User doesn't want to learn about Fraud - respect their choice and exit gracefully
                new_slots = dict(current_slots)
                new_slots["fraud_intro_shown"] = "no"
                new_slots["fraud_example_shown"] = "no"
                new_slots["_fraud_declined"] = "yes"  # Flag to exit the flow
                logger.info("RecSubgraph.extract_slots: Fraud intro declined -> exiting flow gracefully")
                return {
                    "slots": new_slots, 
                    "side_info": None,
                    "messages": [AIMessage(content="No problem at all! If you'd like to learn about Fraud Protect360 later or explore any of our other insurance products, just let me know. I'm here to help! 😊")],
                    "rec_ready": True  # Mark as complete to exit the rec flow
                }
        
        # User responding to example question
        elif pending_slot == "fraud_example_shown":
            if user_intent == "yes":
                new_slots = dict(current_slots)
                new_slots["fraud_example_shown"] = "yes"
                logger.info("RecSubgraph.extract_slots: Fraud example requested")
                return {"slots": new_slots, "side_info": None}
            elif user_intent == "no":
                # User doesn't want the example - offer to just give a quick recommendation instead
                new_slots = dict(current_slots)
                new_slots["fraud_example_shown"] = "no"
                new_slots["_fraud_rec_started"] = "yes"  # Skip to recommendation questions
                logger.info("RecSubgraph.extract_slots: Fraud example declined -> proceeding to questions")
                return {
                    "slots": new_slots, 
                    "side_info": None,
                    "messages": [AIMessage(content="No worries! Let me just ask you a couple of quick questions to find the best Fraud Protect360 plan for you.")]
                }
        
        # User responding to "Would you like me to recommend..." after example
        elif pending_slot == "fraud_ready_for_rec":
            if user_intent == "yes":
                new_slots = dict(current_slots)
                new_slots["_fraud_rec_started"] = "yes"
                logger.info("RecSubgraph.extract_slots: Fraud ready for recommendation questions")
                return {"slots": new_slots, "side_info": None}
            elif user_intent == "no":
                # User doesn't want recommendation right now
                new_slots = dict(current_slots)
                logger.info("RecSubgraph.extract_slots: Fraud recommendation declined")
                return {
                    "slots": new_slots, 
                    "side_info": None,
                    "messages": [AIMessage(content="No problem! Feel free to ask if you have any questions about Fraud Protect360 or any other insurance products.")]
                }
        
        # If intent is "unclear", fall through to normal slot extraction which may handle it
    
    # Get descriptions for context
    descriptions = _slot_descriptions(prod)
    desc_text = "\n".join([f"- {k}: {descriptions.get(k, '')}" for k in required])

    # Build compact, dynamic slot validation rules from configs/slot_validation_rules.yaml
    slot_rules_all = _load_slot_rules() or {}
    prod_key = (prod or "").lower()
    product_rules = slot_rules_all.get(prod_key) or {}

    slot_rule_lines: List[str] = []
    for slot_name in required:
        rule = product_rules.get(slot_name)
        if not isinstance(rule, dict):
            continue
        rtype = str(rule.get("type") or "").lower()

        if rtype == "enum":
            vals = [str(v) for v in rule.get("values", [])]
            if vals:
                slot_rule_lines.append(
                    f"- {slot_name}: enum; valid normalized values: {', '.join(vals)}."
                )
        elif rtype == "integer":
            allowed_vals = rule.get("allowed_values")
            if allowed_vals:
                slot_rule_lines.append(
                    f"- {slot_name}: integer; valid values: {', '.join(str(v) for v in allowed_vals)}."
                )
            else:
                min_val = rule.get("min")
                max_val = rule.get("max")
                if min_val is not None and max_val is not None:
                    slot_rule_lines.append(
                        f"- {slot_name}: integer; valid range: {min_val} to {max_val} (inclusive)."
                    )
                elif min_val is not None:
                    slot_rule_lines.append(
                        f"- {slot_name}: integer; value must be at least {min_val}."
                    )
                elif max_val is not None:
                    slot_rule_lines.append(
                        f"- {slot_name}: integer; value must be at most {max_val}."
                    )
        elif rtype == "set":
            vals = [str(v) for v in rule.get("values", [])]
            if vals:
                slot_rule_lines.append(
                    f"- {slot_name}: multi-select; valid options: {', '.join(vals)}."
                )
        elif rtype == "age":
            bands = [str(b) for b in rule.get("bands", [])]
            min_val = rule.get("numeric_min")
            max_val = rule.get("numeric_max")
            line = f"- {slot_name}: age; acceptable bands: {', '.join(bands)}."
            if min_val is not None and max_val is not None:
                line += f" Numeric ages must be between {min_val} and {max_val}."
            slot_rule_lines.append(line)
        elif rtype == "location":
            allow_city = bool(rule.get("allow_city", True))
            allow_country = bool(rule.get("allow_country", True))
            if allow_city and allow_country:
                slot_rule_lines.append(
                    f"- {slot_name}: location; accept city or country names and keep the user's phrase as the normalized value."
                )
            elif allow_city:
                slot_rule_lines.append(
                    f"- {slot_name}: location; accept city names and keep the user's phrase as the normalized value."
                )
            elif allow_country:
                slot_rule_lines.append(
                    f"- {slot_name}: location; accept country names and keep the user's phrase as the normalized value."
                )

    rules_text = ""
    if slot_rule_lines:
        rules_text = (
            "\n\n## SLOT VALIDATION RULES (per product config)"\
            + "\n" + "\n".join(slot_rule_lines) +
            "\n\nApply these rules STRICTLY when deciding whether to include a slot in 'updates':"\
            "\n- Only include a slot if the user's answer can be confidently normalized to a value that satisfies the rule."\
            "\n- If the input is outside the allowed set or numeric range, do NOT include that slot in 'updates' (leave it empty so it will be re-asked)."\
            "\n- For numeric slots, your final normalized value must be digits only (no '$', commas, 'k', or words)."
        )

    history_ctx = _build_history_context_from_messages(messages, max_pairs=4)

    sys_msg = (
        f"You are extracting structured slots for {prod} insurance recommendation.\n"
        f"Required slots:\n{desc_text}\n"
        f"Current slots (already filled): {current_slots}\n"
        "\n## YOUR TASK"
        "\nAnalyze the conversation and extract ONLY slot values the user has EXPLICITLY provided."
        "\n"
        "\n## SLOT EXTRACTION RULES"
        "\n1. Extract a slot ONLY if the user clearly and explicitly stated that information in their message"
        "\n2. Do NOT infer or assume values:"
        "\n   - If user says 'travel' or 'travel insurance', do NOT assume coverage_scope=self"
        "\n   - If user says 'planning a trip', do NOT assume they're traveling alone"
        "\n   - If user just selects a product, do NOT fill any slots"
        "\n3. Only extract coverage_scope if user EXPLICITLY says: 'just me', 'myself', 'solo', 'alone', 'my family', 'with family', 'group of adults', 'group of families'"
        "\n4. Do NOT re-extract slots already in 'Current slots' unless user is clearly correcting them"
        "\n5. When in doubt, extract NOTHING - we will ask the user"
        "\n6. Do NOT auto-correct invalid values - if user says '260 months' for a field that only accepts 14 or 26, do NOT extract it"
        "\n"
        "\n## CROSS-SLOT EXTRACTION (IMPORTANT)"
        "\nThe user may answer a DIFFERENT slot than what was asked. You MUST detect and extract it correctly."
        "\nExamples:"
        "\n- Bot asks 'Where are you traveling?' but user says 'Family' → This is coverage_scope=family, NOT a destination. Extract coverage_scope."
        "\n- Bot asks 'How long is your trip?' but user says 'Japan' → This is destination=Japan, NOT a duration. Extract destination."
        "\n- Bot asks 'Who is traveling?' but user says '14 days' → This is duration=14, NOT coverage_scope. Extract duration."
        "\n"
        "\nFor each value the user provides, match it to the CORRECT slot based on its meaning, not based on which question was asked."
        "\nYou can extract MULTIPLE slots if the user provides multiple valid values in one message."
    )

    if rules_text:
        sys_msg += rules_text

    sys_msg += (
        "\n\n## DESTINATION SLOT HANDLING"
        "\nFor the 'destination' slot: accept a single location string provided by the user."
        "\nIt can be a city, region, or country name (for example 'Tokyo', 'Japan', 'Bali')."
        "\nStore exactly what the user said for destination; do NOT rewrite cities into countries or guess missing details."
        "\nIf the user mentions multiple distinct places and it is unclear which is primary, skip filling the destination slot."
        "\n"
        "\n## SIDE QUESTION DETECTION"
        "\nSet 'side_question' ONLY when the user is genuinely asking for clarification or information:"
        "\n- 'What does coverage_scope mean?'"
        "\n- 'Why do you need to know my destination?'"
        "\n- 'What's the difference between family and group coverage?'"
        "\n- 'How much does this cost?'"
        "\n- 'What does that mean?' / 'I don't understand'"
        "\n- 'Why did you recommend...' / 'Why this plan?'"
        "\n"
        "\nDo NOT set side_question when:"
        "\n- User is confirming/selecting something (even with '?' like 'travel?' or 'Japan?')"
        "\n- User is providing slot values, even if brief"
        "\n- User is just continuing the conversation normally"
        "\n"
        "\n## PRODUCT SWITCH DETECTION"
        "\nSet 'product_switch_request' ONLY when the user explicitly wants to ABANDON the current "
        "recommendation flow and switch to a DIFFERENT insurance product."
        "\n"
        "\nExamples where you SHOULD set product_switch_request:"
        "\n- 'Actually, I want car insurance instead' → 'car'"
        "\n- 'Forget travel, I need maid insurance' → 'maid'"
        "\n- 'Can we do home insurance instead?' → 'home'"
        "\n- 'I changed my mind, I want personal accident' → 'personalaccident'"
        "\n- 'Switch to fraud protect' → 'fraud'"
        "\n"
        "\nExamples where you should NOT set product_switch_request (just normal answers):"
        "\n- 'I'm going home to visit family' → user is answering destination, 'home' is not a switch"
        "\n- 'I'm traveling with my maid' → user is describing coverage, not switching to maid insurance"
        "\n- 'I'll be driving my car there' → user is explaining travel plans, not switching to car insurance"
        "\n- 'I had an accident last year' → user is sharing context, not switching to personal accident"
        "\n- 'Tell me about car insurance' → this is a side_question, not a switch request"
    )

    user_msg = (
        f"Recent conversation (most recent last):\n{history_ctx}\n\n"
        f"Latest user message:\n{msg}\n\n"
        "Task: Return ONLY slot updates for REQUIRED slots that you can fill confidently."
    )

    try:
        structured = _get_slot_extractor()  # Use cached extractor
        result = await structured.ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=user_msg),
            ],
        )

        new_slots = dict(current_slots)
        updates_log = []
        for update in result.updates:
            if update.slot_name in required and update.value:
                new_slots[update.slot_name] = update.value
                updates_log.append(f"{update.slot_name}={update.value}")
        
        logger.info(
            "RecSubgraph.extract_slots: extracted updates=%s | merged slots=%s", 
            updates_log, new_slots
        )
        
        # Check for Product-Specific Exceptions (e.g., "I have medical insurance" for Early CI)
        # This takes precedence over side_question because it's a specific anticipated interaction.
        slot_config = _slot_config(prod)
        msg_lower = (msg or "").lower()
        
        # Check exceptions for ALL required slots (user might trigger an exception for a slot we haven't asked yet, or the current one)
        for slot_name in required:
            if slot_name in slot_config:
                config = slot_config[slot_name]
                for exc in config.exceptions:
                    if exc.trigger.lower() in msg_lower:
                        logger.info(
                            "RecSubgraph.extract_slots: exception trigger '%s' matched for slot '%s'", 
                            exc.trigger, slot_name
                        )
                        return {"slots": new_slots, "side_info": exc.response}
        
        # Handle genuine side questions (FAQs inside recommendation flow)
        if result.side_question:
            logger.info("RecSubgraph.extract_slots: side_question detected='%s'", result.side_question)
            # Return the question to be handled by the side_info node
            return {"slots": new_slots, "pending_side_question": result.side_question}
        
        # Handle product switch requests
        if result.product_switch_request:
            requested_product = result.product_switch_request.strip().lower()
            # Normalize the product name
            from ..utils.slots import _normalize_product_key
            normalized_new_product = _normalize_product_key(requested_product)
            
            if normalized_new_product and normalized_new_product != prod.lower():
                # User wants to switch to a different product
                logger.info(
                    "RecSubgraph.extract_slots: product_switch_request detected current=%s requested=%s",
                    prod, normalized_new_product
                )
                
                # Build a friendly product name for display
                product_display_names = {
                    "choice": "Choice Protect360",
                    "travel": "Travel Protect360",
                    "maid": "Maid Protect360",
                    "car": "Car Protect360",
                    "personalaccident": "Personal Accident Protect360",
                    "home": "Home Protect360",
                    "early": "Early Critical Illness Protect360",
                    "fraud": "Fraud Protect360",
                    "hospital": "Hospital Cash Protect360",
                }
                current_display = product_display_names.get(prod.lower(), prod)
                new_display = product_display_names.get(normalized_new_product, normalized_new_product)
                
                # Set a flag to indicate product switch was requested
                # The ask_next_slot node will handle this gracefully
                switch_message = (
                    f"I see you'd like to switch to *{new_display}*! 🔄\n\n"
                    f"We were working on your {current_display} recommendation. "
                    f"Would you like me to:\n"
                    f"• Continue with {current_display}\n"
                    f"• Start fresh with {new_display}\n\n"
                    f"Just let me know!"
                )
                
                return {
                    "slots": new_slots,
                    "pending_side_question": None,
                    "product_switch_pending": normalized_new_product,  # Store the requested product
                    "side_info": switch_message,  # Use side_info to display the message
                }
        
        # ==========================================================================
        # CROSS-SLOT EXTRACTION ACKNOWLEDGMENT
        # ==========================================================================
        # Check if user answered a different slot than what was asked
        # This provides better UX by acknowledging what we understood
        
        newly_filled_slots = []
        for update in result.updates:
            slot_name = update.slot_name
            # Check if this is a new slot (not in current_slots or value changed)
            if slot_name in required and update.value:
                old_value = current_slots.get(slot_name)
                if old_value != update.value:
                    newly_filled_slots.append((slot_name, update.value))
        
        cross_slot_ack = None
        if newly_filled_slots and pending_slot:
            # Check if ANY of the filled slots are different from the pending slot
            filled_different = [s for s in newly_filled_slots if s[0] != pending_slot]
            
            if filled_different and pending_slot not in [s[0] for s in newly_filled_slots]:
                # User answered a completely different slot(s) than asked
                # Generate acknowledgment message
                ack_parts = []
                for slot_name, value in filled_different:
                    # Generate friendly acknowledgment based on slot type
                    if slot_name == "coverage_scope":
                        scope_friendly = {
                            "self": "you're traveling solo",
                            "family": "you're traveling with family",
                            "group_adults": "you're traveling with a group of adults",
                            "group_families": "you're traveling with multiple families",
                        }
                        friendly = scope_friendly.get(value.lower(), f"your coverage is for {value}")
                        ack_parts.append(f"Got it, {friendly}! 👍")
                    elif slot_name == "destination":
                        ack_parts.append(f"Great, you're heading to *{value}*! ✈️")
                    elif slot_name == "duration":
                        ack_parts.append(f"Noted, your trip is for *{value} days*! 📅")
                    elif slot_name == "maid_country":
                        ack_parts.append(f"Got it, your helper is from *{value}*! 👍")
                    elif slot_name == "coverage_above_mom_minimum":
                        if value.lower() == "yes":
                            ack_parts.append("Understood, you need coverage above MOM minimum! 👍")
                        else:
                            ack_parts.append("Noted, MOM minimum coverage works for you! 👍")
                    elif slot_name == "desired_amount":
                        ack_parts.append(f"Great, you're looking for *${value}* coverage! 💰")
                    elif slot_name == "coverage_amount":
                        ack_parts.append(f"Noted, *${value}* coverage amount! 💰")
                    elif slot_name == "purchase_frequency":
                        ack_parts.append(f"Got it, you shop online *{value}*! 🛒")
                    else:
                        ack_parts.append(f"Got it, {slot_name} is *{value}*! 👍")
                
                if ack_parts:
                    cross_slot_ack = " ".join(ack_parts)
                    logger.info(
                        "RecSubgraph.extract_slots: cross-slot extraction detected - asked=%s, filled=%s, ack='%s'",
                        pending_slot, [s[0] for s in filled_different], cross_slot_ack
                    )
        
        return {"slots": new_slots, "pending_side_question": None, "side_info": cross_slot_ack}
    except Exception as e:
        logger.warning("Slot extraction failed: %s", e)
        return {}


def _normalize_int_value(raw: str) -> Optional[int]:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        # Preference phrases without explicit numbers (e.g. "maximum coverage")
        high_markers = ["higher", "highest", "max", "maximum", "as high as possible", "best coverage"]
        low_markers = ["lower", "lowest", "min", "minimum", "as low as possible", "budget"]
        if any(m in text for m in high_markers):
            return 3500
        if any(m in text for m in low_markers):
            return 500
        return None
    try:
        val = int(digits)
        # Handle shortcuts like "3k" → 3000
        if "k" in text and val < 1000:
            val *= 1000
        return val
    except Exception:
        return None


def _normalize_enum_slot_value(prod: str, slot_name: str, raw: str, rule: Dict[str, Any]) -> Optional[str]:
    text = str(raw or "").strip().lower()
    if not text:
        return None

    allowed = [str(v).lower() for v in rule.get("values", [])]
    if text in allowed:
        return text

    prod = (prod or "").lower()

    # Generic yes/no style slots
    yes_markers = ["yes", "y ", " y", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "of course", "absolutely", "definitely"]
    no_markers = ["no", "n ", " n", "nope", "nah", "not really", "not now", "skip", "pass", "no thanks", "not interested"]

    if set(["yes", "no"]).issuperset(allowed):
        if any(m in text for m in yes_markers):
            return "yes"
        if any(m in text for m in no_markers):
            return "no"

    # Maid add-ons preference: required / not_required
    if prod == "maid" and slot_name == "add_ons":
        if any(m in text for m in yes_markers) or "add-on" in text or "addon" in text or "add ons" in text or "extras" in text:
            return "required"
        if any(m in text for m in no_markers) or "no extras" in text or "no add" in text:
            return "not_required"

    # Travel / PA coverage scope
    if slot_name == "coverage_scope":
        if any(kw in text for kw in ["just me", "myself", "for me", "solo", "alone", "only me", "self"]):
            if "self" in allowed:
                return "self"
        if any(kw in text for kw in ["family", "my family", "for us", "whole family", "entire family"]):
            if "family" in allowed:
                return "family"
        if prod == "travel":
            if any(kw in text for kw in ["group of adults", "adult group", "group adults", "friends group", "friends trip", "adult friends"]):
                if "group_adults" in allowed:
                    return "group_adults"
            if any(kw in text for kw in ["group of families", "family group", "multiple families"]):
                if "group_families" in allowed:
                    return "group_families"

    # Personal accident risk level
    if prod == "personalaccident" and slot_name == "risk_level":
        if "low" in text:
            return "low" if "low" in allowed else None
        if any(kw in text for kw in ["medium", "mid", "moderate"]):
            return "medium" if "medium" in allowed else None
        if "high" in text:
            return "high" if "high" in allowed else None

    # Fraud purchase frequency
    if prod == "fraud" and slot_name == "purchase_frequency":
        if any(kw in text for kw in ["everyday", "every day", "7 days", "daily"]):
            return "daily" if "daily" in allowed else None
        if any(kw in text for kw in ["once a week", "twice a week", "weekly"]):
            return "weekly" if "weekly" in allowed else None
        if any(kw in text for kw in ["once a month", "twice a month", "monthly"]):
            return "monthly" if "monthly" in allowed else None
        if any(kw in text for kw in ["rarely", "seldom", "occasionally", "few times a year"]):
            return "rarely" if "rarely" in allowed else None

    # Fraud scam experience
    if prod == "fraud" and slot_name == "scam_exp":
        if any(kw in text for kw in ["yes", "yep", "yup", "scammed"]):
            return "yes" if "yes" in allowed else None
        if any(kw in text for kw in ["almost", "nearly", "close call"]):
            return "almost" if "almost" in allowed else None
        if any(kw in text for kw in ["no", "never", "nope", "nah"]):
            return "no" if "no" in allowed else None

    return None


def _parse_simple_int(raw: str) -> Optional[int]:
    """Parse an integer from free text without applying preference phrases."""
    text = str(raw or "").strip().lower()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        val = int(digits)
        if "k" in text and val < 1000:
            val *= 1000
        return val
    except Exception:
        return None


def _normalize_set_value(raw: str, rule: Dict[str, Any]) -> Optional[str]:
    text = str(raw or "").lower()
    allowed = [str(v).lower() for v in rule.get("values", [])]
    if not text or not allowed:
        return None

    selected: List[str] = []
    # Simple synonym handling for home risk_concerns
    for val in allowed:
        if val == "fire" and any(kw in text for kw in ["fire", "fires"]):
            selected.append("fire")
        elif val == "water damage" and any(kw in text for kw in ["water", "flood", "leak", "pipe burst"]):
            selected.append("water damage")
        elif val == "theft" and any(kw in text for kw in ["theft", "burglary", "break-in", "stolen"]):
            selected.append("theft")

    # Handle broad phrases like "all" / "everything"
    if not selected and any(kw in text for kw in ["all", "everything", "both"]):
        selected = [v for v in ["fire", "water damage", "theft"] if v in allowed]

    # Return comma-separated in canonical order
    if not selected:
        return None
    ordered = [v for v in ["fire", "water damage", "theft"] if v in selected]
    return ", ".join(ordered)


def _rec_validate_slots_python(state: AgentState) -> AgentState:
    """Validate and normalize slots using configs/slot_validation_rules.yaml."""
    prod = state.get("product")
    slots = state.get("slots") or {}
    if not prod or not slots:
        return {}

    rules_all = _load_slot_rules() or {}
    prod_key = (prod or "").lower()
    product_rules = rules_all.get(prod_key)
    if not isinstance(product_rules, dict):
        return {}

    new_slots = dict(slots)

    for slot_name, rule in product_rules.items():
        raw_value = _get_slot_value(slots, slot_name)
        if not raw_value:
            continue

        rtype = str(rule.get("type") or "").lower()

        if rtype == "enum":
            normalized = _normalize_enum_slot_value(prod_key, slot_name, raw_value, rule)
            if normalized is None:
                # Invalid enum – clear so the question will be re-asked
                new_slots.pop(slot_name, None)
                logger.info("RecSubgraph.validate: cleared invalid enum slot %s=%s", slot_name, raw_value)
            else:
                new_slots[slot_name] = normalized

        elif rtype == "integer":
            # Personal Accident desired_amount supports preference phrases like "as high as possible".
            if prod_key == "personalaccident" and slot_name == "desired_amount":
                val = _normalize_int_value(raw_value)
            else:
                val = _parse_simple_int(raw_value)
            if val is None:
                new_slots.pop(slot_name, None)
                logger.info("RecSubgraph.validate: cleared non-numeric slot %s=%s", slot_name, raw_value)
                continue

            allowed_vals = rule.get("allowed_values") or []
            min_val = rule.get("min")
            max_val = rule.get("max")

            if allowed_vals:
                if val in allowed_vals:
                    new_slots[slot_name] = str(val)
                else:
                    # Do NOT auto-correct out-of-domain numbers like 260 → 26; just clear
                    new_slots.pop(slot_name, None)
                    logger.info("RecSubgraph.validate: %s=%s outside allowed_values %s", slot_name, val, allowed_vals)
            else:
                if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                    new_slots.pop(slot_name, None)
                    logger.info("RecSubgraph.validate: %s=%s outside range [%s,%s]", slot_name, val, min_val, max_val)
                else:
                    new_slots[slot_name] = str(val)

        elif rtype == "set":
            normalized = _normalize_set_value(raw_value, rule)
            if normalized is None:
                new_slots.pop(slot_name, None)
                logger.info("RecSubgraph.validate: cleared invalid set slot %s=%s", slot_name, raw_value)
            else:
                new_slots[slot_name] = normalized

        elif rtype == "age":
            bands = [str(b).lower() for b in rule.get("bands", [])]
            text = str(raw_value or "").strip().lower()
            if text in bands:
                new_slots[slot_name] = raw_value
                continue

            val = _normalize_int_value(raw_value)
            if val is None:
                new_slots.pop(slot_name, None)
                logger.info("RecSubgraph.validate: cleared non-numeric age %s", raw_value)
                continue

            nmin = rule.get("numeric_min")
            nmax = rule.get("numeric_max")
            if (nmin is not None and val < nmin) or (nmax is not None and val > nmax):
                new_slots.pop(slot_name, None)
                logger.info("RecSubgraph.validate: age=%s outside range [%s,%s]", val, nmin, nmax)
            else:
                new_slots[slot_name] = str(val)

        elif rtype == "location":
            # Accept city or country names as-is, trimming whitespace
            new_slots[slot_name] = " ".join(str(raw_value or "").strip().split())

        elif rtype == "free_text":
            # Keep any non-empty free-text value
            text = str(raw_value or "").strip()
            if not text:
                new_slots.pop(slot_name, None)
            else:
                new_slots[slot_name] = text

    if new_slots != slots:
        logger.info("RecSubgraph.validate: normalized slots from %s to %s", slots, new_slots)
        return {"slots": new_slots}

    return {}


def _rec_validate_slots(state: AgentState) -> AgentState:
    """Lightweight guard rails using configs/slot_validation_rules.yaml.

    Full semantic validation and normalization is done by the LLM in
    _rec_extract_slots using the injected per-slot rules. Here we only:
    - Enforce simple numeric ranges / allowed_values
    - Ensure enums are within the allowed set
    - Drop clearly invalid values so the question will be re-asked
    No synonym or preference handling is done in Python anymore.
    """

    prod = state.get("product")
    slots = state.get("slots") or {}
    if not prod or not slots:
        return {}

    rules_all = _load_slot_rules() or {}
    prod_key = (prod or "").lower()
    product_rules = rules_all.get(prod_key)
    if not isinstance(product_rules, dict):
        return {}

    new_slots = dict(slots)

    # Carry forward any existing validation messages so we can clear them when fixed
    existing_errors = state.get("slot_validation_errors") or {}
    validation_errors = dict(existing_errors)
    errors_changed = False

    for slot_name, rule in product_rules.items():
        raw_value = _get_slot_value(slots, slot_name)
        if raw_value in (None, ""):
            # Clear any stale error for this slot if value is now empty
            if slot_name in validation_errors:
                validation_errors.pop(slot_name, None)
                errors_changed = True
            continue

        rtype = str(rule.get("type") or "").lower()
        text = str(raw_value).strip()

        # Human-friendly label for error messages
        slot_label = slot_name.replace("_", " ")

        if rtype == "enum":
            allowed = [str(v) for v in rule.get("values", [])]
            if allowed and text not in allowed:
                new_slots.pop(slot_name, None)
                msg = (
                    f"Your last answer for {slot_label!r} was not one of the accepted options. "
                    f"Please reply with ONE of: {', '.join(allowed)}."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info(
                    "RecSubgraph.validate: cleared enum slot %s=%s (not in %s)",
                    slot_name,
                    raw_value,
                    allowed,
                )
            else:
                # Clear any previous error if the value is now valid
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True

        elif rtype == "integer":
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I couldn't detect a valid number for {slot_label!r}. "
                    "Please reply with digits only (for example: 14, 26, 500, 2000)."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: cleared non-numeric slot %s=%s", slot_name, raw_value)
                continue

            try:
                val = int(digits)
            except Exception:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I couldn't parse '{raw_value}' as a whole number for {slot_label!r}. "
                    "Please reply with digits only (for example: 14, 26, 500, 2000)."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: failed to parse int for %s=%s", slot_name, raw_value)
                continue

            allowed_vals = rule.get("allowed_values") or []
            min_val = rule.get("min")
            max_val = rule.get("max")

            if allowed_vals:
                if val not in allowed_vals:
                    new_slots.pop(slot_name, None)
                    allowed_str = ", ".join(str(v) for v in allowed_vals)
                    msg = (
                        f"{val} is not an accepted value for {slot_label!r}. "
                        f"Please reply with ONE of: {allowed_str}."
                    )
                    validation_errors[slot_name] = msg
                    errors_changed = True
                    logger.info(
                        "RecSubgraph.validate: %s=%s outside allowed_values %s",
                        slot_name,
                        val,
                        allowed_vals,
                    )
                else:
                    new_slots[slot_name] = str(val)
                    if slot_name in validation_errors:
                        validation_errors.pop(slot_name, None)
                        errors_changed = True
            else:
                if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                    new_slots.pop(slot_name, None)
                    range_msg = None
                    if min_val is not None and max_val is not None:
                        range_msg = f"between {min_val} and {max_val}"
                    elif min_val is not None:
                        range_msg = f"at least {min_val}"
                    elif max_val is not None:
                        range_msg = f"at most {max_val}"
                    msg = (
                        f"{val} is outside the acceptable range for {slot_label!r}. "
                        f"Please provide a number {range_msg}."
                    )
                    validation_errors[slot_name] = msg
                    errors_changed = True
                    logger.info(
                        "RecSubgraph.validate: %s=%s outside range [%s,%s]",
                        slot_name,
                        val,
                        min_val,
                        max_val,
                    )
                else:
                    new_slots[slot_name] = str(val)
                    if slot_name in validation_errors:
                        validation_errors.pop(slot_name, None)
                        errors_changed = True

        elif rtype == "set":
            # Trust LLM normalization but drop empty/whitespace-only values
            if not text:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I didn't catch any clear selections for {slot_label!r}. "
                    "Please mention at least one of the supported options (for example: fire, water damage, theft)."
                )
            
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: cleared empty set slot %s", slot_name)
            else:
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True

        elif rtype == "age":
            # Accept either band labels or numeric ages within range
            bands = [str(b) for b in rule.get("bands", [])]
            if text in bands:
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True
                continue

            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I couldn't detect a valid age for {slot_label!r}. "
                    "Please reply with a whole number of years (for example: 30)."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: cleared non-numeric age %s", raw_value)
                continue

            try:
                val = int(digits)
            except Exception:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I couldn't parse '{raw_value}' as an age in years for {slot_label!r}. "
                    "Please reply with a whole number of years (for example: 30)."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: failed to parse age %s", raw_value)
                continue

            nmin = rule.get("numeric_min")
            nmax = rule.get("numeric_max")
            if (nmin is not None and val < nmin) or (nmax is not None and val > nmax):
                new_slots.pop(slot_name, None)
                msg = (
                    f"{val} is outside the acceptable age range for {slot_label!r}. "
                    f"Please provide an age between {nmin} and {nmax} years."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
                logger.info("RecSubgraph.validate: age=%s outside range [%s,%s]", val, nmin, nmax)
            else:
                new_slots[slot_name] = str(val)
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True

        elif rtype == "location":
            # Accept any non-empty location string; trim whitespace
            if not text:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I didn't catch a clear place for {slot_label!r}. "
                    "Please share a city, region, or country in a few words (for example: Tokyo, Bali, Singapore)."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
            else:
                new_slots[slot_name] = " ".join(text.split())
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True

        elif rtype == "free_text":
            # Keep any non-empty free-text value
            if not text:
                new_slots.pop(slot_name, None)
                msg = (
                    f"I didn't catch any details for {slot_label!r}. "
                    "Please reply with a short phrase or sentence."
                )
                validation_errors[slot_name] = msg
                errors_changed = True
            else:
                if slot_name in validation_errors:
                    validation_errors.pop(slot_name, None)
                    errors_changed = True

    updates: Dict[str, Any] = {}
    if new_slots != slots:
        logger.info("RecSubgraph.validate: adjusted slots from %s to %s", slots, new_slots)
        updates["slots"] = new_slots

    if errors_changed:
        updates["slot_validation_errors"] = validation_errors

    return updates


def _rec_manager(state: AgentState) -> str:
    """Decide next step: ask slot, generate rec, or exit."""
    prod = state.get("product")
    if not prod:
        return "end_turn" # Wait for user reply to product question
        
    required = _required_slots_for_product(prod)
    slots = state.get("slots") or {}
    missing = [s for s in required if not _get_slot_value(slots, s)]
    
    if missing:
        logger.info(
            "RecSubgraph.manager: missing slots %s for product %s -> asking next slot", 
            missing, prod
        )
        return "ask_next_slot"
    
    logger.info("RecSubgraph.manager: all slots filled -> generating recommendation")
    return "generate_rec"

async def _rec_ask_next_slot(state: AgentState) -> AgentState:
    prod = state.get("product")
    required = _required_slots_for_product(prod)
    slots = state.get("slots") or {}
    missing = [s for s in required if not _get_slot_value(slots, s)]
    
    if not missing:
        return {}

    next_slot = missing[0]
    prod_lower = (prod or "").lower()
    messages = list(state.get("messages", []) or [])
    last_user_msg = _get_last_user_message(messages) or ""
    
    # ==========================================================================
    # FRAUD PROTECT360 EDUCATIONAL FLOW
    # ==========================================================================
    if prod_lower == "fraud":
        # Step 1: Introduction question
        if next_slot == "fraud_intro_shown":
            question = "A great choice! Would you like to learn more about our Fraud Protect360 product?"
            logger.info("RecSubgraph.ask_next_slot: Fraud intro question")
            return {
                "messages": [AIMessage(content=question)],
                "pending_slot": next_slot,
                "side_info": None,
                "sources": []
            }
        
        # Step 2: Show intro content and ask about real-life example
        if next_slot == "fraud_example_shown":
            intro_content = (
                "Every day, Singaporeans lose thousands to online scams.\n\n"
                "Fraud Protect360 helps you recover financial losses due to:\n"
                "• Online payment scams\n"
                "• Phishing / malware attacks\n"
                "• Identity theft\n"
                "• Fake e-commerce transactions\n\n"
                "Want to see how it protects you in real life situations?"
            )
            logger.info("RecSubgraph.ask_next_slot: Fraud intro content + example question")
            return {
                "messages": [AIMessage(content=intro_content)],
                "pending_slot": next_slot,
                "side_info": None,
                "sources": []
            }
        
        # Step 3: Show real-life example and ask about recommendation
        if next_slot == "purchase_frequency":
            # Check if we just showed the example (fraud_example_shown is filled)
            example_shown = _get_slot_value(slots, "fraud_example_shown")
            rec_started = _get_slot_value(slots, "_fraud_rec_started")
            
            if example_shown and example_shown.lower() == "yes" and not rec_started:
                example_content = (
                    "Imagine this: you made a purchase on an online platform and did not receive your item, "
                    "and the seller became unresponsive – under our Fraud Protect360 you are covered up to "
                    "$10,000 for your undelivered online purchase!\n\n"
                    "Would you like me to recommend a personalized coverage for you?"
                )
                # We'll treat "yes" to this as starting the actual slot collection
                logger.info("RecSubgraph.ask_next_slot: Fraud example content + recommendation offer")
                return {
                    "messages": [AIMessage(content=example_content)],
                    "pending_slot": "fraud_ready_for_rec",  # Special marker
                    "side_info": None,
                    "sources": []
                }
    
    # ==========================================================================
    # STANDARD SLOT COLLECTION
    # ==========================================================================
    
    # Check if we have a specific hardcoded question for this slot
    slot_config = _slot_config(prod)
    specific_question = None
    if next_slot in slot_config and slot_config[next_slot].question:
        specific_question = slot_config[next_slot].question

    desc_map = _slot_descriptions(prod)
    description = desc_map.get(next_slot, f"information about {next_slot}")
    
    # Check if we're re-asking the same slot (user gave invalid/unclear answer)
    pending_slot = state.get("pending_slot")
    is_reask = (pending_slot == next_slot)

    # Use side_info if available (set by side_info node or exception responses in extract_slots)
    side_info_text = ""
    if state.get("side_info"):
        side_info_text = f"{state['side_info']}\n\nNow, regarding your recommendation: "

    # Any validation guidance for this slot from the previous turn
    slot_errors = state.get("slot_validation_errors") or {}
    slot_error_msg = slot_errors.get(next_slot)
    
    # If we have a specific question, use it directly (bypassing LLM generation for the question part)
    if specific_question:
        question = specific_question
    else:
        sys_msg = (
            "You are helping collect information to recommend a BigTapp insurance plan. "
            "Ask ONE concise, friendly question to collect the requested detail. "
            "Do not explain WHY you are asking; just ask the question."
        )
        user_msg = (
            f"Product: {prod}\nSlot name: {next_slot}\nDescription: {description}\n"
            f"Context (if any): {side_info_text}\n"
            "Please ask the user for this information."
        )
        
        try:
            q_msg = await get_router_llm().ainvoke(
                [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)],
            )
            question = str(getattr(q_msg, "content", "") or "").strip()
        except Exception:
            question = f"Could you please provide details for {next_slot}?"

    # Build a more helpful clarification for re-asks
    if is_reask:
        clarification_parts = []

        if side_info_text:
            clarification_parts.append(side_info_text.strip())

        if slot_error_msg:
            # Use the precise validation guidance from the previous turn
            clarification_parts.append(slot_error_msg.strip())
        else:
            # Build dynamic clarification from slot validation rules
            slot_rules_all = _load_slot_rules() or {}
            prod_key = (prod or "").lower()
            product_rules = slot_rules_all.get(prod_key) or {}
            slot_rule = product_rules.get(next_slot) or {}
            rtype = str(slot_rule.get("type") or "").lower()
            
            if last_user_msg:
                base = f"I didn't quite catch a valid answer from '{last_user_msg}'. "
            else:
                base = "I didn't quite catch a valid answer. "
            
            # Add specific guidance based on slot type
            if rtype == "enum":
                allowed = slot_rule.get("values", [])
                if allowed:
                    friendly_vals = ", ".join(str(v) for v in allowed)
                    base += f"Please reply with one of these options: {friendly_vals}."
            elif rtype == "integer":
                allowed_vals = slot_rule.get("allowed_values")
                if allowed_vals:
                    vals_str = ", ".join(str(v) for v in allowed_vals)
                    base += f"Please reply with one of these values: {vals_str}."
                else:
                    min_val = slot_rule.get("min")
                    max_val = slot_rule.get("max")
                    if min_val is not None and max_val is not None:
                        base += f"Please provide a number between {min_val} and {max_val}."
                    elif min_val is not None:
                        base += f"Please provide a number of at least {min_val}."
                    elif max_val is not None:
                        base += f"Please provide a number no more than {max_val}."
            elif rtype == "age":
                bands = slot_rule.get("bands", [])
                nmin = slot_rule.get("numeric_min")
                nmax = slot_rule.get("numeric_max")
                if bands:
                    bands_str = ", ".join(str(b) for b in bands)
                    base += f"Please reply with an age band ({bands_str}) or a specific age"
                    if nmin is not None and nmax is not None:
                        base += f" between {nmin} and {nmax}."
                    else:
                        base += "."
            elif rtype == "set":
                allowed = slot_rule.get("values", [])
                if allowed:
                    friendly_vals = ", ".join(str(v) for v in allowed)
                    base += f"Please mention one or more of: {friendly_vals}."
            elif next_slot == "destination":
                base += (
                    "For your travel cover, please share your main travel destination "
                    "(city, region, or country). "
                )
            elif next_slot == "maid_country":
                base += "For your helper's cover, please share your helper's country of origin. "
            elif rtype == "location":
                base += "Please share a city, region, or country name."
            else:
                # Final fallback for unknown slot types
                base += "Could you please provide a clearer answer?"

            clarification_parts.append(base)

        clarification = " ".join(part for part in clarification_parts if part).strip()
        if clarification:
            # Keep a single explicit question at the end
            final_content = f"{clarification}\n\n{question}"
        else:
            final_content = question
    else:
        # If side info exists, prepend it to the final message content so Styler sees it
        final_content = f"{side_info_text}{question}" if side_info_text else question
        
    logger.info("RecSubgraph.ask_next_slot: asking for %s -> '%s' (reask=%s)", next_slot, final_content, is_reask)
    
    return {
        "messages": [AIMessage(content=final_content)],
        "pending_slot": next_slot,  # Track what we asked
        "is_slot_reask": is_reask,  # Track if this is a re-ask for styler
        "side_info": None,  # Clear it after using it
        "pending_side_question": None,  # Clear after answering
        "sources": []
    }

async def _rec_generate_recommendation(state: AgentState) -> AgentState:
    prod = state.get("product")
    slots = state.get("slots") or {}
    
    logger.info("RecSubgraph.generate_rec: generating for product=%s with slots=%s", prod, slots)
    
    tier, rec_text = await _generate_recommendation_text_async(prod, slots)
    
    logger.info("RecSubgraph.generate_rec: result tier=%s | text_len=%d", tier, len(rec_text) if rec_text else 0)
    
    answer = rec_text or (
        "Based on what you've shared, I recommend a plan, but I couldn't format the full "
        "explanation. Please try asking again in a slightly different way."
    )
    
    # Handle any pending side info/question that was asked while filling the last slot
    # This ensures the user's question gets answered before we present the recommendation
    side_info = state.get("side_info")
    if side_info:
        answer = f"{side_info}\n\nNow, based on everything you've shared, here's my recommendation:\n\n{answer}"
        logger.info("RecSubgraph.generate_rec: prepended side_info to recommendation")

    return {
        "messages": [AIMessage(content=answer)],
        "rec_ready": True,
        "rec_given": True,
        "pending_slot": None,  # Clear pending slot - slot collection is complete
        "is_slot_reask": None,  # Clear re-ask flag
        "side_info": None,  # Clear side info after using
        "pending_side_question": None,  # Clear pending side question
        "sources": []
    }


async def _rec_side_info(state: AgentState) -> AgentState:
    """Execute side question lookup in parallel."""
    prod = state.get("product")
    question = state.get("pending_side_question")
    
    if not question or not prod:
        return {"side_info": None}
        
    logger.info("RecSubgraph.side_info: looking up '%s' for %s", question, prod)
    answer, _ = await _info_tool_async(prod, question)
    
    return {"side_info": answer, "pending_side_question": None}

def _rec_route_after_extract(state: AgentState) -> List[str]:
    """Route to side_info and/or manager based on extraction result."""
    routes = []
    
    # Always check if we need to proceed with the flow (manager)
    # We always run validation/manager logic
    routes.append("validate_slots")
    
    # If we have a side question, run that in parallel
    if state.get("pending_side_question"):
        routes.append("side_info")
        
    return routes

# Build the subgraph
rec_builder = StateGraph(AgentState)
rec_builder.add_node("ensure_product", _rec_ensure_product)
rec_builder.add_node("extract_slots", _rec_extract_slots)
rec_builder.add_node("validate_slots", _rec_validate_slots)
rec_builder.add_node("side_info", _rec_side_info)
rec_builder.add_node("ask_next_slot", _rec_ask_next_slot)
rec_builder.add_node("generate_rec", _rec_generate_recommendation)

rec_builder.set_entry_point("ensure_product")

rec_builder.add_edge("ensure_product", "extract_slots")

# Parallel routing after extraction
rec_builder.add_conditional_edges(
    "extract_slots",
    _rec_route_after_extract,
    {
        "validate_slots": "validate_slots",
        "side_info": "side_info"
    }
)

# Side info joins back to the flow at the decision points
# We need to make sure side_info finishes before we output the final message
# But validate_slots -> manager -> ask/gen is the main flow.
# We can make side_info edge to a synchronization point or just let it update state.
# Since LangGraph waits for all parallel branches to complete before next step if they converge?
# Actually, we need to ensure side_info is done before ask_next_slot or generate_rec runs?
# Or we can let them run and have a combiner? 
# Simpler approach for now: side_info updates 'side_info' state. 
# validate_slots -> manager -> (ask|gen).
# We need to ensure side_info is complete before ask/gen executes.
# So we can add edges from side_info to the potential next steps? No, that's dynamic.
# Better: side_info -> validate_slots is wrong because validate runs in parallel.
# Let's make side_info -> validate_slots? No, we want parallel.
# 
# Correct Parallel Pattern:
# extract -> [side_info, validate_slots]
# side_info -> join_node
# validate_slots -> manager -> join_node? 
#
# Actually, manager just decides. The actual WORK is in ask_next_slot or generate_rec.
# So:
# extract -> side_info
# extract -> validate_slots -> manager
# 
# manager returns "ask_next_slot" or "generate_rec".
# We want to execute those ONLY after side_info is done (if it was running).
# LangGraph doesn't support "wait for all" easily unless we have a join node.
#
# Let's introduce a 'resolve_next_step' node that acts as the join.
# extract -> side_info -> resolve_next_step
# extract -> validate_slots -> resolve_next_step (this path calculates the decision)
#
# But validate_slots is dummy. Let's move manager logic to resolve_next_step?
#
# Refined Plan:
# 1. extract -> side_info (conditional)
# 2. extract -> validate_slots (always)
# 3. side_info -> resolve_step
# 4. validate_slots -> resolve_step
# 5. resolve_step (runs manager logic) -> ask/gen
#
# This ensures we wait for side_info (if triggered) and validation before deciding.

def _rec_resolve_step(state: AgentState) -> str:
    """Join point: decide next step after side info and validation are done."""
    # This is effectively the manager logic now
    prod = state.get("product")
    if not prod:
        return "end_turn"

    # ------------------------------------------------------------------
    # Choice Protect360 scripted discovery / education (multi-turn)
    # If these flags are set, we already sent a message this turn and must
    # wait for the user's next reply (avoid auto-generating a tier immediately).
    # ------------------------------------------------------------------
    if state.get("choice_info_step"):
        logger.info(
            "RecSubgraph.resolve: choice_info_step=%s -> ending turn (await user reply)",
            state.get("choice_info_step"),
        )
        return "end_turn"
    
    # Check if the flow was already terminated by extract_slots (e.g., Fraud declined)
    # If rec_ready is True but rec_given is False, extract_slots sent a termination message
    rec_ready = state.get("rec_ready", False)
    rec_given = state.get("rec_given", False)
    slots = state.get("slots") or {}
    
    # Handle Fraud declined case - user said "no" to intro/learning
    if slots.get("_fraud_declined") == "yes":
        logger.info("RecSubgraph.resolve: Fraud flow declined -> ending turn")
        return "end_turn"
    
    # If rec_ready is set but rec_given is not, a message was already sent by extract_slots
    # (e.g., when user declines recommendation in fraud flow)
    if rec_ready and not rec_given:
        logger.info("RecSubgraph.resolve: rec_ready=True, rec_given=False -> ending turn (message already sent)")
        return "end_turn"
        
    required = _required_slots_for_product(prod)
    missing = [s for s in required if not _get_slot_value(slots, s)]
    
    if missing:
        logger.info("RecSubgraph.resolve: missing slots %s -> asking next slot", missing)
        return "ask_next_slot"
    
    logger.info("RecSubgraph.resolve: all slots filled -> generating recommendation")
    return "generate_rec"

rec_builder.add_node("resolve_step", lambda x: x) # Dummy join node that passes state? 
# Actually, we can just use the function as the conditional edge from a node.
# But we need a node to be the target of the edges.
# Let's make 'resolve_step' a real node that just logs or passes.
def _resolve_node(state: AgentState) -> AgentState:
    return {} 

rec_builder.add_node("resolve_node", _resolve_node)

rec_builder.add_edge("side_info", "resolve_node")
rec_builder.add_edge("validate_slots", "resolve_node")

rec_builder.add_conditional_edges(
    "resolve_node",
    _rec_resolve_step,
    {
        "end_turn": END,
        "ask_next_slot": "ask_next_slot",
        "generate_rec": "generate_rec"
    }
)

rec_builder.add_edge("ask_next_slot", END)
rec_builder.add_edge("generate_rec", END)

recommendation_subgraph = rec_builder.compile()
