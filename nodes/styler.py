from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from ..state import AgentState
from ..config import _load_knowledge_base
from ..infrastructure import get_router_llm
from ..utils.memory import _build_history_context_from_messages, _get_last_user_message

logger = logging.getLogger(__name__)


async def _style_reply_node(state: AgentState) -> AgentState:
    """Final styling/orchestration node to make replies feel more autonomous.

    It takes the draft reply from previous agents/tools plus a short history
    summary and rewrites the answer in a more human, conversational and
    capability-aware way.
    
    OPTIMIZATION: Skip LLM call for simple/short responses to improve latency.
    """

    messages = list(state.get("messages", []) or [])
    if not messages:
        return {}

    # Find the latest assistant reply to rewrite/polish.
    draft = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            draft = m
            break
    if not draft:
        return {}
    
    draft_content = str(getattr(draft, "content", "") or "").strip()
    draft_len = len(draft_content)
    
    # OPTIMIZATION: Skip styling for very short responses (questions, simple answers)
    # This saves ~1-2s per turn for slot-filling questions
    intent = (state.get("intent") or "").strip().lower()
    is_slot_reask = state.get("is_slot_reask", False)  # Re-ask needs better styling
    pending_slot = state.get("pending_slot")  # Slot question in progress, be very strict
    
    # DEBUG: Log the reask flag
    logger.debug("Agentic.styler.debug: is_slot_reask=%s intent=%s", is_slot_reask, intent)
    
    # Skip styling if:
    # 1. Draft is a short question (< 200 chars) - likely a slot question - BUT NOT if it's a re-ask
    # 2. Draft is a greeting (< 250 chars)
    # 3. Intent is greet/capabilities with short response
    skip_styling = False
    skip_reason = None
    
    # TEMPORARILY DISABLED: Always style slot questions for better invalid input handling
    # if draft_len < 150 and "?" in draft_content and not is_slot_reask:
    #     # Short question - no need to restyle (unless it's a re-ask)
    #     skip_styling = True
    #     skip_reason = "short_question"
    if intent == "greet" and draft_len < 250:
        # Short greeting
        skip_styling = True
        skip_reason = "short_greeting"
    # TEMPORARILY DISABLED: Always style very short responses too
    # elif draft_len < 100 and not is_slot_reask:
    #     # Very short response - keep as is (unless it's a re-ask)
    #     skip_styling = True
    #     skip_reason = "very_short"

    # FRAUD FLOW PRESERVATION:
    # The Fraud Protect360 flow uses carefully scripted educational narratives.
    # We must NOT rewrite these or inject deviations until the final recommendation is given.
    product = (state.get("product") or "").strip().lower()
    rec_given = state.get("rec_given", False)
    
    if product == "fraud" and not rec_given:
        skip_styling = True
        skip_reason = "fraud_educational_flow"
    
    # COMPARE OPTIMIZATION: Compare tool already includes styling in its prompt
    # Skip the separate styler call to save ~3-7s per compare request
    if intent == "compare":
        skip_styling = True
        skip_reason = "compare_already_styled"

    # SUMMARY OPTIMIZATION: Summary tool now embeds WhatsApp-friendly styling
    # in its own prompt, so we can skip the extra styler call here.
    if intent == "summary":
        skip_styling = True
        skip_reason = "summary_already_styled"

    # INFO OPTIMIZATION: Info tool now embeds WhatsApp-friendly styling
    # and flow/naming rules, so we can skip the extra styler call here too.
    if intent == "info":
        skip_styling = True
        skip_reason = "info_already_styled"

    # CAPABILITIES OPTIMIZATION: Capabilities tool now embeds WhatsApp-friendly styling
    if intent == "capabilities":
        skip_styling = True
        skip_reason = "capabilities_already_styled"

    # RECOMMENDATION OPTIMIZATION: Recommendation tool already includes styling in its prompt
    # Skip the separate styler call when recommendation was just generated (rec_given is True)
    if intent == "recommend" and rec_given:
        skip_styling = True
        skip_reason = "recommendation_already_styled"

    # RECOMMENDATION DISCOVERY UX (scripted):
    # Preserve the exact "specific vs customizable" + Choice follow-up prompts.
    if intent == "recommend" and (state.get("product_discovery_step") or state.get("choice_info_step")):
        skip_styling = True
        skip_reason = "recommendation_discovery_flow"
    
    # POLICY SERVICE OPTIMIZATION: Service flow responses are already formatted
    # Skip styling to preserve formatting and save LLM call
    if intent == "policy_service":
        skip_styling = True
        skip_reason = "policy_service_already_styled"

    if skip_styling:
        logger.info(
            "Agentic.styler.skip: reason=%s draft_len=%d (saving LLM call)",
            skip_reason, draft_len
        )
        return {}  # Keep original draft
    
    # Log when we force styling for re-asks
    if is_slot_reask:
        logger.info(
            "Agentic.styler.force_style: slot_reask=True draft_len=%d -> styling for better UX",
            draft_len
        )

    user_text = _get_last_user_message(messages) or ""
    history_ctx = _build_history_context_from_messages(messages[:-1], max_pairs=3)
    kb_text = _load_knowledge_base() or ""

    intent = (state.get("intent") or "").strip().lower()
    product = (state.get("product") or "").strip()

    # Base styler behaviour for most replies
    sys_prompt_base = """You are BigTapp's digital insurance assistant helping customers via WhatsApp.

Your task is to polish the draft reply below so it sounds warm, natural, and helpful.

Guidelines for good responses:
• Sound friendly and conversational, not robotic or scripted
• If the draft answers a side question, explain it clearly first, then smoothly continue
• You can discuss BigTapp products: Choice Protect360, Travel, Maid, Car, Home, Personal Accident, Early Critical Illness, Fraud (Protect360), and Hospital Cash plans
• Keep replies concise and focused on helping the user
• Be honest about limitations and gently steer back to insurance topics when needed

Response quality tips:
• If the draft already provides a recommendation or link, don't ask again if they want help
• When users say "thanks" or "bye", just respond warmly without adding new questions
• Don't repeat questions you just asked
• Don't add justification phrases like "This will help me suggest..." - it's implied
• If asking a question to gather info, don't declare which plan you'll recommend yet
• If the draft asks the user to make a choice, keep it as a choice question
• Vary your openings - don't always start with "Thanks for..." or "Great!"
• Use official product names: Travel Protect360, Maid Protect360, etc.
• Don't mention contact details unless explicitly asked
• BRAND RULE: NEVER output any legacy or internal brand names. If the draft contains any non-BigTapp brand references, replace them with "BigTapp" (company) or "BTBot" (bot). The user must only see BigTapp branding.
"""

    # When we are in slot-collection mode (pending_slot is set), we must be
    # extremely strict: do not add extra questions or new information needs.
    slot_rules = """

When collecting specific information (slot-filling mode):
• Keep the question focused - don't add extra questions beyond what the draft asks
• Preserve the meaning of the draft question exactly
• Skip meta explanations like "This will help me recommend..."
• For re-asks after unclear answers, briefly acknowledge the issue then ask clearly again
• If the draft explains what was unclear and gives examples of needed info, keep that explanation
"""

    sys_prompt = sys_prompt_base
    if pending_slot:
        sys_prompt = sys_prompt_base + slot_rules

    kb_section = kb_text.strip()
    if kb_section:
        kb_section = "\n\nCapabilities / Knowledge Base (for your own reference):\n" + kb_section

    # Make slot behaviour explicit in the instructions we send to the model.
    if pending_slot:
        rewrite_instruction = (
            "Please rewrite or lightly improve this reply following the goals above, "
            "but DO NOT add any new questions or ask for extra details beyond what is already in the draft. "
            "If the draft already explains what was unclear about the previous answer and what information is needed (with examples), preserve that explanation and only tweak phrasing slightly for clarity. "
            "You may rephrase the existing question and, if needed, briefly acknowledge that the previous answer was unclear."
        )
    else:
        rewrite_instruction = (
            "Please rewrite or improve this reply following the goals above. You may add at most one short follow-up "
            "sentence (not a new complex question) to gently suggest relevant insurance help if it is very natural, "
            "but do not be pushy."
        )

    user_prompt = f"""Conversation so far (most recent last):
{history_ctx}

Latest user message:
{user_text}

Current intent: {intent or 'unknown'}
Current product focus (if any): {product or 'none'}
{kb_section}

Draft assistant reply from internal tools/flows:
{draft.content}

{rewrite_instruction}
"""

    try:
        out_msg = await get_router_llm().ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        final_text = str(getattr(out_msg, "content", "") or "").strip()
    except Exception as e:
        logger.warning("Agentic.styler: styling failed, keeping draft reply: %s", e)
        final_text = ""

    if not final_text:
        # Fall back to the original draft if styling fails.
        final_text = str(getattr(draft, "content", "") or "").strip()
        if not final_text:
            return {}
            
    logger.info("Agentic.styler: draft_len=%d -> final_len=%d", len(draft.content), len(final_text))

    return {"messages": [AIMessage(content=final_text)]}
