from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import FeedbackPrediction
from ..infrastructure import get_router_llm
from ..utils.memory import _get_last_turn_from_messages

logger = logging.getLogger(__name__)

FEEDBACK_SYSTEM_PROMPT = """You help the BigTapp insurance chatbot understand how the user
is reacting to its PREVIOUS answer.

You will see the last assistant reply and the user's latest message.

Label the user's latest message as:
- negative_feedback: they say the previous answer was wrong, incomplete,
  off-topic, confusing, or not what they wanted.
- ack: they acknowledge or close (e.g. 'ok', 'thanks', 'great', 'bye') without
  asking for more.
- clarification: they ask you to clarify, simplify, or expand on the last
  answer (e.g. 'can you explain in simpler terms?', 'I don't understand').
- new_question: they move to a new substantive question or different topic.
- other: anything else.

Be strict: only use negative_feedback when the user is clearly unhappy with or
disagreeing with the last answer.
"""

_feedback_structured = None


def _get_feedback_structured():
    """Get cached feedback classifier with structured output."""
    global _feedback_structured
    if _feedback_structured is None:
        _feedback_structured = get_router_llm().with_structured_output(FeedbackPrediction)
    return _feedback_structured


async def _classify_feedback_from_messages_async(
    messages: List[BaseMessage],
) -> Optional[FeedbackPrediction]:
    """Async classify whether the user is giving negative feedback."""

    last_turn = _get_last_turn_from_messages(messages)
    if not last_turn:
        return None
    last_answer = (last_turn.get("assistant") or "").strip()
    last_user = (last_turn.get("user") or "").strip()
    if not last_answer or not last_user:
        return None

    ctx = (
        f"LAST_ANSWER: {last_answer}\n"
        f"USER_MESSAGE: {last_user}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FEEDBACK_SYSTEM_PROMPT),
            ("user", "{context}"),
        ]
    )
    try:
        result = await (prompt | _get_feedback_structured()).ainvoke({"context": ctx})
        if isinstance(result, FeedbackPrediction):
            if result.category == "negative_feedback":
                logger.info("Agentic.feedback: negative feedback detected: %s", result.reason)
            return result
        return FeedbackPrediction.model_validate(result)
    except Exception as e:
        logger.warning("Agentic feedback classification failed: %s", e)
        return None


async def _self_critique_and_rewrite_from_messages_async(
    messages: List[BaseMessage],
    pending_slot: Optional[str] = None,
    product: Optional[str] = None,
) -> Optional[str]:
    """Async self-critique/rewrite based on feedback."""

    last_turn = _get_last_turn_from_messages(messages)
    if not last_turn:
        return None
    last_answer = (last_turn.get("assistant") or "").strip()
    last_user = (last_turn.get("user") or "").strip()
    if not last_answer or not last_user:
        return None

    logger.info("Agentic.feedback: attempting self-critique/rewrite (pending_slot=%s)", pending_slot)

    # Build context-aware system message
    if pending_slot:
        # SLOT COLLECTION MODE - Stay focused, don't deviate!
        slot_label = pending_slot.replace("_", " ")
        sys_msg = (
            f"You are helping the BigTapp insurance chatbot handle user feedback during slot collection.\n\n"
            f"CRITICAL CONTEXT: The bot is currently collecting information for a {product or 'insurance'} recommendation. "
            f"The bot was asking for the '{slot_label}' value.\n\n"
            f"STRICT RULES:\n"
            f"1. DO NOT offer to provide more details, additional info, or coverage options\n"
            f"2. DO NOT ask unrelated questions or deviate from collecting the '{slot_label}'\n"
            f"3. If the user seems confused about valid options, clarify ONLY the valid choices for '{slot_label}'\n"
            f"4. Apologize briefly if needed, then RE-ASK the slot question clearly\n"
            f"5. Keep your response short and focused (2-3 sentences max)\n\n"
            f"Examples of GOOD responses:\n"
            f"- 'Sorry for any confusion! For the policy duration, please choose either 14 months or 26 months.'\n"
            f"- 'I apologize if that wasn't clear. Which country is your helper from?'\n\n"
            f"Examples of BAD responses (DO NOT DO THIS):\n"
            f"- 'Would you like me to explain the coverage options?' (bad)\n"
            f"- 'I can provide more details about the benefits...' (bad)\n"
            f"- 'Let me know if you have any questions!' (bad)\n"
        )
    else:
        # Normal mode - more flexibility
        sys_msg = (
            "You are a self-critique helper for the BigTapp insurance chatbot. "
            "You see the bot's previous answer and the user's feedback. "
            "If the answer was wrong, incomplete, or off-topic, correct it and "
            "provide a clearer, more accurate response. If the user is mainly "
            "asking for clarification, restate the key points more simply. "
            "Stay within BigTapp insurance products and do not invent new products."
        )
    
    user_content = (
        "[Previous answer]\n" + last_answer + "\n\n" +
        "[User feedback]\n" + last_user
    )
    try:
        msg = await get_router_llm().ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=user_content),
            ]
        )
        revised = str(getattr(msg, "content", "") or "").strip()
        if revised:
            logger.info("Agentic.feedback: rewrite successful, length=%d", len(revised))
        return revised or None
    except Exception as e:
        logger.warning("Agentic self-critique failed: %s", e)
        return None
