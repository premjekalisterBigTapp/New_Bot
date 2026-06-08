from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage

from ..state import AgentState
from ..config import _load_life_insurance_kb
from ..infrastructure import get_response_llm, get_chat_llm
from ..utils.memory import _get_last_user_message

logger = logging.getLogger(__name__)


_LIFE_SYSTEM_TEMPLATE = """You are BigTapp's life insurance assistant for the product "Life Protect360".

Answer the user's life-insurance questions using ONLY the KNOWLEDGE BASE below as your source of truth.

RULES:
- Use ONLY facts from the KNOWLEDGE BASE. Do NOT invent plans, figures, premiums, payout amounts, or terms. If a detail is not in the document, say you don't have that specific detail and offer to connect them with an advisor.
- The product is "Life Protect360" by BigTapp. NEVER mention any other insurer, brand, or company name, and NEVER mention "Kotak". Always present everything as Life Protect360 / BigTapp.
- Keep replies concise and conversational (WhatsApp style): short paragraphs or bullet points. You may use *bold* for emphasis. Do not dump the whole document.
- When the user shows buying interest (e.g. "I want this", "how do I buy", "I'm interested"), naturally suggest enhancing cover with the optional Riders (Critical Illness Plus Benefit Rider, Permanent Disability Benefit Rider) as an upsell.
- Do NOT ask for the user's name or personal/contact details.

KNOWLEDGE BASE:
{kb}
"""

_KB_MISSING_REPLY = (
    "I can help you with our life insurance plan, Life Protect360. "
    "Could you tell me what you'd like to know - the plan options, premiums, "
    "payout, or the optional riders?"
)

_LLM_ERROR_REPLY = (
    "Sorry, I'm having trouble pulling up the life insurance details right now. "
    "Please try again in a moment."
)


async def _life_insurance_agent_node(state: AgentState) -> AgentState:
    """Answer life-insurance questions grounded strictly on the brochure KB.

    The reply is returned as plain content so the styler can beautify it like
    any other agent response.
    """
    messages = state.get("messages", []) or []
    user_text = _get_last_user_message(messages)
    logger.info("Agentic.agents: executing life_insurance_agent for query: %s", user_text)

    kb = _load_life_insurance_kb()
    if not kb:
        logger.warning("life_insurance_agent: KB empty/missing; returning safe fallback")
        return {"messages": [AIMessage(content=_KB_MISSING_REPLY)], "sources": []}

    system_prompt = _LIFE_SYSTEM_TEMPLATE.format(kb=kb)

    # Short history window for follow-up context ("how much at 35?").
    history: List[BaseMessage] = [
        m for m in messages[-10:] if isinstance(m, (HumanMessage, AIMessage))
    ]
    if not history:
        history = [HumanMessage(content=user_text or "Tell me about life insurance")]

    llm_messages = [SystemMessage(content=system_prompt)] + history

    try:
        llm = get_response_llm()
    except Exception:
        llm = get_chat_llm()

    try:
        ai_msg = await llm.ainvoke(llm_messages)
        content = getattr(ai_msg, "content", None) or str(ai_msg)
    except Exception as e:
        logger.exception("life_insurance_agent: LLM call failed: %s", e)
        content = _LLM_ERROR_REPLY

    return {"messages": [AIMessage(content=content)], "sources": []}
