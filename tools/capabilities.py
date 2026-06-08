from __future__ import annotations

import logging
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import _load_knowledge_base
from ..infrastructure import get_router_llm

logger = logging.getLogger(__name__)

def _is_services_or_capabilities_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Broad, low-risk heuristics to catch service/capability questions without an extra LLM call.
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
        "what products do you have",
        "which products do you support",
        "supported products",
        "capabilities",
        "features",
    ]
    return any(k in t for k in triggers)

async def _capabilities_tool(question: str) -> str:
    """Answer capability/meta questions using the static knowledge base."""

    # Fast-path: deterministic response for "services / what can you do" questions.
    # This avoids UX loops and avoids spending an LLM call on a simple catalog response.
    if _is_services_or_capabilities_query(question):
        return (
            "Here’s what I can help you with:\n"
            "• Answer coverage/benefit questions\n"
            "• Summarize plans/tiers\n"
            "• Compare plans/tiers (including cross-product comparisons)\n"
            "• Recommend a suitable plan based on your needs\n"
            "• Share purchase links\n"
            "• Policy services: check policy/claim status, update email/mobile/address (with verification)\n\n"
            "Supported products include:\n"
            "• Choice Protect360\n"
            "• Travel Protect360\n"
            "• Maid Protect360\n"
            "• Car Protect360\n"
            "• Personal Accident (Family Protect360)\n"
            "• Home Protect360\n"
            "• Early Critical Illness Protect360\n"
            "• Fraud Protect360\n"
            "• Hospital Cash Protect360"
        )

    kb_text = _load_knowledge_base()
    sys_msg = (
        "You are BTBot. Answer questions about what you can do, "
        "which products you support, and how you help customers. Use only the "
        "knowledge base below; do not invent new capabilities.\n\n"
        "RESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points\n"
        "• Keep responses concise and friendly\n"
        "• NO headers (###), NO tables\n"
        "• Be warm and conversational\n\n"
        "KEY CAPABILITIES:\n"
        "• *Products:* Choice Protect360, Travel Protect360, Maid Protect360, Car Protect360, "
        "Home Protect360, Personal Accident (Family Protect360), Early Critical Illness Protect360, "
        "Fraud Protect360, Hospital Cash Protect360\n"
        "• *Information:* Explain coverage details, compare plans, recommend plans, provide purchase links\n"
        "• *Policy Services:* Check policy status, check claim status, update email/mobile/address\n"
        "• Note: For policy services, customers need to verify their identity with NRIC, name, mobile, and policy number"
    )
    user_parts: List[str] = [f"Question: {question}"]
    if kb_text:
        user_parts.append("")
        user_parts.append("Knowledge Base:")
        user_parts.append(kb_text)
    user_content = "\n".join(user_parts)

    try:
        msg = await get_router_llm().ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=user_content),
            ]
        )
        return str(getattr(msg, "content", "") or "").strip() or (
            "I can help with product information, summaries, comparisons, "
            "recommendations and purchase links for BigTapp insurance plans."
        )
    except Exception as e:
        logger.warning("Capabilities responder failed: %s", e)
        return (
            "I can help with product information, summaries, comparisons, and "
            "recommendations for BigTapp insurance plans."
        )
