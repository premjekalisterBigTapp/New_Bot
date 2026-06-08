from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Local infrastructure imports (lazy singleton - no explicit init needed)
from ..infrastructure import get_response_llm
from .benefits import get_product_benefits
from ..config import _load_summary_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm_async


async def _summary_tool(
    product: Optional[str], tiers: List[str], question: str
) -> tuple[str, List[str]]:
    """Summary tool: product/tier summaries using benefits and templates."""

    prod = _normalize_product_key(product)
    if not prod:
        prod = _normalize_product_key(await _detect_product_llm_async(question))
    if not prod:
        ask = (
            "Which product would you like a summary for: Choice Protect360, Travel, Maid, Car, Personal Accident, "
            "Home, Early, Fraud or Hospital?"
        )
        return ask, []

    try:
        benefits_text = get_product_benefits(prod)
    except Exception:
        benefits_text = ""

    sum_templates = _load_summary_templates()
    tpl = sum_templates.get(prod, {}) if sum_templates else {}

    # Default system prompt with WhatsApp-friendly styling so we can skip the
    # global styler node for summary responses.
    sys_t = tpl.get("system") or (
        "You are BigTapp's Smart Bot summarising insurance plans.\n\n"
        "RESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points and *asterisks* for plan or product names\n"
        "• Keep sentences short and clear, avoid long paragraphs\n"
        "• Use digits for numbers and sums (e.g. $500,000) - NEVER use abbreviations like $500k or $1M\n"
        "• No markdown headers (###) or tables\n"
        "• Be warm, clear and professional – not overly salesy\n\n"
        "STRUCTURE:\n"
        "1. One-line intro acknowledging what you are summarising\n"
        "2. 3–6 key bullet points covering what it protects, major limits and important conditions\n"
        "3. If tiers are relevant, briefly explain how they differ\n"
        "4. Optional short closing question checking if the user wants more detail\n\n"
        "CRITICAL NAMING RULES:\n"
        "• Use full official product names in the final answer where relevant: Travel Protect360, Maid Protect360, Car Protect360, Home Protect360, Personal Accident Protect360, Early Critical Illness Protect360, Fraud Protect360, Hospital Cash Protect360.\n"
        "• Do not shorten them to just 'Travel' or 'Maid' when you first mention them; shortening is fine only after the full name is used.\n\n"
        "Summarise USING ONLY the provided context below. If something is not in the context, do not invent it."
    )

    tiers_txt = ", ".join(tiers) if tiers else ("N/A" if prod in ("car", "early") else "")
    usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
        product=prod,
        tiers=tiers_txt,
        question=question,
        context=benefits_text or "",
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_response_llm()  # Thread-safe singleton
        messages = [SystemMessage(content=sys_t), HumanMessage(content=usr_t)]
        response = await llm.ainvoke(messages)
        answer = str(response.content).strip()
    except Exception as e:
        logger.warning("Tool.summary: LLM call failed: %s", e)
        answer = ""

    if not answer:
        answer = (
            "Here is a concise overview of our {prod} plans. You can also ask about specific benefits or tiers."
        ).format(prod=prod.title())
    return answer, []


async def _summary_tool_async(
    product: Optional[str], tiers: List[str], question: str
) -> tuple[str, List[str]]:
    """Async wrapper for _summary_tool to avoid blocking the event loop."""
    return await _summary_tool(product, tiers, question)
