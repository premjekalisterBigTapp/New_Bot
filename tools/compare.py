"""
Plan comparison tool for comparing different tiers/plans.

This tool:
- Compares plans using benefits data and templates
- Supports product-specific comparison logic
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import List, Optional, Tuple, Dict, Any, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from ..infrastructure import get_response_llm, get_router_llm
from ..infrastructure.metrics import LLM_CALLS_TOTAL, LLM_LATENCY
from .benefits import get_product_benefits
from ..config import _load_cmp_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm_async
from ..utils.products import get_product_names_str, get_product_aliases_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# CROSS-PRODUCT COMPARISON (multi-product / Choice Protect360 components)
# =============================================================================

ChoiceComponent = Literal[
    "base_plan",
    "upgraded_pa",
    "home_contents_addon",
    "hospital_income_addon",
    "annual_travel_addon",
]


class CrossCompareTargets(BaseModel):
    """Targets to compare for cross-product comparisons."""

    products: List[str] = Field(
        default_factory=list,
        description="Products to compare. Use normalized keys like: choice, travel, maid, car, personalaccident, home, early, fraud, hospital.",
    )
    choice_components: List[ChoiceComponent] = Field(
        default_factory=list,
        description=(
            "When Choice Protect360 is involved, specify which parts to focus on: "
            "base_plan, upgraded_pa, home_contents_addon, hospital_income_addon, annual_travel_addon."
        ),
    )
    reason: str = Field(default="", description="Brief reason for the extracted targets.")


_cross_compare_detector = None


def _get_cross_compare_detector():
    global _cross_compare_detector
    if _cross_compare_detector is None:
        _cross_compare_detector = get_router_llm().with_structured_output(CrossCompareTargets)
    return _cross_compare_detector


async def _detect_cross_compare_targets(
    product: Optional[str],
    question: str,
) -> CrossCompareTargets:
    """
    Use LLM to detect whether the user is comparing across products (or Choice components).

    This avoids brittle keyword-only rules and supports mixed requests like:
    - "Compare Choice Protect360 home contents add-on vs Home Protect360"
    - "Travel vs Car insurance"
    """
    known_product = _normalize_product_key(product)
    product_list = get_product_names_str()
    aliases_prompt = get_product_aliases_prompt()

    sys_prompt = f"""You are extracting comparison targets for an insurance chatbot.

Supported products: {product_list}
Aliases:
{aliases_prompt}

Choice Protect360 notes:
- Choice Protect360 is a customizable plan (base Personal Accident coverage) with optional add-ons:
  * Home Contents add-on
  * Hospital Income add-on
  * Annual Travel add-on
  * Upgraded Personal Accident (PA) coverage

TASK:
1) Identify which products are being compared (2+ when cross-product).
2) If the user mentions Choice Protect360 add-ons, set choice_components accordingly.
3) If only one product is being compared (e.g., "compare Gold vs Platinum travel"), you may return that single product.

RULES:
- Prefer normalized product keys in 'products': choice, travel, maid, car, personalaccident, home, early, fraud, hospital.
- Only include a product if it is clearly referenced or strongly implied by the question and context.
- Use known_product='{known_product or ''}' as context when the user uses pronouns like "it" or "this plan".
"""

    try:
        detector = _get_cross_compare_detector()
        result = await detector.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"Known product: {known_product or 'None'}\nUser question: {question}"),
            ]
        )
        # Normalize products to internal keys
        normalized: List[str] = []
        for p in (result.products or []):
            pk = _normalize_product_key(p)
            if pk:
                normalized.append(pk)
        # De-dup while preserving order
        seen = set()
        normalized = [p for p in normalized if not (p in seen or seen.add(p))]
        result.products = normalized
        return result
    except Exception as e:
        logger.warning("Tool.compare.cross_target_detection_failed: %s", e)
        return CrossCompareTargets(products=[known_product] if known_product else [], choice_components=[], reason="detection_failed")


def _extract_choice_context(benefits_text: str, components: List[ChoiceComponent]) -> str:
    """Extract a smaller, component-focused context for Choice Protect360 comparisons."""
    text = str(benefits_text or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lower = [ln.lower() for ln in lines]

    # Split base vs add-ons (we embed an 'Optional Add-ons' header in benefits_raw.json)
    opt_idx = None
    for i, ln in enumerate(lower):
        if "optional add-ons" in ln or "optional add-ons" in ln.replace("-", " "):
            opt_idx = i
            break
    base_lines = lines[:opt_idx] if opt_idx is not None else lines
    addon_lines = lines[opt_idx:] if opt_idx is not None else []

    def _pick_addon(keyword: str) -> List[str]:
        out: List[str] = []
        for ln in addon_lines:
            if keyword.lower() in ln.lower():
                out.append(ln)
        return out

    picked: List[str] = []

    # Default: if no component specified, return full text (but caller may avoid this for token size).
    if not components:
        return text

    if "base_plan" in components or "upgraded_pa" in components:
        # Base plan lines already contain tiered PA benefits; include them in full.
        picked.extend(base_lines)

    if "home_contents_addon" in components:
        picked.append("HOME CONTENTS ADD-ON (Choice Protect360):")
        picked.extend(_pick_addon("home contents optional add-on"))

    if "hospital_income_addon" in components:
        picked.append("HOSPITAL INCOME ADD-ON (Choice Protect360):")
        picked.extend(_pick_addon("hospital income optional add-on"))

    if "annual_travel_addon" in components:
        picked.append("ANNUAL TRAVEL ADD-ON (Choice Protect360):")
        picked.extend(_pick_addon("annual travel optional add-on"))

    # Keep output reasonably compact
    return "\n".join([ln for ln in picked if ln]).strip()


async def _compare_cross_products(
    product: Optional[str],
    question: str,
    *,
    targets: Optional[CrossCompareTargets] = None,
) -> Tuple[str, List[str]]:
    """Cross-product comparison using benefits contexts for 2+ products/components."""
    targets = targets or await _detect_cross_compare_targets(product, question)
    products = targets.products or []

    # If we couldn't identify at least 2 distinct targets, fall back to single-product compare.
    # (Choice-only component comparisons can still be handled if components were identified.)
    distinct_products = list(dict.fromkeys(products))  # stable de-dup
    is_choice_component_compare = ("choice" in distinct_products and bool(targets.choice_components))
    if len(distinct_products) < 2 and not is_choice_component_compare:
        return await _compare_tool(product, [], question)

    # Build contexts
    item_contexts: List[Tuple[str, str]] = []

    for p in distinct_products:
        benefits = ""
        try:
            benefits = get_product_benefits(p)
        except Exception:
            benefits = ""

        if p == "choice" and targets.choice_components:
            benefits = _extract_choice_context(benefits, targets.choice_components)

        # Friendly display name
        display = {
            "choice": "Choice Protect360",
            "travel": "Travel Protect360",
            "maid": "Maid Protect360",
            "car": "Car Protect360",
            "personalaccident": "Personal Accident (Family Protect360)",
            "home": "Home Protect360",
            "early": "Early Critical Illness (Early Protect360 Plus)",
            "fraud": "Fraud Protect360 Plus",
            "hospital": "Hospital Cash (Hospital Protect360)",
        }.get(p, p)

        item_contexts.append((display, benefits))

    # System prompt for cross-product comparisons
    sys_t = (
        "You are BTBot comparing insurance products and/or Choice Protect360 components.\n\n"
        "CRITICAL RULES:\n"
        "- Use ONLY the provided contexts for each item. Do not invent benefits.\n"
        "- Always include full dollar amounts as written in the context (e.g., $100,000 or S$ 100,000).\n"
        "- Be explicit when something is not stated in the context.\n\n"
        "WHATSAPP FORMAT:\n"
        "• Use • bullet points\n"
        "• Use *asterisks* for product/plan names sparingly\n"
        "• No headers (###), no tables\n"
        "• Keep concise\n\n"
        "STRUCTURE:\n"
        "1) One-line intro acknowledging the comparison\n"
        "2) Side-by-side key differences (3–6 bullets)\n"
        "3) Short practical guidance (1–2 lines) based on the user's question\n"
    )

    ctx_blocks: List[str] = []
    for name, ctx in item_contexts:
        ctx_blocks.append(f"[{name} Context]\n{ctx or '(No benefits context available.)'}")

    usr_t = (
        f"User's comparison question:\n{question}\n\n"
        "Use these contexts:\n\n" + "\n\n---\n\n".join(ctx_blocks)
    )

    answer = ""
    llm_start = time.time()
    try:
        llm = get_response_llm()
        if llm:
            response = await llm.ainvoke([SystemMessage(content=sys_t), HumanMessage(content=usr_t)])
            answer = str(response.content).strip()

            llm_duration = time.time() - llm_start
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
        else:
            logger.error("Tool.compare.cross.llm_not_initialized")
    except Exception as e:
        llm_duration = time.time() - llm_start
        logger.error(
            "Tool.compare.cross.llm_failed: duration=%.3fs error=%s\n%s",
            llm_duration, str(e), traceback.format_exc()
        )
        try:
            LLM_CALLS_TOTAL.labels(model="response_llm", status="error").inc()
            LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
        except Exception:
            pass
        answer = ""

    if not answer:
        names = ", ".join(name for name, _ in item_contexts[:3])
        answer = f"I can help compare {names}. Could you share what aspect you care about most (price, medical cover, add-ons, etc.)?"

    return answer, []


async def _compare_tool(
    product: Optional[str], tiers: List[str], question: str
) -> Tuple[str, List[str]]:
    """
    Comparison tool: plan-only comparisons using benefits and templates.
    
    Args:
        product: Product name
        tiers: List of tier names to compare (optional)
        question: User's comparison question
        
    Returns:
        Tuple of (comparison_text, empty_sources_list)
        
    Raises:
        Exception: Re-raises exceptions for the caller to handle
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)
    
    # Attempt product detection from question if not provided
    if not prod:
        logger.debug(
            "Tool.compare.detecting_product: question='%s'",
            (question or "")[:100]
        )
        detected = await _detect_product_llm_async(question)
        prod = _normalize_product_key(detected)
    
    logger.info(
        "Tool.compare.start: raw_product=%s resolved_product=%s tiers=%s question_len=%d",
        raw_product, prod, tiers, len(question or "")
    )
    
    if not prod:
        logger.warning("Tool.compare.no_product: could not determine product")
        return (
            "Which product would you like to compare plans for: Choice Protect360, Travel, Maid, Car, Personal Accident, "
            "Home, Early, Fraud or Hospital?",
            [],
        )

    # Get benefits text for the product
    benefits_text = ""
    benefits_start = time.time()
    try:
        benefits_text = get_product_benefits(prod)
        logger.debug(
            "Tool.compare.benefits_loaded: product=%s len=%d duration=%.3fs",
            prod, len(benefits_text), time.time() - benefits_start
        )
    except Exception as e:
        logger.warning(
            "Tool.compare.benefits_failed: product=%s error=%s",
            prod, str(e)
        )
        benefits_text = ""

    # Load comparison templates
    cmp_templates = _load_cmp_templates()
    tpl = cmp_templates.get(prod, {}) if cmp_templates else {}
    
    # Default system prompt with styling instructions (skips styler node for faster response)
    default_sys = (
        "You are BTBot comparing insurance plans.\n\n"
        "RESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points, *asterisks* for plan names\n"
        "• Numbers as digits ($500,000) - NEVER use abbreviations like $500k or $1M\n"
        "• Clean line breaks between sections\n"
        "• NO headers (###), NO tables\n"
        "• Be warm and conversational, not robotic\n"
        "• Keep response concise but informative\n\n"
        "STRUCTURE:\n"
        "1. Brief intro acknowledging the comparison request\n"
        "2. Key differences between plans with specific amounts\n"
        "3. Simple recommendation based on coverage needs\n"
        "4. Optional: brief closing question about preference\n\n"
        "Compare the plans using only the provided context."
    )
    sys_t = tpl.get("system") or default_sys
    tiers_txt = ", ".join(tiers) if tiers else ""
    usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
        product=prod,
        tiers=tiers_txt,
        question=question,
        context=benefits_text or "",
    )

    # Generate LLM response
    answer = ""
    llm_start = time.time()
    try:
        llm = get_response_llm()
        if llm:
            messages = [SystemMessage(content=sys_t), HumanMessage(content=usr_t)]
            response = await llm.ainvoke(messages)
            answer = str(response.content).strip()
            
            llm_duration = time.time() - llm_start
            logger.info(
                "Tool.compare.llm_response: product=%s answer_len=%d duration=%.3fs",
                prod, len(answer), llm_duration
            )
            
            # Record metrics
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
        else:
            logger.error("Tool.compare.llm_not_initialized")
    except Exception as e:
        llm_duration = time.time() - llm_start
        logger.error(
            "Tool.compare.llm_failed: product=%s duration=%.3fs error=%s\n%s",
            prod, llm_duration, str(e), traceback.format_exc()
        )
        try:
            LLM_CALLS_TOTAL.labels(model="response_llm", status="error").inc()
            LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
        except Exception:
            pass
        answer = ""

    # Fallback answer
    if not answer:
        answer = (
            f"Here is a high-level comparison of the available {prod.title()} plans. "
            "You can ask about a specific benefit if you need more detail."
        )

    total_duration = time.time() - start_time
    logger.info(
        "Tool.compare.completed: product=%s answer_len=%d total_duration=%.3fs",
        prod, len(answer), total_duration
    )
    
    return answer, []


async def _compare_tool_async(
    product: Optional[str], tiers: List[str], question: str
) -> Tuple[str, List[str]]:
    """Async wrapper for the comparison tool."""
    return await _compare_tool(product, tiers, question)


async def _compare_tool_any_async(
    product: Optional[str], tiers: List[str], question: str
) -> Tuple[str, List[str]]:
    """
    Compare tool that supports both:
    - single-product tier comparisons (existing behavior), and
    - cross-product comparisons (new).
    """
    targets = await _detect_cross_compare_targets(product, question)
    products = targets.products or []
    distinct_products = list(dict.fromkeys(products))

    # Cross-product if 2+ products, or if Choice components were explicitly referenced.
    if len(distinct_products) >= 2 or (("choice" in distinct_products) and bool(targets.choice_components)):
        logger.info(
            "Tool.compare.mode: cross_product products=%s choice_components=%s reason=%s",
            distinct_products,
            targets.choice_components,
            (targets.reason or "")[:120],
        )
        return await _compare_cross_products(product, question, targets=targets)

    logger.info(
        "Tool.compare.mode: single_product product=%s inferred_products=%s reason=%s",
        _normalize_product_key(product) or product,
        distinct_products,
        (targets.reason or "")[:120],
    )
    return await _compare_tool(product, tiers, question)
