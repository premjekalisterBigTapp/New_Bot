"""
Recommendation generation tool.

This tool:
- Selects the appropriate tier based on collected slots
- Generates personalized recommendation text using templates
- Includes travel-specific advisory injection
- Comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from ..infrastructure import get_response_llm
from ..infrastructure.metrics import LLM_CALLS_TOTAL, LLM_LATENCY, RECOMMENDATION_GIVEN_TOTAL
from .benefits import get_product_benefits
from ..config import _load_rec_templates
from ..utils.slots import _get_slot_value, _normalize_product_key

logger = logging.getLogger(__name__)


def _get_llm():
    """Get response LLM from local infrastructure (thread-safe singleton)."""
    return get_response_llm()


def _select_tier(product: str, slots: Dict[str, Any]) -> Optional[str]:
    """
    Select the recommended tier based on product and collected slots.
    
    Args:
        product: Normalized product name
        slots: Collected slot values
        
    Returns:
        Recommended tier name or None
    """
    p = _normalize_product_key(product) or (product or "").lower()
    tier: Optional[str] = None
    
    logger.debug(
        "Tool.recommendation.selecting_tier: product=%s slots_keys=%s",
        p, list(slots.keys())
    )
    
    if p == "travel":
        tier = "Gold"
    elif p == "maid":
        coverage_above_mom = (_get_slot_value(slots, "coverage_above_mom_minimum") or "").strip().lower()
        if coverage_above_mom == "yes":
            tier = "Premier"
        elif coverage_above_mom == "no":
            tier = "Enhanced"
        else:
            tier = "Enhanced"  # Default
    elif p == "personalaccident":
        try:
            amount = int(_get_slot_value(slots, "desired_amount"))
            if 500 <= amount <= 1000:
                tier = "Silver"
            elif 1001 <= amount <= 2500:
                tier = "Premier"
            elif 2501 <= amount <= 3500:
                tier = "Platinum"
            else:
                tier = "Premier"  # Default
        except (ValueError, TypeError):
            tier = "Premier"  # Default
    elif p == "home":
        try:
            amount = int(_get_slot_value(slots, "coverage_amount"))
            if amount <= 100000:
                tier = "Silver"
            elif amount <= 200000:
                tier = "Gold"
            else:
                tier = "Platinum"
        except (ValueError, TypeError):
            tier = "Gold"  # Default
    elif p == "early":
        tier = None  # Early CI has its own fixed messaging
    elif p == "fraud":
        freq = _get_slot_value(slots, "purchase_frequency").strip().lower()
        if freq in ("daily", "everyday", "every day"):
            tier = "Platinum"
        else:
            tier = "Gold"
    elif p == "hospital":
        raw = _get_slot_value(slots, "coverage") or ""
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        try:
            val = int(digits) if digits else 0
        except Exception:
            val = 0
        choices = [100, 200, 300]
        if val <= 0:
            sel = 200
        else:
            sel = min(choices, key=lambda x: abs(x - val))
        tier = {100: "Silver", 200: "Premier", 300: "Titanium"}.get(sel, "Premier")
    elif p == "choice":
        # Choice Protect360 is a customizable plan. Default to a popular mid-tier unless
        # the user explicitly asks for a different tier in their question (handled by templates/LLM).
        tier = "Gold"
    elif p == "car":
        tier = "Standard"  # Car has no tier selection
    
    logger.debug(
        "Tool.recommendation.tier_selected: product=%s tier=%s",
        p, tier
    )
    
    return tier


async def _generate_recommendation_text(
    product: str, slots: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """
    Generate final recommendation text and tier.

    Args:
        product: Normalized product name
        slots: Collected slot values
        
    Returns:
        Tuple of (recommended_tier, recommendation_text)
        
    Raises:
        Exception: Re-raises exceptions for the caller to handle
    """
    start_time = time.time()
    
    # Normalize product key so templates/benefits lookups are consistent even when
    # upstream passes display names like "ChoiceProtect360".
    p = _normalize_product_key(product) or (product or "").lower()
    
    logger.info(
        "Tool.recommendation.start: product=%s slots=%s",
        p, {k: v for k, v in slots.items() if not k.startswith("_")}
    )
    
    # Select tier
    tier = _select_tier(p, slots)
    
    # Get benefits text
    benefits_text = ""
    benefits_start = time.time()
    try:
        benefits_text = get_product_benefits(p)
        logger.debug(
            "Tool.recommendation.benefits_loaded: product=%s len=%d duration=%.3fs",
            p, len(benefits_text), time.time() - benefits_start
        )
    except Exception as e:
        logger.warning(
            "Tool.recommendation.benefits_failed: product=%s error=%s",
            p, str(e)
        )

    # Load templates
    rec_templates = _load_rec_templates()
    product_key = p
    tpl = rec_templates.get(product_key, {}) if rec_templates else {}

    # Build prompts based on product type
    sys_t = ""
    usr_t = ""
    
    if product_key == "maid":
        add_ons_pref = _get_slot_value(slots, "add_ons") or "not_required"
        sys_t = (tpl.get("system") or "").format(tier=tier or "", add_ons=add_ons_pref)
        usr_t = (tpl.get("user") or "").format(
            tier=tier or "",
            add_ons=add_ons_pref,
            benefits=benefits_text or "",
        )
    elif product_key == "travel":
        destination = (_get_slot_value(slots, "destination") or "").strip()
        if destination:
            advisory = (
                f"Overseas medical treatment can be expensive for foreigners in {destination} without comprehensive cover. "
                "To ensure peace of mind for you and your travellers, we recommend having comprehensive travel insurance."
            )
        else:
            advisory = (
                "Overseas medical treatment can be expensive for foreigners without comprehensive cover. "
                "To ensure peace of mind for you and your travellers, we recommend having comprehensive travel insurance."
            )
        sys_t = (tpl.get("system") or "").format(tier=tier or "", destination=destination or "")
        usr_t = (tpl.get("user") or "").format(
            tier=tier or "",
            benefits=benefits_text or "",
            advisory=advisory or "",
            destination=destination or "",
        )
        
        # INJECT ADVISORY: Prepend the advisory text to the system prompt for travel only
        sys_t = f"MANDATORY INSTRUCTION: Start your response with this exact advisory: '{advisory}'\n\n{sys_t}"
        
        logger.debug(
            "Tool.recommendation.travel_advisory: destination=%s",
            destination
        )
    else:
        sys_t = (tpl.get("system") or "").format(tier=tier or "")
        usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "")

    # Generate response
    response = ""
    llm_start = time.time()
    
    if product_key == "early":
        # Early CI has its own fixed messaging
        try:
            tpl_e = rec_templates.get("early") or {}
            sys_e = (tpl_e.get("system") or "")
            usr_e = (tpl_e.get("user") or "").format(benefits=benefits_text or "")
            if sys_e and usr_e:
                llm = _get_llm()
                if llm:
                    result = await llm.ainvoke([
                        SystemMessage(content=sys_e),
                        HumanMessage(content=usr_e),
                    ])
                    response = str(result.content).strip()
                    
                    llm_duration = time.time() - llm_start
                    logger.info(
                        "Tool.recommendation.early_llm_response: response_len=%d duration=%.3fs",
                        len(response), llm_duration
                    )
                    
                    try:
                        LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                        LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
                    except Exception:
                        pass
                else:
                    logger.error("Tool.recommendation.llm_not_initialized")
        except Exception as e:
            llm_duration = time.time() - llm_start
            logger.error(
                "Tool.recommendation.early_llm_failed: duration=%.3fs error=%s\n%s",
                llm_duration, str(e), traceback.format_exc()
            )
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="error").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
    elif sys_t and usr_t:
        try:
            llm = _get_llm()
            if llm:
                result = await llm.ainvoke([
                    SystemMessage(content=sys_t),
                    HumanMessage(content=usr_t),
                ])
                response = str(result.content).strip()
                
                llm_duration = time.time() - llm_start
                logger.info(
                    "Tool.recommendation.llm_response: product=%s tier=%s response_len=%d duration=%.3fs",
                    p, tier, len(response), llm_duration
                )
                
                try:
                    LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                    LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
                except Exception:
                    pass
            else:
                logger.error("Tool.recommendation.llm_not_initialized")
        except Exception as e:
            llm_duration = time.time() - llm_start
            logger.error(
                "Tool.recommendation.llm_failed: product=%s duration=%.3fs error=%s\n%s",
                p, llm_duration, str(e), traceback.format_exc()
            )
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="error").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass

    # Record recommendation metric
    if response:
        try:
            RECOMMENDATION_GIVEN_TOTAL.labels(product=p).inc()
        except Exception:
            pass

    total_duration = time.time() - start_time
    logger.info(
        "Tool.recommendation.completed: product=%s tier=%s response_len=%d total_duration=%.3fs",
        p, tier, len(response), total_duration
    )

    return tier, response


async def _generate_recommendation_text_async(
    product: str, slots: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """Async wrapper for recommendation generation."""
    return await _generate_recommendation_text(product, slots)
