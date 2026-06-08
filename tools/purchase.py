"""
Purchase link generation tool.

This tool:
- Returns purchase links for products
- Uses product-specific link configuration
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import _load_purchase_links
from ..utils.slots import _normalize_product_key
from ..infrastructure.metrics import PURCHASE_LINK_GENERATED_TOTAL

logger = logging.getLogger(__name__)


# Friendly product names for user-facing messages
FRIENDLY_NAMES = {
    "choice": "Choice Protect360",
    "travel": "Travel",
    "maid": "Maid",
    "car": "Car",
    "personalaccident": "Personal Accident",
    "home": "Home",
    "early": "Early Critical Illness",
    "fraud": "Fraud Protect360",
    "hospital": "Hospital Protect360",
}


# Cross-sell map: complementary plan suggested after a purchase link is shown.
# Each entry has the product key, friendly display name, and a short benefit blurb.
# Designed to be generic - never references the user by name.
CROSS_SELL_MAP = {
    "travel": {
        "name": "Hospital Protect360",
        "blurb": "Provides daily cash benefit during hospitalisation, so unexpected medical costs while travelling don't impact your savings.",
    },
    "maid": {
        "name": "Home Protect360",
        "blurb": "Protects your home contents and structure against fire, theft, and water damage - a natural pairing with helper coverage.",
    },
    "car": {
        "name": "Fraud Protect360 Plus",
        "blurb": "Covers online shopping fraud, fund transfer scams, and identity theft - an essential digital companion to your motor cover.",
    },
    "home": {
        "name": "Travel Protect360",
        "blurb": "Covers trip cancellations, medical emergencies abroad, and lost baggage - peace of mind whenever you leave home.",
    },
    "hospital": {
        "name": "Early Protect360 Plus",
        "blurb": "Pays a lump sum on diagnosis of critical illness so you can focus on recovery instead of finances.",
    },
    "early": {
        "name": "Hospital Protect360",
        "blurb": "Adds daily cash benefit during hospital stays - a strong complement to your critical illness lump sum.",
    },
    "fraud": {
        "name": "Hospital Protect360",
        "blurb": "Adds daily cash benefit during hospitalisation, covering income loss when you can't work.",
    },
    "personalaccident": {
        "name": "Early Protect360 Plus",
        "blurb": "Pays a lump sum on diagnosis of major illnesses like cancer or heart disease - covering risks beyond accidents.",
    },
    "choice": {
        "name": "Hospital Protect360",
        "blurb": "Adds daily cash benefit during hospitalisation, complementing your customised plan with health protection.",
    },
}


def _build_cross_sell_suggestion(prod: str) -> str:
    """Build a brief, name-free cross-sell suggestion for the given product."""
    cross = CROSS_SELL_MAP.get(prod)
    if not cross:
        return ""
    return (
        f"\n\n💡 *You might also find this useful:*\n\n"
        f"*{cross['name']}* — {cross['blurb']}\n\n"
        f"Would you like to know more?"
    )


def _purchase_tool(product: Optional[str]) -> str:
    """
    Purchase tool: returns purchase link or friendly fallback.
    
    Args:
        product: Product name
        
    Returns:
        Purchase link message or fallback message
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)
    
    logger.info(
        "Tool.purchase.start: raw_product=%s resolved_product=%s",
        raw_product, prod
    )
    
    if not prod:
        logger.warning("Tool.purchase.no_product: could not determine product")
        return (
            "Which product would you like to buy? Available options: Choice Protect360, Travel Protect360, Maid Protect360, Car Protect360, Personal Accident Protect360, "
            "Home Protect360, Fraud Protect360, Early Critical Illness Protect360, Hospital Cash Protect360."
        )

    # Load purchase links
    links = _load_purchase_links()
    link = links.get(prod)
    friendly = FRIENDLY_NAMES.get(prod, product or "this")
    
    if link:
        # Record metric
        try:
            PURCHASE_LINK_GENERATED_TOTAL.labels(product=prod).inc()
        except Exception:
            pass
        
        duration = time.time() - start_time
        logger.info(
            "Tool.purchase.completed: product=%s has_link=True duration=%.3fs",
            prod, duration
        )
        
        base_msg = (
            f"Great! You can visit this link and enter the details to get your quote for {friendly} insurance: {link}\n\n"
            "Make sure to select your preferred plan and add-ons!"
        )
        return base_msg + _build_cross_sell_suggestion(prod)
    
    duration = time.time() - start_time
    logger.warning(
        "Tool.purchase.no_link: product=%s duration=%.3fs",
        prod, duration
    )
    
    return (
        f"I don't have a direct purchase link for the {friendly} plan right now. "
        "Please let me know if you'd like me to connect you with a specialist."
    )
