"""
Benefits Tool for Agentic Chatbot
=================================

Retrieves raw benefits text for insurance products from benefits_raw.json.
This is a standalone copy with local config path.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.tools import tool
from ..config import CONFIG_DIR

logger = logging.getLogger(__name__)

# Cache for benefits data
_benefits_cache: Optional[Dict[str, Any]] = None


def _normalize_product_key(name: str) -> str:
    """Normalize product name to match benefits_raw.json keys."""
    base = (name or "").lower().strip()
    
    # Remove common suffixes
    for suffix in ("_benefits", "-benefits", " benefits"):
        if base.endswith(suffix):
            base = base.replace(suffix, "")
    
    # Keep only alphanumeric
    clean = "".join(ch for ch in base if ch.isalnum())
    
    # Alias mapping
    aliases = {
        "choiceprotect360": "choice",
        "choiceprotect": "choice",
        "choice360": "choice",
        "pa": "personalaccident",
        "personal": "personalaccident",
        "accident": "personalaccident",
        "familyprotect360": "personalaccident",
        "familyprotect": "personalaccident",
        "family": "personalaccident",
        "travelprotect360": "travel",
        "maidprotect360": "maid",
        "carprotect360": "car",
        "homeprotect360": "home",
        "earlyprotect360": "early",
        "earlyci": "early",
        "ci": "early",
        "fraudprotect360": "fraud",
        "hospitalprotect360": "hospital",
        "hospitalincome": "hospital",
    }
    
    return aliases.get(clean, clean)


def _load_benefits_cache() -> Dict[str, Any]:
    """Load benefits data from JSON file."""
    global _benefits_cache
    
    if _benefits_cache is not None:
        return _benefits_cache
    
    # Look in configured path first, then agentic/configs
    config_paths = []
    env_path = os.getenv("AGENTIC_BENEFITS_PATH")
    if env_path:
        config_paths.append(Path(env_path))
    config_paths.append(CONFIG_DIR / "benefits_raw.json")
    
    for json_path in config_paths:
        if json_path.exists():
            try:
                text = json_path.read_text(encoding="utf-8")
                _benefits_cache = json.loads(text) or {}
                logger.info("BenefitsTool: Loaded benefits_raw.json with %d products from %s", 
                           len(_benefits_cache), json_path)
                return _benefits_cache
            except Exception as e:
                logger.error("BenefitsTool: Failed to parse %s - %s", json_path, e)
    
    logger.error(
        "BenefitsTool: benefits_raw.json not found. Checked paths: %s",
        ", ".join(str(p) for p in config_paths),
    )
    _benefits_cache = {}
    return _benefits_cache


def get_product_benefits(product: str, tier: Optional[str] = None) -> str:
    """
    Get raw benefits text for a product.
    
    Args:
        product: Insurance product name
        tier: Optional tier name (not currently used, for future expansion)
        
    Returns:
        Concatenated benefits text or empty string if not found
    """
    data = _load_benefits_cache()
    key = _normalize_product_key(product)
    entry = data.get(key)
    
    if not entry:
        logger.warning("BenefitsTool: No benefits found for product='%s' (key=%s)", product, key)
        return ""
    
    docs = entry.get("docs") or []
    return "\n\n---\n\n".join([str(d or "").strip() for d in docs if str(d or "").strip()])


@tool
def benefits_tool(product: str, tier: Optional[str] = None) -> str:
    """
    Retrieve insurance benefits information for a product.
    
    Use this tool to get detailed coverage information, benefit amounts,
    and policy details for any BigTapp insurance product.
    
    Args:
        product: The insurance product (Travel, Maid, Car, Personal Accident, Home, Fraud, Hospital, Early)
        tier: Optional tier level (Classic, Gold, Platinum, etc.)
        
    Returns:
        Detailed benefits text for the product
    """
    return get_product_benefits(product, tier)


__all__ = ["benefits_tool", "get_product_benefits", "_normalize_product_key"]
