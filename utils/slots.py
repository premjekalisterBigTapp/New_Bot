from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from ..infrastructure import get_router_llm
from .products import PRODUCT_DEFINITIONS, get_product_names_str, get_product_aliases_prompt, get_all_aliases_map, SlotConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ProductDetection(BaseModel):
    product: Optional[str] = Field(
        description=f"The specific insurance product detected from the text ({get_product_names_str()}). None if unclear."
    )

def _normalize_product_key(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # First check if it matches a key directly
    cleaned = str(name).strip().lower()
    if cleaned in PRODUCT_DEFINITIONS:
        return cleaned
    
    # Check aliases
    alias_map = get_all_aliases_map()
    if cleaned in alias_map:
        return alias_map[cleaned]
        
    # Check if it matches a product name value
    for key, prod in PRODUCT_DEFINITIONS.items():
        if prod.name.lower() == cleaned:
            return key
            
    return None

async def _detect_product_llm_async(
    message: str,
    current_product: Optional[str] = None,
) -> Optional[str]:
    """Async detect product using LLM with context awareness."""
    if not message:
        return None

    product_list_str = get_product_names_str()
    aliases_prompt = get_product_aliases_prompt()

    sys_msg = (
        "You are an expert insurance product classifier for BigTapp. "
        f"Identify which product the user is interested in from: {product_list_str}. "
        "\n\nPRODUCT ALIASES (ALWAYS match these exactly):"
        f"\n{aliases_prompt}"
        "\n\nCONTEXT RULES:"
        f"\n- Current Topic: {current_product if current_product else 'None'}"
        "\n- IMPORTANT: If user mentions an alias exactly, map it to the corresponding product."
        "\n- Choice Protect360 (customizable): If the user asks for a customizable/tailored/flexible plan, "
        "or says they want to 'customize coverage', 'pay only for what I need', or an 'all-in-one plan', "
        "classify the product as ChoiceProtect360."
        "\n- IMPORTANT: If Current Topic is ChoiceProtect360, and the user mentions add-ons like "
        "'home contents', 'hospital income', or 'annual travel' WITHOUT explicitly naming another product "
        "(e.g., 'Home Protect360', 'Travel Protect360'), keep the product as ChoiceProtect360."
        "\n- Generic words alone (without specific product indicators) can stay on current topic."
        "\n- Switch products when user explicitly names a different product or its alias."
        "\n\nReturn null if not clearly about one of these."
    )

    try:
        structured = get_router_llm().with_structured_output(ProductDetection)
        result = await structured.ainvoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=message),
            ]
        )
        prod = _normalize_product_key(result.product)

        if not prod:
            logger.debug("Agentic.slots.detect_product_llm_async: no product detected")
            return None

        logger.debug("Agentic.slots.detect_product_llm_async: detected product=%s", prod)
        return prod
    except Exception as e:
        logger.warning("Agentic.slots.detect_product_llm_async failed: %s", e)
        return None

def _get_slot_value(slots: Dict[str, Any], slot_name: str) -> str:
    """Get a simple string value from a slot container."""

    slot_data = slots.get(slot_name)
    if isinstance(slot_data, dict):
        return str(slot_data.get("value") or "")
    return str(slot_data or "")


def _required_slots_for_product(product: Optional[str]) -> List[str]:
    """Return the list of slots required for a recommendation per product."""
    if not product:
        return []
    p = _normalize_product_key(product)
    if p and p in PRODUCT_DEFINITIONS:
        return PRODUCT_DEFINITIONS[p].required_slots
    return []


def _slot_descriptions(product: Optional[str]) -> Dict[str, str]:
    """Short descriptions per slot to help question generation."""
    p = _normalize_product_key(product)
    if p and p in PRODUCT_DEFINITIONS:
        return {k: v.description for k, v in PRODUCT_DEFINITIONS[p].slot_config.items()}
    return {}

def _slot_config(product: Optional[str]) -> Dict[str, SlotConfig]:
    """Return full slot configuration for a product."""
    p = _normalize_product_key(product)
    if p and p in PRODUCT_DEFINITIONS:
        return PRODUCT_DEFINITIONS[p].slot_config
    return {}
