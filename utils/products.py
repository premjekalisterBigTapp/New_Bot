from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ExceptionResponse(BaseModel):
    trigger: str
    response: str

class SlotConfig(BaseModel):
    description: str
    question: Optional[str] = None
    exceptions: List[ExceptionResponse] = []

class ProductDefinition(BaseModel):
    name: str
    aliases: List[str]
    required_slots: List[str]
    slot_config: Dict[str, SlotConfig]

# Central Source of Truth for Product Definitions
PRODUCT_DEFINITIONS: Dict[str, ProductDefinition] = {
    "choice": ProductDefinition(
        name="ChoiceProtect360",
        aliases=[
            "choice protect360",
            "choice protect 360",
            "choice protect",
            "choiceprotect360",
        ],
        # Choice Protect360 is a customizable bundle (base PA + optional add-ons).
        # We currently support it primarily for information, summary and comparison flows.
        # Recommendation flow can still be supported with zero required slots (like Car),
        # relying on templates/benefits context instead of slot-based personalization.
        required_slots=[],
        slot_config={},
    ),
    "travel": ProductDefinition(
        name="Travel",
        aliases=["travel protect360", "travel protect 360", "travel protect"],
        required_slots=["coverage_scope", "destination"],
        slot_config={
            "coverage_scope": SlotConfig(
                description="Coverage for self, family, a group of adults, or a group of families.",
                question="Who will be covered on this trip? Is it just yourself, your family, a group of adults, or a group of families?"
            ),
            "destination": SlotConfig(
                description="Country the user is travelling to.",
                question="Which country will you be traveling to?"
            ),
        }
    ),
    "maid": ProductDefinition(
        name="Maid",
        aliases=["maid protect360", "maid protect 360", "helper insurance"],
        required_slots=[
            "duration_of_insurance",
            "maid_country",
            "coverage_above_mom_minimum",
            "add_ons",
        ],
        slot_config={
            "duration_of_insurance": SlotConfig(description="Policy duration (14 or 26 months)."),
            "maid_country": SlotConfig(description="Helper's country of origin (country name only)."),
            "coverage_above_mom_minimum": SlotConfig(description="Whether user wants coverage beyond MOM minimum (yes/no). MOM minimum is $60,000 medical coverage and a $5,000 security bond."),
            "add_ons": SlotConfig(description="Whether the user wants optional add-on coverages (required/not_required)."),
        }
    ),
    "personalaccident": ProductDefinition(
        name="PersonalAccident",
        aliases=[
            "family protect360", "family protect 360", "family protect", 
            "pa insurance", "personal accident", "pa", "accident plan"
        ],
        required_slots=["coverage_scope", "risk_level", "desired_amount"],
        slot_config={
            "coverage_scope": SlotConfig(description="Coverage for yourself or your family."),
            "risk_level": SlotConfig(description="Occupational risk level: low, medium, or high. Low is mainly office work; high involves heavy physical or hazardous work."),
            "desired_amount": SlotConfig(description="Desired coverage amount between $500 and $3,500. Phrases like 'highest' can map to 3500; 'minimum' can map to 500."),
        }
    ),
    "home": ProductDefinition(
        name="Home",
        aliases=["home protect360", "home protect 360", "home insurance"],
        required_slots=["risk_concerns", "coverage_amount"],
        slot_config={
            "risk_concerns": SlotConfig(description="Specific worries such as fire, water damage, or theft (single, multiple, or 'all')."),
            "coverage_amount": SlotConfig(description="Estimated total value of renovations, contents and valuables (numeric amount)."),
        }
    ),
    "early": ProductDefinition(
        name="Early",
        aliases=[
            "early protect360", "early protect", "critical illness", 
            "early protect 360", "early protect 360 plus", "early protect360 plus"
        ],
        required_slots=["existing_cover", "dependants"],
        slot_config={
            "existing_cover": SlotConfig(
                description="Whether the user already has critical illness coverage (yes/no).",
                question="Do you already have any insurance that pays a lump sum if you’re diagnosed with a critical illness?",
                exceptions=[
                    ExceptionResponse(
                        trigger="medical insurance",
                        response="That’s excellent — medical insurance helps pay your hospital and treatment bills. Critical Illness insurance complements it by giving you a cash payout, which you can use for income replacement, rehabilitation, or other expenses that aren’t covered by hospital plans."
                    ),
                    ExceptionResponse(
                        trigger="hospitalisation insurance",
                        response="That’s excellent — medical insurance helps pay your hospital and treatment bills. Critical Illness insurance complements it by giving you a cash payout, which you can use for income replacement, rehabilitation, or other expenses that aren’t covered by hospital plans."
                    ),
                    ExceptionResponse(
                        trigger="young",
                        response="Serious illnesses can occur at any age. Buying CI protection earlier often means lower premiums and getting covered before any health issues arise."
                    ),
                    ExceptionResponse(
                        trigger="healthy",
                        response="Serious illnesses can occur at any age. Buying CI protection earlier often means lower premiums and getting covered before any health issues arise."
                    ),
                    ExceptionResponse(
                        trigger="never claim",
                        response="The main value is peace of mind — that you and your family are protected from unexpected financial stress. Some plans also offer partial refunds or conversion options at maturity if you’d like to consider them."
                    )
                ]
            ),
            "dependants": SlotConfig(
                description="Whether family members rely on the user's income or care (yes/no).",
                question="Do you have family members who rely on your income or care?"
            ),
        }
    ),
    "car": ProductDefinition(
        name="Car",
        aliases=["car protect360", "car protect 360", "motor insurance"],
        required_slots=[],
        slot_config={}
    ),
    "fraud": ProductDefinition(
        name="Fraud",
        aliases=["fraud protect360", "fraud protect", "scam protection"],
        required_slots=["fraud_intro_shown", "fraud_example_shown", "purchase_frequency", "scam_exp"],
        slot_config={
            "fraud_intro_shown": SlotConfig(
                description="Whether the intro about Fraud Protect360 has been shown (yes/no).",
                question="A great choice! Would you like to learn more about our Fraud Protect360 product?"
            ),
            "fraud_example_shown": SlotConfig(
                description="Whether a real-life example has been shown (yes/no).",
                question="Want to see how it protects you in real life situations?"
            ),
            "purchase_frequency": SlotConfig(
                description="How often the user shops online (daily, weekly, monthly).",
                question="How often do you shop online—daily, weekly, or monthly?"
            ),
            "scam_exp": SlotConfig(
                description="Whether the user has experienced or almost fallen for an online scam (yes/almost/no).",
                question="Have you ever experienced or nearly fallen for an online scam?"
            ),
        }
    ),
    "hospital": ProductDefinition(
        name="Hospital",
        aliases=["hospital protect360", "hospital protect", "hospital cash"],
        required_slots=["age", "support", "coverage"],
        slot_config={
            "age": SlotConfig(description="User age or age band (below 25, 25-35, 36-45, above 45)."),
            "support": SlotConfig(description="Whether the user supports anyone financially (yes/no)."),
            "coverage": SlotConfig(description="Desired daily hospital cash (100, 200, or 300)."),
        }
    ),
}

def get_product_names_str() -> str:
    """Returns a comma-separated string of capitalized product names."""
    return ", ".join(p.name for p in PRODUCT_DEFINITIONS.values())

def get_product_aliases_prompt() -> str:
    """Generates the aliases section for system prompts."""
    lines = []
    for key, prod in PRODUCT_DEFINITIONS.items():
        if prod.aliases:
            # Format: "- 'Alias 1', 'Alias 2' = ProductName"
            alias_str = ", ".join(f"'{a}'" for a in prod.aliases)
            lines.append(f"- {alias_str} = {prod.name}")
    return "\n".join(lines)

def get_all_aliases_map() -> Dict[str, str]:
    """Returns a flattened map of alias -> canonical_key."""
    mapping = {}
    for key, prod in PRODUCT_DEFINITIONS.items():
        for alias in prod.aliases:
            mapping[alias.lower()] = key
    return mapping


# =============================================================================
# POLICY PREFIX → PRODUCT MAPPING
# =============================================================================
# Internal BigTapp policy number prefixes (first 2 chars of policyNo).
# Used to resolve product names deterministically without LLM calls.

POLICY_PREFIX_MAP: Dict[str, str] = {
    "AC": "Accident Protect360",
    "CK": "Choice Protect360",
    "CY": "Fraud Protect360 Plus",
    "DY": "Maid Protect360 PRO",
    "ES": "Early Protect360 Plus",
    "FA": "Family Protect360",
    "HA": "Home ProtectLite",
    "HC": "Home Protect360",
    "HI": "Hospital Protect360",
    "MP": "Car Protect360",
    "TA": "Travel Protect360 (Annual)",
    "TB": "Travel Protect360",
}

# Reverse mapping: prefix → canonical product key (for routing to info agent)
_PREFIX_TO_CANONICAL_KEY: Dict[str, str] = {
    "AC": "personalaccident",
    "CK": "choice",
    "CY": "fraud",
    "DY": "maid",
    "ES": "early",
    "FA": "personalaccident",
    "HA": "home",
    "HC": "home",
    "HI": "hospital",
    "MP": "car",
    "TA": "travel",
    "TB": "travel",
}


def resolve_product_from_policy_no(policy_no: str) -> Optional[str]:
    """Resolve a human-readable product name from a policy number prefix.

    Returns the display name (e.g. 'Travel Protect360') or None if the prefix
    is unrecognized. This is deterministic — no LLM call.
    """
    if not policy_no or len(policy_no) < 2:
        return None
    prefix = policy_no[:2].upper()
    return POLICY_PREFIX_MAP.get(prefix)


def resolve_canonical_key_from_policy_no(policy_no: str) -> Optional[str]:
    """Resolve the canonical product key from a policy number prefix.

    Returns the key used in PRODUCT_DEFINITIONS (e.g. 'travel', 'fraud')
    for routing to the info agent. Returns None if unrecognized.
    """
    if not policy_no or len(policy_no) < 2:
        return None
    prefix = policy_no[:2].upper()
    return _PREFIX_TO_CANONICAL_KEY.get(prefix)
