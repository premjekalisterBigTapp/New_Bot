"""
Policy Service Subgraph for BigTapp Agentic Chatbot
=================================================

Handles policy/claim status checks and customer updates with secure PII handling.
This subgraph ensures:
- Customer validation before any sensitive operations
- PII never sent to LLM (uses placeholders)
- LLM-based intent detection (not keyword-based)
- Proper error handling with user-friendly messages

Supported Actions:
- claim_status: Check status of claims
- policy_status: Check status of a specific policy
- update_email: Update email address
- update_mobile: Update mobile number
- update_address: Update mailing address
- update_payment: Update payment information
- update_insured_address: Update Home Protect insured address
"""

from __future__ import annotations

import logging
import json
import re
from typing import Any, Dict, List, Optional, Literal, Tuple
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..state import AgentState
from ..infrastructure import get_router_llm
from ..utils.pii_masker import get_pii_masker
from ..utils.memory import _get_last_user_message, _build_history_context_from_messages
from ..utils.products import resolve_product_from_policy_no
from ..integrations.bigtapp_api import get_bigtapp_api_client, BigTappApiClient

logger = logging.getLogger("agentic.service_flow")


# =============================================================================
# PYDANTIC MODELS FOR LLM-BASED CLASSIFICATION
# =============================================================================

class ServiceActionDetection(BaseModel):
    """LLM-based detection of what service action the user wants."""
    
    action: Literal[
        "claim_status",
        "policy_status", 
        "update_email",
        "update_mobile",
        "update_address",
        "update_payment",
        "update_insured_address",
        "unclear",
    ] = Field(
        description=(
            "The service action user wants to perform: "
            "'claim_status' - check claim status, "
            "'policy_status' - check specific policy status, "
            "'update_email' - change email address, "
            "'update_mobile' - change phone number, "
            "'update_address' - change mailing address, "
            "'update_payment' - change payment info, "
            "'update_insured_address' - change insured property address, "
            "'unclear' - cannot determine what user wants"
        )
    )
    
    policy_no: Optional[str] = Field(
        default=None,
        description="Policy number if user mentioned one (use placeholder like [POLICY_1] if masked)"
    )
    
    reason: str = Field(
        default="",
        description="Brief explanation of why this action was detected"
    )


class CredentialExtraction(BaseModel):
    """Extract validation credentials from user message (using placeholders).
    
    Also detects user intent to handle side questions, new intents, and exits.
    """
    
    # User intent classification (no extra LLM call - same extraction)
    user_intent: Literal[
        "provide_credential",   # Normal - user is providing requested info
        "side_question",        # User asking a question (hotline, why need this, etc.)
        "new_intent",           # User wants something else (recommend, compare, info, etc.)
        "cancel_exit",          # User wants to stop ("forget it", "cancel", "stop")
        "escalation",           # User explicitly wants human agent ("talk to a person")
    ] = Field(
        default="provide_credential",
        description=(
            "Classify user's primary intent: "
            "'provide_credential' - user is providing the requested info (name, email, mobile, etc.), "
            "'side_question' - user is asking a question instead ('What's your hotline?', 'Why do you need this info?'), "
            "'new_intent' - user wants to do something else entirely ('I want to compare plans', 'Recommend me travel insurance'), "
            "'cancel_exit' - user wants to stop ('Forget it', 'Cancel', 'Nevermind', 'I'll do this later'), "
            "'escalation' - user explicitly wants a human agent ('Talk to a person', 'Speak to an agent', 'Connect me to someone')"
        )
    )
    
    # If side_question, what's the question?
    detected_question: Optional[str] = Field(
        default=None,
        description="If user_intent is 'side_question', capture the question here (e.g., 'What is your hotline number?')"
    )
    
    # If new_intent, what intent?
    detected_intent: Optional[str] = Field(
        default=None,
        description="If user_intent is 'new_intent', identify the intent: 'recommend', 'compare', 'info', 'summary', 'purchase', 'greet', 'capabilities'"
    )
    
    # Credential extraction fields (existing)
    nric_placeholder: Optional[str] = Field(
        default=None,
        description="NRIC placeholder like [NRIC_1] if user provided NRIC"
    )
    
    first_name: Optional[str] = Field(
        default=None,
        description="User's first name if provided"
    )
    
    last_name: Optional[str] = Field(
        default=None,
        description="User's last name if provided"
    )
    
    mobile_placeholder: Optional[str] = Field(
        default=None,
        description="Mobile placeholder like [MOBILE_1] if user provided mobile"
    )
    
    policy_placeholder: Optional[str] = Field(
        default=None,
        description="Policy placeholder like [POLICY_1] if user provided policy number"
    )
    
    email_placeholder: Optional[str] = Field(
        default=None,
        description="Email placeholder like [EMAIL_1] if user provided email"
    )
    
    postal_placeholder: Optional[str] = Field(
        default=None,
        description="Postal code placeholder like [POSTAL_1] if user provided postal code"
    )


class PolicyServiceIntent(BaseModel):
    """
    Policy Service Orchestrator intent classification.
    
    This model classifies user intent WITHIN the service flow context,
    enabling intelligent routing without re-invoking the main orchestrator.
    
    The orchestrator maintains context about:
    - Current service action (policy_status, claim_status, etc.)
    - Credentials collected so far
    - Validation status
    - What question we last asked the user
    
    This allows it to make context-aware routing decisions.
    """
    
    intent: Literal[
        "provide_credential",   # User providing requested info (name, email, mobile)
        "side_question",        # User asking a question (hotline, data safety, why need this)
        "new_service_action",   # User wants different service action (refund, update, etc.)
        "exit_to_info",         # User asking product/general info (exit to info agent)
        "exit_to_recommend",    # User wants recommendation (exit to recommendation flow)
        "exit_to_compare",      # User wants plan comparison (exit to comparison agent)
        "cancel",               # User wants to stop the process
        "escalate",             # User wants human agent
    ] = Field(
        description=(
            "Classify the user's intent in the context of the ongoing service flow: "
            "'provide_credential' - user is answering our question with credential info, "
            "'side_question' - user asks a question but intends to continue the service flow, "
            "'new_service_action' - user wants a DIFFERENT service action (e.g., 'I want a refund instead'), "
            "'exit_to_info' - user asks about products/coverage (general info, not service-related), "
            "'exit_to_recommend' - user wants insurance recommendation, "
            "'exit_to_compare' - user wants to compare plans, "
            "'cancel' - user wants to stop ('forget it', 'cancel', 'never mind'), "
            "'escalate' - user explicitly wants a human agent"
        )
    )
    
    # For side_question: capture the question
    detected_question: Optional[str] = Field(
        default=None,
        description="If side_question, capture the exact question (e.g., 'What is your hotline?')"
    )
    
    # For new_service_action: what service do they want?
    detected_service_action: Optional[str] = Field(
        default=None,
        description="If new_service_action, identify the action: 'refund', 'claim_status', 'policy_status', 'update_email', etc."
    )
    
    # For exit intents: capture the product if mentioned
    detected_product: Optional[str] = Field(
        default=None,
        description="If user mentions a product, capture it: 'travel', 'car', 'home', 'maid', etc."
    )
    
    # Confidence/reasoning
    reason: str = Field(
        default="",
        description="Brief explanation of why this intent was detected"
    )


class ValidationRecoveryDecision(BaseModel):
    """Classification of user response during validation recovery."""

    decision: Literal[
        "retry",
        "reenter",
        "exit_info",
        "exit_recommend",
        "exit_compare",
        "unknown",
    ] = Field(
        description=(
            "User's intent after a validation system error: "
            "'retry' (try again now), "
            "'reenter' (re-enter details), "
            "'exit_info' (stop and ask for other info), "
            "'exit_recommend' (switch to recommendation), "
            "'exit_compare' (switch to comparison), "
            "'unknown' (unclear)."
        )
    )
    reason: str = Field(default="", description="Brief reason for decision")


# Cache structured output models
_action_detector = None
_credential_extractor = None
_service_intent_classifier = None
_validation_recovery_classifier = None


def _get_action_detector():
    """Get cached action detection model."""
    global _action_detector
    if _action_detector is None:
        _action_detector = get_router_llm().with_structured_output(ServiceActionDetection)
    return _action_detector


def _get_credential_extractor():
    """Get cached credential extraction model."""
    global _credential_extractor
    if _credential_extractor is None:
        _credential_extractor = get_router_llm().with_structured_output(CredentialExtraction)
    return _credential_extractor


def _get_service_intent_classifier():
    """Get cached service intent classification model for the Policy Service Orchestrator."""
    global _service_intent_classifier
    if _service_intent_classifier is None:
        _service_intent_classifier = get_router_llm().with_structured_output(PolicyServiceIntent)
    return _service_intent_classifier


def _get_validation_recovery_classifier():
    """Get cached classifier for validation recovery choices."""
    global _validation_recovery_classifier
    if _validation_recovery_classifier is None:
        _validation_recovery_classifier = get_router_llm().with_structured_output(ValidationRecoveryDecision)
    return _validation_recovery_classifier

# =============================================================================
# VALIDATION CREDENTIAL SLOTS
# =============================================================================

VALIDATION_SLOTS = {
    "first_name": {
        "question": "What is your first name as registered with us?",
        "placeholder_prefix": None,  # Not masked
    },
    "last_name": {
        "question": "What is your last name as registered with us?",
        "placeholder_prefix": None,  # Not masked
    },
    "email": {
        "question": "Please provide your registered email address.",
        "placeholder_prefix": "EMAIL",
    },
    "mobile": {
        "question": "Please provide your registered mobile number.",
        "placeholder_prefix": "MOBILE",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_date(date_str: Optional[str]) -> str:
    """Format ISO date string to user-friendly format."""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y")
    except Exception:
        return date_str.split("T")[0] if "T" in str(date_str) else str(date_str)


def _format_policy_list(policies: List[Dict], max_display: int = 10) -> str:
    """Format policy list for user display."""
    if not policies:
        return "You don't have any policies on record."
    
    # Group by status
    active = [p for p in policies if p.get("status", "").lower() in ("active", "pending new business")]
    lapsed = [p for p in policies if p.get("status", "").lower() == "lapsed"]
    other = [p for p in policies if p not in active and p not in lapsed]
    
    lines = ["Here are your policies:\n"]
    
    def format_policy(p):
        status_emoji = "✅" if p in active else "⏸️" if p in lapsed else "📋"
        end_date = _format_date(p.get("policyEndDate"))
        prod_name = _resolve_policy_product_name(p)
        return f"{status_emoji} *{p.get('policyNo', 'N/A')}* - {prod_name}\n   Status: {p.get('status', 'Unknown')} | Ends: {end_date}"
    
    if active:
        lines.append("*Active Policies:*")
        for p in active[:max_display]:
            lines.append(format_policy(p))
        lines.append("")
    
    if lapsed and len(active) < max_display:
        remaining = max_display - len(active)
        lines.append("*Lapsed Policies:*")
        for p in lapsed[:remaining]:
            lines.append(format_policy(p))
    
    total = len(policies)
    displayed = min(total, max_display)
    if total > displayed:
        lines.append(f"\n_Showing {displayed} of {total} policies._")
    
    return "\n".join(lines)


# =============================================================================
# POLICY LIST PAGINATION (policy selection UX)
# =============================================================================

_POLICY_PAGE_DEFAULT = 10
_POLICY_PAGE_MIN = 5
_POLICY_PAGE_MAX = 20  # keep WhatsApp replies readable


def _order_policies_for_display(policies: List[Dict]) -> List[Dict]:
    """Stable ordering: active/pending first, then other, then lapsed (keeps API order within groups)."""
    if not policies:
        return []
    active: List[Dict] = []
    other: List[Dict] = []
    lapsed: List[Dict] = []
    for p in policies:
        status = str(p.get("status") or "").lower().strip()
        if status in ("active", "pending new business"):
            active.append(p)
        elif status == "lapsed":
            lapsed.append(p)
        else:
            other.append(p)
    return active + other + lapsed


def _policy_page_state_keys(slot_name: str) -> Tuple[str, str]:
    # Keep pagination state scoped to the pending_slot that requested it
    return f"_{slot_name}_offset", f"_{slot_name}_page_size"


def _get_policy_page_state(service_slots: Dict[str, Any], slot_name: str) -> Tuple[int, int]:
    off_key, size_key = _policy_page_state_keys(slot_name)
    try:
        offset = int(service_slots.get(off_key, 0) or 0)
    except Exception:
        offset = 0
    try:
        page_size = int(service_slots.get(size_key, _POLICY_PAGE_DEFAULT) or _POLICY_PAGE_DEFAULT)
    except Exception:
        page_size = _POLICY_PAGE_DEFAULT

    if page_size < _POLICY_PAGE_MIN:
        page_size = _POLICY_PAGE_MIN
    if page_size > _POLICY_PAGE_MAX:
        page_size = _POLICY_PAGE_MAX
    if offset < 0:
        offset = 0
    return offset, page_size


def _set_policy_page_state(service_slots: Dict[str, Any], slot_name: str, offset: int, page_size: int) -> None:
    off_key, size_key = _policy_page_state_keys(slot_name)
    service_slots[off_key] = int(max(0, offset))
    service_slots[size_key] = int(max(_POLICY_PAGE_MIN, min(_POLICY_PAGE_MAX, page_size)))


def _clear_policy_page_state(service_slots: Dict[str, Any], slot_name: str) -> None:
    off_key, size_key = _policy_page_state_keys(slot_name)
    service_slots.pop(off_key, None)
    service_slots.pop(size_key, None)


def _extract_policy_no_from_user_msg(pii_mapping: Dict[str, str], user_msg: Optional[str]) -> Optional[str]:
    """Return policy number provided in THIS user message (placeholder-aware; safe against stale values)."""
    pol = _get_latest_value_from_user_message(pii_mapping, user_msg, "POLICY")
    if pol:
        return str(pol).strip().upper()
    # Fallback if masking is bypassed for some reason
    m = re.search(r"\b([A-Za-z]{2}\d{6})\b", user_msg or "")
    if m:
        return m.group(1).strip().upper()
    return None


def _classify_validation_recovery_choice(text: Optional[str]) -> str:
    """
    Heuristic classifier for validation recovery choices (no extra LLM call).
    Returns: retry | reenter | exit_info | exit_recommend | exit_compare | unknown
    """
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not t:
        return "unknown"

    # Numeric shortcuts
    if re.fullmatch(r"[1]\b|one\b", t) or "try again" in t or "retry" in t:
        return "retry"
    if re.fullmatch(r"[2]\b|two\b", t) or "re-enter" in t or "re enter" in t or "reenter" in t or "enter again" in t:
        return "reenter"

    # Flow switching cues
    if "recommend" in t or "recommendation" in t:
        return "exit_recommend"
    if "compare" in t or "comparison" in t:
        return "exit_compare"
    if any(k in t for k in ["other insurance", "other product", "other products", "benefit", "coverage", "info"]):
        return "exit_info"

    # Generic exit / frustration
    if any(k in t for k in ["leave", "stop", "cancel", "quit", "nevermind", "never mind", "forget it"]):
        return "exit_info"

    return "unknown"


async def _classify_validation_recovery_choice_llm(user_msg: Optional[str]) -> str:
    """LLM fallback for validation recovery choices."""
    msg = (user_msg or "").strip()
    if not msg:
        return "unknown"
    sys_prompt = (
        "You are classifying a user's reply after a system error in policy verification.\n"
        "The bot asked:\n"
        "1) Try again now\n"
        "2) Re-enter your details\n\n"
        "Classify the user's intent as one of:\n"
        "- retry (try again now)\n"
        "- reenter (re-enter details)\n"
        "- exit_info (stop and ask about other insurance/info)\n"
        "- exit_recommend (switch to recommendation)\n"
        "- exit_compare (switch to comparison)\n"
        "- unknown\n\n"
        "Use exit_info if the user wants to stop or says they want other insurance info."
    )
    try:
        classifier = _get_validation_recovery_classifier()
        result = await classifier.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"User reply: {msg}"),
            ]
        )
        decision = (result.decision or "unknown").strip().lower()
        logger.info(
            "ServiceFlow.validation_recovery.llm: decision=%s reason='%s'",
            decision,
            (result.reason or "")[:80],
        )
        return decision
    except Exception as e:
        logger.warning("ServiceFlow.validation_recovery.llm_failed: %s", e)
        return "unknown"

def _parse_policy_page_command(user_msg: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse navigation commands while the bot is asking the user to choose a policy.

    Supported (examples):
    - next / nxt / more / continue
    - next 10 / next10
    - prev / previous / back
    - first / start
    - last / end
    - '10' (interpreted as "next 10") to support casual user input
    """
    if not user_msg:
        return None
    raw = str(user_msg).strip().lower()
    if not raw:
        return None

    t = re.sub(r"[^a-z0-9\s]", " ", raw)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None

    # Numeric-only (e.g., "10") => treat as "next 10"
    m = re.fullmatch(r"\d{1,2}", t)
    if m:
        n = int(m.group(0))
        if n > 0:
            return {"cmd": "next", "page_size": n}

    # next / nxt / more / continue
    if re.search(r"\b(next|nxt|more|continue|cont)\b", t):
        # Optional page size override: "next 10", "next10"
        n = None
        m2 = re.search(r"\bnext(\d{1,2})\b", t)
        if m2:
            n = int(m2.group(1))
        else:
            m3 = re.search(r"\b(next|nxt|more|continue|cont)\s+(\d{1,2})\b", t)
            if m3:
                n = int(m3.group(2))
        return {"cmd": "next", "page_size": n}

    if re.search(r"\b(prev|previous|back)\b", t):
        return {"cmd": "prev", "page_size": None}

    if re.search(r"\b(first|start|begin)\b", t):
        return {"cmd": "first", "page_size": None}

    if re.search(r"\b(last|end)\b", t):
        return {"cmd": "last", "page_size": None}

    if re.search(r"\b(show all|all policies|all)\b", t):
        return {"cmd": "all", "page_size": None}

    return None


def _get_validation_creds_for_confirm(state: AgentState) -> Optional[Dict[str, str]]:
    """
    Build best-effort validation credentials for post-timeout confirmation.

    We prefer:
    - existing validated customer_data (already trusted)
    - fall back to service_slots if needed
    """
    customer_data = state.get("customer_data") or {}
    service_slots = state.get("service_slots") or {}

    first_name = (service_slots.get("first_name") or customer_data.get("givenName") or "").strip()
    last_name = (service_slots.get("last_name") or customer_data.get("surname") or "").strip()
    email = (service_slots.get("email") or customer_data.get("email") or "").strip()
    mobile = (service_slots.get("mobile") or customer_data.get("phone") or "").strip()

    if not all([first_name, last_name, email, mobile]):
        return None

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email.strip().lower() if email else email,
        "mobile": _normalize_mobile(mobile) or mobile,
    }


def _resolve_policy_product_name(policy: Dict[str, Any]) -> str:
    """Resolve product name: prefix mapping first, API productName as fallback."""
    policy_no = str(policy.get("policyNo", "") or "")
    resolved = resolve_product_from_policy_no(policy_no)
    if resolved:
        return resolved
    return str(policy.get("productName", "Unknown") or "Unknown").strip()


def _format_policy_line(p: Dict[str, Any]) -> str:
    policy_no = str(p.get("policyNo", "N/A") or "N/A")
    product_name = _resolve_policy_product_name(p)
    if product_name and product_name != "Unknown":
        return f"• *{policy_no}* - {product_name}"
    return f"• *{policy_no}*"


def _render_policy_page(
    policies: List[Dict[str, Any]],
    *,
    title: str,
    offset: int,
    page_size: int,
) -> str:
    total = len(policies or [])
    if total == 0:
        return "You don't have any policies on record."

    page_size = max(_POLICY_PAGE_MIN, min(_POLICY_PAGE_MAX, int(page_size or _POLICY_PAGE_DEFAULT)))
    if offset < 0:
        offset = 0
    if offset >= total:
        offset = max(((total - 1) // page_size) * page_size, 0)

    start = offset
    end = min(offset + page_size, total)

    lines: List[str] = [
        f"{title}\n\nShowing {start + 1}–{end} of {total}:\n",
    ]
    for p in (policies or [])[start:end]:
        lines.append(_format_policy_line(p))

    nav_bits: List[str] = []
    if end < total:
        nav_bits.append("next")
    if start > 0:
        nav_bits.append("prev")

    nav_hint = ""
    if nav_bits:
        nav_hint = f"\n\nType '{' / '.join(nav_bits)}' to navigate pages."

    return (
        "\n".join(lines)
        + "\n\nReply with the *policy number* (e.g., DY300318)."
        + nav_hint
    )


def _format_claim_list(claims: List[Dict]) -> str:
    """Format claim list for user display."""
    if not claims:
        return "You don't have any claims on record."
    
    lines = ["Here are your claims:\n"]
    
    for c in claims:
        status = c.get("status", "Unknown")
        status_emoji = "⏳" if status.lower() == "processing" else "✅" if status.lower() == "approved" else "❌" if status.lower() == "rejected" else "📋"
        prod_name = _resolve_policy_product_name(c)
        lines.append(f"{status_emoji} *{c.get('policyNo', 'N/A')}* ({prod_name}) - Status: {status}")
    
    return "\n".join(lines)


# =============================================================================
# INPUT VALIDATION FUNCTIONS (No PII sent to LLM - all local validation)
# =============================================================================

def _validate_nric(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Singapore NRIC/FIN format.
    Returns (is_valid, error_message).
    Error messages are generic and don't expose the actual value.
    """
    if not value:
        return False, None
    
    value = value.strip().upper()
    
    # Basic format check: S/T/F/G/M + 7 digits + letter
    nric_pattern = r'^[STFGM]\d{7}[A-Z]$'
    if not re.match(nric_pattern, value):
        # Provide helpful feedback without exposing the actual value
        if len(value) < 9:
            return False, "The NRIC/FIN seems too short. It should be 9 characters (e.g., S1234567A)."
        elif len(value) > 9:
            return False, "The NRIC/FIN seems too long. It should be 9 characters (e.g., S1234567A)."
        elif not value[0] in "STFGM":
            return False, "NRIC/FIN should start with S, T, F, G, or M."
        else:
            return False, "Please enter a valid NRIC/FIN in the format S1234567A."
    
    return True, None


def _validate_mobile(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Singapore mobile number format.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    # Clean the value
    cleaned = re.sub(r'[^\d+]', '', value)
    
    # Remove +65 / 65 prefix if present
    if cleaned.startswith('+65'):
        cleaned = cleaned[3:]
    elif cleaned.startswith('65') and len(cleaned) >= 10:
        cleaned = cleaned[2:]
    
    # Should be 8 digits starting with 6, 8, or 9
    if len(cleaned) != 8:
        return False, "Mobile number should be 8 digits (e.g., 91234567, +65 9123 4567, or 65 9123 4567)."
    
    if cleaned[0] not in '689':
        return False, "Singapore mobile numbers start with 6, 8, or 9."
    
    if not cleaned.isdigit():
        return False, "Mobile number should contain only digits."
    
    return True, None


def _normalize_mobile(value: Optional[str]) -> Optional[str]:
    """Normalize a Singapore phone number to 8-digit local format (no +65, no separators)."""
    if not value:
        return value

    cleaned = re.sub(r"[^\d+]", "", str(value))
    if cleaned.startswith("+65"):
        cleaned = cleaned[3:]
    elif cleaned.startswith("65") and len(cleaned) >= 10:
        # Handle "65XXXXXXXX" variants
        cleaned = cleaned[2:]
    return cleaned


def _validate_email(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email format using the email-validator library.
    Returns (is_valid, error_message).
    
    Uses intelligent validation that catches:
    - Consecutive dots (..)
    - Missing @ or domain
    - Invalid characters
    - Too long emails
    - Invalid domain formats
    """
    if not value or not value.strip():
        return False, None
    
    email = value.strip()
    
    try:
        from email_validator import validate_email as email_lib_validate, EmailNotValidError
        # check_deliverability=False for speed (skip DNS lookup)
        email_lib_validate(email, check_deliverability=False)
        return True, None
    except ImportError:
        # Fallback if library not installed - use basic validation
        logger.warning("email-validator library not installed, using basic validation")
        if '@' not in email or '.' not in email.split('@')[-1]:
            return False, "Please provide a valid email address (e.g., user@example.com)."
        return True, None
    except Exception as e:
        # Parse the error message for user-friendly feedback
        error_msg = str(e)
        if "The part after the @-sign" in error_msg:
            return False, "The domain part of your email address is invalid."
        elif "The part before the @-sign" in error_msg:
            return False, "The local part of your email address (before @) is invalid."
        elif "too long" in error_msg.lower():
            return False, "Email address is too long (maximum 254 characters)."
        elif "consecutive" in error_msg.lower() or ".." in email:
            return False, "Email address cannot contain consecutive dots (..)."
        else:
            return False, f"Invalid email format. Please check and try again."


def _mask_mobile_for_display(value: Optional[str]) -> str:
    """Return a safe display string (last 4 digits only)."""
    if not value:
        return "••••"
    digits = re.sub(r"[^\d]", "", str(value))
    return f"••••{digits[-4:]}" if len(digits) >= 4 else "••••"


def _get_whatsapp_mobile_candidate(state: AgentState) -> Optional[str]:
    """
    Extract a valid SG mobile candidate from WhatsApp channel metadata.
    Returns 8-digit local format if valid, otherwise None.
    """
    channel = (state.get("channel") or "").lower()
    if channel != "whatsapp":
        return None
    raw = (state.get("channel_user_id") or "").strip()
    if not raw:
        return None
    normalized = _normalize_mobile(raw)
    is_valid, _ = _validate_mobile(normalized or "")
    if not is_valid:
        return None
    return normalized


def _classify_yes_no_heuristic(text: Optional[str]) -> str:
    """
    Lightweight yes/no classifier to avoid extra LLM calls.
    Returns: "yes", "no", or "unclear".
    """
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not t:
        return "unclear"
    yes_triggers = {
        "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "alright",
        "correct", "that's right", "use it", "use this", "go ahead", "confirm",
    }
    no_triggers = {
        "no", "nope", "nah", "not", "different", "change", "edit",
        "use different", "another", "other number", "no thanks",
    }
    if any(t == k or t.startswith(k + " ") for k in yes_triggers):
        return "yes"
    if any(t == k or t.startswith(k + " ") for k in no_triggers):
        return "no"
    return "unclear"


def _validate_policy_no(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate BigTapp policy number format.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    value = value.strip().upper()
    
    # BigTapp format: 2 letters + 6 digits (e.g., DY300318, HC123456)
    policy_pattern = r'^[A-Z]{2}\d{6}$'
    if not re.match(policy_pattern, value):
        if len(value) < 8:
            return False, "Policy number seems too short. It should be 8 characters (e.g., DY300318)."
        elif len(value) > 8:
            return False, "Policy number seems too long. It should be 8 characters (e.g., DY300318)."
        else:
            return False, "Policy number should be 2 letters followed by 6 digits (e.g., DY300318)."
    
    return True, None


def _validate_name(value: str, field_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate name field.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    value = value.strip()
    
    if len(value) < 1:
        return False, f"Please enter your {field_name}."
    
    # Check for obviously invalid characters
    if any(c.isdigit() for c in value):
        return False, f"Your {field_name} should not contain numbers."
    
    # Check for special characters (allow hyphens, apostrophes, spaces for names like O'Brien, Mary-Jane)
    if re.search(r'[^a-zA-Z\s\'\-]', value):
        return False, f"Your {field_name} contains invalid characters."
    
    return True, None


def _normalize_unit_no(value: Optional[str]) -> Optional[str]:
    """Normalize unit number into a stable 'floor-unit' form (e.g., '#01-01' -> '01-01')."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return ""
    v = v.lstrip("#").strip()
    v = v.replace("/", "-")
    v = re.sub(r"\s+", "", v)
    return v


def _validate_unit_no(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate unit number format for BigTapp address updates.

    Backend frequently expects a 'floor-unit' format (e.g., 01-01, 18-18).
    """
    if value is None:
        return False, None
    v = _normalize_unit_no(value)
    if v is None:
        return False, None
    if not v:
        return False, "Please enter your unit number (e.g., #01-01)."
    if re.match(r"^\d{1,3}-\d{1,3}$", v):
        return True, None
    return False, "Unit number should be in the format #01-01 (e.g., #18-18)."


def _get_latest_from_pii_mapping(pii_mapping: Dict[str, str], prefix: str) -> Optional[str]:
    """
    Get the LATEST value from pii_mapping for a given placeholder prefix.
    
    pii_mapping contains entries like [POLICY_1], [POLICY_2], etc.
    This function finds the highest numbered placeholder (most recent user input)
    and returns its value.
    
    Args:
        pii_mapping: Dict mapping placeholders to original values
        prefix: The prefix to look for, e.g., "[POLICY_", "[EMAIL_", "[POSTAL_"
        
    Returns:
        The original value for the highest numbered placeholder, or None if not found
    """
    latest_placeholder = None
    latest_num = -1
    
    for placeholder in pii_mapping.keys():
        if placeholder.startswith(prefix):
            try:
                # Extract number from placeholder like [POLICY_42] -> 42
                num_str = placeholder.replace(prefix, "").replace("]", "")
                num = int(num_str)
                if num > latest_num:
                    latest_num = num
                    latest_placeholder = placeholder
            except ValueError:
                pass
    
    if latest_placeholder:
        return pii_mapping[latest_placeholder]
    return None


def _get_latest_placeholder_from_text(text: Optional[str], placeholder_prefix: str) -> Optional[str]:
    """
    Extract the latest placeholder (highest numeric suffix) of a given type from the text.

    Example:
      text="my email is [EMAIL_2] and alt [EMAIL_9]" + prefix="EMAIL" -> "[EMAIL_9]"
    """
    if not text:
        return None

    # Placeholders are always in the form: [PREFIX_123]
    pattern = rf"\[{re.escape(placeholder_prefix)}_(\d+)\]"
    nums = re.findall(pattern, text)
    if not nums:
        return None

    try:
        latest_num = max(int(n) for n in nums)
    except ValueError:
        return None

    return f"[{placeholder_prefix}_{latest_num}]"


def _get_latest_value_from_user_message(
    pii_mapping: Dict[str, str],
    user_msg: Optional[str],
    placeholder_prefix: str,
) -> Optional[str]:
    """
    Get a PII value ONLY if the placeholder appears in the CURRENT user message.

    This prevents stale session PII from being reused when the user didn't
    provide the value in this turn (common cause of "correct value then wrong value"
    behaviors during updates).
    """
    ph = _get_latest_placeholder_from_text(user_msg, placeholder_prefix)
    if ph and ph in pii_mapping:
        return pii_mapping[ph]
    return None


# =============================================================================
# POLICY SERVICE ORCHESTRATOR
# =============================================================================
# 
# This is the intelligent entry point for the service subgraph. It:
# 1. Runs on EVERY turn while in service flow
# 2. Has full context about the ongoing flow (action, slots, validation status)
# 3. Classifies user intent in the service context
# 4. Routes to appropriate handler or exits to main orchestrator
#
# This prevents the main orchestrator from re-classifying messages like
# "How can I do a refund?" as policy_service when the user actually wants
# to exit and get info.
# =============================================================================

async def _policy_service_orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Policy Service Orchestrator - Intelligent routing within service flow.
    
    Runs on every turn in service flow. Decides whether to:
    - Continue with credential collection
    - Answer a side question
    - Switch to a different service action
    - Exit to another agent (info, recommend, compare)
    - Cancel the flow
    - Escalate to human agent
    """
    import time
    start_time = time.time()
    
    messages = list(state.get("messages", []) or [])
    service_action = state.get("service_action")
    service_slots = dict(state.get("service_slots") or {})
    pending_slot = state.get("service_pending_slot")
    customer_validated = state.get("customer_validated", False)
    
    # Get last user message
    user_msg = _get_last_user_message(messages)
    
    # =========================================================================
    # CASE 0: Validation recovery after a system error
    # =========================================================================
    if pending_slot == "validation_recovery":
        choice = _classify_validation_recovery_choice(user_msg)
        if choice == "unknown":
            choice = await _classify_validation_recovery_choice_llm(user_msg)
        logger.info("ServiceFlow.validation_recovery.choice: raw='%s' -> %s", (user_msg or "")[:50], choice)

        if choice == "retry":
            # Retry validation with the same details we already have.
            # IMPORTANT: Do NOT run credential extraction on this turn (user typed "1"),
            # otherwise we risk overwriting existing slots with junk and forcing re-entry.
            return {
                "service_pending_slot": None,
                "_orchestrator_route": "detect_action",
            }
        if choice == "reenter":
            # Re-enter details (keep the original service action)
            return {
                "service_slots": {},
                "service_pending_slot": None,
                "_orchestrator_route": "collect_credentials",
            }
        if choice in ("exit_info", "exit_recommend", "exit_compare"):
            exit_intent = {
                "exit_info": "info",
                "exit_recommend": "recommend",
                "exit_compare": "compare",
            }.get(choice, "info")
            logger.info("ServiceFlow.validation_recovery.exit: intent=%s", exit_intent)
            return {
                "service_action": None,
                "service_pending_slot": None,
                "service_exit_intent": exit_intent,
                "service_exit_query": user_msg,
                "messages": [AIMessage(content="Sure — let’s switch. What would you like to know?")],
                "_orchestrator_route": None,  # Exit subgraph
            }

        # Ask again if user didn't choose a valid option
        return {
            "service_pending_slot": "validation_recovery",
            "messages": [AIMessage(content="Please reply with 1 (try again) or 2 (re-enter details). If you'd like to stop, just say what you want to do next (e.g., 'tell me about other insurance').")],
            "_orchestrator_route": None,
        }

    # =========================================================================
    # CASE 0B: Action recovery after a service action error (e.g. update_* failed)
    #
    # Without this, we might ask "Would you like to try again?" but clear
    # service_action/service_pending_slot, causing the next "yes" to be routed
    # as a new top-level intent (often misclassified as recommendation).
    # =========================================================================
    if pending_slot == "action_recovery":
        raw = (user_msg or "").strip().lower()
        m = re.match(r"^\s*(\d)\s*$", raw)
        digit = m.group(1) if m else None

        kind = str(service_slots.get("_action_recovery_kind") or "invalid").strip().lower()
        action = state.get("service_action")

        def _clear_recovery_meta(slots: Dict[str, Any]) -> Dict[str, Any]:
            slots = dict(slots or {})
            slots.pop("_action_recovery_kind", None)
            slots.pop("_action_recovery_last_error", None)
            slots.pop("_action_recovery_request_id", None)
            return slots

        def _reset_inputs_for_action(slots: Dict[str, Any], action_name: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
            """
            Clear only the fields needed to re-enter inputs for the failed action.
            Returns (new_slots, first_pending_slot)
            """
            slots = dict(slots or {})
            # Preserve any pagination meta for policy pickers; we will clear action meta separately.
            if action_name == "update_email":
                return slots, "new_email"
            if action_name == "update_mobile":
                return slots, "new_mobile"
            if action_name == "update_address":
                for k in ("postal_code", "postal_validated", "house_no", "street_name", "unit_no", "building_name"):
                    slots.pop(k, None)
                slots.pop("_postal_suggest_building_name", None)
                return slots, "postal_code"
            if action_name == "update_insured_address":
                # Keep selected policy but re-enter address fields
                for k in ("postal_code", "postal_validated", "house_no", "street_name", "unit_no", "building_name"):
                    slots.pop(k, None)
                slots.pop("_postal_suggest_building_name", None)
                return slots, "postal_code"
            if action_name == "update_payment":
                # Keep selected policy but re-enter payment fields
                for k in ("card_type", "card_no", "card_expiry"):
                    slots.pop(k, None)
                # If no policy selected yet, start there.
                if not slots.get("payment_policy_no"):
                    return slots, "payment_policy_no"
                # Card type is asked first, then card number
                return slots, "card_type"
            return slots, None

        # Interpret user choice. "yes" is mapped based on kind:
        # - system -> retry now
        # - invalid -> re-enter details
        yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure"}
        no_words = {"no", "n", "nope", "cancel", "stop", "never mind", "nevermind"}

        if kind == "system":
            retry_selected = (digit == "1") or ("retry" in raw) or ("try again" in raw) or (raw in yes_words)
            reenter_selected = (digit == "2") or ("re-enter" in raw) or ("reenter" in raw) or ("change" in raw) or ("edit" in raw)
            cancel_selected = (digit == "3") or (raw in no_words)
        else:
            # invalid / validation-style: default "yes" -> re-enter
            reenter_selected = (digit == "1") or ("re-enter" in raw) or ("reenter" in raw) or (raw in yes_words)
            retry_selected = (digit == "2") or ("retry" in raw) or ("try again" in raw) or ("again" == raw)
            cancel_selected = (digit == "3") or (raw in no_words)

        if cancel_selected:
            slots = _clear_recovery_meta(service_slots)
            # Keep validation, but reset current action state and show the action menu again.
            menu = _service_ask_action(state)
            return {
                **menu,
                "service_slots": slots,
                "service_pending_slot": "service_action_choice",
                "messages": [AIMessage(content="No problem — I’ve cancelled that.\n\n" + menu["messages"][0].content)],
                "_orchestrator_route": None,
            }

        if retry_selected:
            slots = _clear_recovery_meta(service_slots)
            return {
                "service_slots": slots,
                "service_pending_slot": None,
                "_orchestrator_route": "execute_action",
            }

        if reenter_selected:
            slots = _clear_recovery_meta(service_slots)
            slots, first_slot = _reset_inputs_for_action(slots, action)
            return {
                "service_slots": slots,
                "service_pending_slot": first_slot,
                "_orchestrator_route": "execute_action",
            }

        # Ask again if user didn't choose a valid option
        if kind == "system":
            prompt = "Please reply with 1 (try again now), 2 (re-enter details), or 3 (cancel)."
        else:
            prompt = "Please reply with 1 (re-enter details), 2 (try again), or 3 (cancel)."
        return {
            "service_pending_slot": "action_recovery",
            "messages": [AIMessage(content=prompt)],
            "_orchestrator_route": None,
        }
    
    # =========================================================================
    # CASE 1: No service_action yet OR we're waiting for an action choice.
    # Instead of blindly running action detection (which returns "unclear" for
    # out-of-scope requests like refund/how-to questions), run the service intent
    # classifier so we can:
    # - exit_to_info (e.g. refund process)
    # - exit_to_recommend / exit_to_compare
    # - cancel / escalate
    # - or start a new_service_action cleanly
    # =========================================================================
    if not service_action or pending_slot == "service_action_choice":
        # If user replied with a numeric menu choice, let detect_action handle it.
        if pending_slot == "service_action_choice":
            m = re.match(r"^\s*(\d)\s*[\).\]]?\s*$", user_msg or "")
            if m:
                logger.info("ServiceOrchestrator.action_choice_numeric: %s -> detect_action", m.group(1))
                return {"_orchestrator_route": "detect_action"}

        try:
            classifier = _get_service_intent_classifier()
            slots_collected = [k for k, v in service_slots.items() if v]
            current_asking = pending_slot or "service action"

            sys_prompt = f"""You are the Policy Service Orchestrator for an insurance chatbot.

CURRENT CONTEXT:
- Service action in progress: {service_action or 'None'}
- Credentials collected so far: {slots_collected or 'none'}
- Currently asking user for: {current_asking}
- Customer validated: {customer_validated}

The user is in an active policy service experience. Classify their intent:

INTENT DEFINITIONS:
1. provide_credential: User is answering our question with requested credential info.
2. side_question: User asks a question but intends to continue the current process.
3. new_service_action: User wants to perform a service operation (claim status, policy status, updates).
4. exit_to_info: User is asking for information/how-to about a process (e.g. refunds, hotline, requirements).
5. exit_to_recommend: User wants an insurance recommendation.
6. exit_to_compare: User wants plan comparison.
7. cancel: User wants to stop the current process.
8. escalate: User explicitly wants a human agent.

CRITICAL RULES:
- Refund is NOT an in-chat service action here. Any refund / cancellation / surrender HOW-TO questions → exit_to_info.
  Example: "Leave it. How can I get a refund?" -> exit_to_info
- If the user asks "how do I / how can I" for a SUPPORTED service operation, treat it as new_service_action
  (they likely want to do it here): claim_status, policy_status, update_email, update_mobile, update_address,
  update_payment, update_insured_address.
- If the user asks for a recommendation while we are asking for a service action, choose exit_to_recommend.
- If the user asks to compare plans, choose exit_to_compare.
- If the user replies with a short name/number/ID when we asked for credentials, choose provide_credential.
- When in doubt between exit intents, prefer exit_to_info.
"""

            result = await classifier.ainvoke(
                [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=f"User message: {user_msg}"),
                ]
            )

            logger.info(
                "ServiceOrchestrator.classify_no_action: intent=%s reason='%s'",
                result.intent,
                (result.reason or "")[:80],
            )

            # Route based on classification (reuse the existing handling below by returning updates)
            # Exit intents:
            if result.intent == "exit_to_info":
                return {
                    "phase": None,
                    "service_action": None,
                    "service_pending_slot": None,
                    "service_exit_intent": "info",
                    "service_exit_query": user_msg,
                    "messages": [AIMessage(content="Sure — I’ll help with that information.")],
                    "_orchestrator_route": None,
                }

            if result.intent == "exit_to_recommend":
                return {
                    "phase": None,
                    "service_action": None,
                    "service_pending_slot": None,
                    "service_exit_intent": "recommend",
                    "service_exit_query": user_msg,
                    "product": result.detected_product or state.get("product"),
                    "messages": [AIMessage(content="Sure — I can help with a recommendation.")],
                    "_orchestrator_route": None,
                }

            if result.intent == "exit_to_compare":
                return {
                    "phase": None,
                    "service_action": None,
                    "service_pending_slot": None,
                    "service_exit_intent": "compare",
                    "service_exit_query": user_msg,
                    "messages": [AIMessage(content="Sure — I can help you compare plans.")],
                    "_orchestrator_route": None,
                }

            if result.intent == "cancel":
                logger.info("ServiceOrchestrator.cancel (no_action path)")
                try:
                    session_id = config["configurable"].get("thread_id")
                    if session_id:
                        get_pii_masker().clear_session(session_id)
                except Exception as e:
                    logger.warning("ServiceOrchestrator.cancel: failed to clear PII session: %s", e)

                return {
                    "phase": None,
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "customer_validated": False,
                    "pii_mapping": {},
                    "messages": [AIMessage(content="No problem — I’ve cancelled that. What would you like to do next?")],
                    "_orchestrator_route": None,
                }

            if result.intent == "escalate":
                logger.info("ServiceOrchestrator.escalate (no_action path)")
                return {
                    "phase": None,
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "customer_validated": False,
                    "live_agent_requested": True,
                    "messages": [AIMessage(content="I understand — I’ll connect you to a customer service representative.")],
                    "_orchestrator_route": None,
                }

            if result.intent == "side_question":
                # Answer and then re-show the action menu (since we don't have a service_action yet)
                question = result.detected_question or user_msg
                logger.info("ServiceOrchestrator.side_question(no_action): '%s'", question)
                from ..tools.info import _info_tool_async
                try:
                    answer, _sources = await _info_tool_async(
                        product=None,
                        question=question,
                    )
                    if not answer or answer.strip() == "":
                        answer = "I’m not sure about that specific question right now."
                except Exception as e:
                    logger.warning("ServiceOrchestrator.side_question(no_action) failed: %s", e)
                    answer = "I apologize — I couldn’t retrieve that information right now."

                action_menu = _service_ask_action(state)
                # Prepend answer then show the menu again
                return {
                    "messages": [AIMessage(content=f"{answer}\n\n{action_menu['messages'][0].content}")],
                    "service_action": None,
                    "service_pending_slot": "service_action_choice",
                    "_orchestrator_route": None,
                }

            if result.intent == "new_service_action":
                detected_action = (result.detected_service_action or "").strip().lower()
                if detected_action:
                    # Refund is info, not a service action
                    if detected_action == "refund":
                        return {
                            "phase": None,
                            "service_action": None,
                            "service_pending_slot": None,
                            "service_exit_intent": "info",
                            "service_exit_query": user_msg,
                            "messages": [AIMessage(content="Sure — I’ll share the refund information.")],
                            "_orchestrator_route": None,
                        }

                    action_map = {
                        "claim": "claim_status",
                        "claim_status": "claim_status",
                        "policy": "policy_status",
                        "policy_status": "policy_status",
                        "update_email": "update_email",
                        "update_mobile": "update_mobile",
                        "email": "update_email",
                        "mobile": "update_mobile",
                        "phone": "update_mobile",
                        "update_address": "update_address",
                        "update_payment": "update_payment",
                        "update_insured_address": "update_insured_address",
                    }
                    normalized_action = action_map.get(detected_action, detected_action)

                    # If user included a specific policy placeholder, capture it for claim/policy status filtering.
                    pii_mapping = state.get("pii_mapping") or {}
                    requested_policy = _get_latest_value_from_user_message(pii_mapping, user_msg, "POLICY")
                    updates: Dict[str, Any] = {
                        "service_action": normalized_action,
                        "service_pending_slot": None,
                        "_orchestrator_route": "collect_credentials" if not customer_validated else "execute_action",
                    }
                    ss: Dict[str, Any] = {}
                    if requested_policy:
                        ss["requested_policy_no"] = str(requested_policy).strip().upper()
                    if normalized_action == "policy_status":
                        ss["_original_question"] = user_msg
                    updates["service_slots"] = ss

                    return updates

                # If classifier couldn't specify which action, fall back to action detector.
                return {"_orchestrator_route": "detect_action"}

            # If user provided credentials but we still don't know the action, ask for the action.
            if result.intent == "provide_credential":
                return {**_service_ask_action(state), "_orchestrator_route": None}

        except Exception as e:
            logger.warning("ServiceOrchestrator.no_action_classify_failed: %s", e)
            return {"_orchestrator_route": "detect_action"}

        # Default fallback
        return {"_orchestrator_route": "detect_action"}
    
    # =========================================================================
    # CASE 2: Already validated - check for exit intent, then execute action
    # =========================================================================
    if customer_validated:
        # Before routing to execute_action, check if user wants to cancel/exit.
        # This handles "Leave it", "Cancel", "Forget it", etc. during update prompts
        # (e.g. when asking for new_email or new_mobile).
        # IMPORTANT: We keep customer validation intact so they don't re-authenticate.
        if user_msg:
            # =================================================================
            # CONFIRMATION LOOP: If we already asked "Are you sure?", handle
            # the user's yes/no response before checking for new exit phrases.
            # =================================================================
            if service_slots.get("_confirm_cancel"):
                user_lower_confirm = (user_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct", "right"}
                _no_words = {"no", "n", "nope", "nah", "not really", "go back", "continue", "back", "resume"}
                if user_lower_confirm in _yes_words:
                    logger.info(
                        "ServiceOrchestrator.cancel_confirmed: action=%s (keeping validation)",
                        service_action,
                    )
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — I've cancelled that.\n\n" + menu["messages"][0].content
                        )],
                        "_orchestrator_route": None,
                    }
                else:
                    # User said no or something else → go back to the update
                    logger.info(
                        "ServiceOrchestrator.cancel_declined: action=%s, resuming pending=%s",
                        service_action, service_slots.get("_cancel_return_slot"),
                    )
                    return_slot = service_slots.get("_cancel_return_slot", pending_slot)
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_cancel", None)
                    service_slots.pop("_cancel_return_slot", None)
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": return_slot,
                        "_orchestrator_route": "execute_action",
                    }

            # --- Exact-match exit phrases (short/single-word safe) ---
            _exit_phrases_exact = {
                "leave it", "cancel", "forget it", "never mind", "nevermind",
                "stop", "no thanks", "no thank you", "skip", "don't bother",
                "i changed my mind", "changed my mind", "quit", "exit",
                "that's okay", "thats okay", "its fine", "it's fine",
                "not now", "later", "maybe later", "i'll do this later",
                "nah", "nvm", "no need", "no its ok", "no it's ok",
                "thanks", "thank you", "ok thanks", "okay thanks",
                "no change", "im good", "i'm good", "all good",
                "that's all", "thats all", "done", "no thanks",
                "not required", "dont want to", "don't want to",
                "no its fine", "no it's fine",
            }
            # --- Substring-match phrases (longer, safe to match inside longer messages) ---
            _exit_phrases_substring = [
                "no need to change", "don't need to change", "dont need to change",
                "i don't want to change", "i dont want to change",
                "no change needed", "don't want to update", "dont want to update",
                "i don't need to", "i dont need to",
                "not needed", "no longer needed", "no longer required",
                "leave it as it is", "keep it as it is", "let it be",
                "i'll do this later", "ill do this later",
                "changed my mind", "i changed my mind",
                "no need for now", "not for now",
            ]
            user_lower = (user_msg or "").strip().lower()
            # Remove placeholders before matching (e.g. [EMAIL_1] should not interfere)
            user_clean = re.sub(r"\[[A-Z]+_\d+\]", "", user_lower).strip()
            
            _is_exit = (
                user_clean in _exit_phrases_exact
                or any(phrase in user_clean for phrase in _exit_phrases_substring)
            )
            if _is_exit:
                logger.info(
                    "ServiceOrchestrator.cancel_request: user='%s' action=%s pending=%s -> asking confirmation",
                    user_clean[:30], service_action, pending_slot,
                )
                # Don't cancel yet – ask for confirmation first
                service_slots = dict(service_slots)
                service_slots["_confirm_cancel"] = True
                service_slots["_cancel_return_slot"] = pending_slot
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_cancel",
                    "messages": [AIMessage(
                        content="Are you sure you want to cancel this operation? (Yes / No)"
                    )],
                    "_orchestrator_route": None,
                }
        
        logger.debug("ServiceOrchestrator: already validated, routing to execute_action")
        return {"_orchestrator_route": "execute_action"}
    
    # =========================================================================
    # CASE 3: First credential collection - no pending slot yet
    # This happens right after action detection, before we've asked for anything
    # =========================================================================
    if not pending_slot and not service_slots:
        # Check if there's a previous AI message asking for credentials
        has_asked_for_credentials = False
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = str(getattr(msg, "content", "") or "").lower()
                if "first name" in content or "verify" in content or "email" in content:
                    has_asked_for_credentials = True
                    break
        
        if not has_asked_for_credentials:
            logger.debug("ServiceOrchestrator: first credential collection, routing to collect_credentials")
            return {"_orchestrator_route": "collect_credentials"}
    
    # =========================================================================
    # CASE 3B: Emoji/symbol-only input during credential collection
    # Route directly to collect_credentials so validation can reject it cleanly
    # =========================================================================
    _cred_slots = {"first_name", "last_name", "email", "mobile"}
    if pending_slot in _cred_slots and user_msg:
        has_alnum = any(c.isalnum() for c in user_msg.strip())
        if not has_alnum:
            logger.info("ServiceOrchestrator: emoji/symbol-only input for %s, routing to collect_credentials", pending_slot)
            return {"_orchestrator_route": "collect_credentials"}

    # =========================================================================
    # CASE 4: Classify user intent in service context
    # =========================================================================
    if not user_msg:
        logger.debug("ServiceOrchestrator: no user message, routing to collect_credentials")
        return {"_orchestrator_route": "collect_credentials"}
    
    try:
        classifier = _get_service_intent_classifier()
        
        # Build context-aware prompt
        slots_collected = [k for k, v in service_slots.items() if v]
        current_asking = pending_slot or "initial credentials"
        
        sys_prompt = f"""You are the Policy Service Orchestrator for an insurance company chatbot.

CURRENT CONTEXT:
- Service action in progress: {service_action}
- Credentials collected so far: {slots_collected or 'none'}
- Currently asking user for: {current_asking}
- Customer validated: {customer_validated}

The user is in an active service flow. Classify their intent:

INTENT DEFINITIONS:

1. provide_credential: User is answering our question with the requested info
   - Short answers like "WL", "TIO", "john@example.com", "81234567" are almost always credentials
   - Names, emails, and phone numbers should be classified as provide_credential
   
2. side_question: User asks a question but intends to CONTINUE the service flow
   - Examples: "Is there a hotline?", "Why do you need my email?", "Is my data safe?"
   - These are questions about the process, not requests to do something else
   
3. new_service_action: User wants to do a DIFFERENT service operation
   - Examples: "I want a refund instead", "Can you update my email?", "Check my claim status"
   - They still want service operations, just a different one
   
4. exit_to_info: User is asking for general information/how-to that is NOT a supported in-chat service operation
   - Examples: "What does travel insurance cover?", "What are the exclusions?", "How can I do a refund?"
   - (Refund is not supported as an action in this service flow, so it's info.)
   
5. exit_to_recommend: User wants an insurance recommendation
   - Examples: "Recommend me car insurance", "Help me choose a plan", "What insurance do I need?"
   
6. exit_to_compare: User wants to compare insurance plans
   - Examples: "Compare Gold and Silver plans", "What's the difference between plans?"
   
7. cancel: User wants to STOP the current process or START OVER
   - Examples: "Forget it", "Cancel", "Never mind", "I'll do this later", "Stop", "Start over", "Restart", "Wrong details"
   
8. escalate: User explicitly wants a HUMAN AGENT
   - Examples: "Talk to a person", "Speak to agent", "I want a human", "Connect me to someone"
   - NOT just asking for phone number (that's side_question)

CRITICAL RULES:
- SHORT ANSWERS (1-3 words) are almost always provide_credential
- "How can I / how do I X?" where X is a SUPPORTED service operation → new_service_action
  (we can do it in-chat): claim_status, policy_status, update_email, update_mobile, update_address, update_payment, update_insured_address
- Refund questions → exit_to_info (refund is not supported as an action)
- "I want to X" where X is a policy operation → new_service_action (they want to do it)
- When in doubt between provide_credential and other intents → prefer provide_credential
- When in doubt between exit_* intents → prefer exit_to_info

Return your classification with reasoning."""

        result = await classifier.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"User message: {user_msg}"),
            ]
        )
        
        duration = time.time() - start_time
        logger.info(
            "ServiceOrchestrator.classify: intent=%s reason='%s' duration=%.3fs",
            result.intent,
            (result.reason or "")[:50],
            duration
        )
        
        # =====================================================================
        # ROUTE BASED ON CLASSIFIED INTENT
        # =====================================================================
        
        if result.intent == "provide_credential":
            return {"_orchestrator_route": "collect_credentials"}
        
        elif result.intent == "side_question":
            # Answer the question inline using RAG
            question = result.detected_question or user_msg
            logger.info("ServiceOrchestrator.side_question: '%s'", question)
            
            from ..tools.info import _info_tool_async
            try:
                answer, sources = await _info_tool_async(
                    product=None,
                    question=question,
                )
                if not answer or answer.strip() == "":
                    answer = "I'm not sure about that specific question. You can reach our support team at 1800-930-9330 for detailed assistance."
            except Exception as e:
                logger.warning("ServiceOrchestrator.side_question failed: %s", e)
                answer = "I apologize, I couldn't find that information right now. You can reach our support team at 1800-930-9330."
            
            # Build response - continue asking for credential if we have a pending slot
            if pending_slot:
                current_question = VALIDATION_SLOTS.get(pending_slot, {}).get("question", "")
                if current_question:
                    response_msg = f"{answer}\n\nNow, {current_question.lower()}"
                else:
                    response_msg = answer
            else:
                # First credential question
                response_msg = f"{answer}\n\nWhat is your first name as registered with us?"
            
            return {
                "messages": [AIMessage(content=response_msg)],
                "service_pending_slot": pending_slot or "first_name",
                "_orchestrator_route": None,  # Handled locally, stop subgraph
            }
        
        elif result.intent == "new_service_action":
            # Switch to different service action - reset slots and detect new action
            detected_action = result.detected_service_action
            logger.info("ServiceOrchestrator.new_service_action: '%s'", detected_action)
            
            # If we detected a specific action, set it directly
            if detected_action:
                # Normalize the action name
                action_map = {
                    "claim": "claim_status",
                    "claim_status": "claim_status",
                    "policy": "policy_status",
                    "policy_status": "policy_status",
                    "update_email": "update_email",
                    "update_mobile": "update_mobile",
                    "email": "update_email",
                    "mobile": "update_mobile",
                    "phone": "update_mobile",
                }

                detected_action_norm = detected_action.lower().strip()

                # Refund is not a service action in this system; provide information instead.
                # (We route to the info agent with product filter disabled.)
                if detected_action_norm == "refund":
                    return {
                        "phase": None,
                        "service_action": None,
                        "service_pending_slot": None,
                        "service_exit_intent": "info",
                        "service_exit_query": user_msg,
                        "messages": [AIMessage(content="Sure — I’ll share the refund information.")],
                        "_orchestrator_route": None,  # Exit subgraph
                    }

                normalized_action = action_map.get(detected_action_norm, detected_action_norm)
                
                return {
                    "service_action": normalized_action,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "_orchestrator_route": "collect_credentials",
                }
            else:
                # Let detect_action figure out what they want
                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "_orchestrator_route": "detect_action",
                }
        
        elif result.intent == "exit_to_info":
            logger.info("ServiceOrchestrator.exit_to_info: query='%s'", user_msg[:50])
            return {
                "phase": None,
                "service_action": None,
                "service_pending_slot": None,
                "service_exit_intent": "info",
                "service_exit_query": user_msg,
                "messages": [AIMessage(content="Let me help you with that information.")],
                "_orchestrator_route": None,  # Exit subgraph
            }
        
        elif result.intent == "exit_to_recommend":
            logger.info("ServiceOrchestrator.exit_to_recommend: product=%s", result.detected_product)
            return {
                "phase": None,
                "service_action": None,
                "service_pending_slot": None,
                "service_exit_intent": "recommend",
                "service_exit_query": user_msg,
                # Pass detected product to supervisor if possible
                "product": result.detected_product or state.get("product"),
                "messages": [AIMessage(content="I can help you find the right insurance plan.")],
                "_orchestrator_route": None,  # Exit subgraph
            }
        
        elif result.intent == "exit_to_compare":
            logger.info("ServiceOrchestrator.exit_to_compare")
            return {
                "phase": None,
                "service_action": None,
                "service_pending_slot": None,
                "service_exit_intent": "compare",
                "service_exit_query": user_msg,
                "messages": [AIMessage(content="I can help you compare plans. Let me gather that information.")],
                "_orchestrator_route": None,  # Exit subgraph
            }
        
        elif result.intent == "cancel":
            logger.info("ServiceOrchestrator.cancel")
            
            # CLEAR PII MAPPING ON CANCEL/RESET
            try:
                session_id = config["configurable"].get("thread_id")
                if session_id:
                    get_pii_masker().clear_session(session_id)
            except Exception as e:
                logger.warning("ServiceOrchestrator.cancel: failed to clear PII session: %s", e)
            
            return {
                "phase": None,
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": None,
                "customer_validated": False,
                "pii_mapping": {},  # Clear persisted PII mapping
                "messages": [AIMessage(content="No problem! I've cancelled the process. Is there anything else I can help you with?")],
                "_orchestrator_route": None,  # Exit subgraph
            }
        
        elif result.intent == "escalate":
            logger.info("ServiceOrchestrator.escalate")
            return {
                "phase": None,
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": None,
                "customer_validated": False,
                "live_agent_requested": True,
                "messages": [AIMessage(content="I understand you'd like to speak with a customer service representative. Let me connect you to our support team.")],
                "_orchestrator_route": None,  # Exit subgraph
            }
        
        else:
            # Unknown intent - default to credential collection
            logger.warning("ServiceOrchestrator.unknown_intent: %s", result.intent)
            return {"_orchestrator_route": "collect_credentials"}
    
    except Exception as e:
        logger.error("ServiceOrchestrator.classify_failed: %s", e)
        # On error, continue with credential collection
        return {"_orchestrator_route": "collect_credentials"}


# =============================================================================
# SERVICE SUBGRAPH NODES
# =============================================================================

async def _service_detect_action(state: AgentState) -> Dict[str, Any]:
    """
    Detect what service action the user wants using LLM.
    
    This uses LLM-based classification, NOT keyword matching.
    """
    messages = list(state.get("messages", []) or [])
    if not messages:
        return {"service_action": "unclear"}
    
    # Get last user message (already masked)
    user_msg = _get_last_user_message(messages)
    if not user_msg:
        return {"service_action": "unclear"}

    # If we just asked the user to choose an action, support numeric selection deterministically.
    # This avoids relying on keyword heuristics and improves reliability for short replies like "1".
    if state.get("service_pending_slot") == "service_action_choice":
        m = re.match(r"^\s*(\d)\s*[\).\]]?\s*$", user_msg or "")
        if m:
            choice = m.group(1)
            choice_map = {
                "1": "claim_status",
                "2": "policy_status",
                "3": "update_email",
                "4": "update_mobile",
                "5": "update_address",
                "6": "update_payment",
                "7": "update_insured_address",
            }
            if choice in choice_map:
                logger.info("ServiceFlow.detect_action: numeric_choice=%s -> action=%s", choice, choice_map[choice])
                return {
                    "service_action": choice_map[choice],
                    "service_pending_slot": None,
                }
    
    # Build context from recent history
    history_ctx = _build_history_context_from_messages(messages, max_pairs=3)
    
    # Check if we already have an action from previous turn
    existing_action = state.get("service_action")
    if existing_action and existing_action != "unclear":
        logger.debug("ServiceFlow.detect_action: using existing action=%s", existing_action)
        return {}  # Keep existing action
    
    # Skip action detection if we're in credential collection phase
    # This prevents the LLM from misinterpreting credential inputs as action requests
    # e.g., user providing mobile number for verification being detected as "update_mobile"
    service_pending_slot = state.get("service_pending_slot")
    credential_slots = {"first_name", "last_name", "email", "mobile"}
    if service_pending_slot in credential_slots:
        logger.debug(
            "ServiceFlow.detect_action: skip detection during credential collection pending_slot=%s",
            service_pending_slot
        )
        return {}  # Keep existing action, don't re-detect
    
    sys_prompt = """You are detecting what policy service action the user wants to perform.

Based on the conversation, determine the most likely action:
- claim_status: User asking about claim status, where is my claim, claim update
- policy_status: User asking about a specific policy's status or details
- update_email: User wants to change their email address
- update_mobile: User wants to change their phone/mobile number
- update_address: User wants to change their mailing/correspondence address (postal address, home address)
- update_payment: User wants to update payment/credit card information
- update_insured_address: User wants to change the insured property address for their Home Protect360 policy
- unclear: Cannot determine what user wants

If the user mentioned a policy number (shown as [POLICY_X] placeholder), extract it.

Be liberal in detecting service actions - if user mentions anything about existing policies, claims, or account updates, detect the appropriate action."""

    user_prompt = f"""Recent conversation:
{history_ctx}

Latest message: {user_msg}

What service action does the user want?"""

    try:
        detector = _get_action_detector()
        result = await detector.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        
        logger.info(
            "ServiceFlow.detect_action: action=%s policy=%s reason='%s'",
            result.action, result.policy_no, result.reason[:50] if result.reason else ""
        )
        
        update = {"service_action": result.action}
        if result.policy_no:
            # Persist the REQUESTED policy number (not the verification policy).
            # Only trust placeholders present in this user message.
            pii_mapping = state.get("pii_mapping") or {}
            policy_ph = str(result.policy_no).strip()
            if policy_ph in (user_msg or "") and policy_ph in pii_mapping:
                service_slots = dict(state.get("service_slots") or {})
                requested_policy = str(pii_mapping.get(policy_ph) or "").strip().upper()
                if requested_policy:
                    service_slots["requested_policy_no"] = requested_policy
                update["service_slots"] = service_slots
        
        return update
        
    except Exception as e:
        logger.error("ServiceFlow.detect_action.failed: %s", e)
        return {"service_action": "unclear"}


def _service_check_validated(state: AgentState) -> Literal["validated", "not_validated", "ask_credentials", "ask_action"]:
    """
    Check if customer is validated and route accordingly.
    """
    is_validated = state.get("customer_validated", False)
    service_action = state.get("service_action")
    
    logger.debug(
        "ServiceFlow.check_validated: validated=%s action=%s",
        is_validated, service_action
    )
    
    # If we don't know what the user wants to do yet, ask for the action first.
    # This avoids collecting sensitive credentials before we've clarified the request.
    if not service_action or service_action == "unclear":
        return "ask_action"
    
    if is_validated:
        return "validated"
    
    # Check if we have enough credentials to attempt validation
    service_slots = state.get("service_slots") or {}
    has_name = bool(service_slots.get("first_name") and service_slots.get("last_name"))
    has_email = bool(service_slots.get("email"))
    has_mobile = bool(service_slots.get("mobile"))
    
    if has_name and has_email and has_mobile:
        return "not_validated"  # Have credentials, try to validate
    
    return "ask_credentials"  # Need to collect credentials


def _service_ask_action(state: AgentState) -> Dict[str, Any]:
    """
    Ask the user what service action they want BEFORE collecting any credentials.

    This prevents poor UX (asking for credentials when the bot hasn't understood the request)
    and avoids getting stuck in loops when an unknown/unsupported action is detected.

    Special case: if the user arrived here via the purchase gate (pending_purchase=True),
    skip the service menu entirely and return them to the sales journey.
    """
    if state.get("pending_purchase"):
        return {
            "service_action": None,
            "service_pending_slot": None,
            "pending_purchase": False,
            "service_exit_intent": "purchase",
            "messages": [AIMessage(
                content=(
                    "✅ Identity verified! Let me send you the payment link now."
                )
            )],
        }

    action_menu = (
        "I can help you with these policy services:\n\n"
        "1) Check claim status\n"
        "2) Check policy status/details\n"
        "3) Update email address\n"
        "4) Update mobile number\n"
        "5) Update mailing address\n"
        "6) Update payment information\n"
        "7) Update insured address (Home Protect360)\n\n"
        "Please reply with the option number, or describe what you’d like to do."
    )

    return {
        # Clear action so the next user turn triggers action detection again.
        "service_action": None,
        # Track that we're awaiting an action choice (not a credential).
        "service_pending_slot": "service_action_choice",
        "messages": [AIMessage(content=action_menu)],
    }


async def _service_collect_credentials(state: AgentState) -> Dict[str, Any]:
    """
    Collect validation credentials from user, extracting from PII mapping.
    
    NOTE: Intent classification is now handled by the Policy Service Orchestrator.
    This function is only called when user_intent == "provide_credential".
    It focuses purely on credential extraction and validation.
    """
    messages = list(state.get("messages", []) or [])
    pii_mapping = state.get("pii_mapping") or {}
    service_slots = dict(state.get("service_slots") or {})
    
    # Get last user message
    user_msg = _get_last_user_message(messages)
    pending_slot = state.get("service_pending_slot")
    validation_error = None  # Track validation errors
    preface = ""  # Optional prefix for next question
    
    slot_errors = {}
    credential_slots = {"first_name", "last_name", "email", "mobile"}
    contains_pii_placeholder = bool(re.search(r"\[(MOBILE|EMAIL|POSTAL|CARD)_\d+\]", user_msg or ""))

    # ------------------------------------------------------------------
    # Guardrail: reject emoji-only or clearly invalid input early
    # ------------------------------------------------------------------
    if pending_slot in credential_slots and user_msg:
        # Strip whitespace and check if input is only emojis/symbols (no alphanumeric content)
        stripped = user_msg.strip()
        has_alnum = any(c.isalnum() for c in stripped)
        if stripped and not has_alnum and not contains_pii_placeholder:
            slot_display = VALIDATION_SLOTS.get(pending_slot, {}).get("question", f"your {pending_slot}")
            logger.info("ServiceFlow.collect_credentials: rejected invalid input (no alphanumeric) for %s", pending_slot)
            return {
                "service_slots": service_slots,
                "service_pending_slot": pending_slot,
                "messages": [AIMessage(content=f"⚠️ That doesn't look like a valid input. Please enter text only.\n\n{slot_display}")],
            }

    # ------------------------------------------------------------------
    # Guardrail: name slot disambiguation using pending_slot + already-collected name
    #
    # Problem observed in production:
    # - When we're asking for last_name and user replies with a short token like "TIO",
    #   the LLM may output first_name="TIO" and overwrite the previously collected first name.
    #
    # Fix:
    # - If pending_slot is a name and the user reply is a short name-only message,
    #   lock the value to the pending slot and prevent LLM name outputs from overwriting
    #   already-collected names.
    # ------------------------------------------------------------------
    locked_name_slot: Optional[str] = None

    def _normalize_name_token(token: str) -> str:
        # Strip common surrounding punctuation but keep internal hyphens/apostrophes.
        return re.sub(r"^[^A-Za-z'\-]+|[^A-Za-z'\-]+$", "", (token or "").strip())

    def _maybe_lock_pending_name() -> None:
        nonlocal locked_name_slot, service_slots
        if pending_slot not in ("first_name", "last_name"):
            return
        text = (user_msg or "").strip()
        if not text:
            return
        # Only lock when the message looks like a pure name answer (avoid mixed credential lines).
        if "[" in text or any(ch.isdigit() for ch in text):
            return

        tokens = [_normalize_name_token(t) for t in re.split(r"\s+", text) if t.strip()]
        tokens = [t for t in tokens if t]
        if not tokens:
            return

        candidate: Optional[str] = None

        if len(tokens) == 1:
            candidate = tokens[0]
        else:
            # If user re-sent both names, use the already-known other name to pick the new one.
            other_slot = "last_name" if pending_slot == "first_name" else "first_name"
            other_val = (service_slots.get(other_slot) or "").strip()
            if other_val:
                other_norm = other_val.upper()
                if tokens[0].upper() == other_norm:
                    candidate = tokens[1]
                elif tokens[1].upper() == other_norm:
                    candidate = tokens[0]
            else:
                # Multi-word names are common (e.g., "WEE LEONG"); keep the full entry.
                candidate = " ".join(tokens)

        if not candidate:
            return

        field_display = "first name" if pending_slot == "first_name" else "last name"
        is_valid, _ = _validate_name(candidate, field_display)
        if not is_valid:
            return

        service_slots[pending_slot] = candidate
        locked_name_slot = pending_slot

    _maybe_lock_pending_name()

    # Helper to validate a specific slot
    def _check_slot(slot, value):
        if slot == "first_name": return _validate_name(value, "first name")
        if slot == "last_name": return _validate_name(value, "last name")
        if slot == "email": return _validate_email(value)
        if slot == "mobile": return _validate_mobile(value)
        return True, None

    # ==========================================================================
    # CREDENTIAL EXTRACTION (LLM-based)
    # ==========================================================================
    
    # Only run LLM extraction when we're actually collecting credentials OR
    # when the user clearly provided credentials in one message (PII placeholders present).
    if user_msg and (pending_slot in credential_slots or contains_pii_placeholder):
        try:
            extractor = _get_credential_extractor()
            
            # Determine context message
            context_msg = f"We are asking the user for their '{pending_slot or 'credentials'}'."

            sys_prompt = f"""You are extracting validation credentials from user input.

CONTEXT: {context_msg}

The user is providing their credentials. Extract the following if present:
- First name (NOT masked - extract actual name)
- Last name (NOT masked - extract actual name)
- Email address (masked as [EMAIL_1] placeholder)
- Mobile number (masked as [MOBILE_1] placeholder)

RULES:
- Email and mobile are masked with placeholders like [EMAIL_1], [MOBILE_1]
- Names (first_name, last_name) are NOT masked - extract the actual names
- Short answers like "WL", "TIO", "John" are likely names
- Numbers like "91234567" are likely mobile numbers
- Addresses containing @ are likely emails

Set user_intent to 'provide_credential' since we are extracting credentials."""

            result = await extractor.ainvoke(
                [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=f"User message: {user_msg}"),
                ]
            )

            logger.debug(
                "ServiceFlow.extract_credentials: email=%s mobile=%s name=%s %s",
                result.email_placeholder,
                result.mobile_placeholder,
                result.first_name,
                result.last_name,
            )

            # ==========================================================================
            # MAP PLACEHOLDERS TO REAL VALUES
            # ==========================================================================

            # Map placeholders to real values and store.
            # IMPORTANT: only trust placeholders that are actually present in this user message
            # to avoid the extractor hallucinating an older placeholder from session context.
            if (
                result.email_placeholder
                and result.email_placeholder in pii_mapping
                and user_msg
                and result.email_placeholder in user_msg
            ):
                service_slots["email"] = pii_mapping[result.email_placeholder]

            if (
                result.mobile_placeholder
                and result.mobile_placeholder in pii_mapping
                and user_msg
                and result.mobile_placeholder in user_msg
            ):
                service_slots["mobile"] = pii_mapping[result.mobile_placeholder]

            if (
                result.postal_placeholder
                and result.postal_placeholder in pii_mapping
                and user_msg
                and result.postal_placeholder in user_msg
            ):
                service_slots["postal_code"] = pii_mapping[result.postal_placeholder]

            # Names are not masked.
            # Guardrails:
            # - Never overwrite an already-collected name unless we are explicitly asking for that slot.
            # - If we are not currently asking for a name, only accept BOTH names when the user clearly
            #   provided credentials (PII placeholders present) and both names are present.
            if locked_name_slot is None:
                if pending_slot == "first_name":
                    if result.first_name:
                        first = str(result.first_name).strip()
                        is_valid, _ = _validate_name(first, "first name")
                        if is_valid:
                            service_slots["first_name"] = first
                elif pending_slot == "last_name":
                    if result.last_name:
                        last = str(result.last_name).strip()
                        is_valid, _ = _validate_name(last, "last name")
                        if is_valid:
                            service_slots["last_name"] = last
                else:
                    # Not currently asking for a name — only accept full name pair,
                    # and only if user clearly provided credentials.
                    if contains_pii_placeholder and result.first_name and result.last_name:
                        if not service_slots.get("first_name") and not service_slots.get("last_name"):
                            first = str(result.first_name).strip()
                            last = str(result.last_name).strip()
                            v1, _ = _validate_name(first, "first name")
                            v2, _ = _validate_name(last, "last name")
                            if v1 and v2:
                                service_slots["first_name"] = first
                                service_slots["last_name"] = last

            # Avoid logging raw names at INFO level
            logger.debug("ServiceFlow.credential_extractor: %s", result)
            
        except Exception as e:
            logger.warning("ServiceFlow.collect_credentials.extraction_failed: %s", e)
            
    # VALIDATE ALL SLOTS (Including newly extracted ones)
    for slot in ["first_name", "last_name", "email", "mobile"]:
        if service_slots.get(slot):
            is_valid, err = _check_slot(slot, service_slots[slot])
            if not is_valid:
                # Avoid logging raw PII values (email/mobile) at INFO level
                logger.info("ServiceFlow.collect: removed invalid %s (%s)", slot, err)
                del service_slots[slot]
                slot_errors[slot] = err

    # ------------------------------------------------------------------
    # Manual fallback for the currently pending credential slot.
    # This prevents the bot from repeatedly asking the same question
    # when the user provides short answers like initials (e.g. "WL").
    # ------------------------------------------------------------------
    
    if pending_slot and not service_slots.get(pending_slot):
        text = (user_msg or "").strip()

        if pending_slot == "email":
            # Only accept email if it was present in THIS user message (avoid stale session PII)
            email = _get_latest_value_from_user_message(pii_mapping, user_msg, "EMAIL")
            if email:
                # Validate the email before accepting it
                is_valid, error_msg = _validate_email(email)
                if is_valid:
                    service_slots["email"] = email.strip().lower()
                    logger.debug("ServiceFlow.fallback: filled email from pii_mapping")
                else:
                    validation_error = error_msg
                    logger.debug("ServiceFlow.validation_failed: email invalid - %s", error_msg)
            
            # If no email found in PII mapping, check if user typed something that looks like an email
            if not service_slots.get("email") and not validation_error and text:
                if "@" in text:  # Looks like they tried to enter an email
                    is_valid, error_msg = _validate_email(text)
                    if is_valid:
                        service_slots["email"] = text.strip().lower()
                    else:
                        validation_error = error_msg

        elif pending_slot == "mobile":
            # Only accept mobile if it was present in THIS user message (avoid stale session PII)
            latest_mobile = _get_latest_value_from_user_message(pii_mapping, user_msg, "MOBILE")
            if latest_mobile:
                # Validate the mobile before accepting it
                is_valid, error_msg = _validate_mobile(latest_mobile)
                if is_valid:
                    service_slots["mobile"] = latest_mobile
                    logger.debug("ServiceFlow.fallback: filled mobile from latest pii_mapping")
                else:
                    validation_error = error_msg
                    logger.debug("ServiceFlow.validation_failed: mobile invalid - %s", error_msg)
            
            # If no mobile found, check if user typed something
            if not service_slots.get("mobile") and not validation_error and text:
                if any(c.isdigit() for c in text):  # Looks like they tried to enter a number
                    validation_error = "Please enter a valid Singapore mobile number (e.g., 91234567 or +65 91234567)."

        elif pending_slot in ("first_name", "last_name") and text:
            # Validate the name before accepting it
            field_display = "first name" if pending_slot == "first_name" else "last name"
            is_valid, error_msg = _validate_name(text, field_display)
            if is_valid:
                service_slots[pending_slot] = text
                logger.debug("ServiceFlow.fallback: filled %s from raw message", pending_slot)
            else:
                validation_error = error_msg
                logger.debug("ServiceFlow.validation_failed: %s invalid - %s", pending_slot, error_msg)

    # ------------------------------------------------------------------
    # WhatsApp mobile confirmation (no extra LLM call)
    # If we can derive a valid SG mobile from the WhatsApp sender,
    # ask the user to confirm it before requesting manual entry.
    # ------------------------------------------------------------------
    if pending_slot == "mobile_confirm":
        candidate = service_slots.get("_whatsapp_mobile_candidate") or _get_whatsapp_mobile_candidate(state)
        if candidate:
            service_slots["_whatsapp_mobile_candidate"] = candidate

        # If user provided a new mobile directly in this reply, accept it.
        if service_slots.get("mobile"):
            is_valid, err = _validate_mobile(service_slots["mobile"])
            if is_valid:
                service_slots["mobile"] = _normalize_mobile(service_slots["mobile"]) or service_slots["mobile"]
                preface = "Got it — I’ll use that number.\n\n"
                pending_slot = None
            else:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "mobile",
                    "messages": [AIMessage(content=f"⚠️ {err}\n\nPlease provide your registered mobile number.")],
                }
        else:
            yn = _classify_yes_no_heuristic(user_msg)
            if yn == "yes" and candidate:
                service_slots["mobile"] = candidate
                service_slots["_whatsapp_mobile_confirmed"] = True
                preface = "Got it — I’ll use that number.\n\n"
                pending_slot = None
            elif yn == "no":
                service_slots["_whatsapp_mobile_declined"] = True
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "mobile",
                    "messages": [AIMessage(content="No problem — please provide your registered mobile number.")],
                }
            else:
                hint = _mask_mobile_for_display(candidate)
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "mobile_confirm",
                    "messages": [AIMessage(content=(
                        f"I can use the mobile number ending {hint} from your WhatsApp. "
                        "Is this your registered mobile number? Reply *Yes* to use it, or send a different number."
                    ))],
                }
    
    # Determine which credential to ask for next
    missing_slots = []
    for slot_name in ["first_name", "last_name", "email", "mobile"]:
        if not service_slots.get(slot_name):
            missing_slots.append(slot_name)
    
    if missing_slots:
        next_slot = missing_slots[0]
        question = VALIDATION_SLOTS[next_slot]["question"]

        # Offer WhatsApp number confirmation before asking for manual mobile entry.
        if (
            next_slot == "mobile"
            and pending_slot not in ("mobile", "mobile_confirm")
            and not service_slots.get("_whatsapp_mobile_declined")
        ):
            candidate = _get_whatsapp_mobile_candidate(state)
            if candidate:
                service_slots["_whatsapp_mobile_candidate"] = candidate
                hint = _mask_mobile_for_display(candidate)
                confirm_msg = (
                    f"I can use the mobile number ending {hint} from your WhatsApp. "
                    "Is this your registered mobile number? Reply *Yes* to use it, or send a different number."
                )
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "mobile_confirm",
                    "messages": [AIMessage(content=confirm_msg)],
                }

        # Create intro message if this is the first credential request
        if len(missing_slots) == 4:  # All slots missing = first time
            intro = "To help you with your request, I'll need to verify your identity first.\n\n"
        else:
            intro = ""

        # If there was a validation error (from current input OR pre-validation of existing slots), show it
        current_error = validation_error if (validation_error and pending_slot == next_slot) else slot_errors.get(next_slot)
        
        if current_error:
            full_question = f"{preface}⚠️ {current_error}\n\n{question}"
            logger.info(
                "ServiceFlow.collect_credentials: validation_error for %s, re-asking",
                next_slot
            )
        else:
            full_question = f"{preface}{intro}{question}"
            logger.info(
                "ServiceFlow.collect_credentials: asking for %s (have: %s)",
                next_slot,
                [s for s in ["first_name", "last_name", "email", "mobile"] if service_slots.get(s)],
            )

        return {
            "service_slots": service_slots,
            "service_pending_slot": next_slot,
            "messages": [AIMessage(content=full_question)],
        }
    
    # All credentials collected; clear any pending slot marker.
    return {
        "service_slots": service_slots,
        "service_pending_slot": None,
    }


async def _service_validate_customer(state: AgentState) -> Dict[str, Any]:
    """
    Call the validation API with collected credentials.
    """
    service_slots = state.get("service_slots") or {}
    
    first_name = service_slots.get("first_name")
    last_name = service_slots.get("last_name")
    email = service_slots.get("email")
    mobile = service_slots.get("mobile")
    
    if not all([first_name, last_name, email, mobile]):
        logger.error("ServiceFlow.validate_customer: missing required fields")
        return {
            "messages": [AIMessage(content="I'm missing some information to verify your identity. Let me ask again.")],
            "service_slots": {},  # Clear and start over
            "service_pending_slot": None,
        }
    
    # Mask email for logging (show first 3 chars + domain)
    email_preview = "?"
    if email and "@" in email:
        local, domain = email.split("@", 1)
        email_preview = f"{local[:3]}***@{domain}" if len(local) >= 3 else f"***@{domain}"

    logger.info(
        "ServiceFlow.validate_customer: attempting validation name=%s %s email=%s mobile=***%s",
        (first_name or "")[:2],
        (last_name or "")[:2],
        email_preview,
        (mobile or "")[-4:],
    )
    
    try:
        # Normalize before sending to API
        mobile_norm = _normalize_mobile(mobile)
        email_norm = email.strip().lower() if email else email

        client = get_bigtapp_api_client()
        result = await client.validate_customer(
            first_name=first_name,
            last_name=last_name,
            email=email_norm,
            mobile=mobile_norm,
        )

        # IMPORTANT: Do not log raw API responses at INFO/WARN/ERROR (even masked).
        # Keep this at DEBUG only for troubleshooting.
        try:
            if logger.isEnabledFor(logging.DEBUG):
                result_json = json.dumps(result, default=str)
                pii_masker = get_pii_masker()
                masked_json, _ = pii_masker.mask(result_json, session_id="service_api_log")
                if len(masked_json) > 2000:
                    masked_json = masked_json[:2000] + "... [TRUNCATED]"
                logger.debug("ServiceFlow.validate_customer.api_raw=%s", masked_json)
        except Exception as log_err:
            logger.debug("ServiceFlow.validate_customer.api_log_failed: %s", log_err)
        
        if result.get("success"):
            customer_data = result.get("data", {})

            # Extract NRIC from the API response (needed for update operations)
            customer_nric = customer_data.get("idCardNumber", "")
            
            # Extract name for greeting
            given_name = customer_data.get("givenName", first_name)

            logger.info(
                "ServiceFlow.validate_customer: SUCCESS for %s nric=***%s",
                given_name,
                customer_nric[-4:] if customer_nric else "?",
            )

            # If the user was routed here via the purchase gate, skip the service
            # menu entirely — verify identity and return directly to sales journey.
            if state.get("pending_purchase"):
                logger.info(
                    "ServiceFlow.validate_customer: pending_purchase=True → returning to sales_agent"
                )
                return {
                    "customer_validated": True,
                    "customer_nric": customer_nric,
                    "customer_data": customer_data,
                    "service_pending_slot": None,
                    "service_slots": {},
                    "pending_purchase": False,
                    "service_exit_intent": "purchase",
                    "messages": [AIMessage(
                        content=(
                            f"✅ *Identity verified!* Welcome, {given_name}! Let me send you the payment link now."
                        )
                    )],
                }

            # Store greeting in service_slots so execute_action can prepend it
            current_slots = dict(state.get("service_slots") or {})
            current_slots["_validated_greeting"] = f"✅ *Customer Validated*\nWelcome, {given_name}! Your identity has been successfully verified."

            return {
                "customer_validated": True,
                "customer_nric": customer_nric,  # Get NRIC from API response
                "customer_data": customer_data,
                "service_pending_slot": None,
                "service_slots": current_slots,
                "messages": [],
            }
        else:
            error_msg = result.get("error", "The details provided don't match our records.")
            # Sanitize error message - remove any NRIC/policy references from API errors
            if "nric" in error_msg.lower() or "policy number" in error_msg.lower():
                error_msg = "The details provided don't match our records."
            status_code = result.get("status_code", 400)
            
            # Check if this is a system/connection error (retryable) vs validation failure
            is_system_error = (
                status_code >= 500 
                or "connect" in error_msg.lower() 
                or "timeout" in error_msg.lower()
            )
            
            if is_system_error:
                final_msg = (
                    "⚠️ *System Error*\n\n"
                    f"{error_msg}\n\n"
                    "I have your details. What would you like to do?\n\n"
                    "1) Try again now\n"
                    "2) Re-enter your details"
                )
                # Keep slots for retry and wait for user choice.
                slots_update = {"service_pending_slot": "validation_recovery"}
            else:
                final_msg = (
                    "❌ *Verification Failed*\n\n"
                    f"{error_msg}\n\n"
                    "Please double-check your details:\n"
                    "• First and last name (as registered)\n"
                    "• Email address\n"
                    "• Mobile number\n\n"
                    "Would you like to try again? Just provide your first name to start."
                )
                # Clear credentials for fresh input on validation mismatch
                # Set pending_slot to first_name so the next input is recognized immediately
                slots_update = {"service_slots": {}, "service_pending_slot": "first_name"}

            logger.warning(
                "ServiceFlow.validate_customer: FAILED - %s (system_error=%s) reply='%s'",
                error_msg, is_system_error,
                final_msg.replace("\n", " ")[:200],
            )

            return {
                "customer_validated": False,
                **slots_update,
                # service_pending_slot is set above for system errors; otherwise keep it None.
                "service_pending_slot": slots_update.get("service_pending_slot"),
                # NOTE: Do NOT clear service_action here!
                # We want to preserve the user's original intent (policy_status, claim_status, etc.)
                # so when they retry, we don't re-detect the action and possibly get it wrong.
                "messages": [AIMessage(content=final_msg)],
            }
            
    except Exception as e:
        logger.exception("ServiceFlow.validate_customer: exception")
        return {
            "messages": [AIMessage(content="I encountered an error while verifying your identity. Please try again later.")],
        }


async def _service_execute_action(state: AgentState) -> Dict[str, Any]:
    """
    Execute the service action after customer is validated.
    """
    action = state.get("service_action")
    customer_nric = state.get("customer_nric")
    customer_data = state.get("customer_data") or {}
    service_slots = state.get("service_slots") or {}
    pii_mapping = state.get("pii_mapping") or {}
    
    # Check if customer was just validated (greeting stored by validate_customer)
    validated_greeting = service_slots.pop("_validated_greeting", None)
    
    if not customer_nric:
        logger.error("ServiceFlow.execute_action: no customer_nric")
        return {"messages": [AIMessage(content="Please verify your identity first.")]}
    
    logger.info("ServiceFlow.execute_action: action=%s", action)
    
    client = get_bigtapp_api_client()
    
    result = await _service_execute_action_inner(state, action, customer_nric, customer_data, service_slots, pii_mapping, client)
    
    # Prepend validated greeting to the response if customer was just validated
    if validated_greeting and "messages" in result:
        for i, msg in enumerate(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                result["messages"][i] = AIMessage(
                    content=f"{validated_greeting}\n\n{msg.content}"
                )
                break
    
    return result


async def _service_execute_action_inner(
    state: AgentState,
    action: str,
    customer_nric: str,
    customer_data: Dict[str, Any],
    service_slots: Dict[str, Any],
    pii_mapping: Dict[str, Any],
    client: BigTappApiClient,
) -> Dict[str, Any]:
    """Inner execution logic for service actions."""
    try:
        # =================================================================
        # CLAIM STATUS
        # =================================================================
        if action == "claim_status":
            result = await client.get_claims(customer_nric)
            
            if result.get("success"):
                claims = result.get("data", [])
                # If the user asked for claim status of a specific policy, filter.
                requested_policy = (service_slots.get("requested_policy_no") or "").strip().upper()
                if requested_policy:
                    filtered = [
                        c for c in claims
                        if str(c.get("policyNo") or "").strip().upper() == requested_policy
                    ]
                    if filtered:
                        response = _format_claim_list(filtered)
                    else:
                        response = (
                            f"I couldn’t find any claims for policy *{requested_policy}*.\n\n"
                            "If you’d like, I can show all your claims—just say “show all claims”."
                        )
                else:
                    response = _format_claim_list(claims)
            else:
                response = f"I couldn't retrieve your claims. {result.get('error', '')}"
            
            # Single-turn action: clear service_action so future messages
            # can trigger a new detection (e.g. switch to policy_status).
            return {
                "messages": [AIMessage(content=response)],
                "service_action": None,
                "service_slots": {},  # Clear to avoid leaking stale requested_policy into next action
                "service_pending_slot": None,
            }
        
        # =================================================================
        # POLICY STATUS
        # =================================================================
        elif action == "policy_status":
            # policy_status is driven off the chatbot policies payload, which
            # has this shape per policy:
            # {"policyNo", "productName", "status", "commencementDate", "policyEndDate", ...}

            policies = customer_data.get("policies", [])
            if not policies:
                # Fallback: fetch via chatbot policies endpoint
                result = await client.get_policies(customer_nric)
                if result.get("success"):
                    policies = result.get("data", [])

            service_pending_slot = state.get("service_pending_slot")

            # =================================================================
            # POLICY QUESTION: Detect if user asked a natural language question
            # about their policies (e.g., "Do I have home insurance?")
            # Only triggers on the FIRST entry (no pending slot yet) to avoid
            # interfering with the policy selection / pagination flow.
            # =================================================================
            _question_check_slots = {None, "service_action_choice", "requested_policy_no"}
            if service_pending_slot in _question_check_slots:
                if service_pending_slot == "requested_policy_no":
                    original_msg = _get_last_user_message(state.get("messages", []) or []) or ""
                else:
                    original_msg = service_slots.get("_original_question") or _get_last_user_message(state.get("messages", []) or []) or ""
                _question_indicators = [
                    "do i have", "do i own", "am i covered", "is there",
                    "what type", "what kind", "how many", "which policies",
                    "any home", "any travel", "any health", "any life",
                    "any motor", "any car", "any fire", "any personal",
                    "have i got", "what insurance", "what policies",
                    "covered for", "insured for",
                ]
                msg_lower = original_msg.lower().strip()
                is_policy_question = (
                    any(ind in msg_lower for ind in _question_indicators)
                    or (msg_lower.endswith("?") and len(msg_lower.split()) >= 4)
                )

                if is_policy_question and policies:
                    try:
                        # -------------------------------------------------------
                        # Disambiguation: if user says "this policy" / "my policy"
                        # and has multiple policies, ask them to specify which one.
                        # -------------------------------------------------------
                        _ambiguous_refs = (
                            "this policy", "that policy", "my policy",
                            "the policy", "this plan", "my plan",
                        )
                        is_ambiguous = (
                            len(policies) > 1
                            and any(ref in msg_lower for ref in _ambiguous_refs)
                        )

                        if is_ambiguous:
                            disambig_lines = ["Which policy are you asking about?\n"]
                            for idx, p in enumerate(policies, 1):
                                prod_name = _resolve_policy_product_name(p)
                                pno = p.get("policyNo", "N/A")
                                st = p.get("status", "Unknown")
                                disambig_lines.append(f"{idx}. *{prod_name}* ({pno}) — {st}")
                            disambig_lines.append("\nPlease reply with the number or product name.")
                            logger.info("ServiceFlow.policy_question: disambiguating, %d policies", len(policies))
                            return {
                                "service_slots": dict(service_slots, _policy_disambig_question=original_msg),
                                "service_pending_slot": "requested_policy_no",
                                "messages": [AIMessage(content="\n".join(disambig_lines))],
                            }

                        # -------------------------------------------------------
                        # Build enriched metadata using prefix-resolved names
                        # -------------------------------------------------------
                        policy_summary = []
                        for p in policies:
                            pno = p.get("policyNo", "N/A")
                            prod = _resolve_policy_product_name(p)
                            st = p.get("status", "Unknown")
                            start = _format_date(p.get("commencementDate")) if p.get("commencementDate") else "N/A"
                            end = _format_date(p.get("policyEndDate")) if p.get("policyEndDate") else "N/A"
                            policy_summary.append(
                                f"• {prod} (Policy {pno}): Status: {st}, Period: {start} – {end}"
                            )
                        policy_data_str = "\n".join(policy_summary)

                        llm = get_router_llm()
                        qa_prompt = (
                            "You are a helpful insurance assistant. The customer asked a question about their policies.\n"
                            "Answer ONLY based on the policy metadata below. Be concise and direct.\n"
                            "If the customer asks about a type of insurance they DON'T have, clearly say so and list what they DO have.\n"
                            "Do NOT make up coverage details, benefits, or features — you only have policy metadata (product name, status, dates).\n"
                            "Do NOT suggest the customer buy anything.\n\n"
                            f"CUSTOMER'S POLICIES (resolved from internal records):\n{policy_data_str}\n\n"
                            f"CUSTOMER'S QUESTION: {original_msg}\n\n"
                            "Answer in 2-3 sentences max. Use bullet points if listing policies."
                        )

                        qa_response = await llm.ainvoke([
                            SystemMessage(content=qa_prompt),
                        ])
                        answer_text = (qa_response.content or "").strip()

                        if answer_text:
                            logger.info("ServiceFlow.policy_question: answered contextually, question='%s'", msg_lower[:50])
                            menu = _service_ask_action(state)
                            return {
                                "service_action": None,
                                "service_slots": {},
                                "service_pending_slot": "service_action_choice",
                                "messages": [AIMessage(
                                    content=f"{answer_text}\n\n{menu['messages'][0].content}"
                                )],
                            }
                    except Exception as e:
                        logger.warning("ServiceFlow.policy_question failed: %s, falling through to normal flow", e)

            # The policy the USER asked about (must not be confused with the verification policy).
            requested_policy: Optional[str] = (
                str(service_slots.get("requested_policy_no") or "").strip().upper()
                if service_slots.get("requested_policy_no")
                else None
            )

            # If we previously asked the user to choose a policy, handle selection OR pagination from THIS turn.
            if not requested_policy and service_pending_slot == "requested_policy_no":
                last_msg = _get_last_user_message(state.get("messages", []) or [])

                # 1) Policy number selection (placeholder-aware)
                selected = _extract_policy_no_from_user_msg(pii_mapping, last_msg)
                if selected:
                    requested_policy = selected
                    # Validate format early for better UX
                    is_valid, err = _validate_policy_no(requested_policy)
                    if not is_valid:
                        # Keep page state so the user can still navigate/choose
                        ordered = _order_policies_for_display(policies or [])
                        offset, page_size = _get_policy_page_state(service_slots, "requested_policy_no")
                        page_text = _render_policy_page(
                            ordered, title="Which policy would you like to check?", offset=offset, page_size=page_size
                        )
                        return {
                            "service_slots": dict(service_slots),
                            "service_pending_slot": "requested_policy_no",
                            "messages": [AIMessage(content=f"⚠️ {err}\n\n{page_text}")],
                        }
                    # Persist for this flow
                    service_slots = dict(service_slots)
                    service_slots["requested_policy_no"] = requested_policy
                    # Once selected, clear pagination state
                    _clear_policy_page_state(service_slots, "requested_policy_no")
                else:
                    # 2) Pagination command (next/prev/10/etc)
                    cmd = _parse_policy_page_command(last_msg)
                    if cmd:
                        ordered = _order_policies_for_display(policies or [])
                        offset, page_size = _get_policy_page_state(service_slots, "requested_policy_no")
                        if cmd.get("page_size"):
                            page_size = int(cmd["page_size"])
                        # Align offset if page_size changed
                        if page_size < _POLICY_PAGE_MIN:
                            page_size = _POLICY_PAGE_MIN
                        if page_size > _POLICY_PAGE_MAX:
                            page_size = _POLICY_PAGE_MAX
                        if page_size:
                            offset = (offset // page_size) * page_size

                        total = len(ordered)
                        if cmd["cmd"] == "next":
                            if offset + page_size < total:
                                offset += page_size
                        elif cmd["cmd"] == "prev":
                            offset = max(0, offset - page_size)
                        elif cmd["cmd"] == "first":
                            offset = 0
                        elif cmd["cmd"] == "last":
                            if total > 0:
                                offset = max(((total - 1) // page_size) * page_size, 0)
                        elif cmd["cmd"] == "all":
                            # Show more per page (bounded) from the beginning
                            offset = 0
                            page_size = min(_POLICY_PAGE_MAX, max(_POLICY_PAGE_MIN, total))

                        _set_policy_page_state(service_slots, "requested_policy_no", offset, page_size)
                        page_text = _render_policy_page(
                            ordered, title="Which policy would you like to check?", offset=offset, page_size=page_size
                        )
                        return {
                            "service_action": "policy_status",
                            "service_slots": dict(service_slots),
                            "service_pending_slot": "requested_policy_no",
                            "messages": [AIMessage(content=page_text)],
                        }

                    # 3) Unknown input: re-show current page with error message
                    ordered = _order_policies_for_display(policies or [])
                    offset, page_size = _get_policy_page_state(service_slots, "requested_policy_no")
                    page_text = _render_policy_page(
                        ordered, title="Which policy would you like to check?", offset=offset, page_size=page_size
                    )
                    return {
                        "service_action": "policy_status",
                        "service_slots": dict(service_slots),
                        "service_pending_slot": "requested_policy_no",
                        "messages": [AIMessage(content=f"⚠️ That doesn't match any of your policies. Please reply with a policy number from the list below.\n\n{page_text}")],
                    }

            # If user didn't specify a policy, ask them to choose (or auto-use the only one).
            if not requested_policy:
                if policies and len(policies) == 1:
                    requested_policy = str(policies[0].get("policyNo") or "").strip().upper() or None

                if not requested_policy:
                    # Show paginated policy list (store pagination state in service_slots)
                    service_slots = dict(service_slots)
                    offset, page_size = _get_policy_page_state(service_slots, "requested_policy_no")
                    # If this is the first time asking, start at the first page
                    if offset == 0 and f"_requested_policy_no_offset" not in service_slots:
                        _set_policy_page_state(service_slots, "requested_policy_no", 0, _POLICY_PAGE_DEFAULT)
                        offset, page_size = _get_policy_page_state(service_slots, "requested_policy_no")
                    ordered = _order_policies_for_display(policies or [])
                    page_text = _render_policy_page(
                        ordered, title="Which policy would you like to check?", offset=offset, page_size=page_size
                    )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "requested_policy_no",
                        "messages": [AIMessage(content=page_text)],
                    }

            matched_policy = None
            if requested_policy and policies:
                req = requested_policy.strip().upper()
                for p in policies:
                    pno = str(p.get("policyNo") or "").strip().upper()
                    if pno and pno == req:
                        matched_policy = p
                        break

            if matched_policy:
                product_name = matched_policy.get("productName") or matched_policy.get("product_name") or "N/A"
                policy_status = matched_policy.get("status") or matched_policy.get("policy_status") or "N/A"

                commencement_raw = matched_policy.get("commencementDate") or matched_policy.get("commencement_date")
                end_raw = matched_policy.get("policyEndDate") or matched_policy.get("policy_end_date")
                commencement = _format_date(commencement_raw) if commencement_raw else "N/A"
                end_date = _format_date(end_raw) if end_raw else "N/A"

                # Status emoji
                status_lower = policy_status.lower()
                if status_lower in ("active", "pending new business"):
                    status_emoji = "✅"
                elif status_lower == "lapsed":
                    status_emoji = "⏸️"
                else:
                    status_emoji = "📋"

                response_lines = [
                    f"📋 *Policy {matched_policy.get('policyNo', requested_policy)}*\n",
                    f"• *Product:* {product_name}",
                    f"• *Status:* {status_emoji} {policy_status}",
                    f"• *Start Date:* {commencement}",
                    f"• *End Date:* {end_date}",
                ]

                # Include any other relevant fields from the API response.
                # Use a blocklist for already-shown keys + known PII keys
                # to prevent sensitive data from leaking into LLM history.
                _excluded_keys = {
                    "policyNo", "productName", "product_name",
                    "status", "policy_status",
                    "commencementDate", "commencement_date",
                    "policyEndDate", "policy_end_date",
                    "insuredName", "insured_name", "beneficiary",
                    "holderName", "holder_name", "policyHolderName",
                    "email", "emailAddress", "email_address",
                    "mobile", "mobileNo", "mobile_no", "phone", "phoneNumber",
                    "nric", "nricFin", "nric_fin", "idNumber", "id_number",
                    "address", "postalCode", "postal_code",
                    "cardNo", "card_no", "creditCardNo",
                    "bankAccount", "bank_account",
                }
                extra_items = {
                    k: v for k, v in matched_policy.items()
                    if k not in _excluded_keys and v
                }
                if extra_items:
                    response_lines.append("")
                    for k in sorted(extra_items.keys()):
                        formatted_key = k.replace("_", " ").title()
                        response_lines.append(f"• *{formatted_key}:* {extra_items[k]}")


                response = "\n".join(response_lines)
            else:
                # Policy not found on this customer's account (or couldn't fetch list).
                # Reset pagination to the first page for clarity.
                service_slots = dict(service_slots)
                _set_policy_page_state(service_slots, "requested_policy_no", 0, _POLICY_PAGE_DEFAULT)
                ordered = _order_policies_for_display(policies or [])
                page_text = _render_policy_page(
                    ordered, title="Which policy would you like to check?", offset=0, page_size=_POLICY_PAGE_DEFAULT
                )
                response = (
                    f"I couldn't find policy *{requested_policy}* on your account.\n\n"
                    f"{page_text}"
                )
                # Stay in policy_status and ask again
                service_slots.pop("requested_policy_no", None)
                return {
                    "service_action": "policy_status",
                    "service_slots": service_slots,
                    "service_pending_slot": "requested_policy_no",
                    "messages": [AIMessage(content=response)],
                }

            # After serving policy_status, clear service_action so the user
            # can ask for a different service action (e.g. claim_status).
            return {
                "messages": [AIMessage(content=response)],
                "service_action": None,
                "service_pending_slot": None,
                "service_slots": {},
            }
        
        # =================================================================
        # UPDATE EMAIL
        # =================================================================
        elif action == "update_email":
            new_email = None
            
            service_pending_slot = state.get("service_pending_slot")
            
            # Handle update confirmation FIRST (before reading new_email)
            if service_pending_slot == "_confirm_update" and service_slots.get("_confirm_update"):
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                user_reply = (last_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct"}
                if user_reply in _yes_words:
                    new_email = service_slots.get("_pending_new_email", "")
                    service_slots = dict(service_slots)
                    service_slots.pop("_pending_new_email", None)
                    logger.info("ServiceFlow.update_email: user confirmed update")
                    # Keep _confirm_update=True so the prompt section is skipped; fall through to API call
                else:
                    logger.info("ServiceFlow.update_email: user declined confirmation")
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_update", None)
                    service_slots.pop("_pending_new_email", None)
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — the email update has been cancelled.\n\n" + menu["messages"][0].content
                        )],
                    }

            elif service_pending_slot == "new_email":
                # We already asked for the new email – only accept it if user provided it THIS turn
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                new_email = _get_latest_value_from_user_message(pii_mapping, last_msg, "EMAIL")
                
                # If no email found AND user typed something, check for decline/exit intent
                # Instead of cancelling directly, route back to orchestrator confirmation
                if not new_email and last_msg and last_msg.strip():
                    _decline_keywords = [
                        "no need", "no change", "don't", "dont", "cancel",
                        "never mind", "nevermind", "forget", "thanks", "thank",
                        "later", "not now", "skip", "leave", "stop", "quit",
                        "exit", "nvm", "nah", "ok", "okay", "done", "fine",
                        "good", "all good", "that's all", "thats all",
                    ]
                    raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", last_msg.strip().lower()).strip()
                    if any(kw in raw_lower for kw in _decline_keywords):
                        logger.info("ServiceFlow.update_email: decline detected '%s', asking confirmation", raw_lower[:30])
                        service_slots = dict(service_slots)
                        service_slots["_confirm_cancel"] = True
                        service_slots["_cancel_return_slot"] = "new_email"
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "_confirm_cancel",
                            "messages": [AIMessage(
                                content="Are you sure you want to cancel the email update? (Yes / No)"
                            )],
                        }
            # else: First time entering update_email - we MUST ask for the new email.
            
            if not new_email:
                return {
                    "service_pending_slot": "new_email",
                    "messages": [AIMessage(content="What would you like your new email address to be?")],
                }
            
            # =====================================================================
            # VALIDATION: Validate email format before processing
            # =====================================================================
            is_valid, error_msg = _validate_email(new_email)
            if not is_valid:
                logger.info("ServiceFlow.update_email.rejected: invalid_format")
                return {
                    "service_pending_slot": "new_email",
                    "messages": [AIMessage(content=f"⚠️ {error_msg}\n\nWhat would you like your new email address to be?")],
                }
            
            # =====================================================================
            # ENHANCEMENT: Compare new email with current email from API
            # If the new email is the same as the existing one, reject the update.
            # =====================================================================
            current_email = (customer_data.get("email") or "").strip().lower()
            new_email_normalized = new_email.strip().lower()
            
            if current_email and new_email_normalized == current_email:
                logger.info("ServiceFlow.update_email.rejected: same_as_current email=%s", 
                           new_email_normalized[:3] + "***" if new_email_normalized else "")
                return {
                    "service_pending_slot": "new_email",
                    "messages": [AIMessage(
                        content=(
                            "⚠️ The email address you provided is the same as your current email on file.\n\n"
                            "Please provide a *different* email address if you wish to update it, "
                            "or let me know if there's something else I can help you with."
                        )
                    )],
                }
            
            # =================================================================
            # CONFIRMATION: Ask user to confirm before updating
            # =================================================================
            if not service_slots.get("_confirm_update"):
                # Mask email for display
                display_email = new_email
                try:
                    masker = get_pii_masker()
                    display_email, _ = masker.mask(new_email, session_id="service_debug")
                except Exception:
                    pass
                service_slots = dict(service_slots)
                service_slots["_confirm_update"] = True
                service_slots["_pending_new_email"] = new_email
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_update",
                    "messages": [AIMessage(
                        content=f"Please confirm: Update your email to *{new_email}*?\n\nReply *Yes* to confirm or *No* to cancel."
                    )],
                }

            # Log masked email going to the API for debugging without exposing PII
            try:
                masker = get_pii_masker()
                masked_email, _ = masker.mask(new_email, session_id="service_debug")
                logger.info("ServiceFlow.update_email.request email=%s", masked_email)
            except Exception as log_err:
                logger.warning("ServiceFlow.update_email.log_failed: %s", log_err)

            result = await client.update_email(customer_nric, new_email)
            
            if result.get("success"):
                # Refresh cached customer data (update endpoint returns the full payload)
                updated_data = result.get("data") if isinstance(result.get("data"), dict) else None
                response = "✅ Your email has been updated successfully!\n\nIs there anything else I can help you with?"
                updates: Dict[str, Any] = {}
                if updated_data:
                    updates["customer_data"] = updated_data

                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "messages": [AIMessage(content=response)],
                    **(updates or {}),
                }

            else:
                # Special case: timeouts can be "unknown outcome" (server may have applied the update).
                # Confirm via validate_customer if possible.
                status_code = int(result.get("status_code", 0) or 0)
                timeout_type = str(result.get("timeout_type") or "")
                request_id = str(result.get("request_id") or "")

                confirmed = False
                confirmed_data: Optional[Dict[str, Any]] = None
                if status_code == 408 or timeout_type:
                    creds = _get_validation_creds_for_confirm(state)
                    if creds:
                        logger.warning(
                            "ServiceFlow.update_email: API timeout (req_id=%s type=%s). Verifying via validate_customer...",
                            request_id,
                            timeout_type or "timeout",
                        )
                        try:
                            verify = await client.validate_customer(
                                first_name=creds["first_name"],
                                last_name=creds["last_name"],
                                email=creds["email"],
                                mobile=creds["mobile"],
                            )
                            if verify.get("success"):
                                v_email = (verify.get("email") or (verify.get("data") or {}).get("email") or "").strip()
                                if v_email and v_email.lower() == str(new_email).strip().lower():
                                    confirmed = True
                                    confirmed_data = verify.get("data") if isinstance(verify.get("data"), dict) else None
                                    logger.info("ServiceFlow.update_email: timeout verified as SUCCESS (req_id=%s)", request_id)
                                else:
                                    logger.warning(
                                        "ServiceFlow.update_email: timeout verify mismatch (req_id=%s) email_match=%s",
                                        request_id,
                                        bool(v_email and v_email.lower() == str(new_email).strip().lower()),
                                    )
                        except Exception as e:
                            logger.warning("ServiceFlow.update_email: timeout verify failed: %s", e)

                if confirmed:
                    response = "✅ Your email has been updated successfully.\n\nIs there anything else I can help you with?"
                    updates = {"customer_data": confirmed_data} if confirmed_data else {}
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": None,
                        "messages": [AIMessage(content=response)],
                        **(updates or {}),
                    }

                # Failure: keep the service flow active and enter action_recovery
                error_msg = result.get("error", "") or "I couldn't update your email."
                kind = "system" if (status_code >= 500 or status_code == 408 or timeout_type) else "invalid"
                if kind == "system":
                    prompt = (
                        f"⚠️ I couldn’t update your email due to a system issue.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Try again now\n"
                        "2) Re-enter the email\n"
                        "3) Cancel"
                    )
                else:
                    prompt = (
                        f"❌ I couldn’t update your email.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Re-enter the email\n"
                        "2) Try again\n"
                        "3) Cancel"
                    )

                ss = dict(state.get("service_slots") or {})
                ss["_action_recovery_kind"] = kind
                ss["_action_recovery_last_error"] = str(error_msg)[:200]
                if request_id:
                    ss["_action_recovery_request_id"] = request_id

                return {
                    "service_action": "update_email",
                    "service_slots": ss,
                    "service_pending_slot": "action_recovery",
                    "messages": [AIMessage(content=prompt)],
                }
        
        # =================================================================
        # UPDATE MOBILE
        # =================================================================
        elif action == "update_mobile":
            new_mobile = None
            
            service_pending_slot = state.get("service_pending_slot")

            # Handle update confirmation FIRST (before reading new_mobile)
            if service_pending_slot == "_confirm_update" and service_slots.get("_confirm_update"):
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                user_reply = (last_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct"}
                if user_reply in _yes_words:
                    new_mobile = service_slots.get("_pending_new_mobile", "")
                    service_slots = dict(service_slots)
                    service_slots.pop("_pending_new_mobile", None)
                    logger.info("ServiceFlow.update_mobile: user confirmed update")
                    # Keep _confirm_update=True so the prompt section is skipped; fall through to API call
                else:
                    logger.info("ServiceFlow.update_mobile: user declined confirmation")
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_update", None)
                    service_slots.pop("_pending_new_mobile", None)
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — the mobile update has been cancelled.\n\n" + menu["messages"][0].content
                        )],
                    }

            elif service_pending_slot == "new_mobile":
                # Only accept it if user provided it THIS turn (avoid reusing validation mobile)
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                new_mobile = _get_latest_value_from_user_message(pii_mapping, last_msg, "MOBILE")

                # Fallback: check if user typed a number directly (should be rare if masking works)
                if not new_mobile and last_msg:
                    text = last_msg.strip()
                    if any(ch.isdigit() for ch in text):
                        new_mobile = text

                # If no mobile found AND user typed something, check for decline/exit intent
                # Instead of cancelling directly, route back to orchestrator confirmation
                if not new_mobile and last_msg and last_msg.strip():
                    _decline_keywords = [
                        "no need", "no change", "don't", "dont", "cancel",
                        "never mind", "nevermind", "forget", "thanks", "thank",
                        "later", "not now", "skip", "leave", "stop", "quit",
                        "exit", "nvm", "nah", "ok", "okay", "done", "fine",
                        "good", "all good", "that's all", "thats all",
                    ]
                    raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", last_msg.strip().lower()).strip()
                    if any(kw in raw_lower for kw in _decline_keywords):
                        logger.info("ServiceFlow.update_mobile: decline detected '%s', asking confirmation", raw_lower[:30])
                        service_slots = dict(service_slots)
                        service_slots["_confirm_cancel"] = True
                        service_slots["_cancel_return_slot"] = "new_mobile"
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "_confirm_cancel",
                            "messages": [AIMessage(
                                content="Are you sure you want to cancel the mobile update? (Yes / No)"
                            )],
                        }
            # else: First time entering update_mobile - we MUST ask for the new number.
            # We should NOT use the validation mobile. The user needs to provide a NEW mobile.
            
            if not new_mobile:
                return {
                    "service_pending_slot": "new_mobile",
                    "messages": [AIMessage(content="What would you like your new mobile number to be?")],
                }

            # Validate + normalize the mobile before sending to API
            is_valid, error_msg = _validate_mobile(new_mobile)
            if not is_valid:
                return {
                    "service_pending_slot": "new_mobile",
                    "messages": [AIMessage(content=f"⚠️ {error_msg}\n\nWhat would you like your new mobile number to be?")],
                }

            new_mobile = _normalize_mobile(new_mobile) or new_mobile
            
            # =====================================================================
            # ENHANCEMENT: Compare new mobile with current mobile from API
            # If the new mobile is the same as the existing one, reject the update.
            # =====================================================================
            current_mobile = (customer_data.get("phone") or "").strip()
            # Normalize current mobile for comparison (remove common prefixes/formatting)
            current_mobile_normalized = _normalize_mobile(current_mobile) or current_mobile
            
            if current_mobile_normalized and new_mobile == current_mobile_normalized:
                logger.info("ServiceFlow.update_mobile.rejected: same_as_current mobile=%s", 
                           new_mobile[:4] + "****" if new_mobile else "")
                return {
                    "service_pending_slot": "new_mobile",
                    "messages": [AIMessage(
                        content=(
                            "⚠️ The mobile number you provided is the same as your current mobile number on file.\n\n"
                            "Please provide a *different* mobile number if you wish to update it, "
                            "or let me know if there's something else I can help you with."
                        )
                    )],
                }
            
            # =================================================================
            # CONFIRMATION: Ask user to confirm before updating
            # =================================================================
            if not service_slots.get("_confirm_update"):
                service_slots = dict(service_slots)
                service_slots["_confirm_update"] = True
                service_slots["_pending_new_mobile"] = new_mobile
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_update",
                    "messages": [AIMessage(
                        content=f"Please confirm: Update your mobile number to *{new_mobile}*?\n\nReply *Yes* to confirm or *No* to cancel."
                    )],
                }

            # Log masked mobile going to the API for debugging without exposing PII
            try:
                masker = get_pii_masker()
                masked_mobile, _ = masker.mask(new_mobile, session_id="service_debug")
                logger.info("ServiceFlow.update_mobile.request mobile=%s", masked_mobile)
            except Exception as log_err:
                logger.warning("ServiceFlow.update_mobile.log_failed: %s", log_err)

            result = await client.update_mobile(customer_nric, new_mobile)
            
            if result.get("success"):
                response = f"✅ Your mobile number has been updated successfully!\n\nIs there anything else I can help you with?"
                updated_data = result.get("data") if isinstance(result.get("data"), dict) else None
                updates: Dict[str, Any] = {}
                if updated_data:
                    updates["customer_data"] = updated_data

                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "messages": [AIMessage(content=response)],
                    **(updates or {}),
                }
            else:
                error_msg = result.get("error", "") or "I couldn't update your mobile number."
                status_code = int(result.get("status_code", 0) or 0)
                timeout_type = str(result.get("timeout_type") or "")
                request_id = str(result.get("request_id") or "")
                kind = "system" if (status_code >= 500 or status_code == 408 or timeout_type) else "invalid"

                if kind == "system":
                    prompt = (
                        f"⚠️ I couldn’t update your mobile number due to a system issue.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Try again now\n"
                        "2) Re-enter the mobile number\n"
                        "3) Cancel"
                    )
                else:
                    prompt = (
                        f"❌ I couldn’t update your mobile number.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Re-enter the mobile number\n"
                        "2) Try again\n"
                        "3) Cancel"
                    )

                ss = dict(state.get("service_slots") or {})
                ss["_action_recovery_kind"] = kind
                ss["_action_recovery_last_error"] = str(error_msg)[:200]
                if request_id:
                    ss["_action_recovery_request_id"] = request_id

                return {
                    "service_action": "update_mobile",
                    "service_slots": ss,
                    "service_pending_slot": "action_recovery",
                    "messages": [AIMessage(content=prompt)],
                }
        
        # =================================================================
        # UPDATE ADDRESS
        # =================================================================
        elif action == "update_address":
            service_pending_slot = state.get("service_pending_slot")
            
            # Handle update confirmation FIRST (before reading slots)
            if service_pending_slot == "_confirm_update" and service_slots.get("_confirm_update"):
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                user_reply = (last_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct"}
                if user_reply in _yes_words:
                    service_slots = dict(service_slots)
                    logger.info("ServiceFlow.update_address: user confirmed update")
                    # Keep _confirm_update=True so the prompt section is skipped; fall through to API call
                else:
                    logger.info("ServiceFlow.update_address: user declined confirmation")
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_update", None)
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — the address update has been cancelled.\n\n" + menu["messages"][0].content
                        )],
                    }

            # Step 1: Collect postal code
            postal_code = service_slots.get("postal_code")
            if not postal_code:
                if service_pending_slot == "postal_code":
                    # Only accept it if user provided it THIS turn (avoid stale session PII)
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    postal_code = _get_latest_value_from_user_message(pii_mapping, last_msg, "POSTAL")

                    # Fallback: if not masked, attempt to parse a 6-digit code from text
                    if not postal_code and last_msg:
                        digits = re.sub(r"\D", "", last_msg)
                        if len(digits) == 6:
                            postal_code = digits
                        elif last_msg.strip():
                            # Check for decline intent before showing invalid postal error
                            _decline_kw = [
                                "no need", "no change", "don't", "dont", "cancel",
                                "never mind", "nevermind", "forget", "thanks", "thank",
                                "later", "not now", "skip", "leave", "stop", "quit",
                                "exit", "nvm", "nah", "done", "fine", "good",
                            ]
                            raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", last_msg.strip().lower()).strip()
                            if any(kw in raw_lower for kw in _decline_kw):
                                logger.info("ServiceFlow.update_address: decline detected '%s', asking confirmation", raw_lower[:30])
                                service_slots = dict(service_slots)
                                service_slots["_confirm_cancel"] = True
                                service_slots["_cancel_return_slot"] = "postal_code"
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "_confirm_cancel",
                                    "messages": [AIMessage(
                                        content="Are you sure you want to cancel the address update? (Yes / No)"
                                    )],
                                }
                            # User provided something but it's not a valid postal code
                            return {
                                "service_slots": service_slots,
                                "service_pending_slot": "postal_code",
                                "messages": [AIMessage(content="⚠️ Invalid postal code. Please enter a valid 6-digit Singapore postal code (e.g., 520161).")],
                            }
            
            if not postal_code:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="What is your new postal code?")],
                }

            # Normalize postal code before API validation
            postal_code = re.sub(r"\D", "", str(postal_code))
            if len(postal_code) != 6:
                service_slots = dict(service_slots)
                service_slots.pop("postal_code", None)
                service_slots.pop("postal_validated", None)
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="⚠️ Invalid postal code. Please enter a valid 6-digit Singapore postal code (e.g., 520161).")],
                }

            # Validate postal code using the API and auto-fill address fields
            if not service_slots.get("postal_validated"):
                postal_result = await client.get_postal_code_info(postal_code)
                if postal_result.get("success"):
                    service_slots["postal_code"] = postal_code
                    service_slots["postal_validated"] = True
                    try:
                        data_obj = postal_result.get("data") if isinstance(postal_result.get("data"), dict) else {}
                        _pr = {**(postal_result or {}), **(data_obj or {})}
                        suggested_building = str(_pr.get("buildingName") or "").strip()
                        suggested_block = str(_pr.get("blockHouseNumber") or _pr.get("blkHouseNo") or "").strip()
                        suggested_street = str(_pr.get("streetName") or "").strip()
                        if suggested_building:
                            service_slots["_postal_suggest_building_name"] = suggested_building
                        if suggested_block:
                            service_slots["_postal_suggest_house_no"] = suggested_block
                        if suggested_street:
                            service_slots["_postal_suggest_street_name"] = suggested_street
                    except Exception:
                        pass
                    logger.info("ServiceFlow.update_address: postal code validated")
                else:
                    # Postal code validation failed
                    error_msg = postal_result.get("error", "We couldn't validate that postal code.")
                    logger.warning(
                        "ServiceFlow.update_address: postal validation failed error=%s",
                        error_msg
                    )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "postal_code",
                        "messages": [AIMessage(
                            content=f"⚠️ {error_msg}\n\nPlease enter a valid 6-digit Singapore postal code."
                        )],
                    }
            
            # Step 2: Collect block/house number
            house_no = service_slots.get("house_no")
            _declined_house_suggestion = False
            if not house_no:
                if service_pending_slot == "house_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_house_no"):
                            text = str(service_slots["_postal_suggest_house_no"]).strip()
                        elif lower in ("no", "n", "na", "nah", "nope") and service_slots.get("_postal_suggest_house_no"):
                            _declined_house_suggestion = True
                            text = ""
                        if text and re.search(r"[A-Za-z0-9]", text):
                            house_no = text
                            service_slots["house_no"] = house_no
            
            if not house_no:
                if _declined_house_suggestion:
                    prompt = "No problem. Please type your block or house number."
                else:
                    suggested = str(service_slots.get("_postal_suggest_house_no") or "").strip()
                    if suggested:
                        prompt = (
                            "What is your block or house number?\n\n"
                            f"For this postal code, I found: *{suggested}*.\n"
                            "Reply *yes* to use it, or type your block/house number."
                        )
                    else:
                        prompt = "What is your block or house number? (e.g., BLK 123 or 45)"
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "house_no",
                    "messages": [AIMessage(content=prompt)],
                }
            
            # Step 3: Collect street name
            street_name = service_slots.get("street_name")
            _declined_street_suggestion = False
            if not street_name:
                if service_pending_slot == "street_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_street_name"):
                            text = str(service_slots["_postal_suggest_street_name"]).strip()
                        elif lower in ("no", "n", "na", "nah", "nope") and service_slots.get("_postal_suggest_street_name"):
                            _declined_street_suggestion = True
                            text = ""
                        if text and re.search(r"[A-Za-z0-9]", text):
                            street_name = text
                            service_slots["street_name"] = street_name
            
            if not street_name:
                if _declined_street_suggestion:
                    prompt = "No problem. Please type your street name."
                else:
                    suggested = str(service_slots.get("_postal_suggest_street_name") or "").strip()
                    if suggested:
                        prompt = (
                            "What is your street name?\n\n"
                            f"For this postal code, I found: *{suggested}*.\n"
                            "Reply *yes* to use it, or type your street name."
                        )
                    else:
                        prompt = "What is your street name?"
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "street_name",
                    "messages": [AIMessage(content=prompt)],
                }
            
            # Step 4: Collect unit number
            unit_no = service_slots.get("unit_no")
            if not unit_no:
                if service_pending_slot == "unit_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        unit_no = last_msg.strip()
                        service_slots["unit_no"] = unit_no
            
            if not unit_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "unit_no",
                    "messages": [AIMessage(content="What is your unit number? (e.g., #10-10)")],
                }

            # Step 5: Collect building name (required)
            building_name = service_slots.get("building_name")
            if not (building_name and str(building_name).strip()):
                _declined_suggestion = False
                if service_pending_slot == "building_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_building_name"):
                            text = str(service_slots.get("_postal_suggest_building_name") or "").strip()
                            lower = text.lower()
                        elif lower in ("no", "n", "na", "nah", "nope") and service_slots.get("_postal_suggest_building_name"):
                            _declined_suggestion = True
                            text = ""
                        if lower in ("skip", "none", "n/a", "-"):
                            text = ""
                        # Basic sanity check: require at least one alphanumeric character.
                        if text and re.search(r"[A-Za-z0-9]", text):
                            building_name = text
                            service_slots["building_name"] = building_name
                        else:
                            building_name = None

                if not (building_name and str(building_name).strip()):
                    if _declined_suggestion:
                        prompt = "No problem. Please type your building name."
                    else:
                        suggested = str(service_slots.get("_postal_suggest_building_name") or "").strip()
                        if suggested:
                            prompt = (
                                "What is your building name?\n\n"
                                f"For this postal code, I found: *{suggested}*.\n"
                                "Reply *yes* to use it, or type your building name."
                            )
                        else:
                            prompt = "What is your building name?"
                        if service_pending_slot == "building_name":
                            prompt = (
                                "⚠️ Please enter a valid building name (it’s required).\n\n"
                                + prompt
                            )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "building_name",
                        "messages": [AIMessage(content=prompt)],
                    }

            building_name = str(building_name).strip()
            
            # =================================================================
            # CONFIRMATION: Ask user to confirm before updating address
            # =================================================================
            if not service_slots.get("_confirm_update"):
                summary = (
                    f"Please confirm: Update your address to the following?\n\n"
                    f"• Block/House No: *{house_no}*\n"
                    f"• Street Name: *{street_name}*\n"
                    f"• Building: *{building_name}*\n"
                    f"• Unit No: *{unit_no}*\n"
                    f"• Postal Code: *{postal_code}*\n\n"
                    f"Reply *Yes* to confirm or *No* to cancel."
                )
                service_slots = dict(service_slots)
                service_slots["_confirm_update"] = True
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_update",
                    "messages": [AIMessage(content=summary)],
                }

            # All required fields collected, call the update API
            result = await client.update_address(
                nric=customer_nric,
                postal_code=postal_code,
                unit_no=unit_no,
                house_no=house_no,
                street_name=street_name,
                building_name=building_name,
            )
            
            if result.get("success"):
                response = "✅ Your address has been updated successfully!\n\nIs there anything else I can help you with?"
                updated_data = result.get("data") if isinstance(result.get("data"), dict) else None
                updates: Dict[str, Any] = {}
                if updated_data:
                    updates["customer_data"] = updated_data

                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "messages": [AIMessage(content=response)],
                    **(updates or {}),
                }
            else:
                error_msg = result.get("error", "") or "I couldn't update your address."
                status_code = int(result.get("status_code", 0) or 0)
                timeout_type = str(result.get("timeout_type") or "")
                request_id = str(result.get("request_id") or "")
                kind = "system" if (status_code >= 500 or status_code == 408 or timeout_type) else "invalid"

                if kind == "system":
                    prompt = (
                        f"⚠️ I couldn’t update your address due to a system issue.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Try again now\n"
                        "2) Re-enter the address details\n"
                        "3) Cancel"
                    )
                else:
                    prompt = (
                        f"❌ I couldn’t update your address.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Re-enter the address details\n"
                        "2) Try again\n"
                        "3) Cancel"
                    )

                ss = dict(service_slots)
                ss["_action_recovery_kind"] = kind
                ss["_action_recovery_last_error"] = str(error_msg)[:200]
                if request_id:
                    ss["_action_recovery_request_id"] = request_id

                return {
                    "service_action": "update_address",
                    "service_slots": ss,
                    "service_pending_slot": "action_recovery",
                    "messages": [AIMessage(content=prompt)],
                }
        
        # =================================================================
        # UPDATE INSURED ADDRESS (Home Protect)
        # =================================================================
        elif action == "update_insured_address":
            service_pending_slot = state.get("service_pending_slot")
            
            # Handle update confirmation FIRST (before reading slots)
            if service_pending_slot == "_confirm_update" and service_slots.get("_confirm_update"):
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                user_reply = (last_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct"}
                if user_reply in _yes_words:
                    service_slots = dict(service_slots)
                    logger.info("ServiceFlow.update_insured_address: user confirmed update")
                    # Keep _confirm_update=True so the prompt section is skipped
                else:
                    logger.info("ServiceFlow.update_insured_address: user declined confirmation")
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_update", None)
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — the insured address update has been cancelled.\n\n" + menu["messages"][0].content
                        )],
                    }

            # Step 1: Select which Home Protect360 policy to update
            policy_no = service_slots.get("insured_policy_no")
            if not policy_no:
                if service_pending_slot == "insured_policy_no":
                    # Handle selection OR pagination from THIS turn.
                    last_msg = _get_last_user_message(state.get("messages", []) or [])

                    # Pre-compute eligible Home policies for validation + display.
                    policies = customer_data.get("policies", [])
                    home_policies = [
                        p
                        for p in policies
                        if "home protect360" in str(p.get("productName", "") or "").lower()
                    ]
                    eligible_set = {
                        str(p.get("policyNo") or "").strip().upper()
                        for p in home_policies
                        if p.get("policyNo")
                    }

                    selected = _extract_policy_no_from_user_msg(pii_mapping, last_msg)
                    if selected:
                        policy_no = selected
                        is_valid, err = _validate_policy_no(policy_no)
                        if not is_valid:
                            ordered = _order_policies_for_display(home_policies)
                            offset, page_size = _get_policy_page_state(service_slots, "insured_policy_no")
                            page_text = _render_policy_page(
                                ordered,
                                title="Which Home Protect360 policy would you like to update the insured address for?",
                                offset=offset,
                                page_size=page_size,
                            )
                            return {
                                "service_slots": dict(service_slots),
                                "service_pending_slot": "insured_policy_no",
                                "messages": [AIMessage(content=f"⚠️ {err}\n\n{page_text}")],
                            }

                        if eligible_set and policy_no not in eligible_set:
                            ordered = _order_policies_for_display(home_policies)
                            offset, page_size = _get_policy_page_state(service_slots, "insured_policy_no")
                            page_text = _render_policy_page(
                                ordered,
                                title="Which Home Protect360 policy would you like to update the insured address for?",
                                offset=offset,
                                page_size=page_size,
                            )
                            return {
                                "service_slots": dict(service_slots),
                                "service_pending_slot": "insured_policy_no",
                                "messages": [AIMessage(
                                    content=(
                                        f"⚠️ Policy *{policy_no}* isn’t a Home Protect360 policy on your account.\n\n{page_text}"
                                    )
                                )],
                            }

                        service_slots = dict(service_slots)
                        service_slots["insured_policy_no"] = policy_no
                        _clear_policy_page_state(service_slots, "insured_policy_no")
                    else:
                        cmd = _parse_policy_page_command(last_msg)
                        if cmd:
                            ordered = _order_policies_for_display(home_policies)
                            offset, page_size = _get_policy_page_state(service_slots, "insured_policy_no")
                            if cmd.get("page_size"):
                                page_size = int(cmd["page_size"])
                            if page_size < _POLICY_PAGE_MIN:
                                page_size = _POLICY_PAGE_MIN
                            if page_size > _POLICY_PAGE_MAX:
                                page_size = _POLICY_PAGE_MAX
                            if page_size:
                                offset = (offset // page_size) * page_size

                            total = len(ordered)
                            if cmd["cmd"] == "next":
                                if offset + page_size < total:
                                    offset += page_size
                            elif cmd["cmd"] == "prev":
                                offset = max(0, offset - page_size)
                            elif cmd["cmd"] == "first":
                                offset = 0
                            elif cmd["cmd"] == "last":
                                if total > 0:
                                    offset = max(((total - 1) // page_size) * page_size, 0)
                            elif cmd["cmd"] == "all":
                                offset = 0
                                page_size = min(_POLICY_PAGE_MAX, max(_POLICY_PAGE_MIN, total))

                            _set_policy_page_state(service_slots, "insured_policy_no", offset, page_size)
                            page_text = _render_policy_page(
                                ordered,
                                title="Which Home Protect360 policy would you like to update the insured address for?",
                                offset=offset,
                                page_size=page_size,
                            )
                            return {
                                "service_action": "update_insured_address",
                                "service_slots": dict(service_slots),
                                "service_pending_slot": "insured_policy_no",
                                "messages": [AIMessage(content=page_text)],
                            }

                        # Unknown input: re-show current page with error message
                        ordered = _order_policies_for_display(home_policies)
                        offset, page_size = _get_policy_page_state(service_slots, "insured_policy_no")
                        page_text = _render_policy_page(
                            ordered,
                            title="Which Home Protect360 policy would you like to update the insured address for?",
                            offset=offset,
                            page_size=page_size,
                        )
                        return {
                            "service_action": "update_insured_address",
                            "service_slots": dict(service_slots),
                            "service_pending_slot": "insured_policy_no",
                            "messages": [AIMessage(content=f"⚠️ That doesn't match any of your policies. Please reply with a policy number from the list below.\n\n{page_text}")],
                        }
            
            if not policy_no:
                # Find Home Protect policies from customer data
                policies = customer_data.get("policies", [])
                home_policies = [
                    p
                    for p in policies
                    if "home protect360" in str(p.get("productName", "") or "").lower()
                ]
                
                if not home_policies:
                    return {
                        "messages": [AIMessage(content="I couldn't find any *Home Protect360* policies on your account. Insured address updates are only available for Home Protect360 policies.")],
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": None,
                    }
                
                # Show paginated list for user to choose
                service_slots = dict(service_slots)
                _set_policy_page_state(service_slots, "insured_policy_no", 0, _POLICY_PAGE_DEFAULT)
                ordered = _order_policies_for_display(home_policies)
                page_text = _render_policy_page(
                    ordered,
                    title="Which Home Protect360 policy would you like to update the insured address for?",
                    offset=0,
                    page_size=_POLICY_PAGE_DEFAULT,
                )
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "insured_policy_no",
                    "messages": [AIMessage(
                        content=page_text
                    )],
                }
            
            # Verify it's an eligible Home Protect policy on the customer's account
            policies = customer_data.get("policies", [])
            home_policies = [
                p
                for p in policies
                if "home protect360" in str(p.get("productName", "") or "").lower()
            ]
            eligible_set = {
                str(p.get("policyNo") or "").strip().upper()
                for p in home_policies
                if p.get("policyNo")
            }
            if eligible_set and str(policy_no).strip().upper() not in eligible_set:
                service_slots = dict(service_slots)
                _set_policy_page_state(service_slots, "insured_policy_no", 0, _POLICY_PAGE_DEFAULT)
                ordered = _order_policies_for_display(home_policies)
                page_text = _render_policy_page(
                    ordered,
                    title="Which Home Protect360 policy would you like to update the insured address for?",
                    offset=0,
                    page_size=_POLICY_PAGE_DEFAULT,
                )
                return {
                    "service_action": "update_insured_address",
                    "service_slots": service_slots,
                    "service_pending_slot": "insured_policy_no",
                    "messages": [AIMessage(content=f"⚠️ Policy *{policy_no}* isn’t eligible for insured address update.\n\n{page_text}")],
                }
            
            # Step 2: Collect postal code
            postal_code = service_slots.get("postal_code")
            if not postal_code:
                if service_pending_slot == "postal_code":
                    # Only accept it if user provided it THIS turn (avoid stale session PII)
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    postal_code = _get_latest_value_from_user_message(pii_mapping, last_msg, "POSTAL")

                    # Fallback: parse digits if not masked
                    if not postal_code and last_msg:
                        digits = re.sub(r"\D", "", last_msg)
                        if len(digits) == 6:
                            postal_code = digits
                        elif last_msg.strip():
                            # Check for decline intent before showing invalid postal error
                            _decline_kw = [
                                "no need", "no change", "don't", "dont", "cancel",
                                "never mind", "nevermind", "forget", "thanks", "thank",
                                "later", "not now", "skip", "leave", "stop", "quit",
                                "exit", "nvm", "nah", "done", "fine", "good",
                            ]
                            raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", last_msg.strip().lower()).strip()
                            if any(kw in raw_lower for kw in _decline_kw):
                                logger.info("ServiceFlow.update_insured_address: decline detected '%s', asking confirmation", raw_lower[:30])
                                service_slots = dict(service_slots)
                                service_slots["_confirm_cancel"] = True
                                service_slots["_cancel_return_slot"] = "postal_code"
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "_confirm_cancel",
                                    "messages": [AIMessage(
                                        content="Are you sure you want to cancel the insured address update? (Yes / No)"
                                    )],
                                }
                            # User provided something but it's not a valid postal code
                            return {
                                "service_slots": service_slots,
                                "service_pending_slot": "postal_code",
                                "messages": [AIMessage(content="⚠️ Invalid postal code. Please enter a valid 6-digit Singapore postal code (e.g., 520161).")],
                            }
            
            if not postal_code:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="What is the new postal code for the insured property?")],
                }

            postal_code = re.sub(r"\D", "", str(postal_code))
            if len(postal_code) != 6:
                service_slots = dict(service_slots)
                service_slots.pop("postal_code", None)
                service_slots.pop("postal_validated", None)
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="⚠️ Invalid postal code. Please enter a valid 6-digit Singapore postal code (e.g., 520161).")],
                }
            service_slots["postal_code"] = postal_code

            # Validate postal code and auto-fill address fields from API
            if not service_slots.get("postal_validated"):
                postal_result = await client.get_postal_code_info(postal_code)
                if postal_result.get("success"):
                    service_slots["postal_validated"] = True
                    try:
                        data_obj = postal_result.get("data") if isinstance(postal_result.get("data"), dict) else {}
                        _pr = {**(postal_result or {}), **(data_obj or {})}
                        suggested_building = str(_pr.get("buildingName") or "").strip()
                        suggested_block = str(_pr.get("blockHouseNumber") or _pr.get("blkHouseNo") or "").strip()
                        suggested_street = str(_pr.get("streetName") or "").strip()
                        if suggested_building:
                            service_slots["_postal_suggest_building_name"] = suggested_building
                        if suggested_block:
                            service_slots["_postal_suggest_house_no"] = suggested_block
                        if suggested_street:
                            service_slots["_postal_suggest_street_name"] = suggested_street
                    except Exception:
                        pass

                    logger.info("ServiceFlow.update_insured_address: postal validated")
                else:
                    error_msg = postal_result.get("error", "We couldn't validate that postal code.")
                    logger.warning(
                        "ServiceFlow.update_insured_address: postal validation failed error=%s",
                        error_msg,
                    )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "postal_code",
                        "messages": [AIMessage(content=f"⚠️ {error_msg}\n\nPlease enter a valid 6-digit Singapore postal code.")],
                    }
            
            # Step 3: Collect block/house number
            house_no = service_slots.get("house_no")
            if not house_no:
                if service_pending_slot == "house_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_house_no"):
                            text = str(service_slots["_postal_suggest_house_no"]).strip()
                        if text and re.search(r"[A-Za-z0-9]", text):
                            house_no = text
                            service_slots["house_no"] = house_no
            
            if not house_no:
                suggested = str(service_slots.get("_postal_suggest_house_no") or "").strip()
                if suggested:
                    prompt = (
                        "What is the block or house number?\n\n"
                        f"For this postal code, I found: *{suggested}*.\n"
                        "Reply *yes* to use it, or type your block/house number."
                    )
                else:
                    prompt = "What is the block or house number? (e.g., BLK 123 or 45)"
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "house_no",
                    "messages": [AIMessage(content=prompt)],
                }
            
            # Step 4: Collect street name
            street_name = service_slots.get("street_name")
            if not street_name:
                if service_pending_slot == "street_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_street_name"):
                            text = str(service_slots["_postal_suggest_street_name"]).strip()
                        if text and re.search(r"[A-Za-z0-9]", text):
                            street_name = text
                            service_slots["street_name"] = street_name
            
            if not street_name:
                suggested = str(service_slots.get("_postal_suggest_street_name") or "").strip()
                if suggested:
                    prompt = (
                        "What is the street name?\n\n"
                        f"For this postal code, I found: *{suggested}*.\n"
                        "Reply *yes* to use it, or type your street name."
                    )
                else:
                    prompt = "What is the street name?"
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "street_name",
                    "messages": [AIMessage(content=prompt)],
                }
            
            # Step 5: Collect unit number
            unit_no = service_slots.get("unit_no")
            if not unit_no:
                if service_pending_slot == "unit_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        candidate = last_msg.strip()
                        is_valid, err = _validate_unit_no(candidate)
                        if not is_valid:
                            return {
                                "service_slots": service_slots,
                                "service_pending_slot": "unit_no",
                                "messages": [AIMessage(content=f"⚠️ {err}\n\nWhat is the unit number? (e.g., #10-10)")],
                            }
                        unit_no = _normalize_unit_no(candidate) or candidate
                        service_slots["unit_no"] = unit_no
            
            if not unit_no:
                addr_bits: List[str] = []
                if service_slots.get("postal_validated"):
                    # Give user context (derived from postal code) to reduce errors.
                    if house_no:
                        addr_bits.append(f"• Block/House: {house_no}")
                    if street_name:
                        addr_bits.append(f"• Street: {street_name}")
                    b = service_slots.get("building_name")
                    if b:
                        addr_bits.append(f"• Building: {b}")
                addr_hint = ""
                if addr_bits:
                    addr_hint = "For that postal code, I found:\n" + "\n".join(addr_bits) + "\n\n"
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "unit_no",
                    "messages": [AIMessage(content=addr_hint + "What is the unit number? (e.g., #10-10)")],
                }

            # Step 6: Collect building name (required)
            building_name = service_slots.get("building_name")
            if not (building_name and str(building_name).strip()):
                _declined_suggestion = False
                if service_pending_slot == "building_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        lower = text.lower()
                        if lower in ("yes", "y") and service_slots.get("_postal_suggest_building_name"):
                            text = str(service_slots.get("_postal_suggest_building_name") or "").strip()
                            lower = text.lower()
                        elif lower in ("no", "n", "na", "nah", "nope") and service_slots.get("_postal_suggest_building_name"):
                            _declined_suggestion = True
                            text = ""
                        if lower in ("skip", "none", "n/a", "-"):
                            text = ""
                        # Basic sanity check: require at least one alphanumeric character.
                        if text and re.search(r"[A-Za-z0-9]", text):
                            building_name = text
                            service_slots["building_name"] = building_name
                        else:
                            building_name = None

                if not (building_name and str(building_name).strip()):
                    if _declined_suggestion:
                        prompt = "No problem. Please type the building name."
                    else:
                        suggested = str(service_slots.get("_postal_suggest_building_name") or "").strip()
                        if suggested:
                            prompt = (
                                "What is the building name?\n\n"
                                f"For this postal code, I found: *{suggested}*.\n"
                                "Reply *yes* to use it, or type the building name."
                            )
                        else:
                            prompt = "What is the building name?"
                        if service_pending_slot == "building_name":
                            prompt = "⚠️ Please enter a valid building name (it’s required).\n\n" + prompt
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "building_name",
                        "messages": [AIMessage(content=prompt)],
                    }

            # Normalize building name
            building_name = building_name or ""

            # =================================================================
            # CONFIRMATION: Ask user to confirm before updating insured address
            # =================================================================
            if not service_slots.get("_confirm_update"):
                summary = (
                    f"Please confirm: Update the insured address for policy *{policy_no}* to:\n\n"
                    f"• Block/House No: *{house_no}*\n"
                    f"• Street Name: *{street_name}*\n"
                    f"• Building: *{building_name}*\n"
                    f"• Unit No: *{unit_no}*\n"
                    f"• Postal Code: *{postal_code}*\n\n"
                    f"Reply *Yes* to confirm or *No* to cancel."
                )
                service_slots = dict(service_slots)
                service_slots["_confirm_update"] = True
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_update",
                    "messages": [AIMessage(content=summary)],
                }

            # Normalize inputs for the Home Protect insured address endpoint
            # (backend expects a strict-ish address format)
            unit_no_api = _normalize_unit_no(unit_no) or str(unit_no).strip().lstrip("#").strip()

            house_raw = str(house_no).strip()
            house_digits = re.findall(r"\d+", house_raw)
            house_no_api = house_digits[0] if house_digits else house_raw

            street_name_api = str(street_name).strip().upper()
            building_name_api = str(building_name).strip().upper() if building_name else ""
            
            # Log the request
            logger.info("ServiceFlow.update_insured_address: calling API")
            
            result = await client.update_insured_address(
                policy_no=policy_no,
                postal_code=postal_code,
                unit_no=unit_no_api,
                house_no=house_no_api,
                street_name=street_name_api,
                building_name=building_name_api,
            )
            
            if result.get("success"):
                response = f"✅ The insured address for policy {policy_no} has been updated successfully!\n\nIs there anything else I can help you with?"
                updated_data = result.get("data") if isinstance(result.get("data"), dict) else None
                updates: Dict[str, Any] = {}
                if updated_data:
                    updates["customer_data"] = updated_data

                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "messages": [AIMessage(content=response)],
                    **(updates or {}),
                }
            else:
                error_msg = result.get("error", "") or "I couldn't update the insured address."
                status_code = int(result.get("status_code", 0) or 0)
                timeout_type = str(result.get("timeout_type") or "")
                request_id = str(result.get("request_id") or "")
                kind = "system" if (status_code >= 500 or status_code == 408 or timeout_type) else "invalid"

                if kind == "system":
                    prompt = (
                        f"⚠️ I couldn’t update the insured address due to a system issue.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Try again now\n"
                        "2) Re-enter the address details\n"
                        "3) Cancel"
                    )
                else:
                    prompt = (
                        f"❌ I couldn’t update the insured address.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Re-enter the address details\n"
                        "2) Try again\n"
                        "3) Cancel"
                    )

                ss = dict(service_slots)
                ss["_action_recovery_kind"] = kind
                ss["_action_recovery_last_error"] = str(error_msg)[:200]
                if request_id:
                    ss["_action_recovery_request_id"] = request_id

                return {
                    "service_action": "update_insured_address",
                    "service_slots": ss,
                    "service_pending_slot": "action_recovery",
                    "messages": [AIMessage(content=prompt)],
                }
        
        # =================================================================
        # UPDATE PAYMENT
        # =================================================================
        elif action == "update_payment":
            service_pending_slot = state.get("service_pending_slot")
            
            # Handle update confirmation FIRST (before reading slots)
            if service_pending_slot == "_confirm_update" and service_slots.get("_confirm_update"):
                last_msg = _get_last_user_message(state.get("messages", []) or [])
                user_reply = (last_msg or "").strip().lower()
                _yes_words = {"yes", "y", "yeah", "yep", "yea", "sure", "ok", "okay", "confirm", "correct"}
                if user_reply in _yes_words:
                    service_slots = dict(service_slots)
                    logger.info("ServiceFlow.update_payment: user confirmed update")
                    # Keep _confirm_update=True so the prompt section is skipped
                else:
                    # Check if user wants to change card type instead of cancelling
                    _change_card_phrases = [
                        "change card", "switch card", "different card",
                        "change type", "switch type", "other card",
                        "wrong card", "not this card", "go back",
                    ]
                    if any(phrase in user_reply for phrase in _change_card_phrases):
                        logger.info("ServiceFlow.update_payment.confirm: change card trigger '%s'", user_reply[:30])
                        service_slots = dict(service_slots)
                        service_slots.pop("card_type", None)
                        service_slots.pop("card_no", None)
                        service_slots.pop("card_expiry", None)
                        service_slots.pop("_confirm_update", None)
                        _CARD_TYPE_DISPLAY_CONFIRM = "VISA, MASTERCARD"
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "card_type",
                            "messages": [AIMessage(
                                content=f"Sure! Which card type would you like to use?\n\nPlease choose: {_CARD_TYPE_DISPLAY_CONFIRM}"
                            )],
                        }
                    logger.info("ServiceFlow.update_payment: user declined confirmation")
                    service_slots = dict(service_slots)
                    service_slots.pop("_confirm_update", None)
                    menu = _service_ask_action(state)
                    return {
                        "service_action": None,
                        "service_slots": {},
                        "service_pending_slot": "service_action_choice",
                        "messages": [AIMessage(
                            content="No problem — the payment update has been cancelled.\n\n" + menu["messages"][0].content
                        )],
                    }

            # Step 1: Select which policy to update payment for
            policy_no = service_slots.get("payment_policy_no")
            if not policy_no:
                if service_pending_slot == "payment_policy_no":
                    # Handle selection OR pagination from THIS turn.
                    last_msg = _get_last_user_message(state.get("messages", []) or [])

                    policies = customer_data.get("policies", [])
                    ordered = _order_policies_for_display(policies or [])

                    selected = _extract_policy_no_from_user_msg(pii_mapping, last_msg)
                    if selected:
                        policy_no = selected
                        is_valid, err = _validate_policy_no(policy_no)
                        if not is_valid:
                            offset, page_size = _get_policy_page_state(service_slots, "payment_policy_no")
                            page_text = _render_policy_page(
                                ordered,
                                title="Which policy would you like to update payment for?",
                                offset=offset,
                                page_size=page_size,
                            )
                            return {
                                "service_slots": dict(service_slots),
                                "service_pending_slot": "payment_policy_no",
                                "messages": [AIMessage(content=f"⚠️ {err}\n\n{page_text}")],
                            }
                        service_slots = dict(service_slots)
                        service_slots["payment_policy_no"] = policy_no
                        _clear_policy_page_state(service_slots, "payment_policy_no")
                    else:
                        cmd = _parse_policy_page_command(last_msg)
                        if cmd:
                            offset, page_size = _get_policy_page_state(service_slots, "payment_policy_no")
                            if cmd.get("page_size"):
                                page_size = int(cmd["page_size"])
                            if page_size < _POLICY_PAGE_MIN:
                                page_size = _POLICY_PAGE_MIN
                            if page_size > _POLICY_PAGE_MAX:
                                page_size = _POLICY_PAGE_MAX
                            if page_size:
                                offset = (offset // page_size) * page_size

                            total = len(ordered)
                            if cmd["cmd"] == "next":
                                if offset + page_size < total:
                                    offset += page_size
                            elif cmd["cmd"] == "prev":
                                offset = max(0, offset - page_size)
                            elif cmd["cmd"] == "first":
                                offset = 0
                            elif cmd["cmd"] == "last":
                                if total > 0:
                                    offset = max(((total - 1) // page_size) * page_size, 0)
                            elif cmd["cmd"] == "all":
                                offset = 0
                                page_size = min(_POLICY_PAGE_MAX, max(_POLICY_PAGE_MIN, total))

                            _set_policy_page_state(service_slots, "payment_policy_no", offset, page_size)
                            page_text = _render_policy_page(
                                ordered,
                                title="Which policy would you like to update payment for?",
                                offset=offset,
                                page_size=page_size,
                            )
                            return {
                                "service_action": "update_payment",
                                "service_slots": dict(service_slots),
                                "service_pending_slot": "payment_policy_no",
                                "messages": [AIMessage(content=page_text)],
                            }

                        # Unknown input: re-show current page with error message
                        offset, page_size = _get_policy_page_state(service_slots, "payment_policy_no")
                        page_text = _render_policy_page(
                            ordered,
                            title="Which policy would you like to update payment for?",
                            offset=offset,
                            page_size=page_size,
                        )
                        return {
                            "service_action": "update_payment",
                            "service_slots": dict(service_slots),
                            "service_pending_slot": "payment_policy_no",
                            "messages": [AIMessage(content=f"⚠️ That doesn't match any of your policies. Please reply with a policy number from the list below.\n\n{page_text}")],
                        }
            
            if not policy_no:
                # Show list of policies for user to choose
                policies = customer_data.get("policies", [])
                if policies:
                    service_slots = dict(service_slots)
                    _set_policy_page_state(service_slots, "payment_policy_no", 0, _POLICY_PAGE_DEFAULT)
                    ordered = _order_policies_for_display(policies or [])
                    page_text = _render_policy_page(
                        ordered,
                        title="Which policy would you like to update payment for?",
                        offset=0,
                        page_size=_POLICY_PAGE_DEFAULT,
                    )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "payment_policy_no",
                        "messages": [AIMessage(
                            content=page_text
                        )],
                    }
                else:
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "payment_policy_no",
                        "messages": [AIMessage(content="Please enter the policy number you want to update payment for.")],
                    }
            
            # =============================================================
            # Card type → digit-length dictionary
            # Each card type maps to: aliases (for matching) + valid lengths
            # =============================================================
            _CARD_TYPE_CONFIG = {
                "VISA":        {"aliases": ["VISA"], "lengths": [16]},
                "MASTERCARD":  {"aliases": ["MASTERCARD", "MASTER CARD", "MASTER"], "lengths": [16]},
            }
            _CARD_TYPE_DISPLAY = ", ".join(sorted(_CARD_TYPE_CONFIG.keys()))

            # ---------------------------------------------------------
            # Step 2: Collect card TYPE first (before card number)
            # ---------------------------------------------------------
            card_type = service_slots.get("card_type")
            if not card_type:
                if service_pending_slot == "card_type":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        raw_input = last_msg.strip().upper()
                        # Exact match against aliases
                        matched_card = None
                        for canonical, cfg in _CARD_TYPE_CONFIG.items():
                            if raw_input in cfg["aliases"] or raw_input == canonical:
                                matched_card = canonical
                                break
                        # Partial match fallback (e.g. "visa card", "my amex")
                        if not matched_card:
                            for canonical, cfg in _CARD_TYPE_CONFIG.items():
                                if any(alias in raw_input for alias in cfg["aliases"]):
                                    matched_card = canonical
                                    break
                        if matched_card:
                            card_type = matched_card
                            service_slots["card_type"] = card_type
                        else:
                            # Check for decline intent before showing card type error
                            _decline_kw = [
                                "no need", "no change", "don't", "dont", "cancel",
                                "never mind", "nevermind", "forget", "thanks", "thank",
                                "later", "not now", "skip", "leave", "stop", "quit",
                                "exit", "nvm", "nah", "done", "fine", "good",
                            ]
                            raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", last_msg.strip().lower()).strip()
                            if any(kw in raw_lower for kw in _decline_kw):
                                logger.info("ServiceFlow.update_payment: decline detected '%s', asking confirmation", raw_lower[:30])
                                service_slots = dict(service_slots)
                                service_slots["_confirm_cancel"] = True
                                service_slots["_cancel_return_slot"] = "card_type"
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "_confirm_cancel",
                                    "messages": [AIMessage(
                                        content="Are you sure you want to cancel the payment update? (Yes / No)"
                                    )],
                                }
                            return {
                                "service_slots": service_slots,
                                "service_pending_slot": "card_type",
                                "messages": [AIMessage(
                                    content=(
                                        f"⚠️ *\"{last_msg.strip()}\"* is not a recognised card type.\n\n"
                                        f"Please choose from one of the following:\n"
                                        f"*{_CARD_TYPE_DISPLAY}*"
                                    )
                                )],
                            }
                    else:
                        # User sent empty / whitespace-only input
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "card_type",
                            "messages": [AIMessage(
                                content=(
                                    "⚠️ No input received.\n\n"
                                    f"Please choose a card type from: *{_CARD_TYPE_DISPLAY}*"
                                )
                            )],
                        }

            if not card_type:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_type",
                    "messages": [AIMessage(
                        content=(
                            "What type of card is this?\n\n"
                            f"Please choose from: *{_CARD_TYPE_DISPLAY}*"
                        )
                    )],
                }

            # Look up the expected digit lengths for the selected card type
            _selected_cfg = _CARD_TYPE_CONFIG.get(card_type, {})
            _expected_lengths = _selected_cfg.get("lengths", [16])
            _lengths_display = " or ".join(str(l) for l in _expected_lengths)

            # ---------------------------------------------------------
            # Step 3: Collect card NUMBER (validated by card-type length)
            # Accepts: 1234567890123456 | 1234 5678 9012 3456 | 1234-5678-9012-3456
            # ---------------------------------------------------------
            card_no = service_slots.get("card_no")
            if not card_no:
                if service_pending_slot == "card_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    # Try PII-masked value first, then fall back to raw input
                    raw_card = _get_latest_value_from_user_message(pii_mapping, last_msg, "CARD")
                    if not raw_card and last_msg:
                        raw_card = last_msg.strip()

                    if raw_card:
                        digits_only = re.sub(r"\D", "", str(raw_card))
                        if digits_only.isdigit() and len(digits_only) in _expected_lengths:
                            card_no = digits_only
                            service_slots["card_no"] = card_no
                        else:
                            # Check if user typed non-numeric junk
                            if not digits_only:
                                raw_upper = re.sub(r"\[[A-Z]+_\d+\]", "", str(raw_card)).strip().upper()
                                # Check if user is trying to change card type
                                switch_card = None
                                for canonical, cfg in _CARD_TYPE_CONFIG.items():
                                    if raw_upper in cfg["aliases"] or raw_upper == canonical:
                                        switch_card = canonical
                                        break
                                if not switch_card:
                                    for canonical, cfg in _CARD_TYPE_CONFIG.items():
                                        if any(alias in raw_upper for alias in cfg["aliases"]):
                                            switch_card = canonical
                                            break
                                if switch_card:
                                    logger.info("ServiceFlow.update_payment.card_no: card type change to %s", switch_card)
                                    service_slots = dict(service_slots)
                                    service_slots["card_type"] = switch_card
                                    service_slots.pop("card_no", None)
                                    new_cfg = _CARD_TYPE_CONFIG[switch_card]
                                    new_lengths = " or ".join(str(l) for l in new_cfg["lengths"])
                                    return {
                                        "service_slots": service_slots,
                                        "service_pending_slot": "card_no",
                                        "messages": [AIMessage(
                                            content=f"Card type changed to *{switch_card}*.\n\nPlease enter your *{switch_card}* card number ({new_lengths} digits)."
                                        )],
                                    }
                                # Check if user wants to change card type without specifying which
                                _change_card_phrases = [
                                    "change card", "switch card", "different card",
                                    "change type", "switch type", "other card",
                                    "wrong card", "not this card", "go back",
                                ]
                                raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", str(raw_card).lower()).strip()
                                if any(phrase in raw_lower for phrase in _change_card_phrases):
                                    logger.info("ServiceFlow.update_payment.card_no: change card trigger '%s'", raw_lower[:30])
                                    service_slots = dict(service_slots)
                                    service_slots.pop("card_type", None)
                                    service_slots.pop("card_no", None)
                                    return {
                                        "service_slots": service_slots,
                                        "service_pending_slot": "card_type",
                                        "messages": [AIMessage(
                                            content=f"Sure! Which card type would you like to use?\n\nPlease choose: {_CARD_TYPE_DISPLAY}"
                                        )],
                                    }
                                # Check for decline intent before showing card error
                                _decline_kw = [
                                    "no need", "no change", "don't", "dont", "cancel",
                                    "never mind", "nevermind", "forget", "thanks", "thank",
                                    "later", "not now", "skip", "leave", "stop", "quit",
                                    "exit", "nvm", "nah", "done", "fine", "good",
                                ]
                                raw_lower = re.sub(r"\[[A-Z]+_\d+\]", "", str(raw_card).lower()).strip()
                                if any(kw in raw_lower for kw in _decline_kw):
                                    logger.info("ServiceFlow.update_payment.card_no: decline detected '%s'", raw_lower[:30])
                                    service_slots = dict(service_slots)
                                    service_slots["_confirm_cancel"] = True
                                    service_slots["_cancel_return_slot"] = "card_no"
                                    return {
                                        "service_slots": service_slots,
                                        "service_pending_slot": "_confirm_cancel",
                                        "messages": [AIMessage(
                                            content="Are you sure you want to cancel the payment update? (Yes / No)"
                                        )],
                                    }
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "card_no",
                                    "messages": [AIMessage(
                                        content=(
                                            "⚠️ That doesn't look like a card number.\n\n"
                                            f"Please enter a valid *{card_type}* card number "
                                            f"({_lengths_display} digits)."
                                        )
                                    )],
                                }
                            else:
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "card_no",
                                    "messages": [AIMessage(
                                        content=(
                                            f"⚠️ *{card_type}* cards require *{_lengths_display} digits*, "
                                            f"but you entered *{len(digits_only)} digits*.\n\n"
                                            "Please enter the correct card number."
                                        )
                                    )],
                                }
                    else:
                        # Empty input
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "card_no",
                            "messages": [AIMessage(
                                content=(
                                    "⚠️ No input received.\n\n"
                                    f"Please enter your *{card_type}* card number ({_lengths_display} digits)."
                                )
                            )],
                        }

            if not card_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_no",
                    "messages": [AIMessage(
                        content=f"Please enter your *{card_type}* card number ({_lengths_display} digits)."
                    )],
                }

            # ---------------------------------------------------------
            # Step 4: Collect card EXPIRY (with format validation)
            # Accepts: MM/YYYY, MM/YY, DD/MM/YYYY
            # ---------------------------------------------------------
            card_expiry = service_slots.get("card_expiry")
            if not card_expiry:
                if service_pending_slot == "card_expiry":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg and last_msg.strip():
                        raw_expiry = last_msg.strip()
                        # Validate format: MM/YYYY, MM/YY, or DD/MM/YYYY
                        expiry_pattern = re.compile(
                            r"^(?:(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})|(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{4})$"
                        )
                        if expiry_pattern.match(raw_expiry):
                            card_expiry = raw_expiry
                            service_slots["card_expiry"] = card_expiry
                        else:
                            raw_lower = raw_expiry.lower().strip()
                            # Check if user wants to change card type
                            _change_card_phrases = [
                                "change card", "switch card", "different card",
                                "change type", "switch type", "other card",
                                "wrong card", "not this card", "go back",
                            ]
                            if any(phrase in raw_lower for phrase in _change_card_phrases):
                                logger.info("ServiceFlow.update_payment.card_expiry: change card trigger '%s'", raw_lower[:30])
                                service_slots = dict(service_slots)
                                service_slots.pop("card_type", None)
                                service_slots.pop("card_no", None)
                                service_slots.pop("card_expiry", None)
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "card_type",
                                    "messages": [AIMessage(
                                        content=f"Sure! Which card type would you like to use?\n\nPlease choose: {_CARD_TYPE_DISPLAY}"
                                    )],
                                }
                            # Check for decline intent before showing expiry error
                            _decline_kw = [
                                "no need", "no change", "don't", "dont", "cancel",
                                "never mind", "nevermind", "forget", "thanks", "thank",
                                "later", "not now", "skip", "leave", "stop", "quit",
                                "exit", "nvm", "nah", "done", "fine", "good",
                            ]
                            if any(kw in raw_lower for kw in _decline_kw):
                                logger.info("ServiceFlow.update_payment.card_expiry: decline detected '%s'", raw_lower[:30])
                                service_slots = dict(service_slots)
                                service_slots["_confirm_cancel"] = True
                                service_slots["_cancel_return_slot"] = "card_expiry"
                                return {
                                    "service_slots": service_slots,
                                    "service_pending_slot": "_confirm_cancel",
                                    "messages": [AIMessage(
                                        content="Are you sure you want to cancel the payment update? (Yes / No)"
                                    )],
                                }
                            return {
                                "service_slots": service_slots,
                                "service_pending_slot": "card_expiry",
                                "messages": [AIMessage(
                                    content=(
                                        f"⚠️ *\"{raw_expiry}\"* is not a valid expiry date format.\n\n"
                                        "Please enter the expiry date in one of these formats:\n"
                                        "• `MM/YYYY` (e.g., 12/2028)\n"
                                        "• `MM/YY` (e.g., 12/28)\n"
                                        "• `DD/MM/YYYY` (e.g., 01/10/2029)"
                                    )
                                )],
                            }
                    else:
                        return {
                            "service_slots": service_slots,
                            "service_pending_slot": "card_expiry",
                            "messages": [AIMessage(
                                content=(
                                    "⚠️ No input received.\n\n"
                                    "Please enter the card expiry date (e.g., 12/2028 or 01/10/2029)."
                                )
                            )],
                        }

            if not card_expiry:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_expiry",
                    "messages": [AIMessage(content="What is the card expiry date? (e.g., 12/2028 or 01/10/2029)")],
                }
            
            # Get payer details from customer data (already validated)
            payer_surname = customer_data.get("surname", "")
            payer_given_name = customer_data.get("givenName", "")
            payer_nric = customer_nric
            
            # =================================================================
            # CONFIRMATION: Ask user to confirm before updating payment
            # =================================================================
            if not service_slots.get("_confirm_update"):
                masked_card = f"****{card_no[-4:]}" if len(card_no) >= 4 else "****"
                summary = (
                    f"Please confirm: Update payment info for policy *{policy_no}*?\n\n"
                    f"• Card Type: *{card_type}*\n"
                    f"• Card Number: *{masked_card}*\n"
                    f"• Expiry: *{card_expiry}*\n\n"
                    f"Reply *Yes* to confirm or *No* to cancel."
                )
                service_slots = dict(service_slots)
                service_slots["_confirm_update"] = True
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "_confirm_update",
                    "messages": [AIMessage(content=summary)],
                }

            # Log all parameters before API call
            logger.info("ServiceFlow.update_payment: calling API card_type=%s", card_type)
            
            # Call the API to update payment info
            result = await client.update_payment_info(
                nric=customer_nric,
                card_no=card_no,
                card_expire=card_expiry,
                credit_card_type=card_type,
                policy_no=policy_no,
                payer_surname=payer_surname,
                payer_given_name=payer_given_name,
                payer_nric=payer_nric,
            )
            
            if result.get("success"):
                # Mask card number for display (show last 4 digits only)
                masked_card = f"****{card_no[-4:]}" if len(card_no) >= 4 else "****"
                response = (
                    f"✅ Your payment information has been updated successfully!\n\n"
                    f"• Policy: {policy_no}\n"
                    f"• Card: {masked_card} ({card_type})\n"
                    f"• Expiry: {card_expiry}\n\n"
                    f"Is there anything else I can help you with?"
                )
                updated_data = result.get("data") if isinstance(result.get("data"), dict) else None
                updates: Dict[str, Any] = {}
                if updated_data:
                    updates["customer_data"] = updated_data

                return {
                    "service_action": None,
                    "service_slots": {},
                    "service_pending_slot": None,
                    "messages": [AIMessage(content=response)],
                    **(updates or {}),
                }
            else:
                error_msg = result.get("error", "") or "I couldn't update your payment information."
                status_code = int(result.get("status_code", 0) or 0)
                timeout_type = str(result.get("timeout_type") or "")
                request_id = str(result.get("request_id") or "")
                kind = "system" if (status_code >= 500 or status_code == 408 or timeout_type) else "invalid"

                if kind == "system":
                    prompt = (
                        f"⚠️ I couldn’t update your payment information due to a system issue.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Try again now\n"
                        "2) Re-enter the payment details\n"
                        "3) Cancel"
                    )
                else:
                    prompt = (
                        f"❌ I couldn’t update your payment information.\n\n{error_msg}\n\n"
                        "What would you like to do?\n"
                        "1) Re-enter the payment details\n"
                        "2) Try again\n"
                        "3) Cancel"
                    )

                ss = dict(service_slots)
                ss["_action_recovery_kind"] = kind
                ss["_action_recovery_last_error"] = str(error_msg)[:200]
                if request_id:
                    ss["_action_recovery_request_id"] = request_id

                return {
                    "service_action": "update_payment",
                    "service_slots": ss,
                    "service_pending_slot": "action_recovery",
                    "messages": [AIMessage(content=prompt)],
                }
        
        # =================================================================
        # UNCLEAR ACTION
        # =================================================================
        else:
            action_menu = (
                "I can help you with these policy services:\n\n"
                "1) Check claim status\n"
                "2) Check policy status/details\n"
                "3) Update email address\n"
                "4) Update mobile number\n"
                "5) Update mailing address\n"
                "6) Update payment information\n"
                "7) Update insured address (Home Protect360)\n\n"
                "Please reply with the option number, or describe what you’d like to do."
            )
            return {
                "messages": [AIMessage(
                    content=action_menu
                )],
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": "service_action_choice",
            }
            
    except Exception as e:
        logger.exception("ServiceFlow.execute_action: exception for action=%s", action)
        return {
            "messages": [AIMessage(content=f"I encountered an error while processing your request. Please try again later.")],
        }


# =============================================================================
# SUBGRAPH CONSTRUCTION
# =============================================================================

def _route_after_orchestrator(state: AgentState) -> str:
    """Route based on orchestrator decision.
    
    The orchestrator sets _orchestrator_route to indicate where to go next.
    If not set (meaning orchestrator handled it directly), end the subgraph.
    """
    route = state.get("_orchestrator_route")
    logger.info("ServiceFlow.route_after_orchestrator: route=%s exit_intent=%s", route, state.get("service_exit_intent"))
    
    if route:
        return route
    return "end"


def _route_after_validation_check(state: AgentState) -> str:
    """Route based on validation check result."""
    # _service_check_validated returns one of:
    #   "validated", "not_validated", "ask_credentials"
    # These values are used directly as keys in the
    # add_conditional_edges mapping for "detect_action".
    return _service_check_validated(state)


def _route_after_validation(state: AgentState) -> str:
    """Route after validation attempt."""
    if state.get("customer_validated"):
        # purchase gate: service_exit_intent already set → supervisor handles return to sales_agent
        if state.get("service_exit_intent") == "purchase":
            return "end"
        return "execute_action"
    else:
        return "end"  # Return message is already set


def _route_after_credentials(state: AgentState) -> str:
    """Route after credential collection."""
    service_slots = state.get("service_slots") or {}
    
    # Check if we have all required credentials
    required = ["first_name", "last_name", "email", "mobile"]
    if all(service_slots.get(s) for s in required):
        return "validate_customer"
    else:
        return "end"  # Still collecting, return question message


# Build the subgraph with Policy Service Orchestrator as entry
_service_builder = StateGraph(AgentState)

# Add nodes - orchestrator is the new entry point
_service_builder.add_node("orchestrator", _policy_service_orchestrator)
_service_builder.add_node("detect_action", _service_detect_action)
_service_builder.add_node("ask_action", _service_ask_action)
_service_builder.add_node("collect_credentials", _service_collect_credentials)
_service_builder.add_node("validate_customer", _service_validate_customer)
_service_builder.add_node("execute_action", _service_execute_action)

# Entry point - orchestrator decides routing
_service_builder.set_entry_point("orchestrator")

# After orchestrator, route based on its decision
_service_builder.add_conditional_edges(
    "orchestrator",
    _route_after_orchestrator,
    {
        "detect_action": "detect_action",
        "collect_credentials": "collect_credentials",
        "execute_action": "execute_action",
        "end": END,
    }
)

# After action detection, check validation status
_service_builder.add_conditional_edges(
    "detect_action",
    _route_after_validation_check,
    {
        "validated": "execute_action",
        "not_validated": "validate_customer",
        "ask_credentials": "collect_credentials",
        "ask_action": "ask_action",
    }
)

# After credential collection
_service_builder.add_conditional_edges(
    "collect_credentials",
    _route_after_credentials,
    {
        "validate_customer": "validate_customer",
        "end": END,
    }
)

# After validation attempt
_service_builder.add_conditional_edges(
    "validate_customer",
    _route_after_validation,
    {
        "execute_action": "execute_action",
        "end": END,
    }
)

# Execute action always ends
_service_builder.add_edge("execute_action", END)

# ask_action prompts the user and ends this turn
_service_builder.add_edge("ask_action", END)

# Compile the subgraph
service_subgraph = _service_builder.compile()

logger.info("ServiceSubgraph: compiled with 6 nodes (orchestrator + ask_action + 4 action nodes)")
