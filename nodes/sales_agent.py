"""
LLM-driven end-to-end sales journey agent.

Handles the full customer journey in a single node:
  Discovery → Explanation → Upsell → Payment link → Payment confirmation
  → Cross-sell → Cross-sell close

The LLM reads the conversation history and decides which stage it is in —
no rule-based state machine, no hard routing conditions.

Purchase gating: only validated customers (customer_validated=True) may proceed
to the payment link step. Anonymous users are redirected to verify their identity.
"""
from __future__ import annotations

import logging
import re
import yaml
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from ..state import AgentState
from ..config import CONFIG_DIR, _load_purchase_links
from ..infrastructure import get_response_llm, get_chat_llm
from ..tools.purchase import CROSS_SELL_MAP, FRIENDLY_NAMES
from ..utils.memory import _get_last_user_message

logger = logging.getLogger(__name__)

# Signals that indicate the user wants to buy / proceed to payment.
# FAST-PATH only — kept deliberately narrow to avoid false positives on
# general interest phrases like "I want to buy insurance" or "sure, tell me more".
# The post-LLM gate is the definitive safety net and catches everything else.
_PURCHASE_INTENT_RE = re.compile(
    r"\b(sign me up|upgrade me|proceed|go ahead|go with|"
    r"i('ll| will) take|i('ll| will) go with|"
    r"let'?s do it|let'?s go|add it)\b"
    r"|^(👍|✅)$",
    re.IGNORECASE,
)

_ANON_GATE_MESSAGE = (
    "To proceed with purchasing a policy, I'll need to verify your identity first. "
    "Could you please share your *first name*, *last name*, *email address*, and "
    "*mobile number* so I can pull up your profile?"
)

_SALES_PRODUCTS_PATH = CONFIG_DIR / "sales_products.yaml"


@lru_cache(maxsize=1)
def _load_sales_products() -> dict:
    """Load and cache the product catalogue YAML."""
    try:
        text = _SALES_PRODUCTS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return data
    except Exception as e:
        logger.warning("sales_agent: failed to load sales_products.yaml: %s", e)
        return {}


def _build_product_knowledge() -> str:
    """Format product catalogue into compact text for the system prompt."""
    products = _load_sales_products()
    if not products:
        return "Product details are currently unavailable."

    lines = []
    for key, p in products.items():
        lines.append(f"\n## {p.get('name', key)}")
        lines.append(p.get("summary", "").strip())
        tiers = p.get("tiers", {})
        if tiers:
            lines.append("Tiers:")
            for tier, desc in tiers.items():
                lines.append(f"  - {tier}: {desc}")
        upsell = p.get("upsell_tier", "")
        upsell_reason = p.get("upsell_reason", "")
        if upsell and upsell_reason:
            lines.append(f"Recommended upsell tier: {upsell} — {upsell_reason}")
    return "\n".join(lines)


def _build_product_links_text() -> str:
    """Format purchase links as key: url pairs for the system prompt."""
    links = _load_purchase_links()
    if not links:
        return "No purchase links configured."
    lines = []
    for key, url in links.items():
        friendly = FRIENDLY_NAMES.get(key, key)
        lines.append(f"- {friendly} ({key}): {url}")
    return "\n".join(lines)


def _build_cross_sell_text() -> str:
    """Format the cross-sell map as readable text for the system prompt."""
    lines = []
    for product_key, info in CROSS_SELL_MAP.items():
        primary = FRIENDLY_NAMES.get(product_key, product_key)
        lines.append(
            f"- After {primary}: suggest *{info['name']}* — {info['blurb']}"
        )
    return "\n".join(lines)


_SYSTEM_PROMPT_TEMPLATE = """\
You are a friendly, knowledgeable BigTapp insurance advisor on WhatsApp.
Your goal is to guide the customer through a complete insurance purchase journey.

SESSION CONTEXT:
- Customer identity: {identity_status}


JOURNEY STAGES — follow these in order based on the conversation history:

STAGE 1 – DISCOVER
Ask 1 or 2 short, friendly questions to understand what the customer needs.
Never dump all questions at once. Listen and respond to what they share.

STAGE 2 – RECOMMEND + SOFT UPSELL (both in ONE message, mandatory)
Based on what the customer shared, recommend ONE tier that best fits their needs.
Present it with 2-3 key benefit highlights, then IMMEDIATELY in the SAME message add
a professional upsell mention and ask if they'd like to proceed.
Do NOT list all tiers. Do NOT show a comparison table. Do NOT ask them to choose from a menu.

Mandatory format (fill in the correct tier names and benefits):
  "Based on what you've shared, I'd recommend our *[RECOMMENDED TIER]* plan — it gives you:
   • [Benefit 1]
   • [Benefit 2]
   • [Benefit 3]

   Based on your selection, *[UPSELL TIER]* can provide you a few more additions —
   [briefly state the 1-2 extra benefits from upsell_reason]. It may be worth considering!

   Would you like to go ahead with [RECOMMENDED TIER]?"

Use the upsell_tier and upsell_reason from PRODUCT KNOWLEDGE for the upsell sentence.
Keep the upsell to 1-2 sentences — professional mention only, not a full comparison.
The default recommendation stays as the base/recommended tier; upsell is advisory only.

STAGE 2b – UPSELL INTEREST (reply-based, triggered when user shows interest in the upsell tier)
If the user says "yes", "tell me more about [UPSELL TIER]", "what does [UPSELL TIER] include?",
or anything that shows interest in the upsell tier (NOT the recommended tier):
→ Explain the upsell tier's benefits in 2-3 bullets using PRODUCT KNOWLEDGE.
→ End with: "Would you like to upgrade to *[UPSELL TIER]* instead?"
Do NOT give the payment link yet. Wait for explicit second confirmation.

If the user says "no", "I'll stick with [RECOMMENDED TIER]", or confirms the original tier:
→ Proceed to Stage 4 with the originally recommended tier's payment link.

STAGE 3 – INFO REQUESTS (answer and return to the journey)
If the customer asks "what does Gold include?", "tell me more about Silver", or similar:
Answer briefly (2-3 bullets) using PRODUCT KNOWLEDGE, then ask:
  "Would you like to go ahead with [TIER NAME]?"
This is NOT agreement to buy — wait for explicit confirmation before Stage 4.

STAGE 4 – PAYMENT
ONLY provide the payment link when the customer clearly agrees to purchase.
Clear purchase signals: "yes", "I'll take it", "sounds good", "let's do it",
"I want to buy", "go ahead", "ok get it", "proceed", "sure", "upgrade me", "sign me up".
Asking "tell me more" or "what does it include?" is NOT agreement to buy.
A first "yes" after an upsell or cross-sell HINT means interest — respond with benefits first (Stage 2b or Stage 6b).
Only give the payment link when the customer confirms AFTER seeing the full benefit explanation.
When they agree, respond with EXACTLY this format (fill in the correct link):
  "Great choice! Here is your payment link:
   [LINK]
   Complete your payment and let me know once it is done!"
Always use the correct link for the product from PRODUCT LINKS below.

STAGE 5 – CONFIRM PAYMENT
When the customer says payment is complete, respond with payment confirmation.
Payment-done signals: "done", "done!", "paid", "payment done", "it's done",
"completed", "payment successful", "i've paid", "payment complete", "ok done".
"done" ALWAYS means the customer has completed the payment — NEVER treat it as a cancellation.
Respond with EXACTLY this format (fill in the product name):
  "Payment confirmed! 🎉 Your [PRODUCT NAME] policy is now being processed.
   You will receive your policy document via email within 24 hours.
   Thank you for choosing BigTapp — you have made a great decision!"

STAGE 6 – CROSS-SELL INTRO
Immediately after confirming payment, in the SAME message or the very next one,
introduce the complementary product in a professional, advisory tone.
Use this format:
  "Based on your current [PRODUCT] plan, I believe *[CROSS-SELL NAME]* would also
   be a great addition for you — it covers [KEY BENEFIT 1] and [KEY BENEFIT 2],
   which pairs very well with what you already have.
   Would you like to know more about it?"
Use the CROSS-SELL PAIRS table below to find the right pairing.
Keep it conversational and advisory — you are recommending as a trusted advisor, not hard-selling.

STAGE 6b – CROSS-SELL EXPLAIN (reply-based, triggered when user shows interest)
If the user says "yes", "tell me more", "sure", "what does it cover?", or any interest signal:
→ Explain the cross-sell product's key benefits in 2-3 bullets using PRODUCT KNOWLEDGE.
→ The recommended tier is already defined in PRODUCT KNOWLEDGE — state it directly.
→ Do NOT ask the customer what coverage level or preferences they want. Recommend the tier proactively.
→ End with: "Would you like to add *[CROSS-SELL NAME]* to your coverage?"
Do NOT give the payment link yet. Wait for explicit second confirmation.

STAGE 7 – CLOSE THE CROSS-SELL
Only after the customer confirms a SECOND time ("yes", "go ahead", "add it", "I want it"):
→ Provide the payment link for the cross-sell product.
Use EXACTLY this format:
  "Great choice! Here is your payment link for [CROSS-SELL NAME]:
   [LINK]
   Complete your payment and let me know once it is done!"
If the customer declines ("no", "maybe later", "not now") → close warmly:
  "No problem at all! Your [ORIGINAL PRODUCT] coverage is all set.
   Feel free to reach out anytime if you need anything else. Have a great day! 😊"

FAREWELLS
If the customer says "bye", "goodbye", "thanks", "thank you", or similar endings:
Respond warmly: "Thank you for chatting with BigTapp! 😊 Have a wonderful day! 👋"
Do NOT restart the greeting or ask another question.

RULES:
- Never ask the customer for their name, email address, mobile number, or any personal/contact details. Identity verification is handled automatically by the system — you do not need to collect or re-confirm it at any point.
- Keep every message short and WhatsApp-friendly (no walls of text).
- Never say you are an AI or a chatbot. Stay in character as an advisor.
- If the customer goes off-topic, gently steer them back to the insurance journey.
- If you are unsure which stage you are at, re-read the full conversation history before replying.
- Never invent prices, limits, or coverage details. Only use the PRODUCT KNOWLEDGE below.
- Use *bold* sparingly for emphasis. Use bullet points (•) for lists.
- Do NOT use headers like ###. Do NOT use markdown tables.
- Never cancel, undo, or acknowledge a cancellation unless the customer explicitly says "cancel".
- 2-STEP CONFIRMATION RULE: A "yes" after an upsell hint (Stage 2) or cross-sell intro (Stage 6)
  means the customer is interested — NOT that they have agreed to buy.
  Always explain the full benefits first (Stage 2b or Stage 6b), then wait for a second
  explicit confirmation before providing any payment link.
  NEVER skip the explanation step and jump straight to a payment link after the first "yes".

PRODUCT LINKS:
{product_links}

CROSS-SELL PAIRS:
{cross_sell_pairs}

PRODUCT KNOWLEDGE:
{product_knowledge}
"""

_FALLBACK_REPLY = (
    "Sorry, I'm having a moment — please try again and I'll be right with you!"
)


@lru_cache(maxsize=1)
def _build_static_prompt_parts() -> tuple:
    """Build and cache the static parts of the system prompt (product data, links, cross-sell)."""
    return (
        _build_product_links_text(),
        _build_cross_sell_text(),
        _build_product_knowledge(),
    )


def _build_system_prompt(customer_validated: bool = False) -> str:
    """Build the system prompt with session-specific context injected."""
    product_links, cross_sell_pairs, product_knowledge = _build_static_prompt_parts()
    identity_status = (
        "VERIFIED — do not ask for any identity details"
        if customer_validated
        else "Not yet verified — identity verification will be triggered automatically when the customer is ready to purchase"
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        identity_status=identity_status,
        product_links=product_links,
        cross_sell_pairs=cross_sell_pairs,
        product_knowledge=product_knowledge,
    )


def _last_ai_text(messages: List[BaseMessage]) -> str:
    """Return the most recent AI message text, or empty string."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(getattr(m, "content", "") or "")
    return ""


def _detect_purchase_signal(user_text: str) -> bool:
    """Return True if the user's message looks like a purchase intent."""
    return bool(_PURCHASE_INTENT_RE.search(user_text))


def _detect_payment_done(user_text: str) -> bool:
    """Return True if the user is confirming they have paid."""
    _DONE_SIGNALS = {
        "done", "done!", "paid", "payment done", "payment done!",
        "it's done", "its done", "completed", "payment successful",
        "i've paid", "ive paid", "payment complete", "ok done",
        "payment confirmed", "i paid", "i have paid",
    }
    return user_text.lower().strip() in _DONE_SIGNALS


async def _write_policy_to_customer(nric: str, product_key: str) -> None:
    """Insert a newly purchased policy into the customer's MongoDB record."""
    import random
    from datetime import datetime, timedelta
    from ..integrations.bigtapp_api import get_bigtapp_api_client, _customer_cache, _persist_customer_to_mongo

    _PRODUCT_META = {
        "travel":   ("Travel Protect360",       "TA", 52.00,  "Annual"),
        "home":     ("Home Protect360",          "HC", 45.00,  "Monthly"),
        "family":   ("Family Protect360",        "FA", 34.90,  "Monthly"),
        "motor":    ("Car Protect360",           "MP", 156.00, "Monthly"),
        "maid":     ("Maid Protect360 PRO",      "DY", 25.00,  "Annual"),
        "early":    ("Early Protect360 Plus",    "ES", 68.50,  "Monthly"),
        "fraud":    ("Fraud Protect360 Plus",    "CY", 18.90,  "Annual"),
        "hospital": ("Hospital Protect360",      "HI", 42.00,  "Monthly"),
        "choice":   ("ChoiceProtect360",         "CK", 15.90,  "Monthly"),
    }

    meta = _PRODUCT_META.get(product_key.lower())
    if not meta:
        logger.warning("sales_agent.write_policy: unknown product_key=%s", product_key)
        return

    product_name, prefix, premium, frequency = meta
    now = datetime.now()
    new_policy = {
        "policyNo": f"{prefix}{random.randint(400000, 499999)}",
        "productName": product_name,
        "status": "Active",
        "commencementDate": now.strftime("%Y-%m-%dT00:00:00"),
        "policyEndDate": (now + timedelta(days=365)).strftime("%Y-%m-%dT00:00:00"),
        "premiumAmount": premium,
        "paymentFrequency": frequency,
    }

    # Update the in-memory cache
    for cd in _customer_cache.values():
        if cd.get("idCardNumber") == nric:
            cd.setdefault("policies", []).append(new_policy)
            _persist_customer_to_mongo(cd)
            logger.info(
                "sales_agent.write_policy: added %s (%s) to nric=***%s",
                new_policy["policyNo"], product_name, nric[-4:],
            )
            return

    logger.warning("sales_agent.write_policy: nric=***%s not found in cache", nric[-4:])


async def _sales_agent_node(state: AgentState) -> AgentState:
    """LLM-driven end-to-end sales journey agent.

    Uses the full conversation history so the LLM can determine the current
    stage (discover / explain / upsell / payment / confirm / cross-sell) without
    any hard-coded state flags.

    Purchase gating: only validated customers may reach the payment link step.
    """
    messages: List[BaseMessage] = state.get("messages", []) or []
    user_text = _get_last_user_message(messages)
    product = state.get("product") or ""
    customer_validated: bool = state.get("customer_validated", False)
    customer_nric: str = state.get("customer_nric") or ""

    logger.info(
        "sales_agent: turn=%d product=%s validated=%s query_len=%d",
        state.get("turn_count", 0),
        product,
        customer_validated,
        len(user_text),
    )

    last_ai = _last_ai_text(messages)
    payment_link_already_sent = "app.bigtapp.com" in last_ai

    # ------------------------------------------------------------------
    # PURCHASE GATE: block unvalidated users from receiving a payment link.
    # Browsing (info / recommendations) is allowed without validation.
    # The moment the user signals intent to actually buy, require identity
    # verification first, then return them to this journey automatically.
    # ------------------------------------------------------------------
    if _detect_purchase_signal(user_text) and not customer_validated:
        logger.info(
            "sales_agent.purchase_gate: purchase signal detected but user not validated → redirecting to identity verification"
        )
        return {
            "messages": [AIMessage(content=_ANON_GATE_MESSAGE)],
            "sources": [],
            "pending_purchase": True,
        }

    system_prompt = _build_system_prompt(customer_validated=customer_validated)

    # Pass the full conversation history so the LLM can track the journey stage.
    # Cap at last 20 messages to avoid prompt overflow.
    history: List[BaseMessage] = [
        m for m in messages[-20:] if isinstance(m, (HumanMessage, AIMessage))
    ]
    if not history:
        history = [HumanMessage(content=user_text or "Hello, I need insurance.")]

    llm_messages = [SystemMessage(content=system_prompt)] + history

    try:
        llm = get_response_llm()
    except Exception:
        llm = get_chat_llm()

    try:
        ai_msg = await llm.ainvoke(llm_messages)
        content = getattr(ai_msg, "content", None) or str(ai_msg)
    except Exception as exc:
        logger.exception("sales_agent: LLM call failed: %s", exc)
        content = _FALLBACK_REPLY

    # ------------------------------------------------------------------
    # POST-LLM PURCHASE GATE (definitive safety net)
    # The pre-LLM regex gate catches obvious patterns like "buy" or "proceed"
    # but misses natural phrases like "go with gold", "yes", "👍", "that one",
    # "gold please", "confirm", "ok", "add it", etc.
    # Here we check the LLM's actual output — if it generated a payment link
    # but the user is not validated, we intercept and redirect regardless of
    # how the user phrased their purchase intent. This is guaranteed to catch
    # 100% of cases because we inspect the output, not predict the input.
    # ------------------------------------------------------------------
    if "app.bigtapp.com" in content and not customer_validated:
        logger.info(
            "sales_agent.purchase_gate.post_llm: LLM generated payment link for unvalidated user → intercepting"
        )
        return {
            "messages": [AIMessage(content=_ANON_GATE_MESSAGE)],
            "sources": [],
            "pending_purchase": True,
        }

    # ------------------------------------------------------------------
    # POST-PAYMENT POLICY WRITE
    # If a payment link was already sent in the previous turn and the user
    # just confirmed payment, add the policy to the validated customer's record.
    # ------------------------------------------------------------------
    if (
        customer_validated
        and customer_nric
        and payment_link_already_sent
        and _detect_payment_done(user_text)
        and product
    ):
        try:
            await _write_policy_to_customer(customer_nric, product)
        except Exception as exc:
            logger.warning("sales_agent: post-payment write failed: %s", exc)

    return {
        "messages": [AIMessage(content=content)],
        "sources": [],
    }
