"""
LLM-driven end-to-end sales journey agent.

Handles the full customer journey in a single node:
  Discovery → Explanation → Upsell → Payment link → Payment confirmation
  → Cross-sell → Cross-sell close

The LLM reads the conversation history and decides which stage it is in —
no rule-based state machine, no hard routing conditions.
"""
from __future__ import annotations

import logging
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
→ Mention the recommended tier for the cross-sell product.
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
- Never ask for the customer's name or personal contact details.
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
def _build_system_prompt() -> str:
    """Build and cache the full system prompt (static after startup)."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        product_links=_build_product_links_text(),
        cross_sell_pairs=_build_cross_sell_text(),
        product_knowledge=_build_product_knowledge(),
    )


async def _sales_agent_node(state: AgentState) -> AgentState:
    """LLM-driven end-to-end sales journey agent.

    Uses the full conversation history so the LLM can determine the current
    stage (discover / explain / upsell / payment / confirm / cross-sell) without
    any hard-coded state flags.
    """
    messages: List[BaseMessage] = state.get("messages", []) or []
    user_text = _get_last_user_message(messages)
    product = state.get("product") or ""
    logger.info(
        "sales_agent: turn=%d product=%s query_len=%d",
        state.get("turn_count", 0),
        product,
        len(user_text),
    )

    system_prompt = _build_system_prompt()

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

    return {
        "messages": [AIMessage(content=content)],
        "sources": [],
    }
