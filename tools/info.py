"""
Information retrieval tool using RAG over Weaviate.

This tool:
- Searches the BigTapp knowledge base using hybrid search (vector + keyword)
- Uses product-specific templates for response generation
- Returns answers with source citations
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

from weaviate.classes.query import TargetVectors, Filter
from langchain_core.messages import SystemMessage, HumanMessage

from ..infrastructure import (
    get_weaviate_client,
    get_embeddings,
    get_response_llm,
    get_router_llm,
    async_weaviate_query,
)
from ..infrastructure.metrics import WEAVIATE_QUERIES_TOTAL, WEAVIATE_LATENCY, LLM_CALLS_TOTAL, LLM_LATENCY
from ..config import _load_ir_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm_async

logger = logging.getLogger(__name__)

WEAVIATE_RETRIEVAL_LIMIT = int(os.environ.get("WEAVIATE_RETRIEVAL_LIMIT", "8"))

_CANONICAL_TO_WEAVIATE_NAME: Dict[str, str] = {
    "choice": "Choice",
    "travel": "Travel",
    "maid": "Maid",
    "personalaccident": "PersonalAccident",
    "home": "Home",
    "early": "Early",
    "car": "Car",
    "fraud": "Fraud",
    "hospital": "Hospital",
}


def _get_models():
    """Get LLM models from local infrastructure (thread-safe singletons)."""
    return get_embeddings(), get_response_llm()


def _is_contact_details_query(q_lower: str) -> bool:
    """
    Heuristic detector for contact-detail requests.

    NOTE: Keep this conservative to avoid triggering on queries like "policy number",
    product names like "Phone Protect360", or policy-commitment questions like
    "What is BigTapp's commitment regarding customer service?".
    """
    if not q_lower:
        return False

    # Exclusion: questions about policies, commitments, or standards are NOT contact requests
    non_contact_cues = (
        "commitment", "committed", "promise", "standard", "policy regarding",
        "what is the", "what does", "what are the", "what happens",
        "emergency assistance", "emergency car", "motor emergency",
    )
    if any(cue in q_lower for cue in non_contact_cues):
        return False

    strong_phrases = (
        "how to contact",
        "how do i contact",
        "contact details",
        "contact number",
        "phone number",
        "hotline",
        "customer care",
        "support number",
        "support hotline",
        "call centre",
        "call center",
        "reach you",
        "reach us",
        "talk to an agent",
        "talk to agent",
        "speak to an agent",
        "speak to agent",
        "human agent",
        "live agent",
        "your number",
    )
    if any(p in q_lower for p in strong_phrases):
        return True

    # "customer service" / "customer support" only when clearly asking for contact
    if ("customer service" in q_lower or "customer support" in q_lower):
        contact_verbs = ("contact", "call", "reach", "speak", "talk", "email", "number", "hotline")
        if any(v in q_lower for v in contact_verbs):
            return True
        return False

    # "call" alone is ambiguous; require explicit "call" as a word (not "called")
    # plus an additional contact cue.
    tokens = q_lower.split()
    if "call" in tokens and any(w in q_lower for w in ("number", "hotline")):
        return True

    return False


def _is_fidrec_query(q_lower: str) -> bool:
    if not q_lower:
        return False
    return "fidrec" in q_lower or "financial industry disputes resolution" in q_lower


async def _confirm_cancel_or_refund_intent(question: str) -> str:
    """
    Light-weight LLM confirmation for cancellation/refund intents.
    Returns one of: "cancellation_process", "premium_refund", "other".
    """
    router_llm = get_router_llm()
    if not router_llm:
        return "other"

    sys_prompt = (
        "You are classifying a user's intent.\n"
        "Return ONLY one label from:\n"
        "- cancellation_process\n"
        "- premium_refund\n"
        "- other\n\n"
        "Choose cancellation_process if the user asks how to cancel/terminate a policy.\n"
        "Choose premium_refund if the user asks about refund of premium/policy refund.\n"
        "If unclear or unrelated, choose other."
    )
    user_prompt = f"User message: {question}"

    try:
        result = await router_llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        label = (getattr(result, "content", "") or "").strip().lower()
        if "cancellation" in label:
            return "cancellation_process"
        if "premium" in label or "refund" in label:
            return "premium_refund"
        if label in {"cancellation_process", "premium_refund", "other"}:
            return label
    except Exception as e:
        logger.warning("Tool.info.cancel_refund_confirm_failed: %s", str(e))
    return "other"


async def _info_tool(product: Optional[str], question: str, conversation_context: Optional[str] = None, skip_product_filter: bool = False) -> Tuple[str, List[str]]:
    """
    Information tool: RAG over Weaviate using ir_response.yaml templates.
    
    Args:
        product: Product name to filter by (optional)
        question: User's question
        conversation_context: Last bot message for context (helps reformulate vague queries)
        skip_product_filter: If True, search across ALL products (useful for general questions)
        
    Returns:
        Tuple of (answer_text, source_files)
        
    Raises:
        Exception: Re-raises exceptions for the caller to handle
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)

    # ---------------------------------------------------------------------
    # Edge-case routing (non-RAG): policy cancellation / premium refund
    # ---------------------------------------------------------------------
    q_lower = (question or "").strip().lower()
    if q_lower:
        # Trigger LLM confirmation only when a simple keyword match hits.
        # Exclude hypothetical/eligibility questions (e.g. "Can I cancel the policy?",
        # "Is premium refunded on cancellation?") — these should go through RAG.
        is_candidate = False
        is_hypothetical = q_lower.startswith(("can i", "can the", "is ", "does ", "what happens", "what is", "how is", "how are"))
        if not is_hypothetical:
            if ("premium refund" in q_lower) or ("refund" in q_lower and "premium" in q_lower):
                is_candidate = True
            elif ("cancellation" in q_lower) or ("cancel" in q_lower and "policy" in q_lower):
                is_candidate = True

        if is_candidate:
            logger.info("Tool.info.cancel_refund_candidate: question='%s'", (question or "")[:120])
            intent = await _confirm_cancel_or_refund_intent(question)
            logger.info("Tool.info.cancel_refund_confirmed: intent=%s", intent)
            if intent == "premium_refund":
                logger.info("Tool.info.cancel_refund_response: type=premium_refund")
                return (
                    "For premium refunds, please email to service@bigtapp.com, with your policy number and our live agent will be calling you within 3 working days.",
                    [],
                )
            if intent == "cancellation_process":
                logger.info("Tool.info.cancel_refund_response: type=cancellation_process")
                return (
                    "For cancellation process, please email to service@bigtapp.com, with your policy number and our live agent will be calling you within 3 working days.",
                    [],
                )

        # Contact details (deterministic): do not rely on product-filtered RAG,
        # because some policy wordings include FIDReC contact details which can be
        # misinterpreted as BigTapp's support number.
        if _is_contact_details_query(q_lower):
            logger.info("Tool.info.contact_details: question='%s'", (question or "")[:120])
            if _is_fidrec_query(q_lower):
                return (
                    "For FIDReC (Financial Industry Disputes Resolution Centre Ltd), you can call (65) 6327 8878 or email info@fidrec.com.sg.",
                    [],
                )
            return (
                "Please contact BigTapp Customer Care Hotline at (65) 6702 0202 (Mon - Fri, 9.00 am - 6.00 pm) or email your questions to service@bigtapp.com.",
                [],
            )
    
    # Attempt product detection from question if not provided
    if not prod:
        logger.debug(
            "Tool.info.detecting_product: question='%s'",
            (question or "")[:100]
        )
        detected = await _detect_product_llm_async(question)
        prod = _normalize_product_key(detected)
    
    # Query reformulation: Let LLM decide if the query needs context-based reformulation
    # Note: Clarifying questions during slot collection are handled by rec_subgraph's side_info.
    # This is for general info queries that may reference previous conversation.
    original_question = question
    
    if conversation_context and len((question or "").split()) <= 6:
        # Short queries may need context - let LLM decide
        try:
            reformulate_prompt = (
                f"CONTEXT: The user is asking for information about insurance.\n"
                f"Conversation context: {conversation_context}\n"
                f"Query: {question}\n\n"
                f"Product context: {prod or 'insurance'}\n\n"
                f"Respond with ONLY the final search query, nothing else."
            )
            router_llm = get_router_llm()
            result = await router_llm.ainvoke([HumanMessage(content=reformulate_prompt)])
            reformulated = str(getattr(result, "content", "") or "").strip()
            if reformulated and len(reformulated) > 5:
                question = reformulated
                if question != original_question:
                    logger.info(
                        "Tool.info.query_reformulated: original='%s' -> reformulated='%s'",
                        original_question, question[:100]
                    )
        except Exception as e:
            logger.warning("Tool.info.reformulation_failed: %s", str(e))
    
    logger.info(
        "Tool.info.start: raw_product=%s resolved_product=%s question_len=%d skip_filter=%s",
        raw_product, prod, len(question or ""), skip_product_filter
    )
    
    # If no product and not skipping filter, ask user to specify product
    if not prod and not skip_product_filter:
        logger.warning("Tool.info.no_product: could not determine product")
        return (
            "Which product would you like to ask about: Choice Protect360, Travel Protect360, Maid Protect360, Car Protect360, Personal Accident Protect360, "
            "Home Protect360, Early Critical Illness Protect360, Fraud Protect360 or Hospital Cash Protect360?",
            [],
        )

    # Initialize Weaviate client
    try:
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
        logger.debug("Tool.info.weaviate_connected: collection=Insurance_Knowledge_Base")
    except Exception as e:
        logger.error(
            "Tool.info.weaviate_init_failed: error=%s\n%s",
            str(e), traceback.format_exc()
        )
        try:
            WEAVIATE_QUERIES_TOTAL.labels(status="error").inc()
        except Exception:
            pass
        raise  # Re-raise for caller to handle

    embeddings, llm = _get_models()

    # Generate embeddings for the question
    emb = None
    emb_start = time.time()
    try:
        if embeddings:
            emb = await embeddings.aembed_query(question)
            logger.debug(
                "Tool.info.embedding_generated: duration=%.3fs",
                time.time() - emb_start
            )
        else:
            logger.error("Tool.info.embeddings_not_initialized")
    except Exception as e:
        logger.error(
            "Tool.info.embedding_failed: error=%s\n%s",
            str(e), traceback.format_exc()
        )
        emb = None

    # Execute Weaviate hybrid search
    objects = []
    weaviate_start = time.time()
    if emb is not None:
        try:
            # Build query parameters
            query_params = {
                "query": question,
                "vector": {
                    "content_vector": emb,
                    "questions_vector": emb,
                },
                "target_vector": TargetVectors.average(["content_vector", "questions_vector"]),
                "limit": WEAVIATE_RETRIEVAL_LIMIT,
                "alpha": 0.7,
                "return_properties": ["content", "product_name", "doc_type", "source_file"],
            }
            
            # Only add product filter if we have a product and not skipping filter
            weaviate_prod = _CANONICAL_TO_WEAVIATE_NAME.get(prod, prod) if prod else None
            if weaviate_prod and not skip_product_filter:
                query_params["filters"] = Filter.by_property("product_name").equal(weaviate_prod)
            
            result = await async_weaviate_query(
                collection.query.hybrid,
                **query_params,
            )
            objects = getattr(result, "objects", []) or []
            
            weaviate_duration = time.time() - weaviate_start
            filter_status = f"product={prod}" if (prod and not skip_product_filter) else "all_products"
            logger.info(
                "Tool.info.weaviate_query: %s hits=%d duration=%.3fs",
                filter_status, len(objects), weaviate_duration
            )

            # Fallback: if a product-filtered search returns 0 hits, try across all products.
            # This improves UX for general questions (e.g., hotline/contact) where the KB
            # may not be cleanly tagged by product_name.
            if not objects and prod and not skip_product_filter:
                try:
                    fb_start = time.time()
                    fb_params = dict(query_params)
                    fb_params.pop("filters", None)
                    fb_result = await async_weaviate_query(
                        collection.query.hybrid,
                        **fb_params,
                    )
                    fb_objects = getattr(fb_result, "objects", []) or []
                    fb_duration = time.time() - fb_start
                    if fb_objects:
                        objects = fb_objects
                        logger.info(
                            "Tool.info.weaviate_fallback: product=%s -> all_products hits=%d duration=%.3fs",
                            prod, len(objects), fb_duration
                        )
                except Exception as fb_err:
                    logger.warning("Tool.info.weaviate_fallback_failed: product=%s error=%s", prod, str(fb_err))
            
            # Record metrics
            try:
                WEAVIATE_QUERIES_TOTAL.labels(status="success").inc()
                WEAVIATE_LATENCY.observe(weaviate_duration)
            except Exception:
                pass
                
        except Exception as e:
            weaviate_duration = time.time() - weaviate_start
            logger.error(
                "Tool.info.weaviate_query_failed: product=%s duration=%.3fs error=%s\n%s",
                prod, weaviate_duration, str(e), traceback.format_exc()
            )
            try:
                WEAVIATE_QUERIES_TOTAL.labels(status="error").inc()
                WEAVIATE_LATENCY.observe(weaviate_duration)
            except Exception:
                pass
            objects = []

    # Handle no results
    if not objects:
        question_preview = (question or "").replace("\n", " ")[:160]
        logger.warning(
            "Tool.info.no_results: product=%s question='%s'",
            prod, question_preview
        )
        return (
            f"I couldn't find that in our {(prod or 'insurance').title()} knowledge base right now. "
            "Could you share a bit more detail (what you’re trying to do), so I can search more precisely?",
            [],
        )

    # Build context from results
    context_str = "\n---\n".join(
        [str(obj.properties.get("content", "") or "") for obj in objects]
    )
    sources = sorted(
        {
            str(obj.properties.get("source_file", "") or "")
            for obj in objects
            if obj.properties.get("source_file")
        }
    )
    
    logger.debug(
        "Tool.info.context_built: product=%s context_len=%d sources=%d",
        prod, len(context_str), len(sources)
    )

    # Load templates and generate response
    ir_templates = _load_ir_templates()
    tpl = ir_templates.get(prod, {}) if ir_templates else {}

    # Base system prompt from templates (if any) or default.
    base_sys = tpl.get("system") or (
        "You are BigTapp's digital insurance assistant answering information questions. "
        "Answer using only the provided context from our official knowledge base."
    )

    # Styling and flow rules inspired by the global styler, so we can safely
    # skip the styler node for pure information flows.
    style_suffix = (
        "\n\nRESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points and *asterisks* for product or plan names where helpful\n"
        "• Keep answers clear, concise and friendly – avoid long paragraphs\n"
        "• Use digits for numbers and sums (e.g. $500,000)\n"
        "• No markdown headers (###) or tables\n"
        "• Be honest about any limits or exclusions in the context – do not invent details beyond it\n\n"
        "FLOW & NAMING RULES:\n"
        "1. Focus on directly answering the user's question first, then optionally add one short follow-up tip or clarification.\n"
        "2. Do not push recommendations or purchase links in pure information responses unless the user explicitly asks.\n"
        "3. Use the full official product names at least once when relevant: Travel Protect360, Maid Protect360, Car Protect360, "
        "Home Protect360, Personal Accident Protect360, Early Critical Illness Protect360, Fraud Protect360, Hospital Cash Protect360.\n"
        "4. You may shorten names (e.g. 'Travel plan') only after using the full name once in the answer.\n"
        "5. Do not mention phone numbers, emails or hotlines unless the user explicitly asks for contact details.\n"
        "6. Do NOT default to openers like 'Good question!', 'Great question!', or 'Thanks for asking'. Lead with the answer most of the time. A brief warm opener is fine occasionally when it genuinely fits, but it should not be the norm.\n"
        "7. Stay strictly within the provided context – if something is not covered, say so clearly.\n"
        "8. NEVER mention FIDReC (Financial Industry Disputes Resolution Centre) in your responses unless the user explicitly asks about FIDReC by name.\n"
        "8b. BRAND RULE: NEVER output any legacy or internal brand names in your response. If the source context contains non-BigTapp brand references, silently replace with 'BigTapp' (company) or 'BTBot' (bot name). The user must only see BigTapp branding.\n"
        "9. When answering about a specific product, do NOT mention, reference, or suggest other BigTapp products unless the user explicitly asks to compare. Stay focused on the product being discussed.\n"
        "10. Always cite specific numbers, limits, coverage amounts, and conditions from the context. Do not paraphrase or approximate numerical facts — use the exact figures provided.\n"
        "11. When the user asks for full benefits, all coverage, or complete details, be EXHAUSTIVE — list every item from the context rather than summarizing. Do not omit details for brevity.\n"
    )

    sys_t = base_sys + style_suffix
    usr_t = (tpl.get("user") or "Question: {question}\n\n[Context]\n{context}").format(
        question=question,
        context=context_str,
    )

    # Generate LLM response
    answer = ""
    llm_start = time.time()
    try:
        if llm:
            result = await llm.ainvoke([
                SystemMessage(content=sys_t),
                HumanMessage(content=usr_t),
            ])
            answer = str(result.content).strip()
            
            llm_duration = time.time() - llm_start
            logger.info(
                "Tool.info.llm_response: product=%s answer_len=%d duration=%.3fs",
                prod, len(answer), llm_duration
            )
            
            # Record metrics
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
        else:
            logger.error("Tool.info.llm_not_initialized")
    except Exception as e:
        llm_duration = time.time() - llm_start
        logger.error(
            "Tool.info.llm_failed: product=%s duration=%.3fs error=%s\n%s",
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
        answer = "I couldn't find precise details. Could you clarify your question?"

    total_duration = time.time() - start_time
    logger.info(
        "Tool.info.completed: product=%s answer_len=%d sources=%d total_duration=%.3fs",
        prod, len(answer), len(sources), total_duration
    )
    
    return answer, sources


async def _info_tool_async(
    product: Optional[str],
    question: str,
    conversation_context: Optional[str] = None,
    skip_product_filter: bool = False,
) -> Tuple[str, List[str]]:
    """Async wrapper for the info tool."""
    return await _info_tool(
        product,
        question,
        conversation_context=conversation_context,
        skip_product_filter=skip_product_filter,
    )
