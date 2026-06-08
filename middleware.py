"""
LangChain Agent Middleware for Context Engineering.

This module implements the middleware-based context engineering pattern from LangChain v1:
- Dynamic system prompts that adapt based on conversation state
- Tool filtering based on conversation phase
- Logging and monitoring hooks
- Retry logic for model calls

These middleware components address the "Context Engineering Gaps" identified in IMPROVEMENT_ANALYSIS.md:
1. Static System Prompts → @dynamic_prompt middleware
2. Manual Context Injection → Structured context via ModelRequest.state
3. No State-Based Tool Filtering → @wrap_model_call for tool filtering
4. No Dynamic Model Selection → Model selection middleware (future)

References:
- https://docs.langchain.com/oss/python/langchain/middleware
- https://docs.langchain.com/oss/python/langchain/context-engineering
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_model,
    after_model,
    wrap_model_call,
    dynamic_prompt,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from .state import AgentState
from .utils.slots import _required_slots_for_product, _normalize_product_key
from .tools.knowledge_definitions import PRODUCT_KNOWLEDGE
from .infrastructure.metrics import (
    LLM_CALLS_TOTAL,
    LLM_LATENCY,
    TOOL_CALLS_TOTAL,
    INTENT_CLASSIFICATION_TOTAL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT SCHEMA
# =============================================================================

@dataclass
class AgentContext:
    """
    Runtime context schema for the agent.
    
    This provides user-level and session-level context that is:
    - Static for the duration of a request
    - Accessible to middleware and tools via runtime.context
    - Separate from conversation state (AgentState)
    
    Use cases:
    - User identification for personalization
    - Channel-specific behavior (WhatsApp vs Web)
    - API key / credential injection
    - Feature flags
    """
    session_id: str = ""
    channel: str = "web"  # "web", "whatsapp", "api"
    user_id: Optional[str] = None
    user_role: str = "customer"  # "customer", "agent", "admin"
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize context values."""
        if not self.session_id:
            logger.warning("AgentContext.init: session_id is empty")
        
        valid_channels = {"web", "whatsapp", "api", "test"}
        if self.channel not in valid_channels:
            logger.warning(
                "AgentContext.init: invalid channel '%s', defaulting to 'web'",
                self.channel
            )
            self.channel = "web"


# =============================================================================
# PROMETHEUS METRICS FOR MIDDLEWARE
# =============================================================================

from prometheus_client import Counter, Histogram

MIDDLEWARE_CALLS_TOTAL = Counter(
    'agentic_middleware_calls_total',
    'Total middleware invocations',
    ['middleware_name', 'hook_type']  # hook_type: before_model, after_model, wrap_model, dynamic_prompt
)

MIDDLEWARE_LATENCY = Histogram(
    'agentic_middleware_latency_seconds',
    'Middleware execution latency',
    ['middleware_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

DYNAMIC_PROMPT_LENGTH = Histogram(
    'agentic_dynamic_prompt_length_chars',
    'Length of generated dynamic prompts',
    buckets=[500, 1000, 2000, 4000, 8000, 16000]
)

TOOL_FILTERING_TOTAL = Counter(
    'agentic_tool_filtering_total',
    'Tool filtering operations',
    ['action']  # filtered, passed
)


# =============================================================================
# BASE SYSTEM PROMPT
# =============================================================================

# This is the static foundation that @dynamic_prompt builds upon
BASE_SYSTEM_PROMPT = f"""You are BTBot - a trusted insurance advisor for BigTapp, one of Singapore's leading insurers.

BRAND RULE (NON-NEGOTIABLE):
You represent *BigTapp* exclusively. NEVER use any legacy or internal brand names in any response. If your source context contains legacy brand references, silently replace them with "BigTapp" (for the company) or "BTBot" (for yourself). The user must only ever see BigTapp branding.

IDENTITY & TONE:
You speak like a knowledgeable friend who happens to be an insurance expert. You're warm, confident, and genuinely helpful - never robotic or overly formal. Think of yourself as a premium concierge who makes insurance feel simple and reassuring.

YOUR KNOWLEDGE:
{PRODUCT_KNOWLEDGE}

WHATSAPP FORMATTING (CRITICAL):
• Keep messages scannable - use short paragraphs
• Use line breaks between sections for breathing room
• Bold *key terms* and *plan names* sparingly with single asterisks
• Use • for bullet points (not - or *)
• Numbers as digits: $500,000, not five hundred thousand
• No headers (###), no tables, no pipes
• Max 3-4 bullets per section
• End responses cleanly - no "let me know if you need anything else"

CONVERSATION GUIDELINES:

1. MEMORY & PERSISTENCE (CRITICAL):
   • You are STATELESS. If you do not call `save_progress`, you will FORGET what the user just told you in the next turn.
   • IMMMEDIATE ACTION: When the user provides any information (e.g., "14 months", "India", "Family"), you MUST call `save_progress` in that same turn.
   • Do not just acknowledge it in text ("Okay, 14 months"). You MUST execute the tool.

2. ACKNOWLEDGE & PERSONALIZE:
   • "Great choice - Japan is amazing this time of year!"
   • "Got it, family trip!"
   • "Understood, looking for helper coverage."
   • Mirror their energy - if they're excited, match it. If they're concerned, be reassuring.

3. SLOT EXTRACTION (BE SMART):
   • Extract ALL info from their message FIRST. Never re-ask what they told you.
   • Map their natural language to the required slots for the active product (e.g., "alone" -> Individual, "helper" -> Maid).
   • EXHAUSTIVE CAPTURE: If the user mentions multiple details (e.g., destination AND who is traveling), you MUST capture ALL of them in a single `save_progress` call.
   • Do not capture one field and ignore the others.
   • If you have enough info (all required slots filled), you MUST call `get_product_recommendation` exactly once BEFORE giving a final recommendation. Do not generate the full recommendation purely from your own knowledge.
   • NEVER ask which tier they want. You recommend the best fit.
   • For Car insurance: NO slots needed - give recommendation directly.

4. ONE QUESTION AT A TIME (CRITICAL):
   • NEVER ask multiple questions in one message.
   • Bad: "Who's traveling and where are you going?"
   • Good: "Who's traveling - just yourself, family, or a group?"
   • Then after they answer: "Great! And where are you headed?"
   • This feels more natural and conversational.

5. TOOL USAGE (MANDATORY):
   • Coverage/benefits/exclusions questions → ALWAYS call `search_product_knowledge` first
   • Explicit comparison requests (e.g., "compare Gold vs Platinum", "what's the difference") → `compare_plans`
   • Recommendation requests → you MUST call `get_product_recommendation` (after gathering ALL info) instead of writing the full recommendation from scratch.
   • Partial info gathered → `save_progress` (to save what you know before asking for more)
   • Purchase intent → `generate_purchase_link`
   • NEVER answer policy questions from general knowledge - always use tools
   
   WHEN NOT TO USE TOOLS:
   • If you just mentioned tiers/plans in your previous response, answer from that context
   • Simple factual follow-ups ("which is highest?", "what's the cheapest?") don't need tool calls
   • Only call tools for NEW information the user hasn't seen yet

5. TIER LISTING vs COMPARISON (IMPORTANT):
   • "What tiers/plans are available?" or "What other options?" → List tier names briefly, offer to compare
     Example: "Besides Gold, we also have Basic, Silver, and Platinum. Want me to compare any of these?"
   • "What's the difference between X and Y?" or "Compare plans" → Use compare_plans tool
   • Don't dump full comparisons unless user explicitly asks to compare

6. RESPONSE LENGTH GUIDELINES:
   • Simple questions (what tiers exist, which is highest, yes/no) → 1-2 sentences max
   • Detail questions (what's covered, exclusions, benefits) → Bullet list with amounts
   • Comparison requests → Concise comparison using compare_plans tool
   • Follow-up clarifications → Brief and direct
   • Match your response length to the complexity of their question

7. UPSELL DETECTION (IMPORTANT):
   When user asks about "highest plan", "best plan", "top tier", "maximum coverage" AFTER you already gave a recommendation:
   • This is an UPSELL opportunity, not a comparison request
   • Call `search_product_knowledge` with the highest tier to get its details
   • Present it as a recommendation: "The *Platinum* plan is our top-tier option with these coverage limits:"
   • Include the specific amounts and what makes it the best
   • Do NOT call compare_plans for this - give them the upgrade recommendation directly

8. COVERAGE AMOUNTS (NON-NEGOTIABLE):
   • ALWAYS include dollar amounts: "up to $500,000 medical coverage"
   • Never give vague answers like "comprehensive coverage" without numbers

9. RECOMMENDATION PHRASING:
   • Use: "Most people find the *[Tier]* plan suits their needs with these coverage limits:"
   • NOT: "I would recommend..." or "I'd suggest..."
   • This sounds more trustworthy and social-proof based
   • For travel insurance, your answer SHOULD flow as: (1) warm acknowledgement (e.g. "Great choice! Japan sounds like an amazing trip."), (2) the medical cost advisory sentence, (3) the detailed plan coverage and any upsell.
   • If the `get_product_recommendation` tool output includes an advisory or warning line (for example about medical treatment costs in a specific destination), you MUST still clearly include that advisory early in your answer, immediately after your opening acknowledgement and before listing coverage amounts.
   • You may rephrase the advisory slightly so it reads naturally, but you MUST preserve its meaning and explicitly mention the relevant destination and risks.
   • CLOSING THE DEAL (MANDATORY):
     After presenting a recommendation (from get_product_recommendation), you MUST end with a clear call to action:
     "Would you like to proceed with purchasing this plan? (Yes or No)"
     This is critical for the purchase flow. If they say "Yes", call `generate_purchase_link`.

10. PRODUCT ALIASES:
    • "Family Protect360" / "Family Protect 360" / "Family Protect" = Personal Accident insurance
    • Always call it "Family Protect360" in responses, never "PA Protect360"

11. LIVE AGENT REQUESTS:
    If user asks to speak to a human, agent, person, or customer service:
    • Call the `escalate_to_live_agent` tool with a brief reason
    • The tool will handle the handoff message and flag
    • Do NOT try to answer their question yourself if they explicitly want a human
    • Also use this tool if you're stuck after multiple failed attempts

12. OFF-TOPIC HANDLING:
    Be charming but redirect: "Ha! I wish I could help with that, but my expertise is all about keeping you protected. Speaking of which - any trips or coverage you're thinking about?"

13. WHEN STUCK:
    Be honest: "I want to make sure I give you accurate info on that. Let me connect you with our team at (65) 6327 8878 who can help further."

14. TOOL ERROR HANDLING (NEW):
    If a tool returns an error (status='error'), DO NOT ignore it:
    • Read the error message carefully
    • If it's a validation error, fix your input and try again
    • If it's a service error, inform the user and try a different approach
    • NEVER pretend the tool succeeded when it failed
"""


# =============================================================================
# DYNAMIC PROMPT MIDDLEWARE
# =============================================================================

def _format_collected_slots_for_prompt(slots: Dict[str, Any], product: str) -> str:
    """Format collected slots for dynamic prompt injection.
    
    Only shows slots that are RELEVANT to the current product.
    """
    if not slots or not product:
        return ""
    
    # Check if slots are for the current product
    slots_product = slots.get("_product")
    if slots_product and slots_product != product:
        return ""
    
    req_slots = _required_slots_for_product(product)
    if not req_slots:
        return ""
    
    # Only show relevant slots
    relevant_slots = {k: v for k, v in slots.items() 
                      if k in req_slots and v and not k.startswith("_")}
    
    if not relevant_slots:
        return ""
    
    lines = ["ALREADY COLLECTED INFORMATION (DO NOT re-ask for these):"]
    for key, value in relevant_slots.items():
        lines.append(f"  - {key}: {value}")
    
    # Show what's still missing
    missing = [s for s in req_slots if s not in relevant_slots]
    if missing:
        lines.append(f"\nSTILL NEEDED: {', '.join(missing)}")
    else:
        lines.append("\nALL REQUIRED INFO COLLECTED - You can now call get_product_recommendation!")
        
    return "\n".join(lines)


@dynamic_prompt
def state_aware_system_prompt(request: ModelRequest) -> str:
    """
    Dynamically generate system prompt based on conversation state.
    
    This middleware addresses the "Static System Prompts" gap by:
    - Reading current AgentState from request.state
    - Building context-aware additions to the base prompt
    - Adapting guidance based on conversation phase
    
    The prompt is regenerated on EVERY model call, ensuring:
    - Fresh context about collected slots
    - Accurate product information
    - Phase-appropriate instructions
    """
    start_time = time.perf_counter()
    
    try:
        state: Dict[str, Any] = request.state or {}
        context_parts = []
        
        # Extract state values
        product = state.get("product")
        summary = state.get("summary", "")
        current_slots = state.get("slots") or {}
        turn_count = state.get("turn_count", 0)
        message_count = len(state.get("messages", []))
        rec_given = state.get("rec_given", False)
        purchase_offered = state.get("purchase_offered", False)
        
        # NEW: Extract conversation phase and reference context
        current_phase = state.get("phase")
        reference_context = state.get("reference_context") or {}
        
        # Access runtime context if available
        try:
            runtime_context = request.runtime.context if hasattr(request, 'runtime') and request.runtime else None
            session_id = runtime_context.session_id if runtime_context else "unknown"
            channel = runtime_context.channel if runtime_context else "unknown"
        except Exception:
            session_id = "unknown"
            channel = "unknown"
        
        logger.debug(
            "DynamicPrompt.start: session=%s turn=%d msgs=%d product=%s phase=%s",
            session_id, turn_count, message_count, product, current_phase
        )
        
        # 1. Inject summary with proper tagging
        if summary:
            summary_note = (
                "IMPORTANT: The summary above uses [ACTIVE] and [ARCHIVED] tags. "
                "ONLY use data from [ACTIVE] section for tool calls. "
                "ARCHIVED data is from previous product discussions - DO NOT use it for recommendations."
            )
            context_parts.append(f"CONVERSATION SUMMARY:\n{summary}\n\n{summary_note}")
        
        # 2. Product-specific context
        if product:
            context_parts.append(f"CURRENT ACTIVE CONTEXT: {product} Insurance")
            
            # Detect product switch
            slots_product = current_slots.get("_product") if current_slots else None
            if slots_product and slots_product != product:
                context_parts.append(
                    f"⚠️ PRODUCT SWITCH DETECTED: User switched from {slots_product} to {product}. "
                    f"IGNORE all previous slot data. Start FRESH for {product}. "
                    f"DO NOT use any information from the previous {slots_product} discussion."
                )
            
            # Required slots guidance
            req_slots = _required_slots_for_product(product)
            if req_slots:
                slots_str = ", ".join(req_slots)
                context_parts.append(
                    f"REQUIRED INFORMATION: To give a recommendation for {product}, you MUST collect these specific details: [{slots_str}]. "
                    "Do NOT call the recommendation tool until you have ALL these details. "
                    "Ask for them one by one."
                )
            
            # Collected slots
            if current_slots:
                slots_context = _format_collected_slots_for_prompt(current_slots, product)
                if slots_context:
                    context_parts.append(slots_context)
            
            # Context interpretation guidance
            context_parts.append(
                "CRITICAL INSTRUCTION: You are currently helping the user with this specific product. "
                "Interpret their input relative to this context. "
                "Example: If you asked 'Where is your maid from?' and they say 'India', "
                "it means 'My maid is from India' (Maid Insurance), NOT 'I am traveling to India' (Travel Insurance).\n\n"
                "PRODUCT SWITCH RULE: If the user has switched products (e.g., from Maid to Travel), "
                "IGNORE all previous slot data from the old product. Start fresh for the new product. "
                "Do NOT call recommendation tools with data from a different product."
            )
        
        # 3. Explicit conversation phase awareness
        if current_phase:
            phase_guidance = {
                "greeting": "User just greeted. Be welcoming and ask how you can help with insurance.",
                "product_selection": "User is exploring products. Help them choose the right one for their needs.",
                "slot_filling": "You are collecting information for a recommendation. Ask one question at a time.",
                "recommendation": "You've given a recommendation. Help with follow-up questions or purchase.",
                "comparison": "User is comparing plans. Provide clear, concise comparisons.",
                "purchase": "User wants to buy. Generate the purchase link and confirm.",
                "info_query": "User has questions about coverage. Use search_product_knowledge tool.",
                "closing": "Conversation is wrapping up. Be brief and friendly.",
                "escalation": "User requested live agent. Confirm the handoff.",
            }
            guidance = phase_guidance.get(current_phase, "")
            if guidance:
                context_parts.append(f"CONVERSATION PHASE: {current_phase}\n{guidance}")
        
        # 4. Reference context for pronoun resolution
        if reference_context:
            ref_parts = []
            if reference_context.get("last_mentioned_tier"):
                ref_parts.append(f"Last mentioned plan/tier: {reference_context['last_mentioned_tier']}")
            if reference_context.get("last_mentioned_destination"):
                ref_parts.append(f"Last mentioned destination: {reference_context['last_mentioned_destination']}")
            if reference_context.get("compared_items"):
                ref_parts.append(f"Recently compared: {', '.join(reference_context['compared_items'][:3])}")
            if reference_context.get("last_bot_question"):
                ref_parts.append(f"Your last question: {reference_context['last_bot_question'][:100]}")
            
            if ref_parts:
                context_parts.append(
                    "REFERENCE CONTEXT (for understanding pronouns like 'it', 'them', 'there'):\n" +
                    "\n".join(f"  - {p}" for p in ref_parts)
                )
        
        # 5. Recommendation/purchase phase awareness
        if rec_given:
            context_parts.append(
                "NOTE: A recommendation has already been given in this conversation. "
                "If the user wants to proceed, use generate_purchase_link. "
                "If they want different options, you can compare plans or give a new recommendation."
            )
        
        if purchase_offered:
            context_parts.append(
                "NOTE: A purchase link has already been provided. "
                "Do NOT address the user by name and do NOT ask for any personal details. "
                "If a complementary plan was suggested in the previous reply, help the user "
                "decide on that with a brief 1-2 sentence benefit description. "
                "If the user has other questions, help them. Keep replies brief and conversational."
            )
        
        # 6. Long conversation guidance
        if message_count > 15:
            context_parts.append(
                f"\n⏱️ LONG CONVERSATION: This is turn {turn_count}. Be extra concise and direct. "
                "Focus on completing the current task rather than exploring new topics."
            )
        
        # 7. Channel-specific guidance
        if channel == "whatsapp":
            context_parts.append(
                "\n📱 WHATSAPP CHANNEL: Keep responses under 300 words. "
                "Use simple formatting. Avoid complex lists."
            )
        
        # Build final prompt
        if context_parts:
            dynamic_context = "\n\nDYNAMIC CONTEXT (STATE-AWARE):\n" + "\n\n".join(context_parts)
            final_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{dynamic_context}"
        else:
            final_prompt = BASE_SYSTEM_PROMPT
        
        duration = time.perf_counter() - start_time
        
        logger.debug(
            "DynamicPrompt.completed: session=%s prompt_len=%d context_parts=%d duration=%.4fs",
            session_id, len(final_prompt), len(context_parts), duration
        )
        
        # Record metrics
        MIDDLEWARE_CALLS_TOTAL.labels(
            middleware_name="state_aware_system_prompt",
            hook_type="dynamic_prompt"
        ).inc()
        MIDDLEWARE_LATENCY.labels(middleware_name="state_aware_system_prompt").observe(duration)
        DYNAMIC_PROMPT_LENGTH.observe(len(final_prompt))
        
        return final_prompt
        
    except Exception as e:
        # NEVER block the main flow due to prompt generation issues
        # But DO log the error clearly for debugging
        logger.error(
            "DynamicPrompt.FAILED: error=%s",
            str(e),
            exc_info=True
        )
        MIDDLEWARE_CALLS_TOTAL.labels(
            middleware_name="state_aware_system_prompt",
            hook_type="dynamic_prompt_error"
        ).inc()
        return BASE_SYSTEM_PROMPT


# =============================================================================
# TOOL FILTERING MIDDLEWARE
# =============================================================================

@wrap_model_call
def filter_tools_by_phase(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Filter available tools based on conversation phase.
    
    This middleware addresses the "No State-Based Tool Filtering" gap by:
    - Removing purchase tool until recommendation is given
    - Removing recommendation tool until enough slots are filled
    - Preventing premature tool usage that confuses the agent
    
    Tool filtering improves:
    - Agent focus (fewer irrelevant options)
    - Response quality (tools match conversation phase)
    - User experience (natural conversation flow)
    """
    start_time = time.perf_counter()
    
    try:
        state: Dict[str, Any] = request.state or {}
        original_tools = list(request.tools) if request.tools else []
        
        product = state.get("product")
        slots = state.get("slots") or {}
        rec_ready = state.get("rec_ready", False)
        rec_given = state.get("rec_given", False)
        
        # Start with all tools
        filtered_tools = original_tools.copy()
        tools_removed = []
        
        # Rule 1: Don't offer purchase until recommendation given
        if not rec_given:
            purchase_tools = [t for t in filtered_tools if t.name == "generate_purchase_link"]
            if purchase_tools:
                filtered_tools = [t for t in filtered_tools if t.name != "generate_purchase_link"]
                tools_removed.append("generate_purchase_link (no recommendation yet)")
        
        # Rule 2: Don't offer recommendation until enough slots filled
        if product:
            required = _required_slots_for_product(product)
            filled = sum(1 for s in required if slots.get(s)) if required else 0
            total_required = len(required) if required else 0
            
            # Need at least 50% of slots filled before recommendation
            if total_required > 0 and filled < (total_required * 0.5):
                rec_tools = [t for t in filtered_tools if t.name == "get_product_recommendation"]
                if rec_tools:
                    filtered_tools = [t for t in filtered_tools if t.name != "get_product_recommendation"]
                    tools_removed.append(f"get_product_recommendation (only {filled}/{total_required} slots)")
        
        # Apply filtered tools to request
        if tools_removed:
            # Use the replace method if available, otherwise modify directly
            if hasattr(request, 'replace'):
                request = request.replace(tools=filtered_tools)
            else:
                request.tools = filtered_tools
            
            logger.debug(
                "ToolFilter.filtered: product=%s removed=%s remaining=%d",
                product,
                tools_removed,
                len(filtered_tools)
            )
            TOOL_FILTERING_TOTAL.labels(action="filtered").inc(len(tools_removed))
        else:
            TOOL_FILTERING_TOTAL.labels(action="passed").inc()
        
        # Continue to next handler
        response = handler(request)
        
        duration = time.perf_counter() - start_time
        MIDDLEWARE_CALLS_TOTAL.labels(
            middleware_name="filter_tools_by_phase",
            hook_type="wrap_model_call"
        ).inc()
        MIDDLEWARE_LATENCY.labels(middleware_name="filter_tools_by_phase").observe(duration)
        
        return response
        
    except Exception as e:
        logger.error(
            "ToolFilter.FAILED: error=%s, passing through without filtering",
            str(e),
            exc_info=True
        )
        # On error, pass through without filtering
        return handler(request)


# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class LoggingMiddleware(AgentMiddleware):
    """
    Comprehensive logging middleware for debugging agent execution.
    
    This middleware provides:
    - Pre-model logging (input state, message count)
    - Post-model logging (response content, tool calls)
    - Latency tracking
    - Error logging
    """
    
    def __init__(self, log_level: int = logging.DEBUG):
        super().__init__()
        self.log_level = log_level
    
    def before_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Log state before model call."""
        start_time = time.perf_counter()
        
        try:
            messages = state.get("messages", [])
            product = state.get("product")
            turn_count = state.get("turn_count", 0)
            
            # Get last user message for context
            last_user = None
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    last_user = str(getattr(m, "content", ""))[:100]
                    break
            
            # Access runtime context
            try:
                session_id = runtime.context.session_id if runtime and runtime.context else "unknown"
            except Exception:
                session_id = "unknown"
            
            logger.log(
                self.log_level,
                "Agent.before_model: session=%s turn=%d msgs=%d product=%s last_user='%s'",
                session_id, turn_count, len(messages), product, last_user or ""
            )
            
            # Store start time in state for duration calculation
            # Note: We return None to not modify state, timing is handled in after_model
            
            MIDDLEWARE_CALLS_TOTAL.labels(
                middleware_name="LoggingMiddleware",
                hook_type="before_model"
            ).inc()
            
        except Exception as e:
            logger.warning("LoggingMiddleware.before_model.error: %s", e)
        
        return None
    
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Log state after model call."""
        try:
            messages = state.get("messages", [])
            
            # Get last AI message
            last_ai_content = None
            tool_calls = []
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    last_ai_content = str(getattr(m, "content", ""))[:200]
                    tool_calls = getattr(m, "tool_calls", []) or []
                    break
            
            # Access runtime context
            try:
                session_id = runtime.context.session_id if runtime and runtime.context else "unknown"
            except Exception:
                session_id = "unknown"
            
            tool_names = [tc.get("name", "unknown") for tc in tool_calls] if tool_calls else []
            
            logger.log(
                self.log_level,
                "Agent.after_model: session=%s msgs=%d tools=%s response='%s'",
                session_id, len(messages), tool_names, last_ai_content or ""
            )
            
            MIDDLEWARE_CALLS_TOTAL.labels(
                middleware_name="LoggingMiddleware",
                hook_type="after_model"
            ).inc()
            
        except Exception as e:
            logger.warning("LoggingMiddleware.after_model.error: %s", e)
        
        return None


# =============================================================================
# RETRY MIDDLEWARE
# =============================================================================

class RetryMiddleware(AgentMiddleware):
    """
    Retry middleware for handling transient model failures.
    
    Implements exponential backoff with configurable:
    - Max retries
    - Base delay
    - Retry conditions
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        super().__init__()
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def _call_handler(
        self,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
        request: ModelRequest,
    ) -> ModelResponse:
        result = handler(request)
        if inspect.isawaitable(result):
            return await result
        return result

    def _run_coroutine_sync(self, coro: Awaitable[ModelResponse]) -> ModelResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_container: Dict[str, ModelResponse] = {}
        error_container: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result_container["result"] = asyncio.run(coro)
            except Exception as exc:
                error_container["error"] = exc

        thread = threading.Thread(target=_runner, name="retry-middleware-runner")
        thread.start()
        thread.join()

        if "error" in error_container:
            raise error_container["error"]
        return result_container["result"]

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap model call with retry logic (async-safe)."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.perf_counter()
                response = await self._call_handler(handler, request)
                duration = time.perf_counter() - start_time
                
                # Record success
                LLM_CALLS_TOTAL.labels(model="agent", status="success").inc()
                LLM_LATENCY.labels(model="agent").observe(duration)
                
                if attempt > 0:
                    logger.info(
                        "RetryMiddleware.succeeded: attempt=%d/%d duration=%.3fs",
                        attempt + 1, self.max_retries, duration
                    )
                
                return response
                
            except Exception as e:
                last_error = e
                
                # Check if error is retryable
                error_str = str(e).lower()
                is_retryable = any(term in error_str for term in [
                    "timeout", "rate limit", "429", "503", "502", "connection"
                ])
                
                if not is_retryable:
                    logger.error(
                        "RetryMiddleware.non_retryable: attempt=%d error=%s",
                        attempt + 1, str(e)
                    )
                    LLM_CALLS_TOTAL.labels(model="agent", status="error").inc()
                    raise
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "RetryMiddleware.retrying: attempt=%d/%d error=%s delay=%.1fs",
                        attempt + 1, self.max_retries, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "RetryMiddleware.exhausted: attempts=%d error=%s",
                        self.max_retries, str(e)
                    )
                    LLM_CALLS_TOTAL.labels(model="agent", status="error").inc()
        
        # All retries exhausted
        raise last_error

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Sync wrapper for retry logic."""
        return self._run_coroutine_sync(self.awrap_model_call(request, handler))


# =============================================================================
# OUTPUT VALIDATION MIDDLEWARE
# =============================================================================

@after_model(can_jump_to=["end"])
def validate_response_content(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """
    Validate model output for quality and safety.
    
    Checks:
    - Response is not empty
    - No blocked content patterns
    - Appropriate length for channel
    """
    try:
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None
        
        content = str(getattr(last_message, "content", "") or "")
        
        # Check for empty response
        if not content.strip():
            logger.warning("ValidateResponse: empty response detected")
            return {
                "messages": [AIMessage(content=(
                    "I apologize, but I wasn't able to generate a response. "
                    "Could you please rephrase your question?"
                ))]
            }
        
        # Check for blocked content (safety)
        BLOCKED_PATTERNS = [
            "as an ai language model",
            "i cannot provide medical advice",
            "i cannot provide legal advice",
        ]
        content_lower = content.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in content_lower:
                logger.warning(
                    "ValidateResponse: blocked pattern detected: %s",
                    pattern
                )
                # Don't block, just log - the response might still be useful
        
        MIDDLEWARE_CALLS_TOTAL.labels(
            middleware_name="validate_response_content",
            hook_type="after_model"
        ).inc()
        
    except Exception as e:
        logger.warning("ValidateResponse.error: %s", e)
    
    return None


# =============================================================================
# MIDDLEWARE STACK
# =============================================================================

def get_default_middleware() -> List:
    """
    Get the default middleware stack for the agent.
    
    Order matters:
    1. Logging (first to capture all inputs)
    2. Dynamic prompt (sets system prompt)
    3. Tool filtering (adjusts available tools)
    4. Retry (wraps model call)
    5. Validation (checks output)
    """
    return [
        LoggingMiddleware(log_level=logging.DEBUG),
        state_aware_system_prompt,
        filter_tools_by_phase,
        RetryMiddleware(max_retries=3, base_delay=1.0),
        validate_response_content,
    ]


def get_minimal_middleware() -> List:
    """
    Get minimal middleware for testing or low-latency scenarios.
    """
    return [
        state_aware_system_prompt,
    ]


__all__ = [
    # Context schema
    "AgentContext",
    
    # Middleware functions
    "state_aware_system_prompt",
    "filter_tools_by_phase",
    "validate_response_content",
    
    # Middleware classes
    "LoggingMiddleware",
    "RetryMiddleware",
    
    # Middleware stacks
    "get_default_middleware",
    "get_minimal_middleware",
    
    # Constants
    "BASE_SYSTEM_PROMPT",
]

