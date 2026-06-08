"""
Agent State Schema for the BigTapp Agentic Chatbot.

This module defines the state schema used throughout the LangGraph workflow.

Key features:
- Inherits from `MessagesState` which provides the `add_messages` reducer
- The `add_messages` reducer enables:
  - Automatic message list management
  - Support for `RemoveMessage` to delete specific messages by ID
  - Support for `REMOVE_ALL_MESSAGES` to clear history
- Enhanced tracking for conversation phase, tools, and memory
- Explicit ConversationPhase enum for state machine behavior

The `messages` field uses the `add_messages` reducer, which means:
- Returning `{"messages": [new_msg]}` appends the message
- Returning `{"messages": [RemoveMessage(id=msg_id)]}` removes that message
- Message IDs are essential for targeted removal

Multi-Turn Conversation Support:
- ConversationPhase enum tracks explicit conversation state
- phase_history tracks phase transitions for debugging
- Pronoun resolution context for reference tracking

References:
- https://docs.langchain.com/oss/python/langgraph/use-graph-api
- https://docs.langchain.com/oss/python/langgraph/add-memory
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# =============================================================================
# CONVERSATION PHASE ENUM
# =============================================================================

class ConversationPhase(str, Enum):
    """
    Explicit conversation phase tracking for state machine behavior.
    
    This addresses the "No Conversation State Machine" issue from IMPROVEMENT_ANALYSIS.md.
    Having an explicit phase enables:
    - Debugging conversation flow issues
    - Analytics on where conversations get stuck
    - Phase-appropriate behavior and tool filtering
    - Enforcing logical conversation flow
    """
    GREETING = "greeting"           # Initial greeting, no product context yet
    PRODUCT_SELECTION = "product_selection"  # User exploring products, no specific one chosen
    SLOT_FILLING = "slot_filling"   # Collecting required information for a product
    RECOMMENDATION = "recommendation"  # Ready to give or just gave a recommendation
    COMPARISON = "comparison"       # Comparing plans/tiers
    PURCHASE = "purchase"           # User wants to buy, generating purchase link
    INFO_QUERY = "info_query"       # Answering general questions about coverage/benefits
    CLOSING = "closing"             # Conversation wrapping up (thanks, bye, etc.)
    ESCALATION = "escalation"       # Escalating to live agent
    SERVICE_FLOW = "service_flow"   # Policy service operations (claim status, updates, etc.)
    
    @classmethod
    def from_intent(cls, intent: str, has_product: bool, rec_given: bool, purchase_offered: bool) -> "ConversationPhase":
        """
        Derive conversation phase from intent and state flags.
        
        This provides a deterministic mapping from intent + state to phase,
        ensuring consistent phase tracking across the system.
        """
        if intent == "greet":
            return cls.GREETING
        elif intent == "purchase":
            return cls.PURCHASE
        elif intent == "compare":
            return cls.COMPARISON
        elif intent in ("info", "summary", "life_insurance"):
            return cls.INFO_QUERY
        elif intent == "recommend":
            if rec_given:
                return cls.RECOMMENDATION
            elif has_product:
                return cls.SLOT_FILLING
            else:
                return cls.PRODUCT_SELECTION
        elif intent == "chat":
            if has_product:
                return cls.SLOT_FILLING
            else:
                return cls.PRODUCT_SELECTION
        elif intent == "capabilities":
            return cls.INFO_QUERY
        elif intent == "policy_service":
            return cls.SERVICE_FLOW
        elif intent == "other":
            return cls.INFO_QUERY
        else:
            return cls.PRODUCT_SELECTION


# =============================================================================
# PRONOUN RESOLUTION CONTEXT
# =============================================================================

class ReferenceContext(BaseModel):
    """
    Context for pronoun resolution in multi-turn conversations.
    
    This addresses the "No Pronoun Resolution" issue from IMPROVEMENT_ANALYSIS.md.
    Tracks recently mentioned entities so pronouns like "it", "them", "that" can be resolved.
    """
    # Most recently mentioned product (for "it", "that product")
    last_mentioned_product: Optional[str] = Field(
        default=None,
        description="Last product explicitly mentioned by user or bot"
    )
    
    # Most recently mentioned tier/plan (for "it", "that plan", "the higher one")
    last_mentioned_tier: Optional[str] = Field(
        default=None,
        description="Last tier/plan explicitly mentioned"
    )
    
    # Recently mentioned destinations (for travel) - for "there", "that place"
    last_mentioned_destination: Optional[str] = Field(
        default=None,
        description="Last destination mentioned for travel insurance"
    )
    
    # Recently mentioned coverage amounts (for "that amount", "the higher coverage")
    last_mentioned_amounts: List[str] = Field(
        default_factory=list,
        description="Recently mentioned coverage amounts for reference"
    )
    
    # Recently compared items (for "them", "those", "the first one", "the second")
    compared_items: List[str] = Field(
        default_factory=list,
        description="Items from recent comparison for ordinal references"
    )
    
    # Last question asked by bot (for contextual answers)
    last_bot_question: Optional[str] = Field(
        default=None,
        description="Last question the bot asked, for interpreting short answers"
    )
    
    # Turn when this context was last updated
    last_updated_turn: int = Field(
        default=0,
        description="Turn count when reference context was last updated"
    )
    
    class Config:
        extra = "allow"
    
    def to_prompt_context(self) -> str:
        """
        Generate prompt context for pronoun resolution.
        
        This is injected into the intent classifier and dynamic prompt
        to help resolve pronouns and references.
        """
        parts = []
        
        if self.last_mentioned_product:
            parts.append(f"Last mentioned product: {self.last_mentioned_product}")
        
        if self.last_mentioned_tier:
            parts.append(f"Last mentioned tier/plan: {self.last_mentioned_tier}")
        
        if self.last_mentioned_destination:
            parts.append(f"Last mentioned destination: {self.last_mentioned_destination}")
        
        if self.compared_items:
            items_str = ", ".join(self.compared_items[:3])  # Max 3 for brevity
            parts.append(f"Recently compared: {items_str}")
        
        if self.last_bot_question:
            parts.append(f"Bot's last question: {self.last_bot_question}")
        
        if not parts:
            return ""
        
        return "REFERENCE CONTEXT (for pronoun resolution):\n" + "\n".join(f"  - {p}" for p in parts)


class IntentPrediction(BaseModel):
    """High-level routing decision for the experimental agent."""

    intent: Literal[
        "info",
        "summary",
        "compare",
        "recommend",
        "purchase",
        "capabilities",
        "greet",
        "chat",
        "policy_service",
        "life_insurance",
        "other",
    ] = Field(
        description=(
            "One of: 'info', 'summary', 'compare', 'recommend', 'purchase', "
            "'capabilities', 'greet', 'chat', 'policy_service', 'life_insurance', 'other'. "
            "Use 'info' for general questions about benefits/coverage, "
            "'summary' for high-level overviews of a product/tiers, "
            "'compare' for differences between plans/tiers, "
            "'recommend' when the user wants a personalised plan suggestion, "
            "'purchase' when they clearly want to buy or get a link. "
            "Use 'policy_service' when user wants to check policy/claim status, "
            "update personal details (email, phone, address), or manage existing policies. "
            "Use 'life_insurance' for ANY question about life insurance / term life / "
            "life cover / 'Life Protect360' (its plans, payout, premiums/cost, riders, "
            "eligibility, free look, maturity, tax, etc.). For life insurance, cost/premium "
            "questions stay 'life_insurance' (NOT 'purchase'). "
            "Use 'capabilities' for questions about what the bot can do; use 'greet' "
            "for very short greetings like 'hi' or 'hello'; use 'chat' for small-talk "
            "or open conversation where the user is not yet asking a concrete "
            "insurance question but may be sharing life context (e.g. travel plans, "
            "new house, new car, family, health)."
        )
    )
    product: Optional[str] = Field(
        default=None,
        description=(
            "Normalized product name if clearly specified. Leave empty "
            "if ambiguous."
        ),
    )
    reset: bool = Field(
        default=False,
        description="True if user explicitly wants to start over, reset, or get a fresh recommendation.",
    )
    reason: str = Field(
        default="",
        description="Short natural-language explanation for the chosen intent.",
    )


class FeedbackPrediction(BaseModel):
    """Classifier for user reactions to the last answer.

    This is used for negative feedback handling and self-correction.
    """

    category: Literal[
        "negative_feedback",
        "ack",
        "clarification",
        "new_question",
        "other",
    ] = Field(
        description=(
            "How the user is reacting to the PREVIOUS answer. "
            "'negative_feedback' = complaining, saying it was wrong, not helpful, "
            "or off-topic. 'ack' = simple thanks/ok/got it/bye. 'clarification' = "
            "they say answer was unclear and ask you to clarify or simplify it. "
            "'new_question' = they move on to a new substantive topic."
        )
    )
    reason: str = Field(
        default="",
        description="Short natural-language explanation of why this category was chosen.",
    )


class RunningSummaryData(BaseModel):
    """
    Structured running summary data for memory management.
    
    This is a simplified version compatible with the LangMem RunningSummary concept,
    but implemented without external dependencies for maximum control.
    """
    summary_text: str = Field(default="", description="The current summary text")
    token_count: int = Field(default=0, description="Approximate token count of the summary")
    last_summarized_at: Optional[str] = Field(default=None, description="ISO timestamp of last summarization")
    messages_summarized: int = Field(default=0, description="Number of messages that have been summarized")
    product_context: Optional[str] = Field(default=None, description="Product context at time of summarization")
    
    class Config:
        extra = "allow"


class AgentState(MessagesState):
    """LangGraph state for the /agent-chat agent.

    messages: list[BaseMessage] is inherited from MessagesState.
    
    Enhanced state schema with comprehensive tracking for:
    - Explicit conversation phase (ConversationPhase enum)
    - Phase history for debugging and analytics
    - Pronoun/reference resolution context
    - Tool usage tracking
    - Recommendation flow state
    - Memory management with token-aware summarization
    - Error tracking for debugging
    """

    # Core conversation tracking
    intent: Optional[str] = None
    product: Optional[str] = None
    tiers: List[str] = Field(default_factory=list)
    slots: Dict[str, Any] = Field(default_factory=dict)
    
    # Explicit conversation phase tracking (addresses "No Conversation State Machine")
    phase: Optional[str] = Field(
        default=None,
        description="Current conversation phase (ConversationPhase enum value)"
    )
    phase_history: List[str] = Field(
        default_factory=list,
        description="History of phase transitions for debugging"
    )
    
    # Turn and flow tracking
    turn_count: int = Field(default=0, description="Number of conversation turns")
    rec_ready: bool = Field(default=False, description="Whether all required slots are collected")
    rec_given: bool = Field(default=False, description="Whether a recommendation has been provided")
    purchase_offered: bool = Field(default=False, description="Whether a purchase link has been offered")

    # Channel / user identifiers (non-LLM, runtime-provided)
    channel: Optional[str] = Field(
        default=None,
        description="Inbound channel name (api, web, whatsapp).",
    )
    channel_user_id: Optional[str] = Field(
        default=None,
        description="Channel-specific user identifier (e.g., WhatsApp phone).",
    )
    
    # Autonomous routing tracking
    live_agent_requested: bool = Field(default=False, description="Whether user requested live agent handoff")
    self_correction_count: int = Field(default=0, description="Number of self-correction attempts in this session")
    last_routing_decision: Optional[str] = Field(default=None, description="Last autonomous routing decision made")
    
    # Tool tracking
    last_tool_called: Optional[str] = Field(default=None, description="Name of the last tool that was called")
    last_tool_status: Optional[str] = Field(default=None, description="Status of the last tool call: 'success' or 'error'")
    tool_call_count: int = Field(default=0, description="Total number of tool calls in this session")
    tool_errors: List[str] = Field(default_factory=list, description="List of recent tool errors for debugging")
    
    # Memory management - Enhanced with token-aware summarization
    sources: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    pending_slot: Optional[str] = None
    side_info: Optional[str] = Field(default=None, description="Temporary answer to a side question during form filling")
    pending_side_question: Optional[str] = Field(default=None, description="Side question detected during slot filling that needs to be answered")
    product_switch_pending: Optional[str] = Field(
        default=None, 
        description="Product the user wants to switch to, pending confirmation"
    )

    # Recommendation flow control
    rec_paused: bool = Field(
        default=False,
        description="If True, an in-progress recommendation flow has been intentionally paused due to a user switching to another task.",
    )
    
    # Slot validation error tracking
    slot_validation_errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable validation error messages per slot for re-ask guidance"
    )
    is_slot_reask: Optional[bool] = Field(
        default=None,
        description="Flag indicating we're re-asking a slot due to invalid/unclear input"
    )
    
    # Legacy summary field (string) - kept for backward compatibility
    summary: str = Field(default="", description="Long-term conversation summary (legacy)")
    has_summary: bool = Field(default=False)
    
    # Structured memory context for token-aware summarization
    # This follows the LangMem pattern of storing RunningSummary in a context dict
    memory_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured memory context containing running summary and metadata"
    )
    
    # Token tracking for memory management
    total_message_tokens: int = Field(default=0, description="Approximate total tokens in message history")
    last_token_count_at: int = Field(default=0, description="Turn count when tokens were last counted")
    
    # Pronoun/reference resolution context (addresses "No Pronoun Resolution")
    reference_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context for pronoun resolution (ReferenceContext data)"
    )
    
    # ==========================================================================
    # POLICY SERVICE FLOW STATE
    # ==========================================================================
    
    # Customer validation state
    customer_validated: bool = Field(
        default=False,
        description="Whether customer has been validated for this session"
    )
    customer_nric: Optional[str] = Field(
        default=None,
        description="Validated customer NRIC (stored securely, never sent to LLM)"
    )
    customer_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cached customer data from validation API response"
    )
    
    # Service action tracking
    service_action: Optional[str] = Field(
        default=None,
        description="Current service action: claim_status, policy_status, update_email, etc."
    )
    service_slots: Dict[str, Any] = Field(
        default_factory=dict,
        description="Collected data for the current service action"
    )
    service_pending_slot: Optional[str] = Field(
        default=None,
        description="Current slot being collected for service action"
    )
    
    # PII mapping for current session
    pii_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="PII placeholder to original value mapping for this session"
    )
    
    # Service flow exit routing - allows service orchestrator to direct supervisor
    service_exit_intent: Optional[str] = Field(
        default=None,
        description="Target intent when exiting service flow: 'info', 'recommend', 'compare', 'summary'. Supervisor routes directly without re-classification."
    )
    service_exit_query: Optional[str] = Field(
        default=None,
        description="Original user query that triggered exit, to be processed by target agent"
    )
    
    # Internal routing for Policy Service Orchestrator (transient - cleared after use)
    _orchestrator_route: Optional[str] = Field(
        default=None,
        description="Internal routing target for service orchestrator: 'detect_action', 'collect_credentials', 'execute_action'. Cleared after routing."
    )
    
    # Info agent control
    info_skip_filter: bool = Field(
        default=False,
        description="If True, info agent skips product filtering. Useful for general service queries."
    )

    # Purchase gate: set True when an unvalidated user signals purchase intent.
    # After identity verification completes, supervisor routes back to sales_agent.
    pending_purchase: bool = Field(
        default=False,
        description="User triggered purchase intent before validation; return to sales_agent after verification."
    )

    # If the info agent needs a product to answer, store the user's original question here.
    # Next user message is expected to be a product selection ("travel", "home", etc.).
    info_pending_question: Optional[str] = Field(
        default=None,
        description="Pending info question awaiting product clarification (stores the original question).",
    )

    # ==========================================================================
    # PRODUCT DISCOVERY / CHOICE PROTECT360 FLOW STATE
    # ==========================================================================

    product_discovery_step: Optional[str] = Field(
        default=None,
        description=(
            "Tracks multi-turn product discovery prompts (e.g., asking whether the user wants a specific "
            "product or a customizable plan like Choice Protect360)."
        ),
    )
    choice_info_step: Optional[str] = Field(
        default=None,
        description="Tracks the Choice Protect360 educational flow step (e.g., awaiting yes/no for 'find out more').",
    )
