# BigTapp Agentic Chatbot

Enterprise-grade LangGraph-based conversational AI for BigTapp Insurance with:
- **Autonomous Agent Routing** - Command-based flow control with self-correction
- **Multi-Turn Conversation Support** - Phase tracking, pronoun resolution, summary-aware intent
- **Token-Aware Memory Management** - Rolling summarization with context window optimization
- **Structured Tool Architecture** - Command-returning tools with proper error handling
- **Middleware-Based Context Engineering** - Dynamic prompts and state-based tool filtering
- **Production Infrastructure** - Redis persistence, MongoDB history, Prometheus metrics

## Architecture

```
agentic/
├── __init__.py              # Main agentic_chat() entry point
├── config.py                # Configuration loaders
├── graph.py                 # LangGraph graph definition with Command routing
├── state.py                 # AgentState with ConversationPhase enum
├── middleware.py            # Context engineering middleware stack
├── configs/                 # YAML templates and configs
├── handlers/                # Channel handlers (WhatsApp)
├── infrastructure/          # Core infrastructure
│   ├── llm.py              # Azure OpenAI LLM setup
│   ├── vector_store.py     # Weaviate client
│   ├── redis_utils.py      # Redis utilities
│   ├── redis_checkpointer.py  # LangGraph Redis persistence
│   ├── session.py          # Session management
│   ├── mongo_history.py    # MongoDB history logging
│   └── metrics.py          # Comprehensive Prometheus metrics
├── integrations/            # Optional integrations
│   └── zoom/               # Zoom Contact Center (live agent)
├── nodes/                   # LangGraph nodes
│   ├── supervisor.py       # Autonomous supervisor with Command routing
│   ├── autonomous_routing.py  # Self-correction & live agent handoff
│   ├── memory_nodes.py     # Token-aware memory compression
│   ├── intent.py           # Enhanced intent classification
│   ├── agents.py           # Specialist agent nodes
│   └── ...
├── tools/                   # LangChain tools
│   ├── unified.py          # Command-returning tools with InjectedState
│   └── tool_node.py        # Instrumented ToolNode with error handling
├── utils/                   # Utility functions
│   ├── messages.py         # Message factory with IDs and metadata
│   ├── memory.py           # History context builders
│   └── slots.py            # Slot validation utilities
└── scripts/                 # Initialization scripts
```

## Key Features

### 1. Autonomous Agent Routing

The supervisor uses `Command(update={...}, goto="node")` for dynamic routing:

```python
# Automatic routing based on conversation state
- Tool errors (≥2) → self_correction node → retry
- Live agent request → live_agent_handoff node
- Negative feedback → self-critique and rewrite
- Intent-based → appropriate specialist agent
```

### 2. Conversation Phase Tracking

Explicit state machine with `ConversationPhase` enum:

```python
class ConversationPhase(str, Enum):
    GREETING = "greeting"
    PRODUCT_SELECTION = "product_selection"
    SLOT_FILLING = "slot_filling"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    PURCHASE = "purchase"
    INFO_QUERY = "info_query"
    CLOSING = "closing"
    ESCALATION = "escalation"
```

Phase is tracked in `AgentState.phase` with full history in `phase_history`.

### 3. Pronoun Resolution

`ReferenceContext` tracks entities for resolving "it", "them", "there":

```python
reference_context = {
    "last_mentioned_product": "Travel",
    "last_mentioned_tier": "Gold",
    "last_mentioned_destination": "Japan",
    "compared_items": ["Gold", "Silver", "Platinum"],
    "last_bot_question": "Where are you traveling to?"
}
```

### 4. Token-Aware Memory Management

- `count_tokens_approximately()` for accurate token counting
- `trim_messages_for_context()` before LLM calls
- Rolling summarization with `[ACTIVE]`/`[ARCHIVED]` tagging
- Safe tool-chain preservation (never splits AIMessage + ToolMessage)

### 5. Command-Returning Tools

Tools use `Command` for direct state updates:

```python
@tool
def save_progress(product, slots, tool_call_id, state) -> Command:
    return Command(
        update={
            "slots": merged_slots,
            "messages": [ToolMessage(content="Saved", status="success", ...)]
        }
    )
```

### 6. Middleware Stack

```python
middleware = [
    state_aware_system_prompt,   # Dynamic prompts with phase/reference context
    filter_tools_by_phase,       # Hide tools based on conversation state
    LoggingMiddleware(),         # Before/after model logging
    RetryMiddleware(),           # Automatic retry with backoff
    validate_response_content,   # Output validation and CTA injection
]
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Required - Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Required - Redis
REDIS_URL=redis://localhost:6379/0

# Required - MongoDB
MONGO_URI=mongodb://localhost:27017
DB_NAME=bigtapp

# Optional - Weaviate (for RAG)
WEAVIATE_URL=http://localhost:8080

# Optional - Tuning
AGENTIC_ROUTER_TEMPERATURE=0.1
AGENTIC_ROUTER_MAX_TOKENS=512
AGENTIC_USE_REDIS_CHECKPOINTER=true
```

### 3. Initialize Infrastructure

```bash
# Verify all services
python -m bigtapp.agentic.scripts.healthcheck

# Initialize MongoDB collections and indexes
python -m bigtapp.agentic.scripts.init_mongodb

# Verify Redis (optional: --clear to reset)
python -m bigtapp.agentic.scripts.init_redis
```

### 4. Run

```python
from bigtapp.agentic import agentic_chat

# Async usage with channel specification
result = await agentic_chat(
    session_id="user123",
    message="Hello!",
    channel="web"  # or "whatsapp", "api"
)

print(result["response"])
print(result["debug_state"])
# {
#     "intent": "greet",
#     "product": None,
#     "phase": "greeting",
#     "rec_ready": False,
#     "live_agent_requested": False,
#     "last_routing_decision": "intent_greet",
#     "self_correction_count": 0
# }
```

## Required Services

| Service | Purpose | Required |
|---------|---------|----------|
| Redis | Session state, LangGraph checkpoints, rate limiting | **Yes** |
| MongoDB | Conversation history persistence | **Yes** |
| Azure OpenAI | LLM for chat and embeddings | **Yes** |
| Weaviate | Vector store for RAG | Optional |
| Zoom Contact Center | Live agent handoff | Optional |

## API Reference

### Main Entry Point

```python
async def agentic_chat(
    session_id: str,
    message: str,
    channel: str = "api"
) -> Dict[str, Any]:
    """
    Process a user message through the agentic chatbot.
    
    Args:
        session_id: Unique session identifier (e.g., "whatsapp_1234567890")
        message: User's message text
        channel: Communication channel ("api", "whatsapp", "web")
        
    Returns:
        {
            "response": str,           # Bot's response
            "sources": str,            # RAG sources (if any)
            "debug_state": {
                "intent": str,         # Detected intent
                "product": str,        # Detected product
                "phase": str,          # Current conversation phase
                "rec_ready": bool,     # Recommendation ready
                "rec_given": bool,     # Recommendation was given
                "purchase_offered": bool,  # Purchase link offered
                "live_agent_requested": bool,  # Live agent handoff
                "last_routing_decision": str,  # Last routing decision
                "self_correction_count": int   # Self-correction attempts
            }
        }
    """
```

### WhatsApp Handler

```python
from bigtapp.agentic.handlers import (
    handle_agentic_whatsapp_verification,
    handle_agentic_whatsapp_message,
)

# FastAPI routes
app.get("/agentic-webhook")(handle_agentic_whatsapp_verification)
app.post("/agentic-webhook")(handle_agentic_whatsapp_message)
```

## Agent State Schema

```python
class AgentState(MessagesState):
    # Core tracking
    intent: Optional[str]
    product: Optional[str]
    slots: Dict[str, Any]
    
    # Conversation phase (state machine)
    phase: Optional[str]           # ConversationPhase enum value
    phase_history: List[str]       # History of phase transitions
    
    # Flow tracking
    turn_count: int
    rec_ready: bool
    rec_given: bool
    purchase_offered: bool
    
    # Autonomous routing
    live_agent_requested: bool
    self_correction_count: int
    last_routing_decision: Optional[str]
    
    # Tool tracking
    last_tool_called: Optional[str]
    last_tool_status: Optional[str]  # "success" or "error"
    tool_call_count: int
    tool_errors: List[str]
    
    # Memory management
    summary: str                    # Rolling conversation summary
    memory_context: Dict[str, Any]  # Structured summary metadata
    total_message_tokens: int
    
    # Pronoun resolution
    reference_context: Dict[str, Any]  # For "it", "them", "there"
```

## Graph Flow

```
START
  ├── compress (parallel) ──────────────────────────────────────┐
  │                                                              │
  └── supervisor ─────────────────────────────────────────────┐  │
        │                                                      │  │
        ├── (Command goto) ──► greet_agent ──────────────────┐│  │
        ├── (Command goto) ──► capabilities_agent ───────────┤│  │
        ├── (Command goto) ──► chat_agent ───────────────────┤│  │
        ├── (Command goto) ──► info_agent ───────────────────┤│  │
        ├── (Command goto) ──► summary_agent ────────────────┤│  │
        ├── (Command goto) ──► compare_agent ────────────────┤│  │
        ├── (Command goto) ──► purchase_agent ───────────────┤│  │
        ├── (Command goto) ──► recommendation (subgraph) ────┤│  │
        ├── (Command goto) ──► self_correction ──► supervisor ││  │
        └── (Command goto) ──► live_agent_handoff ───────────┤│  │
                                                              ││  │
                                                              ▼▼  │
                                                            styler │
                                                              │    │
                                                              ▼    ▼
                                                             END
```

## Metrics

Comprehensive Prometheus metrics for monitoring:

### Message Processing
- `agentic_messages_total` - Total messages by result and product
- `agentic_latency_seconds` - Response latency histogram

### Tool Execution
- `agentic_tool_calls_total` - Tool calls by name and status
- `agentic_tool_latency_seconds` - Tool execution latency
- `agentic_tool_errors_total` - Tool errors by category

### Autonomous Routing
- `agentic_autonomous_routing_total` - Routing decisions by source/target/reason
- `agentic_self_correction_total` - Self-correction attempts
- `agentic_phase_transition_total` - Phase transitions

### Memory Management
- `agentic_memory_summarization_total` - Summarization operations
- `agentic_memory_tokens_before_trim` - Tokens before trimming
- `agentic_memory_messages_pruned_total` - Messages pruned

### Multi-Turn Conversation
- `agentic_phase_transition_total` - Phase transitions by trigger
- `agentic_pronoun_resolution_total` - Pronoun resolution attempts
- `agentic_intent_with_summary_total` - Intent classifications using summary

## Session Management

Sessions are stored in Redis with configurable TTL:

```python
from bigtapp.agentic.infrastructure import SessionManager

manager = SessionManager()

# Get or create session
session = manager.get_session("user123")

# Update session
session["product"] = "Travel"
manager.save_session("user123", session)

# Check live agent status
is_live = manager.is_live_agent_active("user123")
```

## Conversation Persistence

LangGraph conversation state persists in Redis (survives restarts):

```bash
# Enable Redis checkpointer (default: true)
AGENTIC_USE_REDIS_CHECKPOINTER=true
```

Conversation history is logged to MongoDB with message IDs:

```python
from bigtapp.agentic.infrastructure import log_history, get_history

# Log a turn (with message IDs for traceability)
log_history(
    session_id="user123",
    user_message="What's covered?",
    assistant_message="Travel insurance covers...",
    metadata={"product": "Travel", "phase": "info_query"},
    user_message_id="human-abc123",
    assistant_message_id="ai-def456"
)

# Retrieve history
history = get_history("user123", limit=20)
```

## Live Agent Handoff

When the user requests to speak to a human:

1. **Via explicit request**: User says "I want to speak to a human"
2. **Via tool**: Agent calls `escalate_to_live_agent` tool
3. **Via state flag**: `live_agent_requested` is set to `True`

```python
result = await agentic_chat("user123", "I want to speak to a human")
if result["debug_state"].get("live_agent_requested"):
    # Initiate Zoom engagement or other handoff
    pass
```

## Docker Deployment

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: bigtapp

  weaviate:
    image: semitechnologies/weaviate:1.24.1
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  redis_data:
  mongo_data:
  weaviate_data:
```

## Troubleshooting

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping
# Should return: PONG
```

### MongoDB Connection Failed
```bash
# Check MongoDB is running
mongosh --eval "db.adminCommand('ping')"
```

### LLM Initialization Failed
- Verify `AZURE_OPENAI_ENDPOINT` includes trailing slash
- Check API key is valid
- Verify deployment names match your Azure resources

### Weaviate Connection Failed
- Check gRPC port (default: 50051)
- Verify `WEAVIATE_URL` is correct
- RAG features will be disabled if Weaviate is unavailable

### Conversation Context Issues
- Check `phase` and `phase_history` in debug_state
- Verify `summary` is being generated for long conversations
- Check `reference_context` for pronoun resolution issues

### Tool Errors
- Check `tool_errors` list in state for recent errors
- Self-correction triggers after 2+ consecutive tool errors
- Check `last_tool_status` for success/error status

## License

Proprietary - BigTapp Insurance
