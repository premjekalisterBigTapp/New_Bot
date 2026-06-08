"""
LLM Configuration and Initialization for Agentic Chatbot
========================================================

Production-optimized LLM management with:
- Thread-safe singleton pattern (double-checked locking)
- Eager initialization at startup (fail fast, no latency on first request)
- Shared connection pooling via httpx (actually used by LLM instances)
- Multi-provider support (Azure, OpenRouter)
- Cross-platform compatibility (Windows/Linux)
- Configurable retry/timeout logic
- Observability hooks for token tracking

Thread Safety:
    This module uses a double-checked locking pattern to ensure that LLM instances
    are safely shared across threads in LangGraph's parallel execution. The pattern:
    1. First check without lock (fast path for initialized state)
    2. Acquire lock only if not initialized
    3. Second check inside lock to prevent race conditions
    4. Initialize if still needed

Supported Providers (set via LLM_PROVIDER env var):
- "azure" (default): Azure OpenAI
- "openrouter": OpenRouter API (supports many models including free ones)

Usage:
    # At application startup (in main.py lifespan):
    initialize_models()  # Raises on failure - fail fast
    
    # Throughout the application (thread-safe):
    llm = get_chat_llm()  # Returns initialized instance
    llm = get_response_llm()  # Returns initialized instance
    
    # At shutdown (in main.py lifespan):
    await async_cleanup()  # Properly closes async resources
"""

import os
import logging
import threading
import time
from typing import Optional, Union, List, Any, Dict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from dotenv import load_dotenv, find_dotenv
import httpx

load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

# ============================================
# Configuration (loaded once at module import)
# ============================================

# Provider toggle: "azure" or "openrouter"
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER", "azure") or "azure").strip().lower()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

# Embeddings Configuration (using text-embedding-3-large by default)
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT") or AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY") or AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION") or AZURE_OPENAI_API_VERSION
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

# Response LLM Configuration
AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME", "gpt-4o-mini")

# Router LLM Configuration
AZURE_OPENAI_ROUTER_DEPLOYMENT_NAME = os.environ.get(
    "AZURE_OPENAI_ROUTER_DEPLOYMENT_NAME",
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
)

# Temperature settings
AZURE_OPENAI_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2"))
AZURE_OPENAI_RESPONSE_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_RESPONSE_TEMPERATURE", "0.3"))
AZURE_OPENAI_ROUTER_TEMPERATURE = float(os.environ.get("AGENTIC_ROUTER_TEMPERATURE", "0.1"))

# ============================================
# OpenRouter Configuration
# ============================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # Default model
OPENROUTER_RESPONSE_MODEL = os.environ.get("OPENROUTER_RESPONSE_MODEL") or OPENROUTER_MODEL
OPENROUTER_ROUTER_MODEL = os.environ.get("OPENROUTER_ROUTER_MODEL") or OPENROUTER_MODEL
OPENROUTER_TEMPERATURE = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_RESPONSE_TEMPERATURE = float(os.environ.get("OPENROUTER_RESPONSE_TEMPERATURE", "0.3"))
OPENROUTER_ROUTER_TEMPERATURE = float(os.environ.get("AGENTIC_ROUTER_TEMPERATURE", "0.1"))

# ============================================
# Connection Pool & Retry Settings (Configurable)
# ============================================
HTTP_POOL_SIZE = int(os.environ.get("AGENTIC_HTTP_POOL_SIZE", "100"))
HTTP_KEEPALIVE_SIZE = int(os.environ.get("AGENTIC_HTTP_KEEPALIVE_SIZE", "50"))
HTTP_TIMEOUT = float(os.environ.get("AGENTIC_HTTP_TIMEOUT", "60.0"))
HTTP_CONNECT_TIMEOUT = float(os.environ.get("AGENTIC_HTTP_CONNECT_TIMEOUT", "10.0"))

# Retry configuration
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

# Embedding batch size
EMBEDDING_CHUNK_SIZE = int(os.environ.get("EMBEDDING_CHUNK_SIZE", "16"))

# Response LLM output token limit (ensures exhaustive answers aren't truncated)
RESPONSE_LLM_MAX_TOKENS = int(os.environ.get("RESPONSE_LLM_MAX_TOKENS", "4096"))


# ============================================
# Observability: Token Tracking Callback
# ============================================

class TokenTrackingCallback(BaseCallbackHandler):
    """
    Callback handler for tracking LLM token usage and latency.
    
    Integrates with Prometheus metrics if available, otherwise logs to standard logger.
    Thread-safe for concurrent LLM calls.
    """
    
    def __init__(self, llm_name: str = "llm"):
        self.llm_name = llm_name
        self._lock = threading.Lock()
        self._call_start_times: Dict[str, float] = {}
        
        # Try to import Prometheus metrics
        try:
            from agentic.infrastructure import (
                LLM_INPUT_TOKENS_TOTAL,
                LLM_OUTPUT_TOKENS_TOTAL,
                LLM_CALLS_TOTAL,
                LLM_LATENCY,
                LLM_TOKENS_PER_CALL,
            )
            self._input_tokens_counter = LLM_INPUT_TOKENS_TOTAL
            self._output_tokens_counter = LLM_OUTPUT_TOKENS_TOTAL
            self._calls_counter = LLM_CALLS_TOTAL
            self._latency_histogram = LLM_LATENCY
            self._tokens_per_call = LLM_TOKENS_PER_CALL
            self._prometheus_available = True
        except ImportError:
            self._prometheus_available = False
            self._tokens_per_call = None
            logger.debug("Prometheus metrics not available for LLM tracking")
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Record start time for latency tracking."""
        run_id = kwargs.get("run_id")
        if run_id:
            with self._lock:
                self._call_start_times[str(run_id)] = time.perf_counter()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record token usage and latency on completion."""
        run_id = kwargs.get("run_id")
        duration = 0.0
        
        # Calculate latency
        if run_id:
            with self._lock:
                start_time = self._call_start_times.pop(str(run_id), None)
            if start_time:
                duration = time.perf_counter() - start_time
        
        # Extract token usage from response
        input_tokens = 0
        output_tokens = 0
        
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
        
        total_tokens = input_tokens + output_tokens

        # Record metrics
        if self._prometheus_available:
            try:
                if input_tokens > 0:
                    self._input_tokens_counter.inc(input_tokens)
                    if self._tokens_per_call:
                        self._tokens_per_call.labels(token_type="input").observe(input_tokens)
                if output_tokens > 0:
                    self._output_tokens_counter.inc(output_tokens)
                    if self._tokens_per_call:
                        self._tokens_per_call.labels(token_type="output").observe(output_tokens)
                if total_tokens > 0 and self._tokens_per_call:
                    self._tokens_per_call.labels(token_type="total").observe(total_tokens)
                self._calls_counter.labels(model=self.llm_name, status="success").inc()
                if duration > 0:
                    self._latency_histogram.labels(model=self.llm_name).observe(duration)
            except Exception as e:
                logger.debug("Failed to record Prometheus metrics: %s", e)
        
        # Always log at DEBUG level for observability
        logger.debug(
            "LLM.call: model=%s input_tokens=%d output_tokens=%d duration=%.3fs",
            self.llm_name, input_tokens, output_tokens, duration
        )
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Record errors for observability."""
        run_id = kwargs.get("run_id")
        
        # Clean up start time
        if run_id:
            with self._lock:
                self._call_start_times.pop(str(run_id), None)
        
        # Record error metric
        if self._prometheus_available:
            try:
                self._calls_counter.labels(model=self.llm_name, status="error").inc()
            except Exception as e:
                logger.debug("Failed to record error metric: %s", e)
        
        logger.warning("LLM.error: model=%s error=%s", self.llm_name, str(error)[:200])


# ============================================
# Module-level State (Thread-Safe Singletons)
# ============================================

# Thread lock for initialization - ensures only one thread initializes at a time
_init_lock = threading.Lock()

# Initialization flag - checked first without lock for fast path
_initialized: bool = False

# LLM instances - protected by _init_lock during initialization
_chat_llm: Optional[Union[AzureChatOpenAI, ChatOpenAI]] = None
_response_llm: Optional[Union[AzureChatOpenAI, ChatOpenAI]] = None
_router_llm: Optional[Union[AzureChatOpenAI, ChatOpenAI]] = None
_embeddings: Optional[AzureOpenAIEmbeddings] = None

# Shared HTTP clients for connection pooling (actually used by LLM instances)
_http_client: Optional[httpx.Client] = None
_async_http_client: Optional[httpx.AsyncClient] = None

# Provider tracking
_current_provider: Optional[str] = None

# Callback instances for observability
_chat_callback: Optional[TokenTrackingCallback] = None
_response_callback: Optional[TokenTrackingCallback] = None
_router_callback: Optional[TokenTrackingCallback] = None


def _create_http_client() -> httpx.Client:
    """
    Create a sync HTTP client with optimized connection pooling.
    
    Used for LLM instances to share connections and reduce latency.
    """
    return httpx.Client(
        limits=httpx.Limits(
            max_connections=HTTP_POOL_SIZE,
            max_keepalive_connections=HTTP_KEEPALIVE_SIZE,
        ),
        timeout=httpx.Timeout(
            timeout=HTTP_TIMEOUT,
            connect=HTTP_CONNECT_TIMEOUT,
        ),
    )


def _create_async_http_client() -> httpx.AsyncClient:
    """
    Create an async HTTP client with optimized connection pooling.
    
    Used for async LLM calls to share connections and reduce latency.
    """
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=HTTP_POOL_SIZE,
            max_keepalive_connections=HTTP_KEEPALIVE_SIZE,
        ),
        timeout=httpx.Timeout(
            timeout=HTTP_TIMEOUT,
            connect=HTTP_CONNECT_TIMEOUT,
        ),
    )


def _validate_config() -> None:
    """Validate required environment variables based on provider."""
    missing = []
    
    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not OPENROUTER_MODEL:
            missing.append("OPENROUTER_MODEL")
    else:  # azure (default)
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    
    if missing:
        raise ValueError(f"Missing required environment variables for {LLM_PROVIDER}: {missing}")


def _validate_embedding_config() -> bool:
    """
    Validate embedding configuration. Returns True if valid, False otherwise.
    
    Does NOT raise - embeddings are optional for some use cases.
    """
    if not AZURE_OPENAI_EMBEDDING_ENDPOINT:
        logger.warning("AZURE_OPENAI_EMBEDDING_ENDPOINT not set - embeddings disabled")
        return False
    if not AZURE_OPENAI_EMBEDDING_API_KEY:
        logger.warning("AZURE_OPENAI_EMBEDDING_API_KEY not set - embeddings disabled")
        return False
    if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
        logger.warning("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME not set - embeddings disabled")
        return False
    return True


def _initialize_azure() -> None:
    """Initialize Azure OpenAI models with shared connection pool."""
    global _chat_llm, _response_llm, _router_llm, _embeddings, _http_client, _async_http_client
    global _chat_callback, _response_callback, _router_callback
    
    # Create shared HTTP clients for connection pooling
    _http_client = _create_http_client()
    _async_http_client = _create_async_http_client()
    
    # Create observability callbacks
    _chat_callback = TokenTrackingCallback(llm_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
    _response_callback = TokenTrackingCallback(llm_name=AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME)
    _router_callback = TokenTrackingCallback(llm_name=AZURE_OPENAI_ROUTER_DEPLOYMENT_NAME)
    
    # Chat LLM (for routing, intent detection) - with shared http client
    _chat_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        temperature=AZURE_OPENAI_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        callbacks=[_chat_callback],
    )
    logger.info("Agentic Chat LLM initialized (Azure): %s", AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
    
    # Response LLM (for generating user-facing responses) - with shared http client
    _response_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME,
        temperature=AZURE_OPENAI_RESPONSE_TEMPERATURE,
        max_tokens=RESPONSE_LLM_MAX_TOKENS,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        callbacks=[_response_callback],
    )
    logger.info("Agentic Response LLM initialized (Azure): %s", AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME)

    # Router LLM (for intent/classification) - with shared http client
    _router_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_ROUTER_DEPLOYMENT_NAME,
        temperature=AZURE_OPENAI_ROUTER_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        callbacks=[_router_callback],
    )
    logger.info("Agentic Router LLM initialized (Azure): %s", AZURE_OPENAI_ROUTER_DEPLOYMENT_NAME)
    
    # Embeddings - validate config before attempting initialization
    if _validate_embedding_config():
        _embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
            api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            chunk_size=EMBEDDING_CHUNK_SIZE,
        )
        logger.info("Agentic Embeddings initialized (Azure): %s", AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)
    else:
        _embeddings = None
        logger.warning("Embeddings not initialized - configuration incomplete")


def _initialize_openrouter() -> None:
    """Initialize OpenRouter models with shared connection pool."""
    global _chat_llm, _response_llm, _router_llm, _embeddings, _http_client, _async_http_client
    global _chat_callback, _response_callback, _router_callback
    
    # Create shared HTTP clients for connection pooling
    _http_client = _create_http_client()
    _async_http_client = _create_async_http_client()
    
    # Create observability callbacks
    _chat_callback = TokenTrackingCallback(llm_name=OPENROUTER_MODEL)
    _response_callback = TokenTrackingCallback(llm_name=OPENROUTER_RESPONSE_MODEL)
    _router_callback = TokenTrackingCallback(llm_name=OPENROUTER_ROUTER_MODEL)
    
    # Chat LLM via OpenRouter (OpenAI-compatible API) - with shared http client
    _chat_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://bigtapp.com"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "BigTapp Agentic Chatbot"),
        },
        callbacks=[_chat_callback],
    )
    logger.info("Agentic Chat LLM initialized (OpenRouter): %s", OPENROUTER_MODEL)
    
    # Response LLM via OpenRouter - with shared http client
    _response_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_RESPONSE_MODEL,
        temperature=OPENROUTER_RESPONSE_TEMPERATURE,
        max_tokens=RESPONSE_LLM_MAX_TOKENS,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://bigtapp.com"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "BigTapp Agentic Chatbot"),
        },
        callbacks=[_response_callback],
    )
    logger.info("Agentic Response LLM initialized (OpenRouter): %s", OPENROUTER_RESPONSE_MODEL)

    # Router LLM via OpenRouter - with shared http client
    _router_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_ROUTER_MODEL,
        temperature=OPENROUTER_ROUTER_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        request_timeout=HTTP_TIMEOUT,
        http_client=_http_client,
        http_async_client=_async_http_client,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://bigtapp.com"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "BigTapp Agentic Chatbot"),
        },
        callbacks=[_router_callback],
    )
    logger.info("Agentic Router LLM initialized (OpenRouter): %s", OPENROUTER_ROUTER_MODEL)
    
    # Embeddings - OpenRouter doesn't support embeddings, try Azure fallback
    if _validate_embedding_config():
        _embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
            api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            chunk_size=EMBEDDING_CHUNK_SIZE,
        )
        logger.info("Agentic Embeddings initialized (Azure fallback): %s", AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)
    else:
        _embeddings = None
        logger.warning("Embeddings not available - OpenRouter doesn't support embeddings and Azure not configured")


def initialize_models() -> None:
    """
    Initialize LLM and embedding models eagerly at startup.
    
    MUST be called once at application startup. Raises on failure (fail fast).
    Thread-safe and idempotent - safe to call multiple times from multiple threads.
    
    Thread Safety:
        Uses double-checked locking pattern:
        1. Fast path: Check _initialized without lock (most calls will return here)
        2. Slow path: Acquire lock, check again, initialize if needed
    
    Provider is selected via LLM_PROVIDER env var:
    - "azure" (default): Azure OpenAI
    - "openrouter": OpenRouter API
    
    Raises:
        ValueError: If required environment variables are missing
        Exception: If model initialization fails (fail fast - no fallbacks)
    """
    global _chat_llm, _response_llm, _router_llm, _embeddings, _current_provider, _initialized
    
    # Fast path: already initialized (no lock needed)
    if _initialized:
        logger.debug("LLM models already initialized (provider: %s)", _current_provider)
        return
    
    # Slow path: acquire lock and double-check
    with _init_lock:
        # Double-check inside lock to prevent race condition
        if _initialized:
            logger.debug("LLM models already initialized by another thread (provider: %s)", _current_provider)
            return
        
        logger.info("Initializing LLM models (provider: %s)...", LLM_PROVIDER)
        
        # Validate configuration - raises on error (fail fast)
        _validate_config()
        
        # Initialize based on provider - no try/except, let errors propagate (fail fast)
        if LLM_PROVIDER == "openrouter":
            _initialize_openrouter()
            _current_provider = "openrouter"
        else:
            _initialize_azure()
            _current_provider = "azure"
        
        # Mark as initialized AFTER successful initialization
        # This ensures _initialized is only True when models are ready
        _initialized = True
        
        logger.info(
            "All agentic LLM models initialized successfully (provider: %s, thread: %s)",
            _current_provider,
            threading.current_thread().name
        )



def get_chat_llm() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """
    Get the chat LLM instance (thread-safe).
    
    This function is safe to call from any thread after initialize_models()
    has been called at application startup.
    
    Returns:
        The initialized chat LLM instance
        
    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    # Check the initialization flag (thread-safe read)
    if not _initialized:
        raise RuntimeError(
            f"LLM not initialized. Call initialize_models() at application startup. "
            f"(Current thread: {threading.current_thread().name})"
        )
    
    # At this point, _chat_llm is guaranteed to be set (invariant maintained by initialize_models)
    # But we still check for defensive programming
    if _chat_llm is None:
        raise RuntimeError(
            f"LLM initialization incomplete - _chat_llm is None despite _initialized=True. "
            f"This indicates a bug in initialization logic. (Thread: {threading.current_thread().name})"
        )
    
    return _chat_llm


def get_response_llm() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """
    Get the response LLM instance (thread-safe).
    
    This function is safe to call from any thread after initialize_models()
    has been called at application startup.
    
    Returns:
        The initialized response LLM instance
        
    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    # Check the initialization flag (thread-safe read)
    if not _initialized:
        raise RuntimeError(
            f"LLM not initialized. Call initialize_models() at application startup. "
            f"(Current thread: {threading.current_thread().name})"
        )
    
    # At this point, _response_llm is guaranteed to be set (invariant maintained by initialize_models)
    # But we still check for defensive programming
    if _response_llm is None:
        raise RuntimeError(
            f"LLM initialization incomplete - _response_llm is None despite _initialized=True. "
            f"This indicates a bug in initialization logic. (Thread: {threading.current_thread().name})"
        )
    
    return _response_llm


def get_router_llm() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """
    Get the router LLM instance (thread-safe).

    This function is safe to call from any thread after initialize_models()
    has been called at application startup.

    Returns:
        The initialized router LLM instance

    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    if not _initialized:
        raise RuntimeError(
            f"LLM not initialized. Call initialize_models() at application startup. "
            f"(Current thread: {threading.current_thread().name})"
        )

    if _router_llm is None:
        raise RuntimeError(
            "LLM initialization incomplete - _router_llm is None despite _initialized=True. "
            f"(Thread: {threading.current_thread().name})"
        )

    return _router_llm


def get_embeddings() -> Optional[AzureOpenAIEmbeddings]:
    """
    Get the embeddings instance (thread-safe).
    
    Note: Returns None if embeddings are not configured.
    
    Returns:
        The initialized embeddings instance, or None if not available
        
    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    # Check the initialization flag (thread-safe read)
    if not _initialized:
        raise RuntimeError(
            f"LLM not initialized. Call initialize_models() at application startup. "
            f"(Current thread: {threading.current_thread().name})"
        )
    
    # _embeddings can legitimately be None (e.g., config not set)
    return _embeddings


# ============================================
# Cleanup (Async-Safe)
# ============================================

async def async_cleanup() -> None:
    """
    Async cleanup for resources on shutdown.
    
    This is the preferred cleanup method when running in an async context
    (e.g., FastAPI lifespan). Properly closes async HTTP clients.
    
    Should be called during application shutdown.
    """
    global _http_client, _async_http_client, _chat_llm, _response_llm, _router_llm, _embeddings
    global _initialized, _current_provider, _chat_callback, _response_callback, _router_callback
    
    with _init_lock:
        # Close sync HTTP client
        if _http_client:
            try:
                _http_client.close()
                logger.debug("Sync HTTP client closed")
            except Exception as e:
                logger.warning("Error closing sync HTTP client: %s", e)
            _http_client = None
        
        # Close async HTTP client properly
        if _async_http_client:
            try:
                await _async_http_client.aclose()
                logger.debug("Async HTTP client closed")
            except Exception as e:
                logger.warning("Error closing async HTTP client: %s", e)
            _async_http_client = None
        
        # Reset LLM instances
        _chat_llm = None
        _response_llm = None
        _router_llm = None
        _embeddings = None
        _current_provider = None
        _chat_callback = None
        _response_callback = None
        _router_callback = None
        
        # Reset initialization flag LAST
        _initialized = False
        
        logger.info("LLM resources cleaned up (thread: %s)", threading.current_thread().name)


def cleanup() -> None:
    """
    Synchronous cleanup for resources on shutdown.
    
    Use async_cleanup() when in an async context (preferred).
    This method is provided for backwards compatibility and non-async contexts.
    
    Note: Cannot properly close async HTTP client in sync context.
    """
    global _http_client, _async_http_client, _chat_llm, _response_llm, _router_llm, _embeddings
    global _initialized, _current_provider, _chat_callback, _response_callback, _router_callback
    
    with _init_lock:
        # Close sync HTTP client
        if _http_client:
            try:
                _http_client.close()
                logger.debug("Sync HTTP client closed")
            except Exception as e:
                logger.warning("Error closing sync HTTP client: %s", e)
            _http_client = None
        
        # For async client, we can't await in sync context
        # Set to None and let garbage collection handle it
        # This is not ideal but safe - connections will timeout naturally
        if _async_http_client:
            logger.warning(
                "Async HTTP client cannot be properly closed in sync context. "
                "Use async_cleanup() in async contexts for proper cleanup."
            )
            _async_http_client = None
        
        # Reset LLM instances
        _chat_llm = None
        _response_llm = None
        _router_llm = None
        _embeddings = None
        _current_provider = None
        _chat_callback = None
        _response_callback = None
        _router_callback = None
        
        # Reset initialization flag LAST
        _initialized = False
        
        logger.info("LLM resources cleaned up (sync) (thread: %s)", threading.current_thread().name)


def is_initialized() -> bool:
    """
    Check if LLM models have been initialized (thread-safe).
    
    Useful for health checks and startup verification.
    
    Returns:
        True if initialize_models() has been called successfully
    """
    return _initialized



__all__ = [
    "initialize_models",
    "get_chat_llm",
    "get_response_llm", 
    "get_router_llm",
    "get_embeddings",
    "is_initialized",
    "cleanup",
    "async_cleanup",
    "LLM_PROVIDER",
    "TokenTrackingCallback",
]
