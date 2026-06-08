"""
Vector Store (Weaviate) Client for Agentic Chatbot

Production Features:
- Async wrapper via asyncio.to_thread (Weaviate v4 is sync-only)
- Configurable timeouts (init, query, insert)
- Configurable gRPC security
- Health check for monitoring
- Connection pooling via httpx/grpc

Environment Variables:
    WEAVIATE_URL: Weaviate HTTP endpoint (required)
    WEAVIATE_API_KEY: API key for authentication (optional)
    WEAVIATE_GRPC_PORT: gRPC port (default: 50051)
    WEAVIATE_GRPC_SECURE: Use TLS for gRPC (default: false)
    WEAVIATE_TIMEOUT_INIT: Init timeout seconds (default: 30)
    WEAVIATE_TIMEOUT_QUERY: Query timeout seconds (default: 45)
    WEAVIATE_TIMEOUT_INSERT: Insert timeout seconds (default: 120)
"""

from __future__ import annotations

import asyncio
import os
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ============================================
# Weaviate Import with Graceful Degradation
# ============================================
try:
    import weaviate
    from weaviate.auth import AuthApiKey
    import weaviate.classes as wvc
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None
    AuthApiKey = None
    wvc = None

# ============================================
# Configuration
# ============================================
WEAVIATE_URL = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_GRPC_SECURE = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() in ("true", "1", "yes")

# Timeouts (seconds)
TIMEOUT_INIT = int(os.getenv("WEAVIATE_TIMEOUT_INIT", "30"))
TIMEOUT_QUERY = int(os.getenv("WEAVIATE_TIMEOUT_QUERY", "45"))
TIMEOUT_INSERT = int(os.getenv("WEAVIATE_TIMEOUT_INSERT", "120"))

# ============================================
# Global Client State
# ============================================
_weaviate_client = None
_weaviate_url_cached = None


def initialize_weaviate() -> None:
    """
    Initialize Weaviate client at startup.
    
    MUST be called once at application startup. Raises on failure (fail fast).
    Idempotent - safe to call multiple times.
    
    Raises:
        RuntimeError: If Weaviate package not installed or URL not configured
        Exception: If connection fails
    """
    global _weaviate_client, _weaviate_url_cached
    
    if not WEAVIATE_AVAILABLE:
        raise RuntimeError("Weaviate package not installed. Install with: pip install weaviate-client")
    
    # Idempotent: skip if already initialized
    if _weaviate_client is not None:
        logger.debug("Weaviate client already initialized")
        return
    
    if not WEAVIATE_URL:
        raise RuntimeError(
            "WEAVIATE_URL not configured. Set WEAVIATE_URL environment variable."
        )
    
    # Suppress httpx INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.WARNING)
    
    # Disable version check
    os.environ["WEAVIATE_SKIP_INIT_CHECKS"] = "true"
    
    parsed_url = urlparse(WEAVIATE_URL)
    http_host = parsed_url.hostname
    if not http_host:
        raise RuntimeError(f"Invalid WEAVIATE_URL: could not parse hostname from '{WEAVIATE_URL}'")
    http_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 8080)
    http_secure = parsed_url.scheme == "https"
    
    auth_credentials = None
    if WEAVIATE_API_KEY:
        auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY)
    
    # No try/except - let errors propagate for fail-fast
    _weaviate_client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=http_host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_GRPC_SECURE,
        auth_credentials=auth_credentials,
        additional_config=wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(
                init=TIMEOUT_INIT,
                query=TIMEOUT_QUERY,
                insert=TIMEOUT_INSERT,
            ),
        ),
        skip_init_checks=True,
    )
    
    _weaviate_url_cached = WEAVIATE_URL
    
    logger.info(
        "Weaviate client initialized: %s (grpc_port=%d, grpc_secure=%s, timeouts=%d/%d/%d)",
        WEAVIATE_URL,
        WEAVIATE_GRPC_PORT,
        WEAVIATE_GRPC_SECURE,
        TIMEOUT_INIT,
        TIMEOUT_QUERY,
        TIMEOUT_INSERT,
    )


def get_weaviate_client():
    """
    Get the Weaviate client instance.
    
    Raises:
        RuntimeError: If initialize_weaviate() was not called at startup
    """
    if _weaviate_client is None:
        raise RuntimeError(
            "Weaviate not initialized. Call initialize_weaviate() at application startup."
        )
    return _weaviate_client


def get_weaviate_health() -> Dict[str, Any]:
    """
    Health check for Weaviate connection.
    
    Returns:
        Dict with status, connectivity info, and latency
    """
    if not WEAVIATE_AVAILABLE:
        return {"status": "disabled", "reason": "weaviate-client not installed"}
    
    if _weaviate_client is None:
        return {"status": "not_initialized", "reason": "initialize_weaviate() not called"}
    
    try:
        start = time.time()
        is_ready = _weaviate_client.is_ready()
        latency_ms = (time.time() - start) * 1000
        
        if is_ready:
            return {
                "status": "healthy",
                "url": _weaviate_url_cached,
                "latency_ms": round(latency_ms, 2),
            }
        else:
            return {
                "status": "unhealthy",
                "reason": "is_ready() returned False",
                "url": _weaviate_url_cached,
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "url": _weaviate_url_cached,
        }


async def async_weaviate_query(func, *args, **kwargs) -> Any:
    """
    Run a synchronous Weaviate operation in a thread pool.
    
    Use this wrapper for all Weaviate calls in async contexts to prevent
    blocking the event loop.
    
    Example:
        results = await async_weaviate_query(
            collection.query.near_text,
            query="travel insurance",
            limit=5,
        )
    """
    return await asyncio.to_thread(func, *args, **kwargs)


def close_weaviate_client() -> None:
    """Close the Weaviate client connection."""
    global _weaviate_client, _weaviate_url_cached
    
    if _weaviate_client is not None:
        _weaviate_client.close()
        logger.info("Weaviate client connection closed")
        _weaviate_client = None
        _weaviate_url_cached = None


__all__ = [
    "WEAVIATE_AVAILABLE",
    "initialize_weaviate",
    "get_weaviate_client",
    "get_weaviate_health",
    "async_weaviate_query",
    "close_weaviate_client",
]
