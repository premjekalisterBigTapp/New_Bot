#!/usr/bin/env python3
"""
BigTapp Agentic Chatbot - Standalone FastAPI Application
========================================================

Production-ready FastAPI server for the agentic chatbot.
Can be deployed independently from the legacy BigTapp system.

Usage:
    # From the project root directory
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
    
    # With auto-reload for development
    python -m uvicorn main:app --reload --port 8000

    # Run directly
    python main.py
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# =============================================================================
# PATH AND PACKAGE SETUP
# =============================================================================
# Ensure the agentic package is importable regardless of how the script is run.
# This handles both `python main.py` and `python -m uvicorn main:app` cases.

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent

# Add parent directory to path so 'agentic' can be imported as a package
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Also ensure current directory is in path for direct module imports
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables early
load_dotenv()

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# LOCAL/PACKAGE IMPORTS
# =============================================================================
import agentic
from agentic.handlers import (
    agentic_whatsapp_handler,
    close_agentic_whatsapp_client,
    handle_agentic_whatsapp_message,
    handle_agentic_whatsapp_verification,
)
from agentic.infrastructure import (
    BackgroundLogger,
    SessionManager,
    WEAVIATE_AVAILABLE,
    close_mongo,
    close_async_redis,
    get_async_redis,
    get_response_llm,
    get_router_llm,
    get_weaviate_client,
    initialize_models,
    initialize_weaviate,
    llm_async_cleanup,
)
from agentic.infrastructure.background_logger import set_background_logger
from agentic.infrastructure.idle_monitor import (
    ENABLE_IDLE_FAREWELL,
    idle_monitor_loop,
    set_whatsapp_handler,
)
from agentic.infrastructure.metrics import AGENTIC_LATENCY, AGENTIC_MESSAGES_TOTAL

# Get the main chat function
agentic_chat = agentic.agentic_chat

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
# Request timeout in seconds (prevents hanging requests from exhausting workers)
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "60.0"))
# Maximum error message length to log (prevents log flooding)
MAX_ERROR_LOG_LENGTH = 500


# ============================================
# Lifespan Management
# ============================================

# Module-level singleton for session manager (initialized at startup)
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton SessionManager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events - fail fast on initialization errors."""
    global _session_manager
    
    # Startup - eager initialization, fail fast on errors
    logger.info("Starting BigTapp Agentic Chatbot...")
    
    # Initialize LLM models in thread pool to avoid blocking event loop
    await asyncio.to_thread(initialize_models)
    logger.info("LLM models initialized")
    
    # Initialize Weaviate client for RAG in thread pool (non-fatal)
    if WEAVIATE_AVAILABLE:
        try:
            await asyncio.to_thread(initialize_weaviate)
            logger.info("Weaviate client initialized")
        except Exception as e:
            logger.warning("Weaviate initialization failed (non-fatal): %s", e)
            logger.warning("RAG features will be unavailable")
    else:
        logger.warning("Weaviate not available - RAG features disabled")
    
    # Initialize SessionManager singleton at startup
    _session_manager = SessionManager()
    logger.info("SessionManager initialized")

    # Warm async Redis connection
    try:
        await get_async_redis()
        logger.info("Async Redis client initialized")
    except Exception as e:
        logger.error("Async Redis initialization failed: %s", e)
        raise
    
    # Start background logger for non-blocking MongoDB writes
    bg_logger = BackgroundLogger()
    await bg_logger.start()
    set_background_logger(bg_logger)  # Set module-level reference for enqueue_log
    logger.info("Background logger started")
    
    # Register WhatsApp handler for idle monitor
    set_whatsapp_handler(agentic_whatsapp_handler)
    
    # Start idle monitor background task
    idle_monitor_task = None
    if ENABLE_IDLE_FAREWELL:
        idle_monitor_task = asyncio.create_task(idle_monitor_loop())
        logger.info("Idle monitor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BigTapp Agentic Chatbot...")
    
    # Cancel idle monitor
    if idle_monitor_task:
        idle_monitor_task.cancel()
        try:
            await idle_monitor_task
        except asyncio.CancelledError:
            pass
    
    # Stop background logger (drains pending logs)
    await bg_logger.stop()
    logger.info("Background logger stopped")
    
    await close_agentic_whatsapp_client()
    
    # Async cleanup of LLM resources (properly closes httpx clients)
    await llm_async_cleanup()
    logger.info("LLM resources cleaned up")
    
    # Close Redis connection pools
    await close_async_redis()
    
    # Close MongoDB connection pool
    close_mongo()
    
    logger.info("Shutdown complete")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="BigTapp Agentic Chatbot",
    description="Production-ready LangGraph-based insurance chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[str] = ""
    debug_state: Optional[dict] = None


# ============================================
# Health & Metrics Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bigtapp-agentic",
        "version": "1.0.0",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies all dependencies without blocking event loop."""
    checks = {}
    
    # Check Redis (async)
    try:
        redis = await get_async_redis()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        checks["redis"] = f"error: {error_msg}"
    
    # Check if router/response LLMs are initialized (these are used at runtime)
    try:
        router_llm = get_router_llm()
        checks["llm_router"] = "ok" if router_llm else "not initialized"
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        checks["llm_router"] = f"error: {error_msg}"

    try:
        response_llm = get_response_llm()
        checks["llm_response"] = "ok" if response_llm else "not initialized"
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        checks["llm_response"] = f"error: {error_msg}"

    llm_ok = checks.get("llm_router") == "ok" and checks.get("llm_response") == "ok"
    checks["llm"] = "ok" if llm_ok else "not ready"
    
    # Check Weaviate
    try:
        if WEAVIATE_AVAILABLE:
            client = get_weaviate_client()
            checks["weaviate"] = "ok" if client else "not initialized"
        else:
            checks["weaviate"] = "not available"
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        checks["weaviate"] = f"error: {error_msg}"
    
    # Check MongoDB (for conversation history persistence)
    try:
        from agentic.infrastructure import mongo_history
        await asyncio.to_thread(mongo_history._init_if_needed)
        # Access _client AFTER init (not at import time)
        if mongo_history._client is not None:
            # Ping MongoDB to verify connection
            await asyncio.to_thread(mongo_history._client.admin.command, "ping")
            checks["mongodb"] = "ok"
        else:
            checks["mongodb"] = "not configured"
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        checks["mongodb"] = f"error: {error_msg}"
    
    # BigTapp API is mocked (LLM-generated demo data) — always report ok
    checks["bigtapp_api"] = "ok (demo mode)"
    
    # Determine overall readiness
    # Core dependencies: redis + router/response LLMs must be ok
    core_ok = checks.get("redis") == "ok" and llm_ok
    optional_statuses = ["ok", "not available", "not configured", "unreachable", "timeout"]
    optional_ok = all(
        v in optional_statuses or v.startswith("error")
        for k, v in checks.items()
        if k not in ("redis", "llm", "llm_router", "llm_response")
    )
    
    return {
        "ready": core_ok,
        "checks": checks,
    }


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint (non-blocking)."""
        # Run synchronous prometheus serialization in thread pool
        content = await asyncio.to_thread(generate_latest)
        return Response(
            content=content,
            media_type=CONTENT_TYPE_LATEST,
        )


# ============================================
# Chat Endpoint
# ============================================

@app.post("/agent-chat", response_model=ChatResponse)
async def agent_chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the agentic chatbot.
    
    Args:
        request: ChatRequest with session_id and message
        
    Returns:
        ChatResponse with bot response and debug info
    """
    start_time = time.time()
    session_id = request.session_id
    
    try:
        # Apply timeout to prevent hanging requests from exhausting workers
        result = await asyncio.wait_for(
            agentic_chat(session_id, request.message),
            timeout=CHAT_TIMEOUT_SECONDS
        )
        
        # Record metrics
        latency = time.time() - start_time
        AGENTIC_LATENCY.labels(endpoint="agent-chat").observe(latency)
        AGENTIC_MESSAGES_TOTAL.labels(
            result="ok",
            product=result.get("debug_state", {}).get("product") or "unknown"
        ).inc()
        
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources", ""),
            debug_state=result.get("debug_state"),
        )
    
    except asyncio.TimeoutError:
        latency = time.time() - start_time
        logger.error(
            "Chat timeout after %.2fs",
            latency,
            extra={"session_id": session_id}
        )
        AGENTIC_LATENCY.labels(endpoint="agent-chat").observe(latency)
        AGENTIC_MESSAGES_TOTAL.labels(result="timeout", product="unknown").inc()
        return ChatResponse(
            response="I'm sorry, the request took too long. Please try again.",
            debug_state={"error": "timeout", "timeout_seconds": CHAT_TIMEOUT_SECONDS},
        )
        
    except Exception as e:
        latency = time.time() - start_time
        # Truncate error message to prevent log flooding
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        logger.error(
            "Chat error: %s",
            error_msg,
            extra={"session_id": session_id},
            exc_info=True
        )
        AGENTIC_LATENCY.labels(endpoint="agent-chat").observe(latency)
        AGENTIC_MESSAGES_TOTAL.labels(result="error", product="unknown").inc()
        return ChatResponse(
            response="I'm sorry, something went wrong. Please try again.",
            debug_state={"error": error_msg},
        )


# ============================================
# WhatsApp Webhook Endpoints
# ============================================

@app.get("/webhook/whatsapp")
async def whatsapp_verification(request: Request):
    """WhatsApp webhook verification (GET)."""
    return await handle_agentic_whatsapp_verification(request)


@app.post("/webhook/whatsapp")
async def whatsapp_message(request: Request):
    """WhatsApp message handler (POST)."""
    return await handle_agentic_whatsapp_message(request)


# ============================================
# Session Management Endpoints
# ============================================

@app.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a session to initial state."""
    try:
        session_manager = get_session_manager()
        await session_manager.reset_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} reset"}
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        logger.error("Session reset error: %s", error_msg, extra={"session_id": session_id})
        return {"status": "error", "message": error_msg}


@app.get("/session/{session_id}")
async def get_session(
    session_id: str,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")
):
    """Get session state (for debugging). Protected by admin key."""
    # Simple security check for production
    admin_secret = os.getenv("ADMIN_API_KEY")
    if admin_secret and x_admin_key != admin_secret:
        return Response(content="Unauthorized", status_code=403)

    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(session_id)
        return {"status": "ok", "session": session}
    except Exception as e:
        error_msg = str(e)[:MAX_ERROR_LOG_LENGTH]
        logger.error("Session get error: %s", error_msg, extra={"session_id": session_id})
        return {"status": "error", "message": error_msg}


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Uvicorn does not allow reload=True with workers > 1
    if reload_enabled:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
        )
