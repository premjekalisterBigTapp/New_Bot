"""
MongoDB conversation history persistence for the agentic chatbot.

Production Features:
- Configurable connection timeouts (fast fail, no hanging)
- UTC timestamps for consistent storage
- Connection pool sizing
- Proper shutdown handling

Environment Variables:
    MONGO_URI: MongoDB connection string
    DB_NAME: Database name
    AGENTIC_HISTORY_COLLECTION: Collection name (default: agentic_conversation_history)
    MONGO_CONNECT_TIMEOUT_MS: Connection timeout (default: 3000)
    MONGO_SERVER_SELECTION_TIMEOUT_MS: Server selection timeout (default: 5000)
    MONGO_MAX_POOL_SIZE: Max connection pool size (default: 50)
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "").lower()
COLLECTION_NAME = os.getenv("AGENTIC_HISTORY_COLLECTION", "agentic_conversation_history")

# Timeout configuration (milliseconds)
CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "3000"))
SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))
MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "50"))
INIT_RETRY_SECONDS = float(os.getenv("MONGO_INIT_RETRY_SECONDS", "30"))

# ============================================
# Lazy Initialization State
# ============================================
_client = None
_db = None
_collection = None
_initialized = False
_init_disabled_reason: Optional[str] = None
_last_init_attempt = 0.0


def _init_if_needed() -> bool:
    """
    Initialize MongoDB connection lazily.
    
    Returns True if initialized successfully, False otherwise.
    """
    global _initialized, _client, _db, _collection, _init_disabled_reason, _last_init_attempt
    
    if _initialized:
        return _db is not None

    if _init_disabled_reason:
        return False

    now = time.time()
    if now - _last_init_attempt < INIT_RETRY_SECONDS:
        return False

    _last_init_attempt = now
    
    try:
        from pymongo import MongoClient
    except ImportError:
        _init_disabled_reason = "pymongo_not_installed"
        logger.warning("Agentic Mongo history: pymongo not installed; history persistence disabled")
        return False
    
    if not MONGO_URI or not DB_NAME:
        _init_disabled_reason = "not_configured"
        logger.warning("Agentic Mongo history: MONGO_URI/DB_NAME not set; history persistence disabled")
        return False
    
    try:
        _client = MongoClient(
            MONGO_URI,
            tz_aware=True,
            connectTimeoutMS=CONNECT_TIMEOUT_MS,
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
            maxPoolSize=MAX_POOL_SIZE,
        )
        
        # Verify connection
        _client.admin.command("ping")
        
        _db = _client[DB_NAME]
        _collection = _db[COLLECTION_NAME]
        _initialized = True
        _init_disabled_reason = None
        
        logger.info(
            "Agentic Mongo history initialized: db='%s', collection='%s', pool_size=%d",
            DB_NAME,
            COLLECTION_NAME,
            MAX_POOL_SIZE,
        )
        return True
        
    except Exception as e:
        logger.error("Agentic Mongo history initialization failed: %s", e)
        _client = None
        _db = None
        _collection = None
        _initialized = False
        return False


def get_mongo_client():
    """Get the MongoDB client (for health checks)."""
    _init_if_needed()
    return _client


def log_history(
    session_id: str, 
    user_message: str, 
    assistant_message: str, 
    ts: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a conversation turn to MongoDB.
    
    Args:
        session_id: Unique session identifier
        user_message: User's message
        assistant_message: Bot's response
        ts: Unix timestamp (defaults to now)
        metadata: Optional dict with additional info (product, intent, etc.)
    """
    if not _init_if_needed():
        return
    
    if ts is None:
        ts = time.time()
    
    # Store as UTC datetime (MongoDB preferred format)
    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
    
    doc = {
        "session_id": session_id,
        "timestamp": timestamp,
        "ts_unix": ts,  # Also store Unix timestamp for easier querying
        "user": user_message,
        "assistant": assistant_message,
    }
    
    if metadata:
        doc["metadata"] = metadata
    
    _collection.insert_one(doc)


def get_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history for a session from MongoDB.
    
    Args:
        session_id: Unique session identifier
        limit: Maximum number of turns to retrieve
        
    Returns:
        List of conversation turns, oldest first
    """
    if not _init_if_needed():
        return []
    
    cursor = _collection.find(
        {"session_id": session_id}
    ).sort("timestamp", 1).limit(limit)
    
    return list(cursor)


def clear_history(session_id: str) -> int:
    """
    Clear conversation history for a session from MongoDB.
    
    Returns:
        Number of documents deleted
    """
    if not _init_if_needed():
        return 0
    
    result = _collection.delete_many({"session_id": session_id})
    deleted = result.deleted_count
    
    if deleted > 0:
        logger.info("Agentic Mongo history: cleared %d documents for session %s", deleted, session_id)
    
    return deleted


def get_history_count(session_id: str) -> int:
    """Get the number of history entries for a session."""
    if not _init_if_needed():
        return 0
    
    return _collection.count_documents({"session_id": session_id})


def get_mongo_health() -> Dict[str, Any]:
    """Health check for MongoDB connection."""
    if not _init_if_needed():
        return {"status": "disabled", "reason": "not configured or pymongo not installed"}
    
    try:
        # Fast ping check
        start = time.time()
        _client.admin.command("ping")
        latency_ms = (time.time() - start) * 1000
        
        return {
            "status": "healthy",
            "database": DB_NAME,
            "collection": COLLECTION_NAME,
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def close_mongo() -> None:
    """Close MongoDB connection (call on shutdown)."""
    global _client, _db, _collection, _initialized, _init_disabled_reason
    
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        _collection = None
        logger.info("MongoDB connection closed")
    
    _initialized = False
    _init_disabled_reason = None
