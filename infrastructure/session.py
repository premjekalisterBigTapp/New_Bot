"""
Production-grade session management for the agentic chatbot.

Production Features:
- AsyncRedisLock for atomic read-modify-write operations (prevents race conditions)
- UTC timestamps (floats) for fast, timezone-free comparisons
- Integrates with idle_monitor ZSET for efficient idle tracking

Uses Redis for session state and integrates with MongoDB for history persistence.
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any, Optional

from .redis_utils import AsyncSessionCache, AsyncRedisLock
from .metrics import SESSION_CACHE_HITS, SESSION_CACHE_MISSES
from .idle_monitor import track_activity, remove_from_tracker

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
SESSION_IDLE_RESET_SECONDS = int(os.getenv("SESSION_IDLE_RESET_SECONDS", os.getenv("SESSION_CACHE_TTL_SECONDS", "900")))
SESSION_LOCK_TTL = float(os.getenv("SESSION_LOCK_TTL_SECONDS", "5.0"))
SESSION_LOCK_WAIT = float(os.getenv("SESSION_LOCK_WAIT_SECONDS", "3.0"))

DEFAULT_SESSION_FIELDS = {
    "product": None,
    "slots": {},
    "recommended_tier": None,
    "live_agent_status": False,
    "intent": None,
}


class SessionManager:
    """
    Redis-backed session manager for the agentic chatbot.
    
    Production Features:
    - Atomic updates via AsyncRedisLock (prevents race conditions)
    - UTC timestamps (floats) for fast comparisons
    - Singleton pattern for connection reuse
    """
    _instance: Optional["SessionManager"] = None
    _cache: Optional[AsyncSessionCache] = None

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._cache = AsyncSessionCache(prefix="agentic:session")
            logger.info("Agentic SessionManager initialized (Redis-backed)")
        return cls._instance

    def _new_session(self, session_id: str, now: float) -> Dict[str, Any]:
        """Create a new session with default fields."""
        session = {
            "session_id": session_id,
            "history": [],
            "created_at": now,
            "last_active": now,
        }
        session.update(DEFAULT_SESSION_FIELDS)
        return session

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Fetch session from Redis; create if missing.
        Performs idle reset if session exceeds idle timeout.
        """
        now = time.time()
        cached = await self._cache.get(session_id)
        
        if cached:
            SESSION_CACHE_HITS.inc()
        else:
            SESSION_CACHE_MISSES.inc()
            logger.info("Creating new agentic session: %s", session_id)
            session_data = self._new_session(session_id, now)
            await self._cache.set(session_id, session_data)
            return session_data

        # Idle reset check (fast float comparison)
        last_active = cached.get("last_active")
        
        # Handle legacy ISO string format (migration)
        if isinstance(last_active, str):
            # Legacy format - treat as expired for simplicity
            last_active = 0.0
        
        if not isinstance(last_active, (int, float)):
            last_active = 0.0
        
        idle_duration = now - last_active
        
        if idle_duration > SESSION_IDLE_RESET_SECONDS:
            logger.info(
                "Idle reset: session %s inactive %.0fs (threshold: %ds)",
                session_id,
                idle_duration,
                SESSION_IDLE_RESET_SECONDS,
            )
            reset_state = self._new_session(session_id, now)
            # Preserve original creation time
            if cached.get("created_at"):
                reset_state["created_at"] = cached["created_at"]
            await self._cache.set(session_id, reset_state)
            return reset_state

        return cached

    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Persist session state to Redis with atomic locking.
        
        Uses AsyncRedisLock to prevent race conditions from concurrent updates.
        """
        if not session_data:
            raise ValueError(f"Cannot save empty session data for {session_id}")
        
        async with AsyncRedisLock(f"session:{session_id}", ttl_seconds=SESSION_LOCK_TTL, wait_timeout=SESSION_LOCK_WAIT):
            session_state = dict(session_data)
            session_state["last_active"] = time.time()
            await self._cache.set(session_id, session_state)
        
        # Track activity for idle monitor (O(log N) ZSET update)
        await track_activity(session_id)
        
        logger.debug("Saved agentic session: %s", session_id)

    async def add_history_entry(
        self, 
        session_id: str, 
        user_message: str, 
        bot_response: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add conversation turn to in-session Redis cache (keep last 5 turns).
        
        Uses atomic locking to prevent race conditions.
        
        Note: MongoDB persistence is handled centrally by agentic_chat() via
        BackgroundLogger. This method only updates the Redis session cache
        for quick in-session context retrieval.
        """
        now = time.time()
        
        async with AsyncRedisLock(f"session:{session_id}", ttl_seconds=SESSION_LOCK_TTL, wait_timeout=SESSION_LOCK_WAIT):
            cached = await self._cache.get(session_id) or self._new_session(session_id, now)
            
            # Update in-session history (rolling window for quick context)
            hist = cached.get("history", [])
            hist.append({
                "timestamp": now,
                "user": user_message,
                "assistant": bot_response[:200] if len(bot_response) > 200 else bot_response,
            })
            
            # Keep only last 5 turns
            if len(hist) > 5:
                hist = hist[-5:]
            
            cached["history"] = hist
            cached["last_active"] = now
            await self._cache.set(session_id, cached)
        
        # Track activity for idle monitor (O(log N) ZSET update)
        await track_activity(session_id)
        
        logger.debug("Session history updated in Redis for %s", session_id)

    async def reset_session(self, session_id: str) -> None:
        """Reset session to defaults and clear history."""
        now = time.time()
        
        async with AsyncRedisLock(f"session:{session_id}", ttl_seconds=SESSION_LOCK_TTL, wait_timeout=SESSION_LOCK_WAIT):
            cached = await self._cache.get(session_id)
            new_state = self._new_session(session_id, now)
            
            # Preserve original creation time
            if cached and cached.get("created_at"):
                new_state["created_at"] = cached["created_at"]
            
            await self._cache.set(session_id, new_state)
        
        # Remove from idle tracker (prevents pending farewell after reset)
        await remove_from_tracker(session_id)
        
        logger.info("Reset agentic session: %s", session_id)

    async def delete_session(self, session_id: str) -> None:
        """Completely remove session from Redis."""
        await self._cache.delete(session_id)
        
        # Remove from idle tracker
        await remove_from_tracker(session_id)
        
        logger.info("Deleted agentic session: %s", session_id)

    async def update_field(self, session_id: str, field: str, value: Any) -> None:
        """
        Update a single field in the session atomically.
        
        Uses locking to prevent race conditions.
        """
        async with AsyncRedisLock(f"session:{session_id}", ttl_seconds=SESSION_LOCK_TTL, wait_timeout=SESSION_LOCK_WAIT):
            session = await self._cache.get(session_id)
            if session is None:
                session = self._new_session(session_id, time.time())
            
            session[field] = value
            session["last_active"] = time.time()
            await self._cache.set(session_id, session)
        
        # Track activity
        await track_activity(session_id)

    async def get_field(self, session_id: str, field: str, default: Any = None) -> Any:
        """Get a single field from the session without locking."""
        session = await self.get_session(session_id)
        return session.get(field, default)

    async def is_live_agent_active(self, session_id: str) -> bool:
        """Check if live agent mode is active for this session."""
        session = await self.get_session(session_id)
        val = session.get("live_agent_status")
        if isinstance(val, str):
            return val.strip().lower() in ("on", "true", "yes", "1")
        return bool(val)

    async def set_live_agent_status(self, session_id: str, status: bool) -> None:
        """Set live agent status for this session."""
        await self.update_field(session_id, "live_agent_status", status)

    async def touch_session(self, session_id: str) -> None:
        """Update last_active timestamp without modifying other fields."""
        async with AsyncRedisLock(f"session:{session_id}", ttl_seconds=SESSION_LOCK_TTL, wait_timeout=SESSION_LOCK_WAIT):
            session = await self._cache.get(session_id)
            if session:
                session["last_active"] = time.time()
                await self._cache.set(session_id, session)
        
        await track_activity(session_id)
