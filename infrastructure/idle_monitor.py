"""
Session Idle Monitor for Agentic Chatbot
=========================================

Production-grade background task that monitors WhatsApp sessions for inactivity
and sends a farewell message before the session times out.

Architecture:
    Uses a Redis Sorted Set (ZSET) for O(log N) idle detection instead of O(N) key scanning.
    - Key: `agentic:activity_tracker`
    - Score: Unix timestamp of last activity
    - Member: session_id (e.g., "whatsapp_1234567890")
    
    When a user sends a message, their session is added/updated in the ZSET.
    The monitor queries for sessions with score < (now - IDLE_FAREWELL_SECONDS).
    This is O(log N + M) where M is the number of idle sessions.

Environment Variables:
    ENABLE_IDLE_FAREWELL: Enable/disable the monitor (default: false)
    IDLE_FAREWELL_SECONDS: Seconds of inactivity before sending farewell (default: 0 = disabled)
    IDLE_FAREWELL_MESSAGE: Custom farewell message
    IDLE_MONITOR_POLL_SECONDS: How often to scan for idle sessions (default: 60)
    IDLE_MONITOR_BATCH_SIZE: Max sessions to process per scan (default: 100)
    IDLE_MONITOR_CONCURRENCY: Max parallel farewell sends (default: 10)
"""

import asyncio
import logging
import os
import time
from typing import List, Optional

from .redis_utils import get_async_redis, AsyncRedisLock, session_lock_key
from .metrics import AGENTIC_MESSAGES_TOTAL

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

def _env_flag(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default)
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


ENABLE_IDLE_FAREWELL = _env_flag("ENABLE_IDLE_FAREWELL", "false")
IDLE_FAREWELL_SECONDS = int(os.getenv("IDLE_FAREWELL_SECONDS", "0") or "0")
IDLE_FAREWELL_MESSAGE = os.getenv(
    "IDLE_FAREWELL_MESSAGE",
    "It looks like you haven't sent any new questions for a while, so I'll close this chat now. "
    "If you need anything else, just message me again. Have a great day!",
)
IDLE_MONITOR_POLL_SECONDS = int(os.getenv("IDLE_MONITOR_POLL_SECONDS", "60") or "60")
IDLE_MONITOR_BATCH_SIZE = int(os.getenv("IDLE_MONITOR_BATCH_SIZE", "100") or "100")
IDLE_MONITOR_CONCURRENCY = int(os.getenv("IDLE_MONITOR_CONCURRENCY", "10") or "10")

# Redis key for the activity tracker ZSET
ACTIVITY_TRACKER_KEY = "agentic:activity_tracker"
# Redis key prefix for farewell-sent markers (prevents duplicate farewells)
FAREWELL_SENT_PREFIX = "agentic:farewell_sent:"

# WhatsApp handler reference (set by main.py on startup)
_whatsapp_handler = None


# ============================================
# Activity Tracking (Called by SessionManager)
# ============================================

async def track_activity(session_id: str, timestamp: Optional[float] = None) -> None:
    """
    Track session activity in the ZSET for O(log N) idle detection.
    
    Called by SessionManager when a session is accessed or updated.
    
    Args:
        session_id: The session identifier (e.g., "whatsapp_1234567890")
        timestamp: Unix timestamp of activity (defaults to now)
    """
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        return
    
    # Only track WhatsApp sessions
    if not session_id.startswith("whatsapp_"):
        return
    
    try:
        redis = await get_async_redis()
        ts = timestamp if timestamp is not None else time.time()
        await redis.zadd(ACTIVITY_TRACKER_KEY, {session_id: ts})
    except Exception as e:
        logger.warning("IdleMonitor: Failed to track activity: %s", e)


async def remove_from_tracker(session_id: str) -> None:
    """
    Remove a session from the activity tracker.
    
    Called when a session is explicitly reset (e.g., user says "hi").
    """
    try:
        redis = await get_async_redis()
        await redis.zrem(ACTIVITY_TRACKER_KEY, session_id)
        # Also clear farewell marker
        await redis.delete(f"{FAREWELL_SENT_PREFIX}{session_id}")
    except Exception as e:
        logger.warning("IdleMonitor: Failed to remove from tracker: %s", e)


# ============================================
# WhatsApp Handler Registration
# ============================================

def set_whatsapp_handler(handler) -> None:
    """Set the WhatsApp handler reference for sending farewell messages."""
    global _whatsapp_handler
    _whatsapp_handler = handler
    logger.info("IdleMonitor: WhatsApp handler registered")


# ============================================
# Farewell Processing
# ============================================

async def _send_farewell_to_session(session_id: str) -> bool:
    """
    Send farewell message to a single session.
    
    Returns True if farewell was sent successfully, False otherwise.
    Thread-safe via Redis lock.
    """
    if _whatsapp_handler is None:
        logger.warning("IdleMonitor: No WhatsApp handler registered")
        return False
    
    if not session_id.startswith("whatsapp_"):
        return False
    
    phone = session_id[len("whatsapp_"):]
    lock_key = session_lock_key(session_id)
    farewell_marker_key = f"{FAREWELL_SENT_PREFIX}{session_id}"
    
    try:
        # Use a short lock to prevent race with active message processing
        async with AsyncRedisLock(lock_key, ttl_seconds=10.0, wait_timeout=0.5):
            redis = await get_async_redis()

            # Double-check: farewell not already sent
            if await redis.exists(farewell_marker_key):
                return False

            # Double-check: session still idle (user didn't just send a message)
            score = await redis.zscore(ACTIVITY_TRACKER_KEY, session_id)
            if score is None:
                # Session was removed from tracker (e.g., user reset)
                return False

            idle_duration = time.time() - float(score)
            if idle_duration < IDLE_FAREWELL_SECONDS:
                # Session became active again
                return False

            # Check if live agent is active (from session data)
            session_key = f"agentic:session:{session_id}"
            raw_session = await redis.get(session_key)
            if raw_session:
                try:
                    import orjson
                    session_data = orjson.loads(raw_session)
                except Exception:
                    import json
                    session_data = json.loads(raw_session)

                las = session_data.get("live_agent_status")
                if isinstance(las, str):
                    is_live = las.strip().lower() in ("on", "true", "yes", "1")
                else:
                    is_live = bool(las)

                if is_live:
                    logger.debug("IdleMonitor: Session %s is in live_agent state; skipping", session_id)
                    return False

            if not hasattr(_whatsapp_handler, '_send_message_async'):
                logger.warning("IdleMonitor: WhatsApp handler missing _send_message_async")
                return False
            await _whatsapp_handler._send_message_async(phone, IDLE_FAREWELL_MESSAGE)

            # Mark farewell as sent (set with TTL matching session TTL)
            await redis.set(farewell_marker_key, "1", ex=1800)  # 30 min TTL

            # Remove from activity tracker (no need to check again)
            await redis.zrem(ACTIVITY_TRACKER_KEY, session_id)
            
            AGENTIC_MESSAGES_TOTAL.labels(result="idle_farewell", product="none").inc()
            logger.info("IdleMonitor: Sent farewell to %s", session_id)
            return True
            
    except TimeoutError:
        # Lock acquisition failed - session is being processed elsewhere
        logger.debug("IdleMonitor: Lock timeout for %s; skipping", session_id)
        return False
    except Exception as e:
        logger.error("IdleMonitor: Error processing %s: %s", session_id, e)
        return False


async def _process_idle_sessions_batch(session_ids: List[str]) -> int:
    """
    Process a batch of idle sessions with controlled concurrency.
    
    Returns the number of farewells sent successfully.
    """
    if not session_ids:
        return 0
    
    # Use semaphore to limit concurrent WhatsApp API calls
    semaphore = asyncio.Semaphore(IDLE_MONITOR_CONCURRENCY)
    
    async def process_with_semaphore(session_id: str) -> bool:
        async with semaphore:
            return await _send_farewell_to_session(session_id)
    
    # Process all sessions concurrently (up to IDLE_MONITOR_CONCURRENCY at a time)
    results = await asyncio.gather(
        *[process_with_semaphore(sid) for sid in session_ids],
        return_exceptions=True
    )
    
    # Count successes (True results)
    success_count = sum(1 for r in results if r is True)
    error_count = sum(1 for r in results if isinstance(r, Exception))
    
    if error_count > 0:
        logger.warning("IdleMonitor: %d errors in batch of %d sessions", error_count, len(session_ids))
    
    return success_count


# ============================================
# Main Scanning Logic (O(log N) with ZSET)
# ============================================

async def run_idle_farewell_scan_once() -> int:
    """
    Single scan pass to detect and process idle sessions.
    
    Uses ZRANGEBYSCORE for O(log N + M) complexity where M is idle sessions.
    
    Returns the number of farewells sent.
    """
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        return 0
    
    try:
        redis = await get_async_redis()
    except Exception as e:
        logger.error("IdleMonitor: Failed to get Redis client: %s", e)
        raise  # Don't swallow - let caller handle
    
    # Calculate the cutoff timestamp
    cutoff_timestamp = time.time() - IDLE_FAREWELL_SECONDS
    
    # Query for sessions that have been idle longer than the threshold
    # ZRANGEBYSCORE returns members with score between min and max
    # We want sessions with last_active < cutoff (i.e., idle too long)
    try:
        idle_sessions = await redis.zrangebyscore(
            ACTIVITY_TRACKER_KEY,
            "-inf",
            cutoff_timestamp,
            start=0,
            num=IDLE_MONITOR_BATCH_SIZE,
        )
    except Exception as e:
        logger.error("IdleMonitor: Failed to query idle sessions: %s", e)
        raise
    
    if not idle_sessions:
        logger.debug("IdleMonitor: No idle sessions found")
        return 0
    
    # Decode bytes to strings if necessary
    session_ids = []
    for sid in idle_sessions:
        if isinstance(sid, bytes):
            session_ids.append(sid.decode("utf-8"))
        elif isinstance(sid, str):
            session_ids.append(sid)
    
    logger.info("IdleMonitor: Found %d idle sessions to process", len(session_ids))
    
    # Process the batch
    success_count = await _process_idle_sessions_batch(session_ids)
    
    logger.info("IdleMonitor: Sent %d farewells in this scan", success_count)
    return success_count


# ============================================
# Background Loop
# ============================================

async def idle_monitor_loop() -> None:
    """
    Background loop to periodically scan for idle sessions.
    Start this in FastAPI lifespan.
    """
    logger.info(
        "IdleMonitor: Starting (enabled=%s, idle_seconds=%d, poll_interval=%ds, batch_size=%d, concurrency=%d)",
        ENABLE_IDLE_FAREWELL,
        IDLE_FAREWELL_SECONDS,
        IDLE_MONITOR_POLL_SECONDS,
        IDLE_MONITOR_BATCH_SIZE,
        IDLE_MONITOR_CONCURRENCY,
    )
    
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        logger.info("IdleMonitor: Disabled; exiting loop")
        return
    
    try:
        while True:
            try:
                await run_idle_farewell_scan_once()
            except Exception as e:
                logger.error("IdleMonitor: Error in scan loop: %s", e)
                # Continue running - don't let one error stop the monitor
            
            await asyncio.sleep(IDLE_MONITOR_POLL_SECONDS)
            
    except asyncio.CancelledError:
        logger.info("IdleMonitor: Loop cancelled; shutting down")
        raise


# ============================================
# Exports
# ============================================

__all__ = [
    # Loop control
    "idle_monitor_loop",
    "run_idle_farewell_scan_once",
    # Handler registration
    "set_whatsapp_handler",
    # Activity tracking (called by SessionManager)
    "track_activity",
    "remove_from_tracker",
    # Configuration
    "ENABLE_IDLE_FAREWELL",
    "IDLE_FAREWELL_SECONDS",
    "IDLE_FAREWELL_MESSAGE",
    "ACTIVITY_TRACKER_KEY",
]
