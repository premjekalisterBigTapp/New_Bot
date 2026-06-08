"""
Redis utilities for production-grade session, rate limiting, deduplication, and locking.

Production Features:
- Explicit connection pool sizing to prevent connection exhaustion
- Exponential backoff for lock acquisition (no CPU-hogging spin locks)
- Cached Lua scripts via EVALSHA for lock release performance
- Module-level serialization setup (no per-call hasattr checks)
- Configurable via environment variables

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://127.0.0.1:6379/0)
    AGENTIC_SESSION_TTL_SECONDS: Session TTL (default: 900)
    AGENTIC_REDIS_MAX_CONNECTIONS: Max pool connections (default: 100)
    RL_WINDOW_SECONDS: Rate limit window (default: 60)
    RL_MAX_MESSAGES: Rate limit max per window (default: 10)
    DEDUPE_TTL_SECONDS: Deduplication TTL (default: 86400)
    ORDER_TTL_SECONDS: Order guard TTL (default: 86400)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import logging
from typing import Any, Dict, Optional, AsyncContextManager

logger = logging.getLogger(__name__)

# ============================================
# Serialization Setup (Module-level, one-time)
# ============================================
try:
    import orjson
    
    def _json_dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, default=str)
    
    def _json_loads(data: bytes | str) -> Any:
        return orjson.loads(data)
    
    _USING_ORJSON = True
except ImportError:
    import json
    
    def _json_dumps(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode("utf-8")
    
    def _json_loads(data: bytes | str) -> Any:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    
    _USING_ORJSON = False

# ============================================
# Redis Import
# ============================================
try:
    import redis
    import redis.asyncio as redis_async
    from redis.asyncio import ConnectionPool as AsyncConnectionPool
except ImportError as e:
    raise ImportError("redis package is required. Install with 'pip install redis'.") from e

# ============================================
# Configuration (Read once at module load)
# ============================================
_REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
_SESSION_TTL = int(os.getenv("AGENTIC_SESSION_TTL_SECONDS", os.getenv("SESSION_CACHE_TTL_SECONDS", "1200")))
_MAX_CONNECTIONS = int(os.getenv("AGENTIC_REDIS_MAX_CONNECTIONS", "100"))
_RL_WINDOW = int(os.getenv("RL_WINDOW_SECONDS", "60"))
_RL_MAX = int(os.getenv("RL_MAX_MESSAGES", "10"))
_DEDUPE_TTL = int(os.getenv("DEDUPE_TTL_SECONDS", "86400"))
_ORDER_TTL = int(os.getenv("ORDER_TTL_SECONDS", "86400"))

# ============================================
# Singleton Async Client with Connection Pool
# ============================================
_async_pool: Optional[AsyncConnectionPool] = None
_async_client: Optional[redis_async.Redis] = None
_async_init_lock: Optional[asyncio.Lock] = None

# Lua script for atomic lock release (cached SHA)
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""
_async_release_lock_sha: Optional[str] = None


async def get_async_redis() -> redis_async.Redis:
    """
    Return a singleton async Redis client with explicit connection pooling.

    Raises RuntimeError if connection fails.
    """
    global _async_pool, _async_client, _async_release_lock_sha

    if _async_client is not None:
        return _async_client

    global _async_init_lock
    if _async_init_lock is None:
        _async_init_lock = asyncio.Lock()

    async with _async_init_lock:
        if _async_client is not None:
            return _async_client

        _async_pool = AsyncConnectionPool.from_url(
            _REDIS_URL,
            max_connections=_MAX_CONNECTIONS,
            decode_responses=False,
        )

        _async_client = redis_async.Redis(connection_pool=_async_pool)

        await _async_client.ping()
        _async_release_lock_sha = await _async_client.script_load(_RELEASE_LOCK_SCRIPT)

        logger.info(
            "Agentic Redis async connected: %s (pool_size=%d, orjson=%s)",
            _REDIS_URL,
            _MAX_CONNECTIONS,
            _USING_ORJSON,
        )

    return _async_client


async def get_async_redis_health() -> Dict[str, Any]:
    """Async health check for Redis connection."""
    try:
        client = await get_async_redis()
        info = await client.info("server")
        clients = await client.info("clients")
        memory = await client.info("memory")
        return {
            "status": "healthy",
            "redis_version": info.get("redis_version"),
            "connected_clients": clients.get("connected_clients"),
            "used_memory_human": memory.get("used_memory_human"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ============================================
# Distributed Lock (with exponential backoff)
# ============================================

class AsyncRedisLock(AsyncContextManager["AsyncRedisLock"]):
    """
    Async Redis-based distributed lock with token verification.

    Uses exponential backoff with asyncio.sleep to avoid blocking the event loop.
    """

    def __init__(
        self,
        key: str,
        ttl_seconds: float = 10.0,
        wait_timeout: float = 5.0,
        retry_base_ms: float = 20.0,
        retry_max_ms: float = 200.0,
    ):
        self._client: Optional[redis_async.Redis] = None
        self._key = f"agentic:lock:{key}"
        self._ttl_ms = int(ttl_seconds * 1000)
        self._wait_timeout = wait_timeout
        self._retry_base_ms = retry_base_ms
        self._retry_max_ms = retry_max_ms
        self._token = str(uuid.uuid4())
        self._acquired = False

    async def __aenter__(self) -> "AsyncRedisLock":
        self._client = await get_async_redis()
        deadline = time.monotonic() + self._wait_timeout
        attempt = 0

        while time.monotonic() < deadline:
            if await self._client.set(self._key, self._token, nx=True, px=self._ttl_ms):
                self._acquired = True
                return self

            delay_ms = min(self._retry_base_ms * (2 ** attempt), self._retry_max_ms)
            jitter = delay_ms * 0.25 * (0.5 - (time.monotonic() % 1))
            delay_seconds = (delay_ms + jitter) / 1000.0

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            await asyncio.sleep(min(delay_seconds, remaining))
            attempt += 1

        raise TimeoutError(f"Failed to acquire RedisLock for {self._key} within {self._wait_timeout}s")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._acquired or not self._client:
            return

        global _async_release_lock_sha

        if _async_release_lock_sha:
            try:
                await self._client.evalsha(_async_release_lock_sha, 1, self._key, self._token)
            except redis.exceptions.NoScriptError:
                _async_release_lock_sha = await self._client.script_load(_RELEASE_LOCK_SCRIPT)
                await self._client.evalsha(_async_release_lock_sha, 1, self._key, self._token)
        else:
            await self._client.eval(_RELEASE_LOCK_SCRIPT, 1, self._key, self._token)


# ============================================
# Session Cache
# ============================================

class AsyncSessionCache:
    """Async JSON-based session cache in Redis with TTL."""

    def __init__(self, prefix: str = "agentic:session", ttl_seconds: Optional[int] = None):
        self._client: Optional[redis_async.Redis] = None
        self._ttl = ttl_seconds if ttl_seconds is not None else _SESSION_TTL
        self._prefix = prefix

    async def _get_client(self) -> redis_async.Redis:
        if self._client is None:
            self._client = await get_async_redis()
        return self._client

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        client = await self._get_client()
        raw = await client.get(self._key(session_id))
        if not raw:
            return None
        return _json_loads(raw)

    async def set(self, session_id: str, data: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        client = await self._get_client()
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        payload = _json_dumps(data)

        if ttl > 0:
            await client.set(self._key(session_id), payload, ex=ttl)
        else:
            await client.set(self._key(session_id), payload)

    async def delete(self, session_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(session_id))

    async def exists(self, session_id: str) -> bool:
        client = await self._get_client()
        return bool(await client.exists(self._key(session_id)))

    async def touch(self, session_id: str, ttl_seconds: Optional[int] = None) -> bool:
        client = await self._get_client()
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        return bool(await client.expire(self._key(session_id), ttl))


# ============================================
# Rate Limiter (Fixed Window)
# ============================================

class AsyncRateLimiter:
    """Async fixed-window rate limiter using Redis INCR + EXPIRE."""

    def __init__(
        self,
        window_seconds: int = _RL_WINDOW,
        max_messages: int = _RL_MAX,
        scope: str = "agentic",
    ):
        self._client: Optional[redis_async.Redis] = None
        self._window = window_seconds
        self._max = max_messages
        self._scope = scope

    async def _get_client(self) -> redis_async.Redis:
        if self._client is None:
            self._client = await get_async_redis()
        return self._client

    async def allow(self, key: str) -> bool:
        client = await self._get_client()
        redis_key = f"agentic:rl:{self._scope}:{key}"

        pipe = client.pipeline(transaction=True)
        pipe.incr(redis_key)
        pipe.ttl(redis_key)
        results = await pipe.execute()

        count = results[0]
        ttl = results[1]

        if ttl == -1:
            await client.expire(redis_key, self._window)

        return count <= self._max

    async def get_remaining(self, key: str) -> int:
        client = await self._get_client()
        redis_key = f"agentic:rl:{self._scope}:{key}"
        count = await client.get(redis_key)
        if count is None:
            return self._max
        if isinstance(count, bytes):
            count = int(count.decode("utf-8"))
        return max(0, self._max - int(count))


# ============================================
# Deduplicator
# ============================================

class AsyncDeduplicator:
    """Async deduplicator for message IDs within a TTL window using SETNX."""

    def __init__(self, ttl_seconds: int = _DEDUPE_TTL, scope: str = "agentic"):
        self._client: Optional[redis_async.Redis] = None
        self._ttl = ttl_seconds
        self._scope = scope

    async def _get_client(self) -> redis_async.Redis:
        if self._client is None:
            self._client = await get_async_redis()
        return self._client

    async def is_new(self, message_id: str) -> bool:
        client = await self._get_client()
        key = f"agentic:dedupe:{self._scope}:{message_id}"
        created = await client.set(key, b"1", nx=True, ex=self._ttl)
        return bool(created)

    async def mark_seen(self, message_id: str) -> None:
        client = await self._get_client()
        key = f"agentic:dedupe:{self._scope}:{message_id}"
        await client.set(key, b"1", ex=self._ttl)


# ============================================
# Order Guard
# ============================================

class AsyncOrderGuard:
    """Async order guard to ensure non-decreasing timestamp order per user."""

    def __init__(self, ttl_seconds: int = _ORDER_TTL, scope: str = "agentic"):
        self._client: Optional[redis_async.Redis] = None
        self._ttl = ttl_seconds
        self._scope = scope

    async def _get_client(self) -> redis_async.Redis:
        if self._client is None:
            self._client = await get_async_redis()
        return self._client

    async def allow(self, user_key: str, ts: int) -> bool:
        client = await self._get_client()
        key = f"agentic:order:{self._scope}:{user_key}"

        script = """
        local last = redis.call('get', KEYS[1])
        if last and tonumber(ARGV[1]) < tonumber(last) then
            return 0
        end
        redis.call('set', KEYS[1], ARGV[1])
        redis.call('expire', KEYS[1], ARGV[2])
        return 1
        """

        result = await client.eval(script, 1, key, str(ts), str(self._ttl))
        return bool(result)


# ============================================
# Utility Functions
# ============================================

def session_lock_key(session_id: str) -> str:
    """Generate a lock key for a session."""
    return f"session:{session_id}"


async def close_async_redis() -> None:
    """Close async Redis connection (call on shutdown)."""
    global _async_client, _async_pool

    if _async_client is not None:
        await _async_client.close()
        _async_client = None

    if _async_pool is not None:
        await _async_pool.disconnect(inuse_connections=True)
        _async_pool = None

    logger.info("Async Redis connection closed")


__all__ = [
    "get_async_redis",
    "get_async_redis_health",
    "close_async_redis",
    "AsyncRedisLock",
    "AsyncSessionCache",
    "AsyncRateLimiter",
    "AsyncDeduplicator",
    "AsyncOrderGuard",
    "session_lock_key",
]
