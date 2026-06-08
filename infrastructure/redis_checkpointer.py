"""
Redis-based LangGraph Checkpointer for production conversation persistence.

Production Features:
- Native async methods via redis.asyncio (no event loop blocking)
- Batched deletion to prevent OOM during thread cleanup
- Reuses redis_utils connection pool
- Configurable TTL via environment variable

Environment Variables:
    AGENTIC_CHECKPOINT_TTL_SECONDS: Checkpoint TTL (default: 86400 = 24h)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, AsyncIterator, List

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .redis_utils import get_async_redis

logger = logging.getLogger(__name__)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Synchronous RedisCheckpointer methods cannot be called from a running event loop. "
        "Use the async methods instead."
    )

# ============================================
# Serialization (Module-level, one-time)
# ============================================
try:
    import orjson
    
    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, default=str)
    
    def _loads(data: bytes | str) -> Any:
        return orjson.loads(data)
    
    _USING_ORJSON = True
except ImportError:
    import json
    
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode("utf-8")
    
    def _loads(data: bytes | str) -> Any:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    
    _USING_ORJSON = False

# ============================================
# Configuration
# ============================================
CHECKPOINT_TTL_SECONDS = int(os.getenv("AGENTIC_CHECKPOINT_TTL_SECONDS", "86400"))
DELETE_BATCH_SIZE = 500  # Max keys to delete per batch


class RedisCheckpointer(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph.
    
    Production Features:
    - Uses shared connection pool from redis_utils
    - Native async via redis.asyncio (doesn't block event loop)
    - Batched deletion to prevent OOM and Redis blocking
    - Automatic TTL-based expiration
    """

    def __init__(
        self,
        prefix: str = "agentic:checkpoint",
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._async_client = None

    async def _get_async_client(self):
        """Lazy async Redis client initialization (uses shared async pool)."""
        if self._async_client is None:
            self._async_client = await get_async_redis()
        return self._async_client

    # ========================================
    # Key Generation
    # ========================================
    
    def _checkpoint_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _metadata_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:meta:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self._prefix}:index:{thread_id}:{checkpoint_ns}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _type_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:type:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    # ========================================
    # Sync Methods (called by LangGraph)
    # ========================================

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config (sync wrapper)."""
        return _run_sync(self.aget_tuple(config))

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread (sync wrapper)."""
        if config is None:
            return

        return iter(
            _run_sync(self._alist_to_list(config, filter=filter, before=before, limit=limit))
        )

    async def _alist_to_list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[CheckpointTuple]:
        if config is None:
            return []

        items: List[CheckpointTuple] = []
        async for item in self.alist(config, filter=filter, before=before, limit=limit):
            items.append(item)
        return items

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store a checkpoint (sync wrapper)."""
        return _run_sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint (sync wrapper)."""
        _run_sync(self.aput_writes(config, writes, task_id))

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread (sync wrapper)."""
        _run_sync(self.adelete_thread(thread_id))

    # ========================================
    # Async Methods (Native redis.asyncio)
    # ========================================

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async get a checkpoint tuple."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        client = await self._get_async_client()

        if checkpoint_id is None:
            index_key = self._index_key(thread_id, checkpoint_ns)
            result = await client.zrevrange(index_key, 0, 0)
            if not result:
                return None
            checkpoint_id = result[0]
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")

        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)
        type_key = self._type_key(thread_id, checkpoint_ns, checkpoint_id)
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        pipe = client.pipeline(transaction=False)
        pipe.get(checkpoint_key)
        pipe.get(metadata_key)
        pipe.get(type_key)
        pipe.lrange(writes_key, 0, -1)
        results = await pipe.execute()

        checkpoint_data = results[0]
        metadata_data = results[1]
        checkpoint_type = results[2]
        pending_writes_data = results[3]

        if not checkpoint_data:
            return None

        if checkpoint_type:
            if isinstance(checkpoint_type, bytes):
                checkpoint_type = checkpoint_type.decode("utf-8")
        else:
            first_byte = checkpoint_data[0] if isinstance(checkpoint_data, bytes) else ord(checkpoint_data[0])
            checkpoint_type = "json" if first_byte in (0x7B, 0x5B) else "bytes"

        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_data))

        if metadata_data:
            if isinstance(metadata_data, bytes):
                metadata_data = metadata_data.decode("utf-8")
            metadata = _loads(metadata_data)
        else:
            metadata = {}

        parent_checkpoint_id = metadata.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        pending_writes = []
        for write_data in pending_writes_data:
            if isinstance(write_data, bytes):
                write_data = write_data.decode("utf-8")
            try:
                task_id, channel, value = _loads(write_data)
                pending_writes.append((task_id, channel, value))
            except (ValueError, TypeError):
                continue

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async list checkpoints for a thread."""
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        client = await self._get_async_client()
        index_key = self._index_key(thread_id, checkpoint_ns)

        if before:
            before_id = before["configurable"]["checkpoint_id"]
            rank = await client.zrevrank(index_key, before_id)
            if rank is None:
                return
            start = rank + 1
        else:
            start = 0

        end = start + (limit - 1) if limit else -1
        checkpoint_ids = await client.zrevrange(index_key, start, end)

        for checkpoint_id in checkpoint_ids:
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")
            checkpoint_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            result = await self.aget_tuple(checkpoint_config)
            if result:
                yield result

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Async store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        client = await self._get_async_client()

        checkpoint_type, checkpoint_data = self.serde.dumps_typed(checkpoint)

        meta_to_store = dict(metadata) if metadata else {}
        if parent_checkpoint_id:
            meta_to_store["parent_checkpoint_id"] = parent_checkpoint_id
        metadata_data = _dumps(meta_to_store)

        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)
        type_key = self._type_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._index_key(thread_id, checkpoint_ns)

        score = time.time()

        pipe = client.pipeline(transaction=True)
        pipe.set(checkpoint_key, checkpoint_data, ex=self._ttl)
        pipe.set(metadata_key, metadata_data, ex=self._ttl)
        pipe.set(type_key, checkpoint_type, ex=self._ttl)
        pipe.zadd(index_key, {checkpoint_id: score})
        pipe.expire(index_key, self._ttl)
        await pipe.execute()

        logger.debug("Stored checkpoint %s for thread %s", checkpoint_id, thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async store pending writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        client = await self._get_async_client()
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        pipe = client.pipeline(transaction=True)
        for channel, value in writes:
            write_data = _dumps([task_id, channel, value])
            pipe.rpush(writes_key, write_data)
        pipe.expire(writes_key, self._ttl)
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        """Async delete all checkpoints for a thread."""
        client = await self._get_async_client()

        if isinstance(thread_id, dict):
            thread_id = thread_id.get("configurable", {}).get("thread_id", str(thread_id))

        patterns = [
            f"{self._prefix}:{thread_id}:*",
            f"{self._prefix}:meta:{thread_id}:*",
            f"{self._prefix}:type:{thread_id}:*",
            f"{self._prefix}:writes:{thread_id}:*",
            f"{self._prefix}:index:{thread_id}:*",
        ]

        total_deleted = 0

        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=DELETE_BATCH_SIZE)
                if keys:
                    await client.delete(*keys)
                    total_deleted += len(keys)
                if cursor == 0:
                    break

        if total_deleted > 0:
            logger.info("Deleted %d checkpoint keys for thread %s", total_deleted, thread_id)
