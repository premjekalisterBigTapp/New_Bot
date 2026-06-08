"""
Production-grade background logging for MongoDB conversation history.

Production Features:
- Bounded async queue to prevent memory issues under load
- Graceful shutdown with reliable queue drain
- Automatic retry with exponential backoff
- Prometheus metrics for observability
- Thread pool offloading for sync MongoDB operations
- UTC timestamps (float) for consistency

Environment Variables:
    BG_LOGGER_QUEUE_SIZE: Max queue size (default: 1000)
    BG_LOGGER_DRAIN_TIMEOUT: Shutdown drain timeout (default: 10.0)
    BG_LOGGER_MAX_RETRIES: Max retry attempts (default: 3)
    BG_LOGGER_RETRY_DELAY: Base retry delay seconds (default: 0.1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .mongo_history import log_history

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
DEFAULT_QUEUE_SIZE = int(os.getenv("BG_LOGGER_QUEUE_SIZE", "1000"))
DEFAULT_DRAIN_TIMEOUT = float(os.getenv("BG_LOGGER_DRAIN_TIMEOUT", "10.0"))
DEFAULT_MAX_RETRIES = int(os.getenv("BG_LOGGER_MAX_RETRIES", "3"))
DEFAULT_RETRY_DELAY = float(os.getenv("BG_LOGGER_RETRY_DELAY", "0.1"))
DEFAULT_OVERFLOW_CONCURRENCY = int(os.getenv("BG_LOGGER_OVERFLOW_CONCURRENCY", "5"))

# ============================================
# Prometheus Metrics
# ============================================
try:
    from prometheus_client import Counter, Gauge, Histogram
    
    BG_LOG_QUEUE_SIZE = Gauge(
        "agentic_bg_log_queue_size",
        "Current items in background log queue",
    )
    BG_LOG_ENQUEUED_TOTAL = Counter(
        "agentic_bg_log_enqueued_total",
        "Total conversation logs enqueued",
        ["status"],  # success, overflow, dropped
    )
    BG_LOG_PROCESSED_TOTAL = Counter(
        "agentic_bg_log_processed_total",
        "Total conversation logs processed",
        ["status"],  # success, error
    )
    BG_LOG_LATENCY = Histogram(
        "agentic_bg_log_latency_seconds",
        "Time from enqueue to successful write",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    BG_LOG_RETRY_TOTAL = Counter(
        "agentic_bg_log_retry_total",
        "Total retry attempts for failed log writes",
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


@dataclass
class LogEntry:
    """Immutable log entry for queue processing."""
    session_id: str
    user_message: str
    assistant_message: str
    metadata: Optional[Dict[str, Any]]
    timestamp: float  # Unix timestamp (time.time())
    enqueue_time: float  # time.perf_counter() for latency tracking
    retry_count: int = 0


class BackgroundLogger:
    """
    Production-grade background logger for MongoDB conversation history.
    
    Features:
    - Async queue-based processing (non-blocking)
    - Bounded queue with configurable size
    - Graceful shutdown with reliable drain
    - Automatic retry with exponential backoff
    - Thread pool offloading for sync MongoDB
    """
    
    def __init__(
        self,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_retry_delay: float = DEFAULT_RETRY_DELAY,
        overflow_concurrency: int = DEFAULT_OVERFLOW_CONCURRENCY,
    ):
        self._queue: asyncio.Queue[Optional[LogEntry]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._drain_timeout = drain_timeout
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._max_queue_size = max_queue_size
        self._overflow_limit = max(1, overflow_concurrency)
        self._overflow_semaphore = asyncio.Semaphore(self._overflow_limit)
        
        logger.info(
            "BackgroundLogger initialized: queue=%d, drain_timeout=%.1fs, retries=%d overflow=%d",
            max_queue_size,
            drain_timeout,
            max_retries,
            overflow_concurrency,
        )
    
    async def start(self) -> None:
        """Start the background worker task."""
        if self._running:
            logger.debug("BackgroundLogger already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="background_logger_worker",
        )
        logger.info("BackgroundLogger worker started")
    
    async def stop(self) -> None:
        """
        Stop the background worker gracefully.
        
        Drains pending logs up to drain_timeout seconds.
        """
        if not self._running:
            logger.debug("BackgroundLogger not running")
            return
        
        queue_size = self._queue.qsize()
        logger.info(
            "BackgroundLogger stopping: queue_size=%d, drain_timeout=%.1fs",
            queue_size,
            self._drain_timeout,
        )
        
        self._running = False
        
        # Signal worker to stop by sending None sentinel
        # Use wait-based put during shutdown (blocking is acceptable)
        try:
            await asyncio.wait_for(
                self._queue.put(None),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            logger.warning("BackgroundLogger: could not send stop signal, queue full")
        
        # Wait for worker to finish
        if self._worker_task:
            try:
                await asyncio.wait_for(
                    self._worker_task,
                    timeout=self._drain_timeout,
                )
                logger.info("BackgroundLogger worker stopped gracefully")
            except asyncio.TimeoutError:
                remaining = self._queue.qsize()
                logger.warning(
                    "BackgroundLogger drain timeout: %d entries may be lost",
                    remaining,
                )
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            finally:
                self._worker_task = None
    
    async def enqueue(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Enqueue a conversation log entry for background processing.
        
        Non-blocking. Falls back to sync logging if queue is full.
        
        Returns:
            True if enqueued successfully, False if fallback was used
        """
        now = time.time()
        
        entry = LogEntry(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata,
            timestamp=now,
            enqueue_time=time.perf_counter(),
        )
        
        try:
            self._queue.put_nowait(entry)
            
            if METRICS_AVAILABLE:
                BG_LOG_ENQUEUED_TOTAL.labels(status="success").inc()
                BG_LOG_QUEUE_SIZE.set(self._queue.qsize())
            
            logger.debug(
                "BackgroundLogger.enqueue: session=%s queue_size=%d",
                session_id,
                self._queue.qsize(),
            )
            return True
            
        except asyncio.QueueFull:
            # Backpressure: queue full, fall back to bounded background processing
            logger.warning(
                "BackgroundLogger queue full (%d), overflow fallback: session=%s",
                self._max_queue_size,
                session_id,
            )
            
            if METRICS_AVAILABLE:
                BG_LOG_ENQUEUED_TOTAL.labels(status="overflow").inc()

            if self._overflow_semaphore.locked():
                if METRICS_AVAILABLE:
                    BG_LOG_ENQUEUED_TOTAL.labels(status="dropped").inc()
                logger.error(
                    "BackgroundLogger overflow saturated (%d); dropping log: session=%s",
                    self._overflow_limit,
                    session_id,
                )
                return False

            async def _overflow_worker(entry: LogEntry) -> None:
                async with self._overflow_semaphore:
                    try:
                        await self._process_entry(entry)
                    except Exception as exc:
                        logger.error("BackgroundLogger.overflow_error: session=%s err=%s", entry.session_id, exc)

            asyncio.create_task(_overflow_worker(entry), name="background_logger_overflow")
            logger.debug("BackgroundLogger.overflow_enqueued: session=%s", session_id)
            return False
    
    async def _worker_loop(self) -> None:
        """Background worker that processes log entries from the queue."""
        logger.debug("BackgroundLogger worker loop starting")
        
        while self._running or not self._queue.empty():
            try:
                # Wait for entry with timeout to check _running periodically
                try:
                    entry = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                # None sentinel signals shutdown
                if entry is None:
                    self._queue.task_done()
                    if not self._running:
                        break
                    continue
                
                # Process the entry
                await self._process_entry(entry)
                self._queue.task_done()
                
                if METRICS_AVAILABLE:
                    BG_LOG_QUEUE_SIZE.set(self._queue.qsize())
                    
            except asyncio.CancelledError:
                logger.debug("BackgroundLogger worker cancelled")
                raise
            except Exception as e:
                logger.error("BackgroundLogger worker error: %s", str(e)[:200])
        
        logger.debug("BackgroundLogger worker loop exiting")
    
    async def _process_entry(self, entry: LogEntry) -> None:
        """Process a single log entry with retry logic."""
        while entry.retry_count <= self._max_retries:
            try:
                # Run sync MongoDB operation in thread pool
                await asyncio.to_thread(
                    log_history,
                    session_id=entry.session_id,
                    user_message=entry.user_message,
                    assistant_message=entry.assistant_message,
                    ts=entry.timestamp,
                    metadata=entry.metadata,
                )
                
                # Success
                if METRICS_AVAILABLE:
                    latency = time.perf_counter() - entry.enqueue_time
                    BG_LOG_LATENCY.observe(latency)
                    BG_LOG_PROCESSED_TOTAL.labels(status="success").inc()
                
                logger.debug(
                    "BackgroundLogger.processed: session=%s latency=%.3fs retries=%d",
                    entry.session_id,
                    time.perf_counter() - entry.enqueue_time,
                    entry.retry_count,
                )
                return
                
            except Exception as e:
                entry.retry_count += 1
                
                if entry.retry_count > self._max_retries:
                    # Max retries exceeded
                    logger.error(
                        "BackgroundLogger.max_retries: session=%s error=%s",
                        entry.session_id,
                        str(e)[:200],
                    )
                    if METRICS_AVAILABLE:
                        BG_LOG_PROCESSED_TOTAL.labels(status="error").inc()
                    return
                
                # Exponential backoff
                delay = self._base_retry_delay * (2 ** (entry.retry_count - 1))
                
                logger.warning(
                    "BackgroundLogger.retry: session=%s attempt=%d/%d delay=%.2fs",
                    entry.session_id,
                    entry.retry_count,
                    self._max_retries,
                    delay,
                )
                
                if METRICS_AVAILABLE:
                    BG_LOG_RETRY_TOTAL.inc()
                
                await asyncio.sleep(delay)
    
    @property
    def queue_size(self) -> int:
        """Current number of entries in the queue."""
        return self._queue.qsize()
    
    @property
    def is_running(self) -> bool:
        """Whether the background worker is running."""
        return self._running


# ============================================
# Module-level Singleton
# ============================================

_background_logger: Optional[BackgroundLogger] = None


def set_background_logger(bg_logger: BackgroundLogger) -> None:
    """Set the module-level background logger instance."""
    global _background_logger
    _background_logger = bg_logger
    logger.debug("Background logger reference set")


def get_background_logger() -> Optional[BackgroundLogger]:
    """Get the module-level background logger instance."""
    return _background_logger


async def enqueue_log(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enqueue a log entry for background processing.
    
    Uses the module-level BackgroundLogger set at startup.
    Falls back to sync logging if BackgroundLogger not available.
    """
    if _background_logger is None or not _background_logger.is_running:
        # BackgroundLogger not initialized, use sync fallback in thread
        logger.debug(
            "BackgroundLogger not available, sync fallback: session=%s",
            session_id,
        )
        async def _fallback() -> None:
            try:
                await asyncio.to_thread(
                    log_history,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    ts=time.time(),
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning("BackgroundLogger fallback failed: %s", exc)

        asyncio.create_task(_fallback(), name="background_logger_fallback")
        return
    
    await _background_logger.enqueue(
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        metadata=metadata,
    )
