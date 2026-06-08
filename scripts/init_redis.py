#!/usr/bin/env python3
"""
Redis Initialization Script for Agentic Chatbot
================================================

This script verifies Redis connectivity and clears any stale
agentic session/checkpoint data if needed.

Usage:
    python -m bigtapp.agentic.scripts.init_redis

Or directly:
    python init_redis.py
"""

import os
import sys
import logging

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from pathlib import Path

# Explicitly load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:
    logger.error("redis is required. Install with: pip install redis")
    sys.exit(1)


def init_redis(clear_data: bool = False):
    """Initialize and verify Redis connection for the agentic chatbot."""
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    logger.info("Connecting to Redis: %s", redis_url)
    
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("Successfully connected to Redis")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        sys.exit(1)
    
    # Get Redis info
    info = client.info()
    logger.info("Redis version: %s", info.get("redis_version"))
    logger.info("Connected clients: %d", info.get("connected_clients", 0))
    logger.info("Used memory: %s", info.get("used_memory_human", "N/A"))
    
    # Count agentic keys
    agentic_keys = list(client.scan_iter("agentic:*"))
    logger.info("Existing agentic keys: %d", len(agentic_keys))
    
    if agentic_keys:
        # Group by prefix
        prefixes = {}
        for key in agentic_keys:
            prefix = key.split(":")[1] if ":" in key else "other"
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        logger.info("Key breakdown:")
        for prefix, count in sorted(prefixes.items()):
            logger.info("  - agentic:%s: %d keys", prefix, count)
    
    if clear_data:
        if agentic_keys:
            logger.warning("Clearing %d agentic keys...", len(agentic_keys))
            for key in agentic_keys:
                client.delete(key)
            logger.info("All agentic keys cleared")
        else:
            logger.info("No agentic keys to clear")
    
    # Verify write access
    test_key = "agentic:_init_test_"
    client.set(test_key, "test_value", ex=10)
    value = client.get(test_key)
    assert value == "test_value", "Failed to verify write access"
    client.delete(test_key)
    logger.info("Write access verified")
    
    logger.info("=" * 50)
    logger.info("Redis initialization completed successfully!")
    logger.info("URL: %s", redis_url)
    logger.info("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize Redis for agentic chatbot")
    parser.add_argument("--clear", action="store_true", help="Clear all agentic data")
    args = parser.parse_args()
    
    init_redis(clear_data=args.clear)


if __name__ == "__main__":
    main()
