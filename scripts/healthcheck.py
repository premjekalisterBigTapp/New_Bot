#!/usr/bin/env python3
"""
Health Check Script for Agentic Chatbot
=======================================

Verifies all required services are running and configured correctly.

Usage:
    python -m bigtapp.agentic.scripts.healthcheck

Or directly:
    python healthcheck.py
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


def check_redis() -> bool:
    """Check Redis connectivity."""
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("[OK] Redis: Connected to %s", redis_url)
        return True
    except ImportError:
        logger.error("[FAIL] Redis: redis package not installed")
        return False
    except Exception as e:
        logger.error("[FAIL] Redis: %s", e)
        return False


def check_mongodb() -> bool:
    """Check MongoDB connectivity."""
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("DB_NAME", "bigtapp")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        logger.info("[OK] MongoDB: Connected to %s/%s", mongo_uri, db_name)
        return True
    except ImportError:
        logger.error("[FAIL] MongoDB: pymongo package not installed")
        return False
    except Exception as e:
        logger.error("[FAIL] MongoDB: %s", e)
        return False


def check_weaviate() -> bool:
    """Check Weaviate connectivity."""
    try:
        import weaviate
        from urllib.parse import urlparse
        
        weaviate_url = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT") or "http://localhost:8080"
        parsed = urlparse(weaviate_url)
        
        # Simple HTTP health check
        import urllib.request
        health_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 8080}/v1/.well-known/ready"
        urllib.request.urlopen(health_url, timeout=5)
        
        logger.info("[OK] Weaviate: Available at %s", weaviate_url)
        return True
    except ImportError:
        logger.warning("[WARN] Weaviate: weaviate package not installed (RAG disabled)")
        return True  # Not critical
    except Exception as e:
        logger.warning("[WARN] Weaviate: %s (RAG may be disabled)", e)
        return True  # Not critical


def check_azure_openai() -> bool:
    """Check Azure OpenAI configuration."""
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    ]
    
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        logger.error("[FAIL] Azure OpenAI: Missing env vars: %s", missing)
        return False
    
    # Try to initialize
    try:
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            max_retries=1,
        )
        # Quick test
        response = llm.invoke("Say 'OK' if you can hear me.")
        if response and response.content:
            logger.info("[OK] Azure OpenAI: Connected to %s", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))
            return True
    except Exception as e:
        logger.error("[FAIL] Azure OpenAI: %s", e)
        return False
    
    return False


def check_whatsapp_config() -> bool:
    """Check WhatsApp/Meta configuration."""
    required = [
        "META_ACCESS_TOKEN",
        "META_PHONE_NUMBER_ID",
    ]
    
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        logger.warning("[WARN] WhatsApp: Missing env vars: %s (WhatsApp disabled)", missing)
        return True  # Not critical
    
    logger.info("[OK] WhatsApp: Configured")
    return True


def main():
    logger.info("=" * 60)
    logger.info("BigTapp Agentic Chatbot - Health Check")
    logger.info("=" * 60)
    
    checks = [
        ("Redis", check_redis),
        ("MongoDB", check_mongodb),
        ("Weaviate", check_weaviate),
        ("Azure OpenAI", check_azure_openai),
        ("WhatsApp", check_whatsapp_config),
    ]
    
    results = {}
    for name, check_fn in checks:
        logger.info("")
        logger.info("Checking %s...", name)
        results[name] = check_fn()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)
    
    all_passed = True
    critical_failed = False
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", name, status)
        if not passed:
            all_passed = False
            if name in ["Redis", "Azure OpenAI"]:
                critical_failed = True
    
    logger.info("")
    
    if critical_failed:
        logger.error("Critical services failed! Please fix before running.")
        sys.exit(1)
    elif all_passed:
        logger.info("All checks passed! Ready to run.")
        sys.exit(0)
    else:
        logger.warning("Some optional services unavailable. Limited functionality.")
        sys.exit(0)


if __name__ == "__main__":
    main()
