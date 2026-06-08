"""Test script to verify idle monitor configuration and behavior."""
import sys
import asyncio
sys.path.insert(0, '..')

from dotenv import load_dotenv
load_dotenv()

from infrastructure.redis_utils import get_async_redis
from infrastructure.idle_monitor import (
    ENABLE_IDLE_FAREWELL,
    IDLE_FAREWELL_SECONDS,
    IDLE_MONITOR_POLL_SECONDS,
)

print("=" * 60)
print("IDLE MONITOR CONFIGURATION CHECK")
print("=" * 60)

print(f"\n1. Configuration Values:")
print(f"   ENABLE_IDLE_FAREWELL    = {ENABLE_IDLE_FAREWELL}")
print(f"   IDLE_FAREWELL_SECONDS   = {IDLE_FAREWELL_SECONDS}")
print(f"   IDLE_MONITOR_POLL_SECONDS = {IDLE_MONITOR_POLL_SECONDS}")

if not ENABLE_IDLE_FAREWELL:
    print("\n❌ ISSUE: Idle farewell is DISABLED!")
    print("   Set ENABLE_IDLE_FAREWELL=true in .env")

if IDLE_FAREWELL_SECONDS <= 0:
    print("\n❌ ISSUE: IDLE_FAREWELL_SECONDS must be > 0!")
    print("   Current value:", IDLE_FAREWELL_SECONDS)

print(f"\n2. Redis Connection Check:")


async def _check_redis() -> None:
    try:
        redis = await get_async_redis()
        print(f"   ✅ Redis connected: {await redis.ping()}")
        
        # Check for sessions
        keys = []
        async for key in redis.scan_iter(match="agentic:session:*", count=100):
            keys.append(key)
        print(f"\n3. Session Scan:")
        print(f"   Found {len(keys)} total sessions")
        
        whatsapp_sessions = [k for k in keys if b"whatsapp_" in k or "whatsapp_" in str(k)]
        print(f"   Found {len(whatsapp_sessions)} WhatsApp sessions")
        
        # Show sample sessions
        if whatsapp_sessions:
            print(f"\n4. Sample Session Data:")
            for key in whatsapp_sessions[:3]:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                data = await redis.get(key)
                if data:
                    import json
                    try:
                        session = json.loads(data)
                        last_active = session.get("last_active", "N/A")
                        idle_farewell_sent = session.get("idle_farewell_sent", False)
                        live_agent_status = session.get("live_agent_status", False)
                        print(f"\n   Session: {key}")
                        print(f"   - last_active: {last_active}")
                        print(f"   - idle_farewell_sent: {idle_farewell_sent}")
                        print(f"   - live_agent_status: {live_agent_status}")
                    except Exception as e:
                        print(f"   Error parsing session {key}: {e}")
        else:
            print("\n   ⚠️ No WhatsApp sessions found in Redis")
            print("   (This is expected if no users have chatted recently)")
            
    except Exception as e:
        print(f"   ❌ Redis error: {e}")


asyncio.run(_check_redis())

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
