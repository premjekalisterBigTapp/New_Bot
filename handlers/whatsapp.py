"""
Production-Grade WhatsApp Handler for Agentic Chatbot
=====================================================

Standalone WhatsApp handler that uses the agentic infrastructure
(Redis session, MongoDB history, LangGraph with Redis checkpointer).
Includes full Zoom live agent handoff support.
"""

import os
import re
import logging
import uuid
import hmac
import hashlib
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from functools import partial
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import io

from fastapi import Request, Response
import httpx

try:
    from zoneinfo import ZoneInfo
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    from datetime import timezone, timedelta
    SGT_TZ = timezone(timedelta(hours=8))

from ..infrastructure import (
    SessionManager,
    AsyncRateLimiter,
    AsyncDeduplicator,
    AsyncOrderGuard,
    WA_MESSAGES_PROCESSED_TOTAL,
    WA_MESSAGE_LATENCY,
    LIVE_AGENT_HANDOFFS,
)
from .. import agentic_chat

# Import Zoom engagement manager (local integration)
try:
    from ..integrations.zoom.engagement import EngagementManager
    ZOOM_AVAILABLE = True
except ImportError:
    EngagementManager = None
    ZOOM_AVAILABLE = False

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("ZOOM_BASE_URL", "https://us01cciapi.zoom.us")

# Greeting message
GREETING_MESSAGE = (
    "Hi. I'm your assistant for today. "
    "Here to guide you through BigTapp insurance products and services, "
    "answer your questions instantly, and make things easier for you. How can I help you today?"
)

# Farewell message
FAREWELL_MESSAGE = (
    "Thank you for chatting with BigTapp! 😊 "
    "It was a pleasure assisting you. "
    "If you ever need help with your insurance, don't hesitate to reach out. "
    "Have a wonderful day! 👋"
)

# Words that trigger a farewell response (not a session reset)
_FAREWELL_KEYWORDS = frozenset({
    "bye", "goodbye", "good bye", "byebye", "bye bye",
    "cya", "see you", "see ya", "take care", "farewell",
    "thanks bye", "thank you bye", "ok bye", "okay bye",
    "tq bye", "ok tq", "ok thank you",
})


class AgenticWhatsAppHandler:
    """
    Production-grade WhatsApp handler for the agentic chatbot.
    
    Features:
    - Redis-backed session management
    - MongoDB conversation history persistence
    - Rate limiting, deduplication, ordering
    - Zoom live agent handoff support
    - Robust error handling with retries
    """
    
    def __init__(self):
        self.verify_token = os.environ.get("META_VERIFY_TOKEN")
        self.access_token = os.environ.get("META_ACCESS_TOKEN")
        self.phone_number_id = os.environ.get("META_PHONE_NUMBER_ID")
        self.max_message_length = 4096

        # Async HTTP client with connection pooling
        self._http: Optional[httpx.AsyncClient] = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
        )

        # Redis-backed controls
        self.rate_limiter = AsyncRateLimiter(scope="agentic_wa")
        self.deduper = AsyncDeduplicator(scope="agentic_wa")
        self.order_guard = AsyncOrderGuard(scope="agentic_wa")
        
        # Session manager
        self._session_manager = SessionManager()
        
        logger.info("AgenticWhatsAppHandler initialized")

    def verify_webhook(self, request: Request) -> Response:
        """Verify webhook subscription with Meta."""
        try:
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            challenge = request.query_params.get("hub.challenge")
            
            if not all([mode, token, challenge]):
                logger.warning("Missing webhook verification parameters")
                return Response(content="Missing parameters", status_code=400)
            
            if mode == "subscribe" and token == self.verify_token:
                logger.info("Webhook verification successful")
                return Response(content=challenge, status_code=200)
            
            logger.warning("Webhook verification failed")
            return Response(content="Verification failed", status_code=403)
            
        except Exception as e:
            logger.error(f"Webhook verification error: {e}")
            return Response(content="Internal error", status_code=500)

    def extract_message_data(self, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Extract message and user info from WhatsApp webhook data."""
        try:
            value = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
            
            # Skip status updates
            if 'statuses' in value:
                return None, None, {}

            # Extract message
            try:
                message = value['messages'][0]['text']['body']
                user_phone = value['messages'][0]['from']
            except (KeyError, IndexError):
                return None, None, {}
            
            # Extract metadata
            metadata = {}
            try:
                msg_data = value['messages'][0]
                metadata = {
                    'message_id': msg_data.get('id'),
                    'timestamp': msg_data.get('timestamp'),
                    'type': msg_data.get('type', 'text'),
                    'from_name': value.get('contacts', [{}])[0].get('profile', {}).get('name', 'Unknown')
                }
            except Exception:
                pass
            
            # Clean message
            message = re.sub(r'\s+', ' ', message.strip())
            if len(message) > self.max_message_length:
                message = message[:self.max_message_length]
            
            # Validate phone
            user_phone = re.sub(r'[^\d+]', '', user_phone)
            if len(user_phone) < 8 or len(user_phone) > 15:
                return None, None, {}
            
            return message, user_phone, metadata
            
        except Exception as e:
            logger.error(f"Error extracting message: {e}")
            return None, None, {}

    async def handle_message(self, message: str, user_phone: str, metadata: Dict[str, Any]) -> str:
        """Process message through the agentic chat system."""
        start_time = time.perf_counter()

        try:
            logger.info("AgenticWA: Processing from %s, msg_len=%d", user_phone, len(message))
            
            session_id = f"whatsapp_{user_phone}"
            
            # NOTE: Greeting reset ("hi", "hello", "hey") is now handled in 
            # _process_and_respond() BEFORE lock acquisition to prevent nested lock deadlock.
            
            # Check if live agent is active
            if await self._session_manager.is_live_agent_active(session_id):
                logger.info(f"AgenticWA: Live agent active for {session_id}")
                return "You're currently connected to a live agent. Please continue your conversation with them, or say 'hi' to start a new chat with the bot."

            # Process through agentic chat with WhatsApp channel context
            # This enables channel-specific behavior in middleware (e.g., shorter responses)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = await agentic_chat(
                    session_id,
                    message,
                    channel="whatsapp",
                    channel_user_id=user_phone,
                )
            
            response = str(result.get("response") or "")
            debug_state = result.get("debug_state", {})
            
            # Check if live agent handoff was requested
            live_agent_requested = debug_state.get("live_agent_requested", False)
            
            if live_agent_requested:
                await self._session_manager.set_live_agent_status(session_id, True)
                LIVE_AGENT_HANDOFFS.inc()
                logger.info(f"AgenticWA: Live agent requested for {session_id}")
            
            # Log conversation history
            await self._session_manager.add_history_entry(
                session_id,
                message,
                response,
                metadata={
                    "product": debug_state.get("product"),
                    "intent": debug_state.get("intent"),
                    "live_agent_requested": live_agent_requested,
                },
            )

            if not response:
                response = "I'm sorry, I couldn't process your request. Please try again."
            
            # Truncate if needed
            if len(response) > self.max_message_length:
                response = response[:self.max_message_length - 50] + "...\n\n(Message truncated)"
            
            logger.info(f"AgenticWA: Response for {user_phone}: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"AgenticWA: Error processing message from {user_phone}: {e}")
            return "I'm sorry, there was an error processing your message. Please try again later."
        finally:
            duration = time.perf_counter() - start_time
            try:
                WA_MESSAGE_LATENCY.observe(duration)
            except Exception:
                pass

    async def _send_message_async(self, recipient: str, message: str) -> None:
        """Send WhatsApp message with retries."""
        if not self.phone_number_id or not self.access_token:
            logger.error("META_PHONE_NUMBER_ID or META_ACCESS_TOKEN not configured")
            return

        url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": message}
        }

        for attempt in range(3):
            try:
                response = await self._http.post(url, headers=headers, json=payload)
                response.raise_for_status()
                logger.info(f"Message sent to {recipient}")
                return
            except httpx.HTTPError as e:
                logger.warning(f"Send attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        
        logger.error(f"Failed to send message to {recipient} after 3 attempts")

    async def _send_image_with_caption_async(self, recipient: str, image_url: str, caption: str) -> None:
        """Send WhatsApp image with caption using a publicly accessible URL."""
        if not self.phone_number_id or not self.access_token:
            logger.error("META_PHONE_NUMBER_ID or META_ACCESS_TOKEN not configured")
            return

        url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "image",
            "image": {
                "link": image_url,
                "caption": caption
            }
        }

        for attempt in range(3):
            try:
                response = await self._http.post(url, headers=headers, json=payload)
                response.raise_for_status()
                logger.info(f"Image sent to {recipient}")
                return
            except httpx.HTTPError as e:
                logger.warning(f"Image send attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        
        logger.error(f"Failed to send image to {recipient} after 3 attempts")

    async def _process_and_respond(self, message: str, user_phone: str, metadata: Dict[str, Any]) -> None:
        """Process message and send response."""
        # Rate limiting
        if not await self.rate_limiter.allow(user_phone):
            await self._send_message_async(user_phone, "You're sending messages too quickly. Please wait a moment.")
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="rate_limited").inc()
            return

        session_id = f"whatsapp_{user_phone}"
        
        # Handle greeting reset BEFORE acquiring lock to avoid nested lock deadlock
        # reset_session() also acquires a lock on the same key, causing deadlock if nested
        if message.lower().strip() in ("hi", "hello", "hey"):
            logger.info(f"AgenticWA: Received greeting - resetting session {session_id}")
            await self._session_manager.reset_session(session_id)
            
            # Clear LangGraph checkpointer
            try:
                from ..graph import _get_checkpointer
                checkpointer = _get_checkpointer()
                if hasattr(checkpointer, "adelete_thread"):
                    await checkpointer.adelete_thread(session_id)
                    logger.info(f"AgenticWA: Cleared LangGraph history for {session_id}")
                elif hasattr(checkpointer, "delete_thread"):
                    await asyncio.to_thread(checkpointer.delete_thread, session_id)
                    logger.info(f"AgenticWA: Cleared LangGraph history for {session_id}")
            except Exception as e:
                logger.warning(f"AgenticWA: Failed to clear checkpointer: {e}")
            
            # Clear PII session mapping
            try:
                from ..utils.pii_masker import get_pii_masker
                get_pii_masker().clear_session(session_id)
                logger.info(f"AgenticWA: Cleared PII session mapping for {session_id}")
            except Exception as e:
                logger.warning(f"AgenticWA: Failed to clear PII session mapping: {e}")
            
            await self._send_message_async(user_phone, GREETING_MESSAGE)
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="greeting_reset").inc()
            return

        # Handle farewell — send goodbye message without resetting the session
        if message.lower().strip() in _FAREWELL_KEYWORDS:
            logger.info(f"AgenticWA: Received farewell from {session_id}")
            await self._send_message_async(user_phone, FAREWELL_MESSAGE)
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="farewell").inc()
            return
        
        try:
            # NOTE: We removed the outer AsyncRedisLock here because session methods
            # (add_history_entry, set_live_agent_status, update_field, etc.) have their own
            # internal locking. Having an outer lock caused nested lock deadlocks.
            
            # Check if live agent is active BEFORE processing
            if await self._session_manager.is_live_agent_active(session_id):
                if ZOOM_AVAILABLE and EngagementManager:
                    manager = EngagementManager.get_by_session(user_phone)
                    if manager:
                        if manager.is_agent_connected:
                            await manager.send_message(message)
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_forwarded").inc()
                        else:
                            await self._send_message_async(user_phone, "You're in a queue. Please wait while we connect you to an agent.")
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_queue").inc()
                    else:
                        # Manager not found - reset status
                        await self._session_manager.set_live_agent_status(session_id, False)
                        await self._send_message_async(user_phone, "The agent has disconnected. Please say 'hi' to restart.")
                        WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_error").inc()
                else:
                    await self._send_message_async(user_phone, "Live agent support is currently unavailable. Please say 'hi' to continue with the bot.")
                    WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_unavailable").inc()
                return

            # Process message through agentic bot
            response = await self.handle_message(message, user_phone, metadata)
            
            if response and response.strip():
                # Send all responses (including purchase links) as plain text.
                # The image-with-caption path was removed because the placeholder
                # logo URL caused purchase replies to fail/hang on WhatsApp.
                await self._send_message_async(user_phone, response)
                WA_MESSAGES_PROCESSED_TOTAL.labels(result="ok").inc()
            else:
                WA_MESSAGES_PROCESSED_TOTAL.labels(result="empty").inc()

            # Check if live agent was requested AFTER processing
            if await self._session_manager.is_live_agent_active(session_id):
                if ZOOM_AVAILABLE and EngagementManager:
                    if user_phone not in EngagementManager._active_engagements:
                        logger.info(f"AgenticWA: Creating Zoom engagement for {user_phone}")
                        callback = partial(self.handle_agent_response, user_phone)
                        
                        temp_name = uuid.uuid4().hex[:6]
                        temp_email = f"{temp_name}@bigtapptest.com"

                        manager = EngagementManager.create_and_register(
                            session_id=user_phone,
                            nick_name=temp_name,
                            email=temp_email,
                            base_api_url=BASE_URL,
                            on_agent_message_callback=callback
                        )
                        asyncio.create_task(manager.initiate_engagement())

        except Exception as e:
            logger.error(f"AgenticWA: Error processing message from {user_phone}: {e}")
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="error").inc()

    async def handle_agent_response(self, user_phone: str, message) -> None:
        """Callback for agent messages from Zoom."""
        logger.info(f"AgenticWA: Agent message for {user_phone}: {message}")
        
        # Forward to customer
        if isinstance(message, str):
            await self._send_message_async(user_phone, message)
        elif isinstance(message, dict):
            text = message.get("text") or message.get("message") or str(message)
            await self._send_message_async(user_phone, text)
        
        session_id = f"whatsapp_{user_phone}"
        
        # Check for chat closed
        if isinstance(message, str) and message == "This chat has been closed.":
            logger.info(f"AgenticWA: Agent closed chat for {user_phone}")
            await self._session_manager.set_live_agent_status(session_id, False)
            await self.close_engagement_and_cleanup(user_phone)
        
        # Check for consumer_disconnected
        if isinstance(message, dict) and message.get("event") == "consumer_disconnected":
            logger.info(f"AgenticWA: Agent ended chat for {user_phone}")
            await self._session_manager.set_live_agent_status(session_id, False)
            await self.close_engagement_and_cleanup(user_phone)

    async def close_engagement_and_cleanup(self, session_id: str) -> None:
        """Close and cleanup Zoom engagement."""
        if not ZOOM_AVAILABLE or not EngagementManager:
            return
        manager = EngagementManager.get_by_session(session_id)
        if manager:
            manager.unregister(session_id)
            await manager.close()
            logger.info(f"AgenticWA: Cleaned up Zoom engagement for {session_id}")

    async def process_webhook(self, request: Request) -> Response:
        """Main webhook processing - acknowledge immediately, process in background."""
        try:
            # Verify signature
            raw_body = await request.body()
            app_secret = os.environ.get("META_APP_SECRET")
            sig_header = request.headers.get("X-Hub-Signature-256")
            
            if app_secret:
                if not sig_header or not sig_header.startswith("sha256="):
                    logger.error("Webhook signature missing")
                    return Response(status_code=403)
                expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
                provided = sig_header.split("=", 1)[1]
                if not hmac.compare_digest(expected, provided):
                    logger.error("Webhook signature mismatch")
                    return Response(status_code=403)

            data = await request.json()
            message, user_phone, metadata = self.extract_message_data(data)
            
            if message and user_phone:
                # Deduplication
                message_id = metadata.get('message_id', '')
                if message_id and not await self.deduper.is_new(message_id):
                    logger.info(f"Duplicate message: {message_id}")
                    WA_MESSAGES_PROCESSED_TOTAL.labels(result="duplicate").inc()
                    return Response(status_code=200)
                
                # Ordering
                try:
                    ts = int(metadata.get('timestamp', time.time()))
                except Exception:
                    ts = int(time.time())
                if not await self.order_guard.allow(user_phone, ts):
                    logger.info(f"Out-of-order message for {user_phone}")
                    WA_MESSAGES_PROCESSED_TOTAL.labels(result="out_of_order").inc()
                    return Response(status_code=200)

                # Process in background
                asyncio.create_task(self._process_and_respond(message, user_phone, metadata))
            
            return Response(status_code=200)
            
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="error").inc()
            return Response(status_code=200)

    def get_health_status(self) -> Dict[str, Any]:
        """Health check status."""
        return {
            "status": "healthy",
            "handler": "agentic",
            "timestamp": datetime.now(SGT_TZ).isoformat(),
            "verify_token_configured": bool(self.verify_token),
            "zoom_available": ZOOM_AVAILABLE,
        }


# Global handler instance
agentic_whatsapp_handler = AgenticWhatsAppHandler()


# Convenience functions for FastAPI routes
async def handle_agentic_whatsapp_verification(request: Request) -> Response:
    """Handle WhatsApp webhook verification."""
    return agentic_whatsapp_handler.verify_webhook(request)


async def handle_agentic_whatsapp_message(request: Request) -> Response:
    """Handle incoming WhatsApp messages."""
    return await agentic_whatsapp_handler.process_webhook(request)


async def close_agentic_whatsapp_client():
    """Close HTTP client on shutdown."""
    try:
        if agentic_whatsapp_handler._http:
            await agentic_whatsapp_handler._http.aclose()
            agentic_whatsapp_handler._http = None
            logger.info("Closed agentic WhatsApp HTTP client")
    except Exception as e:
        logger.error(f"Failed to close HTTP client: {e}")
