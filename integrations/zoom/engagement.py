from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Callable, Awaitable
import httpx
from pydantic import BaseModel, Field
from .websocket import WebSocketManager

logger = logging.getLogger(__name__)


class AuthResponse(BaseModel):
    token: str
    user_id: str = Field(..., alias="userId")
    resource: str


class IncomingResponse(BaseModel):
    engagement_id: str = Field(..., alias="engagementId")
    zm_aid: str = Field(..., alias="zmAid")


class EngagementManager:
    """
    Manages the entire lifecycle of a single customer support engagement.
    """

    _active_engagements: Dict[str, EngagementManager] = {}

    def __init__(
        self,
        base_api_url: str,
        nick_name: str,
        email: str,
        on_agent_message_callback: Callable[[dict | str], Awaitable[None]],
    ):
        """
        Initializes the EngagementManager for a new conversation.

        Args:
            base_api_url (str): The base URL for the Zoom Contact Center APIs.
            on_agent_message_callback (Callable): An async callback function to send
                                                   messages/events back to the primary orchestrator.
        """
        self.base_api_url = base_api_url
        self.zoom_api_key = os.environ.get("ZOOM_API_KEY")
        self.on_agent_message_callback = on_agent_message_callback

        self.http_client = httpx.AsyncClient()
        self.ws_manager: WebSocketManager | None = None

        # State for the engagement
        self.auth_token: str | None = None
        self.user_id: str | None = None
        self.user_email: str = email
        self.user_name: str = nick_name
        self.resource: str | None = None
        self.engagement_id: str | None = None
        self.session_id: str | None = None
        self.chat_session_id: str | None = None
        self.zm_aid: str | None = None
        self.is_agent_connected = False

    @classmethod
    def create_and_register(
        cls, 
        session_id: str, 
        nick_name: str, 
        email: str, 
        base_api_url: str, 
        on_agent_message_callback: Callable
    ) -> EngagementManager:
        """A factory method to create, register, and return a new engagement manager."""
        if session_id in cls._active_engagements:
            return cls._active_engagements[session_id]
            
        new_manager = cls(
            nick_name=nick_name,
            email=email,
            base_api_url=base_api_url,
            on_agent_message_callback=on_agent_message_callback
        )
        cls._active_engagements[session_id] = new_manager
        return new_manager

    @classmethod
    def get_by_session(cls, session_id: str) -> EngagementManager | None:
        """Fetches an active engagement manager by its session ID."""
        return cls._active_engagements.get(session_id)

    @classmethod
    def unregister(cls, session_id: str):
        """Removes an engagement manager from the registry when it's done."""
        if session_id in cls._active_engagements:
            del cls._active_engagements[session_id]
            logger.info(f"Unregistered engagement session: {session_id}")

    async def _handle_websocket_message(self, message: dict | str):
        """Internal callback to process messages from the WebSocketManager."""

        if isinstance(message, str) and (message.startswith("ping") or message.startswith("pong")):
            logger.info(f"Received heartbeat from server: {message}")
            return

        try:
            data = json.loads(message)
            logger.info(f"Received structured message from WebSocket: {data}")

        # try:
            if data.get("type") == "push" and "id" in data:
                ack_msg = {
                    "id": data["id"],
                    "type": "push-ack"
                }
                await self.ws_manager.send_json(ack_msg)
                logger.info(f"Sent push-ack for id={data['id']}")

            message_name = data.get("name")

            # 1. Ignore typing indicators and participant info immediately
            if message_name in ["/chat/typingIndicator", "cci/participant/info"]:
                asyncio.create_task(self.user_ready_ack())
                # asyncio.create_task(self.populate_chat_session_id())
                logger.info(f"Ignoring event: {message_name}")
                return

            # 2. Process the main '/chat/message' events
            if message_name == "/chat/message":
                event_str = data.get("event")
                if not event_str:
                    logger.warning(
                        "'/chat/message' received without 'event' field. No message to forward to whatsapp handler. Ignoring..."
                    )
                    return

                # The 'event' field is a JSON string, so it needs to be decoded
                event_data = json.loads(event_str)
                event_name = event_data.get("eventName")
                msg_type = event_data.get("type")
                if(event_data.get("chatSessionId") and not self.chat_session_id):
                    self.chat_session_id = event_data.get("chatSessionId")

                # 1. Agent sends a message
                if event_name == "send_msg" and msg_type == "1":
                    try:
                        # The 'text' field is another layer of JSON string
                        text_payload_str = event_data.get("text", "{}")
                        text_payload = json.loads(text_payload_str)
                        # Extract the actual message from the nested structure
                        text = text_payload.get("ops", [{}])[0].get("insert", "").strip()
                        if text:
                            await self.on_agent_message_callback(text)
                    except (json.JSONDecodeError, IndexError, KeyError) as e:
                        logger.error(f"Error parsing agent message text content: {e}")

                # 2. Agent joins the chat
                elif event_name == "send_msg" and msg_type == "2":
                    logger.info("Agent has connected to the engagement.")
                    self.is_agent_connected = True
                    # asyncio.create_task(self.populate_chat_session_id())
                    text = event_data.get("text")
                    if text:
                        await self.on_agent_message_callback(text)
                    asyncio.create_task(self.user_ready_ack())

                # 3. Chat has ended
                elif event_name == "chat_ended":
                    await self.on_agent_message_callback("This chat has been closed.")

                else:
                    logger.debug(
                        f"Received unhandled '/chat/message' with eventName: '{event_name}' and type: '{msg_type}'"
                    )
            else:
                # logger.debug(f"Received unhandled event type: {event}")
                logger.debug(f"Received unhandled message name: {message_name}")

        except json.JSONDecodeError:
            logger.warning(
                f"Received a non-JSON event string that was not handled: {message}"
            )

    async def get_auth_token(self) -> bool:
        """Calls the auth endpoint to retrieve a JWT token for the session."""
        url = f"{self.base_api_url}/v1/auth/token/generate/in/visitor/mode"
        payload = {
            "customerId": None,
            "customerName": self.user_name,
            "email": self.user_email,
            "apiKey": self.zoom_api_key,
        }
        try:
            logger.info("Requesting auth token...")
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()

            resp_json = response.json()
            auth_data = {
                "result": {
                    "token": resp_json.get("result", {}).get("token"),
                    "customer": {
                        "id": resp_json.get("result", {}).get("customer", {}).get("id")
                    },
                    "zmAid": resp_json.get("result", {}).get("zmAid"),
                    "zpnsJwtToken": resp_json.get("result", {}).get("zpnsJwtToken"),
                }
            }

            self.auth_token = auth_data["result"]["token"]
            self.user_id = auth_data["result"]["customer"]["id"]
            self.resource = os.environ.get("ZOOM_SDK_RESOURCE")
            self.zm_aid = auth_data["result"]["zmAid"]
            self.zpns_jwt_token = auth_data["result"]["zpnsJwtToken"]

            logger.info("Successfully retrieved auth token")
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error getting auth token: {e}")
            return False

    async def initiate_engagement(self) -> bool:
        """
        Orchestrates the full engagement initiation process:
        1. Gets an auth token.
        2. Calls the 'incoming' API to get engagement details.
        3. Establishes the WebSocket connection.
        """
        if not await self.get_auth_token():
            return False

        url = f"{self.base_api_url}/v1/livechat/customer/incoming"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {
            "entryPointType": "entryId",
            "entryPoint": self.zoom_api_key,
            "preChatSurveyCustomerInfo": {
                "nickName": self.user_name or "Guest User",
                "email": self.user_email,
            },
            "featureOption": "7",
            "customerBrowserInfo": {
                "deviceDetailInDTO": {
                    "deviceId": "",
                    "deviceName": "",
                    "deviceType": "",
                    "deviceModel": "",
                    "os": "Windows 10",
                    "browser": "Chrome",
                    "browserVersion": "139",
                    "mobileAppName": "",
                    "source": "Web",
                },
                "referrer": "",
                "engagementStartPage": "",
                "engagementStartPageTitle": "",
            },
            "consumerWebsiteData": [],
        }
        try:
            logger.info("Initiating engagement...")

            # Establish the WebSocket connection
            self.ws_manager = WebSocketManager(
                zm_aid=self.zm_aid, on_message_callback=self._handle_websocket_message
            )
            await self.ws_manager.connect(
                self.zpns_jwt_token, self.user_id, self.resource
            )

            # Initiate the engagement to a Zoom agent via API upon successful WS connection
            response = await self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            incoming_data = response.json()
            self.engagement_id = incoming_data.get("result", {}).get("engagementId")
            self.session_id = incoming_data.get("result", {}).get("sessionId")
            asyncio.create_task(self.user_ready_ack())
            # asyncio.create_task(self.populate_chat_session_id())
            logger.info(f"Engagement created with ID: {self.engagement_id}")

            return True

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error initiating engagement: {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during WebSocket connection: {e}"
            )
            return False

    async def user_ready_ack(self):
        """
        Sends the 'connected' acknowledgement after the user successfully joins ws channel.
        This enables the ws channel to receive the agent messages.
        """
        if not self.engagement_id or not self.auth_token or not self.session_id:
            logger.error("Cannot send connection ack: missing request body parameters.")
            return

        url = f"{self.base_api_url}/v1/livechat/customer/connected"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {
            "engagementId": self.engagement_id,
            "sessionId": self.session_id,
            "protocol": "ZPNS",
        }
        try:
            logger.info(
                "Sending user connection ack. Subscribing to agent messages in ws channel..."
            )
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info("Connection acknowledgement sent successfully.")
            return True

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error sending connection acknowledgement: {e}")

    async def populate_chat_session_id(self) -> None:
        """
        Fetches the engagement info and extracts the chat_session_id.\
        This is needed to send a message to the chat.
        """
        if not self.engagement_id or not self.auth_token:
            logger.error("Cannot request engagement data: missing request body parameters.")
            return

        url = f"{self.base_api_url}/v1/engagement/content/sdk/history/timestamp/engagement"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        params = {
            "group": self.engagement_id,
        }
        try:
            logger.info(
                "Requesting engagement data to extract chatSessionId..."
            )
            response = await self.http_client.get(url, params=params,headers=headers)
            response.raise_for_status()
            data = response.json()
            result_string = data.get("result")

            if not isinstance(result_string, str) or not result_string.strip():
                logger.error(
                    "'result' key is missing, not a string, or contains only whitespace."
                )
                return

            # Safely attempt to decode the string
            try:
                nested_data = json.loads(result_string)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode JSON from 'result' string. Content: '{result_string}'"
                )
                return

            if not nested_data.get("chatSessionId"):
                logger.error("'chatSessionId' not found inside the 'result' data.")
                return None

            self.chat_session_id = nested_data.get("chatSessionId")
            logger.info(f"Extracted chatSessionId: {self.chat_session_id}")
            return None

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error sending connection acknowledgement: {e}")

    async def send_message(self, text: str):
        """Sends a message from the customer to the Zoom agent."""
        if not self.engagement_id or not self.auth_token:
            logger.error("Cannot send message: missing engagement_id or auth_token.")
            return

        if not self.is_agent_connected:
            logger.warning(
                "Attempted to send message before agent was connected. Message not sent."
            )
            return

        url = f"{self.base_api_url}/v1/livechat/message/send"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {
            "source": "WEB_CHAT",
            "text": text,
            "originalText": text,
            "messageTempId": f"temp_{uuid.uuid4().hex[:20]}",
            "uniqueId": f"temp_{uuid.uuid4().hex[:20]}",
            "chatSessionId": self.chat_session_id,
            "engagementId": self.engagement_id,
            "messageSeqId": int(time.time() * 1000),
            "messagingSeqInfo": {"instanceId": uuid.uuid4().hex[:20], "seqId": 1},
            "pushSelf": True,
            "featureOption": "7",
        }
        try:
            logger.info(f"Sending message to agent: '{text}'")
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info("Message sent successfully.")

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error sending message: {e}")

    async def format_and_send_chat_history(self, chat_history: list[dict]):
        formatted_chat_history = []
        for message in chat_history:
            sender = message.get("sender", "Unknown")
            text = message.get("text", "")
            formatted_text = f"[{sender}]: {text}"
            formatted_chat_history.append({"text": formatted_text})

        for message in formatted_chat_history:
            await self.send_message(message["text"])

    async def close(self):
        """Gracefully closes all connections."""
        logger.info("Closing engagement manager...")
        if self.ws_manager:
            await self.ws_manager.close()
        await self.http_client.aclose()
        logger.info("Engagement manager closed.")
