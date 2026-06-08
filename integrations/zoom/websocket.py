import asyncio
import json
import logging
import uuid
from typing import Callable, Awaitable
import time

import websockets

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages the WebSocket connection lifecycle for a single Zoom user session.

    This class handles connecting, sending the initial authorization message,
    listening for incoming messages, sending periodic pings to keep the
    connection alive, and gracefully disconnecting.
    """

    def __init__(self, zm_aid: str, on_message_callback: Callable[[str], Awaitable[None]]):
        """
        Initializes the WebSocketManager.

        Args:
            zm_aid (str): The Zoom a_id required for the WebSocket URL.
            on_message_callback (Callable): An async function that will be called
                                            when a message is received from the WebSocket.
        """
        self.zm_aid = zm_aid
        self.websocket_url = f"wss://zpns.zoom.us/ws?zm_aid={self.zm_aid}"
        self.on_message_callback = on_message_callback

        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.is_connected = False
        self._listener_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None

    async def connect(self, jwt_token: str, user_id: str, resource: str) -> None:
        """
        Establishes and authorizes the WebSocket connection.

        This method connects to the server, sends the initial login payload,
        and starts background tasks for listening to messages and sending pings.

        Args:
            jwt_token (str): The JWT token obtained from the Zoom auth API.
            user_id (str): The user identifier for the login payload.
            resource (str): The resource string for the login payload.
        """
        if self.is_connected:
            logger.warning("WebSocket is already connected.")
            return

        try:
            logger.info(f"Connecting to WebSocket at {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            self.is_connected = True
            logger.info("WebSocket connection established successfully.")

            # Authorize the connection by sending the login payload
            await self._send_login_payload(jwt_token, user_id, resource)

            # Start concurrent tasks for listening and pinging
            self._listener_task = asyncio.create_task(self._listen_for_messages())
            self._ping_task = asyncio.create_task(self._send_pings())

        except (websockets.exceptions.InvalidURI, websockets.exceptions.WebSocketException) as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            self.websocket = None
            # Re-raise or handle the exception as needed by the orchestrator
            raise

    async def _send_login_payload(self, jwt_token: str, user_id: str, resource: str):
        """Constructs and sends the initial login message."""
        login_payload = {
            "type": "login",
            "authtype": "jwt",
            "id": str(uuid.uuid4()),
            "token": jwt_token,
            "user": user_id,
            "option": 192,
            "resource": resource
        }
        logger.info(f"Sending login payload...")
        await self.send_json(login_payload)

    # Need to make this handling more robust in case there is a failure in forwarding message
    # after the push-ack is received from callback.
    async def _listen_for_messages(self):
        """Listens for incoming messages and passes them to the callback."""
        logger.info("Starting message listener...")
        try:
            async for message in self.websocket:
                await self.on_message_callback(message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e.code} {e.reason}")

        except asyncio.CancelledError:
            logger.warning("Task was cancelled explicitly")
            raise

        except Exception as e:
            logger.exception(f"Unexpected error in listener: {e}")
            
        finally:
            logger.info("Message listener stopped.")
            # self.is_connected = False


    async def _send_pings(self):
        """Sends a ping every 30 seconds to keep the connection alive."""
        logger.info("Starting ping task...")
        while self.is_connected:
            try:
                timestamp = int(time.time() * 1000)
                ping_message = f"ping {timestamp}"
                
                logger.debug(f"Sending application ping: {ping_message}")
                if self.websocket:
                    await self.websocket.send(ping_message)
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Ping task cancelled.")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Could not send ping, connection is closed.")
                break
        logger.info("Ping task stopped.")


    async def send_json(self, data: dict):
        """Sends a Python dictionary as a JSON string over the WebSocket."""
        if not self.is_connected or not self.websocket:
            logger.error("Cannot send message, WebSocket is not connected.")
            return
        
        try:
            await self.websocket.send(json.dumps(data))
            logger.info(f"Sent message to ws: {data}")
        except websockets.exceptions.ConnectionClosed:
            logger.error("Failed to send message, connection closed.")
            self.is_connected = False


    async def close(self):
        """Gracefully closes the WebSocket connection and cleans up tasks."""
        if not self.is_connected:
            logger.info("Connection is already closed.")
            return

        logger.info("Closing WebSocket connection...")
        self.is_connected = False # Signal tasks to stop

        if self._ping_task:
            self._ping_task.cancel()
        
        if self.websocket:
            await self.websocket.close(code=1000, reason="Client initiated disconnect")
            self.websocket = None

        if self._listener_task:
            if asyncio.current_task() == self._listener_task:
                self._listener_task.cancel()
            else:
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass
            self._listener_task = None

        logger.info("WebSocket connection and all tasks have been closed.")