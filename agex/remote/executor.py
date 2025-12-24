"""
Remote task execution logic.

This module provides the core execution machinery for remote agent tasks,
separating concerns from the decorator itself.
"""

import base64
import json
from dataclasses import dataclass
from typing import Any, Callable

import cloudpickle
import httpx

from agex.agent.base import BaseAgent
from agex.remote.serialize import serialize_agent


class RemoteExecutionError(Exception):
    """Raised when the remote task execution fails."""

    def __init__(self, message: str, remote_traceback: str | None = None):
        super().__init__(message)
        self.remote_traceback = remote_traceback

    def __str__(self) -> str:
        """Include remote traceback in string representation for debugging."""
        base = super().__str__()
        if self.remote_traceback:
            return f"{base}\n\nRemote Traceback:\n{self.remote_traceback}"
        return base


class RemoteTimeoutError(TimeoutError):
    """Raised when the remote execution times out."""

    pass


@dataclass
class SSEEvent:
    """Parsed SSE event."""

    event_type: str | None
    data: dict | None


class RemoteTaskExecutor:
    """
    Handles the execution of agent tasks on a remote server.

    This class encapsulates:
    - Agent serialization
    - HTTP transport (sync and async)
    - SSE stream parsing
    - Callback invocation
    """

    def __init__(
        self,
        agent: BaseAgent,
        task_name: str,
        url: str,
        default_state: str | None = None,
        timeout: float = 300.0,
        retries: int = 0,
        _http_client: Any | None = None,
    ):
        self.agent = agent
        self.task_name = task_name
        self.url = url
        self.default_state = default_state
        self.timeout = timeout
        self.retries = retries
        # Test hook: allows injecting a custom HTTP client (e.g., TestClient)
        self._http_client = _http_client

    def execute_sync(
        self,
        args: tuple,
        kwargs: dict,
        state_uri: str | None,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Execute the task synchronously."""
        payload = self._build_payload(args, kwargs, state_uri)

        # Use injected client if provided (for testing)
        if self._http_client is not None:
            return self._execute_with_client(
                self._http_client, payload, on_token, on_event
            )

        # Normal path: create httpx client
        transport = httpx.HTTPTransport(retries=self.retries)
        with httpx.Client(transport=transport, timeout=self.timeout) as client:
            return self._execute_with_client(client, payload, on_token, on_event)

    def _execute_with_client(
        self,
        client,
        payload: dict,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Execute using the provided HTTP client."""
        try:
            # Check if client supports streaming (httpx) vs request (TestClient)
            if hasattr(client, "stream"):
                with client.stream(
                    "POST",
                    self.url,
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    return self._process_sync_stream(response, on_token, on_event)
            else:
                # TestClient uses .post() with stream=True in response
                response = client.post(
                    "/execute",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                )
                response.raise_for_status()
                # TestClient returns response.text directly, parse it
                return self._process_text_response(response.text, on_token, on_event)

        except httpx.TimeoutException as e:
            raise RemoteTimeoutError(
                f"Remote execution timed out after {self.timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RemoteExecutionError(
                f"HTTP Error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RemoteExecutionError(f"Connection error: {e}") from e

    def _process_text_response(
        self,
        text: str,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Process SSE response from text (for TestClient)."""
        current_event_type = None

        for line in text.split("\n"):
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                current_event_type = event.data
                continue
            if event.event_type == "__reset__":
                current_event_type = None
                continue
            if event.data is None:
                continue

            result = self._handle_event(event, on_token, on_event)
            if result is not None:
                return result

        raise RemoteExecutionError("Connection closed without returning a result.")

    async def execute_async(
        self,
        args: tuple,
        kwargs: dict,
        state_uri: str | None,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Execute the task asynchronously."""
        payload = self._build_payload(args, kwargs, state_uri)

        # Use injected client if provided (for testing)
        if self._http_client is not None:
            return await self._execute_async_with_client(
                self._http_client, payload, on_token, on_event
            )

        transport = httpx.AsyncHTTPTransport(retries=self.retries)
        async with httpx.AsyncClient(
            transport=transport, timeout=self.timeout
        ) as client:
            return await self._execute_async_with_client(
                client, payload, on_token, on_event
            )

    async def _execute_async_with_client(
        self,
        client,
        payload: dict,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Execute using the provided Async HTTP client."""
        try:
            async with client.stream(
                "POST",
                self.url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                return await self._process_async_stream(response, on_token, on_event)

        except httpx.TimeoutException as e:
            raise RemoteTimeoutError(
                f"Remote execution timed out after {self.timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RemoteExecutionError(
                f"HTTP Error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RemoteExecutionError(f"Connection error: {e}") from e

    def _build_payload(self, args: tuple, kwargs: dict, state_uri: str | None) -> dict:
        """Build the JSON request payload."""
        agent_payload = serialize_agent(self.agent)
        return {
            "agent_payload": base64.b64encode(agent_payload).decode("utf-8"),
            "task_name": self.task_name,
            "args": args,
            "kwargs": kwargs,
            "state_uri": state_uri or self.default_state,
        }

    def _process_sync_stream(
        self,
        response,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Process SSE stream synchronously."""
        current_event_type = None

        for line in response.iter_lines():
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                current_event_type = event.data
                continue
            if event.event_type == "__reset__":
                current_event_type = None
                continue
            if event.data is None:
                continue

            result = self._handle_event(event, on_token, on_event)
            if result is not None:
                return result

        raise RemoteExecutionError("Connection closed without returning a result.")

    async def _process_async_stream(
        self,
        response,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Process SSE stream asynchronously."""
        current_event_type = None

        async for line in response.aiter_lines():
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                current_event_type = event.data
                continue
            if event.event_type == "__reset__":
                current_event_type = None
                continue
            if event.data is None:
                continue

            result = self._handle_event(event, on_token, on_event)
            if result is not None:
                return result

        raise RemoteExecutionError("Connection closed without returning a result.")

    def _parse_sse_line(self, line: str, current_event_type: str | None) -> SSEEvent:
        """Parse a single SSE line."""
        line = line.strip()

        if not line:
            # Empty line resets event type per SSE spec
            return SSEEvent(event_type="__reset__", data=None)

        if line.startswith("event: "):
            # Track event type for next data line
            return SSEEvent(event_type="__set_type__", data=line[7:])

        if line.startswith("data: "):
            data_str = line[6:]
            try:
                event_data = json.loads(data_str)
            except json.JSONDecodeError:
                return SSEEvent(event_type=None, data=None)

            # Use explicit type from JSON, fall back to SSE event field
            event_type = event_data.get("type") or current_event_type
            return SSEEvent(event_type=event_type, data=event_data)

        return SSEEvent(event_type=None, data=None)

    def _handle_event(
        self,
        event: SSEEvent,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any | None:
        """
        Handle a parsed SSE event.

        Returns the final result if this is a result event, otherwise None.
        Raises RemoteExecutionError for error events.
        """
        event_type = event.event_type
        event_data = event.data
        event_payload_b64 = event_data.get("payload") if event_data else None

        if event_type == "token":
            if on_token and event_payload_b64:
                token_chunk = cloudpickle.loads(base64.b64decode(event_payload_b64))
                on_token(token_chunk)

        elif event_type == "event":
            if on_event and event_payload_b64:
                base_event = cloudpickle.loads(base64.b64decode(event_payload_b64))
                on_event(base_event)

        elif event_type == "result":
            return cloudpickle.loads(base64.b64decode(event_payload_b64))

        elif event_type == "error":
            msg = event_data.get("message", "Unknown remote error")
            tb = event_data.get("traceback")
            raise RemoteExecutionError(msg, remote_traceback=tb)

        return None
