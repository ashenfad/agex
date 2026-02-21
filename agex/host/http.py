"""
HTTP host implementation.

Executes agent tasks on a remote HTTP server.
"""

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

try:
    import cloudpickle
except ImportError:
    cloudpickle = None  # type: ignore
import httpx
from gitkv import Store

from .base import Host

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.state.config import StateConfig


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
    data: dict[str, Any] | str | None  # str when __set_type__, dict for JSON data


class HTTP(Host):
    """
    HTTP remote execution host.

    Sends agent tasks to a remote HTTP server for execution.
    Supports SSE streaming for real-time event and token callbacks.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 300.0,
        retries: int = 0,
        _http_client: Any | None = None,
    ):
        if cloudpickle is None:
            raise ImportError(
                "HTTP host requires 'cloudpickle'. Install it with: pip install agex[http]"
            )

        """
        Initialize the HTTP host.

        Args:
            url: The remote server URL (e.g., "https://compute.example.com/execute")
            timeout: Client-side HTTP timeout in seconds
            retries: Number of connection retries for network failures only
            _http_client: Test hook for injecting a custom HTTP client
        """
        self._validate_url(url)
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._http_client = _http_client

    def dump_config(self) -> dict[str, Any]:
        """Serialize HTTP host configuration."""
        return {
            "provider": "http",
            "url": self.url,
            "timeout": self.timeout,
            "retries": self.retries,
        }

    @staticmethod
    def _validate_url(url: str) -> None:
        """Validate that url looks like a proper HTTP(S) URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid URL '{url}': must start with 'http://' or 'https://'"
            )
        if not parsed.netloc:
            raise ValueError(
                f"Invalid URL '{url}': missing host (e.g., 'example.com:8000')"
            )

    def validate_state(self, config: "StateConfig | None") -> None:
        """Validate that the state config is compatible with HTTP execution."""
        if config is None:
            return  # Ephemeral is always valid

        # HTTP host only supports disk storage (server resolves it)
        if config.storage not in (None, "disk"):
            raise ValueError(
                f"HTTP host does not support storage '{config.storage}'. "
                f"Supported: disk (server resolves state)"
            )

    def resolve_state(self, config: "StateConfig | None", session: str) -> Store:
        """
        Placeholder for HTTP host - state is resolved server-side.

        This should not be called directly; the server handles state resolution.
        """
        raise NotImplementedError(
            "HTTP host resolves state server-side. "
            "This method should not be called directly."
        )

    def execute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute the task on the remote server synchronously."""
        payload = self._build_payload(agent, task_name, args, kwargs, session)

        # Use injected client if provided (for testing)
        if self._http_client is not None:
            return self._execute_with_client(
                self._http_client, payload, on_token, on_event
            )

        # Normal path: create httpx client
        transport = httpx.HTTPTransport(retries=self.retries)
        with httpx.Client(transport=transport, timeout=self.timeout) as client:
            return self._execute_with_client(client, payload, on_token, on_event)

    async def aexecute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute the task on the remote server asynchronously."""
        payload = self._build_payload(agent, task_name, args, kwargs, session)

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

    def _build_payload(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
    ) -> dict:
        """Build the JSON request payload."""
        from .serialize import serialize_agent

        agent_payload = serialize_agent(agent)

        # Include state config if present
        state_config = None
        if agent._state_config is not None:
            state_config = agent._state_config.dump_config()

        # Serialize args and kwargs with cloudpickle to support non-JSON types
        # (dataclasses, custom objects, etc.)
        args_payload = cloudpickle.dumps(args)
        kwargs_payload = cloudpickle.dumps(kwargs)

        return {
            "agent_payload": base64.b64encode(agent_payload).decode("utf-8"),
            "task_name": task_name,
            "args": base64.b64encode(args_payload).decode("utf-8"),
            "kwargs": base64.b64encode(kwargs_payload).decode("utf-8"),
            "session": session,
            "state_config": state_config,
        }

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
            # For streaming responses, read body before accessing .text
            try:
                error_text = e.response.text
            except httpx.ResponseNotRead:
                error_text = "(response body not available)"
            raise RemoteExecutionError(
                f"HTTP Error {e.response.status_code}: {error_text}"
            ) from e
        except httpx.RequestError as e:
            raise RemoteExecutionError(f"Connection error: {e}") from e

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
            # For streaming responses, read body before accessing .text
            try:
                error_text = e.response.text
            except httpx.ResponseNotRead:
                error_text = "(response body not available)"
            raise RemoteExecutionError(
                f"HTTP Error {e.response.status_code}: {error_text}"
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
        current_event_type: str | None = None

        for line in text.split("\n"):
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                # event.data is a string when __set_type__
                current_event_type = str(event.data) if event.data else None
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

    def _process_sync_stream(
        self,
        response,
        on_token: Callable | None,
        on_event: Callable | None,
    ) -> Any:
        """Process SSE stream synchronously."""
        current_event_type: str | None = None

        for line in response.iter_lines():
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                # event.data is a string when __set_type__
                current_event_type = str(event.data) if event.data else None
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
        current_event_type: str | None = None

        async for line in response.aiter_lines():
            event = self._parse_sse_line(line, current_event_type)
            if event.event_type == "__set_type__":
                # event.data is a string when __set_type__
                current_event_type = str(event.data) if event.data else None
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

        # Only dict data has payload - skip string/None
        if not isinstance(event_data, dict):
            return None

        event_payload_b64 = event_data.get("payload")

        if event_type == "token":
            if on_token and event_payload_b64:
                token_chunk = cloudpickle.loads(base64.b64decode(event_payload_b64))
                on_token(token_chunk)

        elif event_type == "event":
            if on_event and event_payload_b64:
                base_event = cloudpickle.loads(base64.b64decode(event_payload_b64))
                on_event(base_event)

        elif event_type == "result":
            if event_payload_b64:
                return cloudpickle.loads(base64.b64decode(event_payload_b64))

        elif event_type == "error":
            msg = event_data.get("message", "Unknown remote error")
            tb = event_data.get("traceback")
            raise RemoteExecutionError(msg, remote_traceback=tb)

        return None
