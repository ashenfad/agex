"""
FastAPI application factory for remote agent execution.

This module provides create_app() for building the server application
and run_server() for convenient startup. It serves as a reference
implementation for hosting agex agents remotely.

Modern FastAPI patterns used:
- Lifespan context manager for configuration
- Dependency injection for shared resources
- sse-starlette for proper SSE handling
- Pydantic models for request validation
- Type hints throughout
"""

import asyncio
import base64
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agex.host import Local, prepare_agent
from agex.server.helpers import (
    execute_worker,
    format_error_data,
    format_event_data,
    format_result_data,
    format_token_data,
)
from agex.state.config import StateConfig

# =============================================================================
# Request/Response Models
# =============================================================================


class ExecuteRequest(BaseModel):
    """Request body for /execute endpoint."""

    agent_payload: str  # Base64-encoded cloudpickle bytes
    task_name: str
    args: str  # Base64-encoded cloudpickled tuple
    kwargs: str  # Base64-encoded cloudpickled dict
    session: str = "default"
    state_config: dict[str, Any] | None = None  # Serialized StateConfig


# =============================================================================
# Application State (Dependency Injection)
# =============================================================================


@dataclass
class AppState:
    """Shared application state, injected into endpoints via Depends()."""

    state_dir: str


def get_app_state(request: Request) -> AppState:
    """Dependency that retrieves app state from the request context."""
    return request.app.state.app_state


# =============================================================================
# SSE Event Generator
# =============================================================================


async def generate_execution_events(
    request: ExecuteRequest,
    app_state: AppState,
) -> AsyncGenerator[dict[str, str], None]:
    """
    Execute an agent task and yield SSE events.

    This is an async generator that:
    1. Deserializes the agent from the payload
    2. Resolves the state URI if provided
    3. Executes the task in a thread pool
    4. Streams events back as they occur

    Yields:
        SSE event dicts with 'data' key for EventSourceResponse
    """
    try:
        # 1. Prepare agent (deserialize, rehydrate LLM, force Local host)
        import cloudpickle

        agent_bytes = base64.b64decode(request.agent_payload)
        agent = prepare_agent(agent_bytes)

        # Deserialize args and kwargs
        args = cloudpickle.loads(base64.b64decode(request.args))
        kwargs = cloudpickle.loads(base64.b64decode(request.kwargs))

        # 2. Resolve state from config + session using Local host
        local_host = Local()
        config = None
        if request.state_config:
            config = StateConfig.from_config(request.state_config)

        state = local_host.resolve_state(config, request.session)

        # 3. Find the task on the agent
        task_func = agent._tasks.get(request.task_name)
        if task_func is None:
            yield {
                "data": format_error_data(
                    f"Task '{request.task_name}' not found on agent"
                )
            }
            return

        # 4. Create queue for thread-to-async bridging
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        FINISHED = object()  # Sentinel for completion

        # 5. Check if task is async
        is_async_task = getattr(task_func, "__agex_is_async__", False)

        if is_async_task:
            # For async tasks, run directly in the event loop
            async def async_execute():
                try:
                    exec_kwargs = dict(kwargs)

                    def on_token(token_chunk):
                        queue.put_nowait(("token", token_chunk))

                    def on_event(event):
                        queue.put_nowait(event)

                    exec_kwargs["on_event"] = on_event
                    exec_kwargs["on_token"] = on_token

                    result = await task_func(*args, **exec_kwargs)
                    queue.put_nowait((FINISHED, result))
                except Exception as e:
                    queue.put_nowait((FINISHED, e))

            # Start async task
            asyncio.create_task(async_execute())
        else:
            # For sync tasks, use thread pool
            loop.run_in_executor(
                None,
                execute_worker,
                task_func,
                args,
                kwargs,
                state,
                queue,
                loop,
                FINISHED,
            )

        # 6. Stream events from queue
        async for event_data in _stream_from_queue(queue, FINISHED):
            yield {"data": event_data}

    except Exception as e:
        yield {
            "data": format_error_data(
                f"Execution error: {e}",
                traceback.format_exc(),
            )
        }


async def _stream_from_queue(
    queue: asyncio.Queue[Any],
    finished_sentinel: object,
) -> AsyncGenerator[str, None]:
    """
    Stream formatted SSE data from the execution queue.

    Handles three types of queue items:
    - (finished_sentinel, result): Task completed
    - ("token", token_chunk): Token from LLM
    - event: Agent event object
    """
    import inspect

    while True:
        item = await queue.get()

        # Check for completion
        if isinstance(item, tuple) and item[0] is finished_sentinel:
            result = item[1]
            if isinstance(result, Exception):
                raise result

            # Handle async results (if task returned coroutine)
            if inspect.isawaitable(result):
                result = await result

            yield format_result_data(result)
            break

        # Handle tokens
        elif isinstance(item, tuple) and item[0] == "token":
            yield format_token_data(item[1])

        # Handle normal events
        else:
            yield format_event_data(item)


# =============================================================================
# Application Factory
# =============================================================================


def create_app(
    state_dir: str = "/var/agex/state",
) -> FastAPI:
    """
    Create a FastAPI application for remote agent execution.

    The server reconstructs LLM clients from the serialized agent configuration.
    Ensure the server environment has the appropriate API keys set (e.g.,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).

    Args:
        state_dir: Base directory for disk:// state URIs.

    Returns:
        A configured FastAPI application.

    Example:
        from agex.server import create_app

        app = create_app(state_dir="/data/agex/sessions")
        # Run with: uvicorn myapp:app
    """
    # Create shared state
    app_state = AppState(
        state_dir=state_dir,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for app startup/shutdown."""
        yield
        # Cleanup would go here if needed

    app = FastAPI(
        title="Agex Remote Execution Server",
        description="Execute agex agent tasks remotely via SSE streaming",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store state on app for dependency injection
    app.state.app_state = app_state

    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------

    @app.post("/execute")
    async def execute(
        request: ExecuteRequest,
        app_state: AppState = Depends(get_app_state),
    ) -> EventSourceResponse:
        """
        Execute an agent task and stream results via Server-Sent Events.

        The agent is deserialized from the payload, the task is located,
        and execution proceeds with events streamed back to the client.

        Returns:
            EventSourceResponse with streaming execution events
        """
        return EventSourceResponse(
            generate_execution_events(request, app_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


# =============================================================================
# Server Runner
# =============================================================================


def run_server(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
    **uvicorn_kwargs: Any,
) -> None:
    """
    Run the server using uvicorn.

    Args:
        app: The FastAPI application
        host: Host to bind to
        port: Port to bind to
        **uvicorn_kwargs: Additional arguments passed to uvicorn.run()

    Example:
        from agex.server import create_app, run_server

        app = create_app()
        run_server(app, host="0.0.0.0", port=8000)
    """
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
