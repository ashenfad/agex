"""
FastAPI application factory for remote agent execution.

This module provides create_app() for building the server application
and run_server() for convenient startup.
"""

import base64
import json
import traceback
from typing import Any

import cloudpickle
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agex.llm import LLMClient
from agex.remote.serialize import deserialize_agent
from agex.server.state import InvalidStateURIError, resolve_state_uri


class ExecuteRequest(BaseModel):
    """Request body for /execute endpoint."""

    agent_payload: str  # Base64-encoded cloudpickle bytes
    task_name: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    state_uri: str | None = None


def create_app(
    llm_client: LLMClient | None = None,
    state_base_path: str = "/var/agex/state",
) -> FastAPI:
    """
    Create a FastAPI application for remote agent execution.

    Args:
        llm_client: Optional LLM client to inject into deserialized agents.
            If not provided, agents will use their serialized config to connect.
        state_base_path: Base directory for disk:// state URIs.

    Returns:
        A configured FastAPI application.

    Example:
        from agex import connect_llm
        from agex.server import create_app

        app = create_app(llm_client=connect_llm())
        # Run with: uvicorn myapp:app
    """
    app = FastAPI(
        title="Agex Remote Execution Server",
        description="Execute agex agent tasks remotely",
        version="0.1.0",
    )

    @app.post("/execute")
    async def execute(request: ExecuteRequest):
        """
        Execute an agent task and stream results via SSE.

        The agent is deserialized from the payload, the task is located,
        and execution proceeds with events streamed back to the client.
        """

        async def event_generator():
            try:
                # 1. Deserialize agent
                agent_bytes = base64.b64decode(request.agent_payload)
                agent = deserialize_agent(agent_bytes, llm_client=llm_client)

                # 2. Resolve state if provided
                state = None
                if request.state_uri:
                    try:
                        state = resolve_state_uri(request.state_uri, state_base_path)
                    except InvalidStateURIError as e:
                        yield _format_error_event(str(e))
                        return

                # 3. Find the task on the agent
                task_func = _find_task(agent, request.task_name)
                if task_func is None:
                    yield _format_error_event(
                        f"Task '{request.task_name}' not found on agent"
                    )
                    return

                # 4. Set up event handlers that stream to client
                def on_token(token_chunk):
                    # We can't yield from here directly, so we'll use a different approach
                    # For now, we'll collect and send after completion
                    pass

                def on_event(event):
                    pass

                # 5. Execute the task
                # Note: Event streaming during execution requires a more complex
                # architecture (queue + async generator). For v1, we run to completion
                # and stream the result.
                try:
                    kwargs = dict(request.kwargs)
                    if state is not None:
                        kwargs["state"] = state

                    result = task_func(*request.args, **kwargs)

                    # Handle async tasks
                    import inspect

                    if inspect.isawaitable(result):
                        result = await result

                    # Stream result
                    yield _format_result_event(result)

                except Exception as e:
                    yield _format_error_event(str(e), traceback.format_exc())

            except Exception as e:
                yield _format_error_event(
                    f"Deserialization error: {e}", traceback.format_exc()
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app


def _find_task(agent, task_name: str):
    """
    Find a task function on the agent by name.

    Tasks are callable attributes that were created by @agent.task.
    """
    # Check if the agent has the task as a direct attribute or method
    # Tasks are typically accessed via the pattern used in the original code
    # We need to search through registered functions
    main_ns = agent._policy.namespaces.get("__main__")
    if main_ns and task_name in main_ns.fn_objects:
        return main_ns.fn_objects[task_name]

    return None


def _format_result_event(result: Any) -> str:
    """Format a result as an SSE event."""
    payload = base64.b64encode(cloudpickle.dumps(result)).decode("utf-8")
    data = json.dumps({"type": "result", "payload": payload})
    return f"data: {data}\n\n"


def _format_error_event(message: str, tb: str | None = None) -> str:
    """Format an error as an SSE event."""
    data = json.dumps({"type": "error", "message": message, "traceback": tb})
    return f"data: {data}\n\n"


def _format_token_event(token_chunk: Any) -> str:
    """Format a token chunk as an SSE event."""
    payload = base64.b64encode(cloudpickle.dumps(token_chunk)).decode("utf-8")
    data = json.dumps({"type": "token", "payload": payload})
    return f"data: {data}\n\n"


def _format_event(event: Any) -> str:
    """Format an agent event as an SSE event."""
    payload = base64.b64encode(cloudpickle.dumps(event)).decode("utf-8")
    data = json.dumps({"type": "event", "payload": payload})
    return f"data: {data}\n\n"


def run_server(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
    **uvicorn_kwargs,
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
    import uvicorn

    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
