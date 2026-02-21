"""
Remote task execution runner.

This module provides the single entry point for executing serialized agent tasks.
Any remote host backend (HTTP server, Modal function, Beam function) should use
these functions to run tasks on the server side.

The runner handles:
- Agent deserialization with Local host override
- LLM client rehydration from config
- State resolution from config + session
- Task execution with callbacks
"""

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.state.config import StateConfig


def prepare_agent(payload: bytes) -> "BaseAgent":
    """
    Deserialize and prepare an agent for remote execution.

    This handles all the setup needed to run an agent on a remote host:
    - Deserializes from cloudpickle bytes
    - Rehydrates LLM client from serialized config
    - Overrides host to Local (so nested calls don't bounce back over network)
    - Re-registers agent in the new process's global registry

    Args:
        payload: Cloudpickle bytes from serialize_agent()

    Returns:
        A fully prepared Agent ready to execute tasks

    Raises:
        ValueError: If payload is not a valid Agent
        RuntimeError: If LLM cannot be reconstructed
        ImportError: If cloudpickle is not available
    """
    from agex.host.serialize import _get_cloudpickle, _is_agent_like

    cloudpickle = _get_cloudpickle()
    agent = cloudpickle.loads(payload)

    # Validate it's an agent
    if not _is_agent_like(agent):
        raise ValueError(f"Deserialized object is not an Agent: {type(agent)}")

    # Rehydrate LLM from serialized config
    if hasattr(agent, "_llm_config") and agent._llm_config:
        try:
            from agex.llm import LLM

            agent.llm = LLM.from_config(agent._llm_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to reconstruct LLM from config: {e}. "
                f"Ensure the server has the appropriate API keys set."
            ) from e
    else:
        raise RuntimeError(
            "Agent has no LLM configuration. Ensure the agent has an "
            "llm set before serialization."
        )

    # Force Local host for server-side execution
    # When running on a remote server, all task execution must stay local
    # to prevent infinite network bouncing (agent -> server -> agent -> server...)
    # If sub-agents need to run on different hosts, they should be configured
    # with their own host and called directly, not through hierarchical execution.
    from agex.host.local import Local

    agent._host = Local()

    # Fingerprint recomputes lazily on first access

    return agent


def execute_task(
    agent: "BaseAgent",
    task_name: str,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    session: str = "default",
    state_config: "StateConfig | None" = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
) -> Any:
    """
    Execute a task on a prepared agent.

    Args:
        agent: A prepared agent from prepare_agent()
        task_name: Name of the task to execute
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        session: Session ID for state isolation
        state_config: Optional state configuration
        on_event: Callback for agent events
        on_token: Callback for LLM tokens

    Returns:
        The task result

    Raises:
        KeyError: If task_name not found on agent
    """
    if kwargs is None:
        kwargs = {}

    # Find the task
    task_func = agent._tasks.get(task_name)
    if task_func is None:
        raise KeyError(f"Task '{task_name}' not found on agent '{agent.name}'")

    # Build execution kwargs - session is passed via kwargs for state resolution
    exec_kwargs = dict(kwargs)
    exec_kwargs["session"] = session
    if on_event is not None:
        exec_kwargs["on_event"] = on_event
    if on_token is not None:
        exec_kwargs["on_token"] = on_token

    # Execute
    return task_func(*args, **exec_kwargs)


async def aexecute_task(
    agent: "BaseAgent",
    task_name: str,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    session: str = "default",
    state_config: "StateConfig | None" = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
) -> Any:
    """
    Execute an async task on a prepared agent.

    Same as execute_task but for async tasks.
    """
    if kwargs is None:
        kwargs = {}

    # Find the task
    task_func = agent._tasks.get(task_name)
    if task_func is None:
        raise KeyError(f"Task '{task_name}' not found on agent '{agent.name}'")

    # Build execution kwargs - session is passed via kwargs for state resolution
    exec_kwargs = dict(kwargs)
    exec_kwargs["session"] = session
    if on_event is not None:
        exec_kwargs["on_event"] = on_event
    if on_token is not None:
        exec_kwargs["on_token"] = on_token

    # Execute
    return await task_func(*args, **exec_kwargs)


# Convenience function that combines prepare + execute
def run_remote_task(
    agent_payload: bytes,
    task_name: str,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    session: str = "default",
    state_config: "StateConfig | None" = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
) -> Any:
    """
    The single entry point for executing a serialized agent task.

    This is what any remote host backend (HTTP, Modal, Beam) should call
    to run a task on the server side. It handles everything:

    1. Deserializes the agent from cloudpickle bytes
    2. Rehydrates the LLM client from config
    3. Forces Local host for nested calls
    4. Resolves state from config + session
    5. Executes the task with callbacks

    Args:
        agent_payload: Cloudpickle bytes from serialize_agent()
        task_name: Name of the task to execute
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        session: Session ID for state isolation
        state_config: Optional state configuration
        on_event: Callback for agent events
        on_token: Callback for LLM tokens

    Returns:
        The task result

    Example (Modal function):
        @modal.function()
        def run_agent_task(payload: bytes, task_name: str, **kwargs):
            return run_remote_task(payload, task_name, **kwargs)

    Example (Beam function):
        @beam.function()
        def run_agent_task(payload: bytes, task_name: str, **kwargs):
            return run_remote_task(payload, task_name, **kwargs)
    """
    agent = prepare_agent(agent_payload)
    return execute_task(
        agent=agent,
        task_name=task_name,
        args=args,
        kwargs=kwargs,
        session=session,
        state_config=state_config,
        on_event=on_event,
        on_token=on_token,
    )


async def arun_remote_task(
    agent_payload: bytes,
    task_name: str,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    session: str = "default",
    state_config: "StateConfig | None" = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
) -> Any:
    """
    Async version of run_remote_task for async tasks.
    """
    agent = prepare_agent(agent_payload)
    return await aexecute_task(
        agent=agent,
        task_name=task_name,
        args=args,
        kwargs=kwargs,
        session=session,
        state_config=state_config,
        on_event=on_event,
        on_token=on_token,
    )
