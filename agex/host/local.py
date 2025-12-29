"""
Local host implementation.

Executes agent tasks in the current process using the agent's task loop.
"""

import os
from typing import TYPE_CHECKING, Any, Callable

from .base import Host

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.state import State
    from agex.state.config import StateConfig


class Local(Host):
    """
    Local execution host.

    Runs agent tasks in the current process. This is the default host
    and provides the same behavior as calling tasks without any host configuration.

    For memory storage, each agent has isolated state. For shared state
    across agents, use disk storage with a shared path.
    """

    def __init__(self):
        # Per-agent session cache for memory-backed states
        # Key format: "{type}:{session}"
        self._session_cache: dict[str, "State"] = {}

    def dump_config(self) -> dict[str, Any]:
        """Serialize local host configuration."""
        return {"provider": "local"}

    def validate_state(self, config: "StateConfig | None") -> None:
        """Validate that the state config is compatible with local execution."""
        if config is None:
            return  # Ephemeral is always valid

        if config.storage not in (None, "memory", "disk"):
            raise ValueError(
                f"Local host does not support storage '{config.storage}'. "
                f"Supported: memory, disk"
            )

        if config.storage == "disk" and not config.path:
            raise ValueError("Disk storage requires 'path' parameter")

    def resolve_state(
        self, config: "StateConfig | None", session: str, fingerprint: str = ""
    ) -> "State":
        """Create or retrieve a State instance for this session."""
        from agex.state import Live
        from agex.state.kv import Disk, Memory

        # Ephemeral: fresh Live instance per call
        if config is None:
            return Live()

        # For memory storage, use session cache
        if config.storage == "memory":
            cache_key = f"{config.type}:{session}"
            if cache_key not in self._session_cache:
                self._session_cache[cache_key] = self._create_state(config, Memory())
            return self._session_cache[cache_key]

        # For disk storage, create with session-namespaced path
        if config.storage == "disk":
            path = os.path.expanduser(config.path or "")
            session_path = os.path.join(path, "sessions", session)
            kv = Disk(session_path)
            return self._create_state(config, kv)

        # Fallback for unspecified storage (treat as memory)
        cache_key = f"{config.type}:{session}"
        if cache_key not in self._session_cache:
            self._session_cache[cache_key] = self._create_state(config, Memory())
        return self._session_cache[cache_key]

    def _create_state(self, config: "StateConfig", kv: Any) -> "State":
        """Create a state instance from config and KV store."""
        from agex.state import Live, Versioned
        from agex.state.gc import GCVersioned

        if config.type == "versioned":
            state: "State" = Versioned(store=kv)
            # Wrap with GC if high_water_bytes is set
            if config.high_water_bytes is not None:
                state = GCVersioned(
                    state,
                    high_water_bytes=config.high_water_bytes,
                    low_water_bytes=config.low_water_bytes,
                )
            return state
        elif config.type == "live":
            return Live()
        else:  # ephemeral
            return Live()

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
        """Execute the task locally using the agent's task loop."""
        # Resolve state from agent's config
        state = self.resolve_state(agent._state_config, session)

        # Get the registered task
        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        # Call the task directly - it handles the loop internally
        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return task_fn(*args, **call_kwargs)

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
        """Execute the task locally using the agent's async task loop."""
        # Resolve state from agent's config
        state = self.resolve_state(agent._state_config, session)

        # Get the registered task
        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        # Call the task directly - it handles the loop internally
        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return await task_fn(*args, **call_kwargs)

    def state(
        self,
        config: "StateConfig | None",
        session: str,
        fingerprint: str = "",
    ) -> "State":
        """Get state for client-side access.

        For Local host, this is the same as resolve_state since we have
        direct access to the state.
        """
        return self.resolve_state(config, session, fingerprint)
