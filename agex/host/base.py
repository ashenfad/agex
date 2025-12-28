"""
Host abstraction for agent task execution.

This module defines the Host ABC that encapsulates where/how agent tasks execute.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.host.dependencies import Dependencies
    from agex.state import State
    from agex.state.config import StateConfig


class Host(ABC):
    """
    Abstract base class for execution hosts.

    A Host determines where and how agent tasks are executed:
    - Local: runs in the current process
    - HTTP: sends to a remote HTTP server
    - Modal/Beam: serverless execution (future)
    """

    @abstractmethod
    def dump_config(self) -> dict[str, Any]:
        """
        Serialize host configuration for transport.

        Returns:
            Dictionary with 'provider' key and provider-specific config
        """
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Host":
        """
        Create a Host from serialized configuration.

        Args:
            config: Dictionary from dump_config()

        Returns:
            Reconstructed Host instance
        """
        provider = config.get("provider")
        if provider == "local":
            from agex.host.local import Local

            return Local()
        elif provider == "http":
            from agex.host.http import HTTP

            return HTTP(
                url=config["url"],
                timeout=config.get("timeout", 300.0),
                retries=config.get("retries", 0),
            )
        elif provider == "modal":
            from agex.host.modal import Modal

            return Modal(
                app=config.get("app"),
                secrets=config.get("secrets"),
                gpu=config.get("gpu"),
                memory=config.get("memory"),
                timeout=config.get("timeout", 300.0),
                scaledown_window=config.get("scaledown_window", 300),
            )
        else:
            raise ValueError(f"Unknown host provider: {provider}")

    @abstractmethod
    def validate_state(self, config: "StateConfig | None") -> None:
        """
        Validate state config is compatible with this host.

        Called at Agent creation time for early failure.

        Args:
            config: State configuration to validate (None = ephemeral)

        Raises:
            ValueError: If the config is not compatible with this host
        """
        ...

    @abstractmethod
    def resolve_state(self, config: "StateConfig | None", session: str) -> "State":
        """
        Create or retrieve a State instance for this session.

        Args:
            config: State configuration (None = ephemeral)
            session: Session identifier for state isolation

        Returns:
            A State instance for this session
        """
        ...

    @abstractmethod
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
        """
        Execute a task synchronously.

        Args:
            agent: The agent to execute the task on
            task_name: Name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            session: Session identifier for state resolution
            on_event: Optional event callback
            on_token: Optional token callback

        Returns:
            The task result
        """
        ...

    @abstractmethod
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
        """
        Execute a task asynchronously.

        Args:
            agent: The agent to execute the task on
            task_name: Name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            session: Session identifier for state resolution
            on_event: Optional event callback
            on_token: Optional token callback

        Returns:
            The task result
        """
        ...

    def warmup(self, deps: "Dependencies") -> None:
        """
        Pre-warm the host for faster cold starts.

        For serverless hosts (Modal, Beam), this builds the container image
        and starts a warm instance. For local/HTTP hosts, this is a no-op.

        Args:
            deps: Dependencies inferred from agent registrations
        """
        pass  # Default no-op for Local/HTTP
