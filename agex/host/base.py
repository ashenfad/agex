"""
Host abstraction for agent task execution.

This module defines the Host ABC that encapsulates where/how agent tasks execute.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
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
