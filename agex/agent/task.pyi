"""
Type stubs for agex.agent.task module.

This provides static type information for the task decorator to help
type checkers understand that decorated functions preserve their return types.
"""

from typing import Any, Callable, Generator, Protocol, TypeVar, Union, overload

from agex.agent.base import BaseAgent
from agex.agent.events import BaseEvent
from agex.agent.loop import TaskLoopMixin

# Type variable to preserve the return type of decorated functions
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

class TaskCallable(Protocol[T]):
    """Protocol for task-decorated functions that includes the stream method."""

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...
    @property
    def stream(self) -> Callable[..., Generator[BaseEvent, None, None]]: ...

class TaskMixin(TaskLoopMixin, BaseAgent):
    # Overloads for different usage patterns of the task decorator

    @overload
    def task(self, func: Callable[..., T]) -> TaskCallable[T]:
        """Naked decorator: @agent.task"""
        ...

    @overload
    def task(self, primer: str) -> Callable[[Callable[..., T]], TaskCallable[T]]:
        """Parameterized decorator: @agent.task("primer")"""
        ...

    @overload
    def task(self, *, primer: str) -> Callable[[Callable[..., T]], TaskCallable[T]]:
        """Keyword decorator: @agent.task(primer="...")"""
        ...

    @overload
    def task(self, *, setup: str) -> Callable[[Callable[..., T]], TaskCallable[T]]:
        """Setup decorator: @agent.task(setup="...")"""
        ...

    @overload
    def task(
        self, *, primer: str, setup: str
    ) -> Callable[[Callable[..., T]], TaskCallable[T]]:
        """Primer and setup decorator: @agent.task(primer="...", setup="...")"""
        ...

    def task(
        self,
        primer_or_func: Union[str, Callable[..., T], None] = None,
        /,
        *,
        primer: str | None = None,
        setup: str | None = None,
    ) -> Union[TaskCallable[T], Callable[[Callable[..., T]], TaskCallable[T]]]:
        """
        Decorator to mark a function as an agent task.

        The return type is preserved from the original function.
        """
        ...

def clear_dynamic_dataclass_registry() -> None:
    """Clear the dynamic dataclass registry."""
    ...
