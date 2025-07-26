"""
Type stubs for agex.agent.task module.

This provides static type information for the task decorator to help
type checkers understand that decorated functions preserve their return types.
"""

from typing import Any, Callable, TypeVar, Union, overload

from agex.agent.base import BaseAgent
from agex.agent.loop import TaskLoopMixin

# Type variable to preserve the return type of decorated functions
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

class TaskMixin(TaskLoopMixin, BaseAgent):
    # Overloads for different usage patterns of the task decorator

    @overload
    def task(self, func: Callable[..., T]) -> Callable[..., T]:
        """Naked decorator: @agent.task"""
        ...

    @overload
    def task(self, primer: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Parameterized decorator: @agent.task("primer")"""
        ...

    @overload
    def task(self, *, primer: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Keyword decorator: @agent.task(primer="...")"""
        ...

    @overload
    def task(self, *, setup: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Setup decorator: @agent.task(setup="...")"""
        ...

    @overload
    def task(
        self, *, primer: str, setup: str
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Primer and setup decorator: @agent.task(primer="...", setup="...")"""
        ...

    def task(
        self,
        primer_or_func: Union[str, Callable[..., T], None] = None,
        /,
        *,
        primer: str | None = None,
        setup: str | None = None,
    ) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
        """
        Decorator to mark a function as an agent task.

        The return type is preserved from the original function.
        """
        ...

def clear_dynamic_dataclass_registry() -> None:
    """Clear the dynamic dataclass registry."""
    ...
