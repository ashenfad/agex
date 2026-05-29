"""
Type stubs for agex.agent.task module.

Provides static type information for the task decorator: a decorated function
becomes a ``Task[T]`` that preserves the wrapped function's return type ``T``
on its call *and* on its control methods (``resume``/``aresume``/``cancel``).

Parameters are intentionally not statically checked on ``__call__`` (consistent
with the prior ``Callable[..., T]`` typing) — the framework also injects
``session``/``on_event``/``on_token`` keyword arguments at call time.
"""

from typing import Any, Callable, Generic, TypeVar, Union, overload

from agex.agent.base import BaseAgent
from agex.agent.loop import TaskLoopMixin
from agex.agent.permission import PermissionResponse

T = TypeVar("T")

class Task(Generic[T]):
    """Typed view of a function decorated with ``@agent.task``.

    For a sync task, ``T`` is the function's return type; for an async task,
    ``T`` is its ``Coroutine[..., R]`` (so ``await task(...)`` yields ``R``).
    The control methods return ``T`` to match — use ``resume`` for sync tasks
    and ``aresume`` for async ones.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...
    def resume(self, *, response: PermissionResponse, session: str = ...) -> T: ...
    def aresume(self, *, response: PermissionResponse, session: str = ...) -> T: ...
    def cancel(self, session: str = ...) -> None: ...

class TaskMixin(TaskLoopMixin, BaseAgent):
    # Overloads for different usage patterns of the task decorator

    @overload
    def task(self, func: Callable[..., T]) -> Task[T]:
        """Naked decorator: @agent.task"""
        ...

    @overload
    def task(self, primer: str) -> Callable[[Callable[..., T]], Task[T]]:
        """Parameterized decorator: @agent.task("primer")"""
        ...

    @overload
    def task(self, *, primer: str) -> Callable[[Callable[..., T]], Task[T]]:
        """Keyword decorator: @agent.task(primer="...")"""
        ...

    @overload
    def task(self, *, setup: str) -> Callable[[Callable[..., T]], Task[T]]:
        """Setup decorator: @agent.task(setup="...")"""
        ...

    @overload
    def task(self, *, primer: str, setup: str) -> Callable[[Callable[..., T]], Task[T]]:
        """Primer and setup decorator: @agent.task(primer="...", setup="...")"""
        ...

    def task(
        self,
        primer_or_func: Union[str, Callable[..., T], None] = None,
        /,
        *,
        primer: str | None = None,
        setup: str | None = None,
        on_conflict: str = "retry",
        max_conflict_retries: int = 3,
    ) -> Union[Task[T], Callable[[Callable[..., T]], Task[T]]]:
        """
        Decorator to mark a function as an agent task.

        The return type is preserved from the original function, and the
        result exposes ``resume``/``aresume``/``cancel``.

        Args:
            on_conflict: How to handle concurrency conflicts ('retry' or 'abandon')
            max_conflict_retries: Max retry attempts for 'retry' strategy (default: 3)
        """
        ...

def clear_dynamic_dataclass_registry() -> None:
    """Clear the dynamic dataclass registry."""
    ...
