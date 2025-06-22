from dataclasses import dataclass
from typing import Any, Callable


class _AgentExit(Exception):
    """Base class for agent exit signals. Should not be caught by user code."""

    pass


@dataclass
class ExitSuccess(_AgentExit):
    """Signal that the agent has completed its task successfully."""

    result: Any


@dataclass
class ExitFail(_AgentExit):
    """Signal that the agent has failed and cannot complete its task."""

    reason: str


@dataclass
class ExitClarify(_AgentExit):
    """Signal that the agent needs clarification from the user."""

    question: str


@dataclass
class Task:
    """A placeholder for a task definition."""

    # For now, we don't need any fields.
    # This will be fleshed out later.
    pass


def task(func):
    """A decorator to mark a function as an agent task."""
    return Task()


@dataclass
class RegisteredItem:
    public: bool  # expose existence to agent
    document: bool  # expose documentation to agent


@dataclass
class RegisteredFn(RegisteredItem):
    fn: Callable


@dataclass
class RegisteredClass(RegisteredItem):
    cls: type
    constructable: bool
    allowed_methods: set[str]  # perhaps make this a set of RegisteredFn?
    allowed_attrs: set[str]


class Agent:
    def __init__(self, primer: str | None = None):
        self.primer = primer
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.fn = self.make_fn_decorator()
        self.cls = self.make_cls_decorator()
        self.task = self.make_task_decorator()

    def make_fn_decorator(self):
        pass

    def make_cls_decorator(self):
        pass

    def make_task_decorator(self):
        pass
