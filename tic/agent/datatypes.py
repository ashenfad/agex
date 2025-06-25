from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Union

Pattern = Union[str, Iterable[str], Callable[[str], bool]]
Visibility = Literal["high", "medium", "low"]
RESERVED_NAMES = {"dataclass", "dataclasses"}


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
class MemberSpec:
    visibility: Visibility | None = None
    docstring: str | None = None
    constructable: bool | None = None


@dataclass
class AttrDescriptor:
    # A descriptor to hold metadata until the class is processed.
    default: Any
    visibility: Visibility


@dataclass
class RegisteredItem:
    visibility: Visibility


@dataclass
class RegisteredFn(RegisteredItem):
    fn: Callable
    docstring: str | None


@dataclass
class RegisteredClass(RegisteredItem):
    """Represents a registered class and its members."""

    cls: type
    constructable: bool
    # 'visibility' on RegisteredItem is the default.
    attrs: dict[str, MemberSpec] = field(default_factory=dict)
    methods: dict[str, MemberSpec] = field(default_factory=dict)


@dataclass
class RegisteredModule(RegisteredItem):
    """Represents a registered module with its selected members."""

    name: str  # The name the agent will use to import it
    module: ModuleType
    fns: dict[str, MemberSpec] = field(default_factory=dict)
    consts: dict[str, MemberSpec] = field(default_factory=dict)
    classes: dict[str, RegisteredClass] = field(default_factory=dict)
