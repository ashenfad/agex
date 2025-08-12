from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Mapping, Union

Pattern = Union[str, Iterable[str], Callable[[str], bool]]
Visibility = Literal["high", "medium", "low"]
RESERVED_NAMES = {"dataclass", "dataclasses"}


class _AgentExit(Exception):
    """Base class for agent exit signals. Should not be caught by agent code."""

    pass


# Task control classes for improved iterative workflow
@dataclass
class TaskSuccess(_AgentExit):
    """Signal that the agent has completed its task successfully."""

    result: Any = None


@dataclass
class TaskFail(_AgentExit):
    """Signal that the agent has failed and cannot complete its task."""

    message: str


@dataclass
class TaskClarify(_AgentExit):
    """Signal that the agent needs more information to complete its task."""

    message: str


@dataclass
class TaskContinue(_AgentExit):
    """Signal that the agent wants to continue to the next iteration."""

    observations: tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class TaskTimeout(_AgentExit):
    """Signal that task could not be completed within limits."""

    message: str


@dataclass
class LLMFail(_AgentExit):
    """Uncatchable signal that the LLM call failed (after retries)."""

    message: str
    provider: str | None = None
    model: str | None = None
    retries: int = 0


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


@dataclass
class RegisteredObject(RegisteredItem):
    """Represents a live Python object registered with the agent."""

    # The mandatory, agent-facing namespace (e.g., 'db').
    # This is also used as the key in the host-side registry.
    name: str

    # A dictionary of exposed methods, reusing MemberSpec for consistency.
    methods: dict[str, MemberSpec] = field(default_factory=dict)

    # A dictionary for exposed read-only attributes/properties.
    properties: dict[str, MemberSpec] = field(default_factory=dict)

    # Exception mappings: external exception type -> agex exception type
    # This allows live objects to map their library-specific exceptions
    # to user-catchable agex exceptions
    exception_mappings: dict[type, type] = field(default_factory=dict)


@dataclass
class StateType(Mapping):
    """Represents the agent's state at a particular moment in time."""
