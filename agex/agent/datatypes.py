from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Union

Pattern = Union[str, Iterable[str], Callable[[str], bool]]
Visibility = Literal["high", "medium", "low"]
RESERVED_NAMES = {"dataclass", "dataclasses"}


class _AgentExit(BaseException):
    """Base class for agent exit signals.

    Inherits from BaseException (not Exception) so that agent code using
    ``except Exception`` cannot catch these signals. sandtrap's sandbox
    excludes BaseException from the namespace, making these uncatchable.
    """

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
class TaskTimeout(_AgentExit):
    """Signal that task could not be completed within limits."""

    message: str


@dataclass
class TaskCancelled(_AgentExit):
    """Signal that task was cancelled via external request."""

    message: str
    task_name: str
    iterations_completed: int = 0


@dataclass
class LLMFail(_AgentExit):
    """Uncatchable signal that the LLM call failed (after retries)."""

    message: str
    provider: str | None = None
    model: str | None = None
    retries: int = 0


@dataclass
class FileAction:
    """DEPRECATED — superseded by FileWriteEmission in agex.agent.emissions.

    Kept alive during Phase 1 of the retooling so legacy code still
    imports. Phase 2 rewrites the execution loop and renderer to use
    FileWriteEmission directly, after which this goes away.
    """

    path: str
    content: str
    mode: Literal["write", "append"] = "write"


@dataclass
class EditAction:
    """DEPRECATED — superseded by FileEditEmission in agex.agent.emissions.

    Kept alive during Phase 1 of the retooling so legacy code still
    imports. Phase 2 rewrites the execution loop and renderer to use
    FileEditEmission directly, after which this goes away.
    """

    path: str
    search: str
    content: str
    operation: Literal["replace", "insert-after", "insert-before"] = "replace"
    match_all: bool = False


class UnpicklableVariableError(Exception):
    """Raised when attempting to access a variable that was not persisted due to being unpicklable."""

    def __init__(self, marker: "UnpicklableMarker | str"):
        if isinstance(marker, str):
            super().__init__(marker)
            self.marker = None
        else:
            super().__init__(str(marker))
            self.marker = marker


class UnpicklableMarker:
    """Marker for variables that couldn't be persisted due to being unpicklable.

    Stored in the namespace in place of the original object. Any access
    (attribute, call, iteration, comparison, etc.) raises a descriptive
    UnpicklableVariableError so the agent knows what happened.
    """

    def __init__(self, variable_name: str, type_name: str, original_exception: str):
        # Use object.__setattr__ to bypass our __getattr__
        object.__setattr__(self, "variable_name", variable_name)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "original_exception", original_exception)

    def _raise(self):
        raise UnpicklableVariableError(
            f"Variable '{self.variable_name}' (type: {self.type_name}) was not "
            f"persisted because it could not be saved between turns. "
            f"Re-create it. Original error: {self.original_exception}"
        )

    def __getattr__(self, name):
        # Allow pickle and introspection protocols to work normally
        if name.startswith("__"):
            raise AttributeError(name)
        self._raise()

    def __reduce__(self):
        return (
            UnpicklableMarker,
            (self.variable_name, self.type_name, self.original_exception),
        )

    def __call__(self, *args, **kwargs):
        self._raise()

    def __iter__(self):
        self._raise()

    def __bool__(self):
        self._raise()

    def __eq__(self, other):
        self._raise()

    def __ne__(self, other):
        self._raise()

    def __lt__(self, other):
        self._raise()

    def __le__(self, other):
        self._raise()

    def __gt__(self, other):
        self._raise()

    def __ge__(self, other):
        self._raise()

    def __len__(self):
        self._raise()

    def __getitem__(self, key):
        self._raise()

    def __contains__(self, item):
        self._raise()

    def __str__(self):
        return (
            f"<UnpicklableMarker: '{self.variable_name}' ({self.type_name}) "
            f"was not persisted — re-create it. "
            f"Original error: {self.original_exception}>"
        )

    def __repr__(self):
        return self.__str__()


@dataclass
class MemberSpec:
    visibility: Visibility | None = None
    docstring: str | None = None
    constructable: bool | None = None
    host_fs_access: bool = False
    network_access: bool = False


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
