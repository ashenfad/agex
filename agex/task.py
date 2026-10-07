"""Tasks: a typed call that runs an agent on a world of its own.

``@agent.task`` turns a function's signature and docstring into a task;
the docstring is the job, and the body stays empty. Calling the task
runs the agent on a fresh world, a scratch one built from
``agent.profile`` or a fork of ``world=``, and hands back the value the
agent's code passed to ``task.success``, checked against the return
annotation. The caller's world is never touched.

    @agent.task
    def grade(responses: list[Response]) -> Report:
        \"""Grade the quiz.\"""

    report = grade(rs)                         # the value, or TaskFailed
    out = grade.run(rs, world=ws, keep=True)   # an Outcome; out.ref names the fork

Inside the world, agent code sees each argument bound by its own name,
the record and enum types the signature names, and ``task``:
``task.success(value)`` ends the task with its value, and
``task.fail(reason)`` ends it without one. A value that doesn't fit is a
``TypeError`` at the call, so the agent fixes it in the same script. A
model that stops without either call is nudged to finish, twice; then
the task fails.

Inputs are values: a task can't change what its caller passed. Data is
deep-copied, a table shallow-copied (pandas' copy-on-write keeps the
task's writes to itself), and an array passed as a read-only view. A
live input (a client, a callable) is a capability: it is bound as a host
object of the task's world, under that world's host-object policy.

Values travel only within the process so far, so a world whose code
runs elsewhere (process isolation, a dud machine) is refused before any
model call.
"""

from __future__ import annotations

import ast
import asyncio
import collections.abc
import copy
import dataclasses
import datetime
import decimal
import enum
import functools
import inspect
import pathlib
import textwrap
import types
import typing
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import reprobate
from nontainer import NotSupportedError, Profile, Store, Workspace
from nontainer.executor import LocalExecutor
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.errors import PydanticSchemaGenerationError

from .agent import Outcome, Session, Status, _block
from .record import Message, Text, new_id

if TYPE_CHECKING:
    from .agent import Agent

__all__ = [
    "NUDGES",
    "RESERVED",
    "Kind",
    "Task",
    "TaskError",
    "TaskFailed",
    "TaskInterrupted",
    "TaskObject",
    "TaskSpec",
    "ValueSpec",
    "kinds_of",
]

Kind = Literal["data", "bytes", "table", "array", "live", "any"]
"""What a type needs carried: plain data (scalars, containers, records),
bytes, a table (a DataFrame, an Arrow table), an array, a live object
(a callable, a client, a generator), or anything (``Any``, no
annotation)."""

RESERVED = frozenset({"task", "host", "world", "keep"})
"""Names a task's parameters can't take: ``task`` and ``host`` are bound
in the world, and ``world`` and ``keep`` are the call's own."""

NUDGES = 2
"""How many times a model that stops without finishing is sent back to
finish, before the task fails."""

PREVIEW = 600
"""How many characters of each input the task's brief shows."""

NUDGE = (
    "The task isn't finished. In run_python, call `task.success(value)` "
    "with the result, or `task.fail(reason)` if it can't be done."
)

FINISHED = "task.success: the task is done."
GAVE_UP = "task.fail: the task is over."

INSTRUCTIONS = """\
You are doing a task. Its inputs are bound by name in every run_python \
call, along with the types it names and `task`:
- `task.success(value)` hands back the result and ends the task. The value \
must have the task's return type; one that doesn't raises TypeError right \
there, so fix it and call again.
- `task.fail(reason)` ends the task without a result, saying why it can't \
be done.
Either call ends the script it is in: nothing after it runs."""

_SCALARS: tuple[type, ...] = (
    bool,
    int,
    float,
    str,
    decimal.Decimal,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    uuid.UUID,
    pathlib.PurePath,
)

_CONTAINERS = frozenset(
    {
        list,
        tuple,
        set,
        frozenset,
        dict,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Set,
        collections.abc.MutableSet,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
    }
)

_EMPTY = inspect.Parameter.empty


# -- what a type needs carried ---------------------------------------------------------


def _root(tp: type) -> str:
    return (tp.__module__ or "").split(".")[0]


def _is_table_type(tp: type) -> bool:
    return (
        (_root(tp) == "pandas" and tp.__name__ in ("DataFrame", "Series"))
        or (_root(tp) == "pyarrow" and tp.__name__ in ("Table", "RecordBatch"))
        or hasattr(tp, "__arrow_c_stream__")
    )


def _is_array_type(tp: type) -> bool:
    return _root(tp) == "numpy" and tp.__name__ == "ndarray"


def _fields(tp: type) -> dict[str, Any] | None:
    """The field annotations of a record type (a dataclass, a pydantic
    model, a TypedDict or a NamedTuple), or ``None`` for any other."""
    if dataclasses.is_dataclass(tp):
        try:
            return typing.get_type_hints(tp)
        except Exception:  # noqa: BLE001 - an unresolvable name is just unknown
            return {f.name: Any for f in dataclasses.fields(tp)}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return {name: f.annotation for name, f in tp.model_fields.items()}
    if typing.is_typeddict(tp) or (
        isinstance(tp, type) and issubclass(tp, tuple) and hasattr(tp, "_fields")
    ):
        try:
            return typing.get_type_hints(tp)
        except Exception:  # noqa: BLE001
            return dict.fromkeys(getattr(tp, "__annotations__", {}), Any)
    return None


def kinds_of(tp: Any) -> frozenset[Kind]:
    """What values of ``tp`` need carried, as the set of kinds its parts
    are: ``list[Response]`` is data, ``dict[str, DataFrame]`` data and a
    table, ``Callable[[int], int]`` live."""
    return frozenset(_kinds(tp, frozenset()))


def _kinds(tp: Any, seen: frozenset[type]) -> set[Kind]:
    if tp is Any or tp is object or tp is _EMPTY:
        return {"any"}
    if tp is None or tp is type(None):
        return {"data"}
    if isinstance(tp, (TypeVar, str, typing.ForwardRef)):
        return {"any"}
    supertype = getattr(tp, "__supertype__", None)  # a NewType
    if supertype is not None:
        return _kinds(supertype, seen)
    origin = typing.get_origin(tp)
    if origin is not None:
        args = typing.get_args(tp)
        if origin is typing.Annotated:
            return _kinds(args[0], seen)
        if origin is Literal:
            return {"data"}
        if origin is Union or origin is types.UnionType:
            return _all(set(), args, seen)
        if origin in _CONTAINERS:
            inner = [a for a in args if a is not Ellipsis]
            if not inner:
                return {"data", "any"}
            return _all({"data"}, inner, seen)
        if isinstance(origin, type) and _fields(origin) is not None:
            return _kinds(origin, seen)
        return {"live"}
    if not isinstance(tp, type):
        return {"live"}
    if issubclass(tp, enum.Enum) or issubclass(tp, _SCALARS):
        return {"data"}
    if issubclass(tp, (bytes, bytearray, memoryview)):
        return {"bytes"}
    if tp in (list, tuple, set, frozenset, dict):
        return {"data", "any"}
    if tp in seen:
        return {"data"}
    fields = _fields(tp)
    if fields is not None:
        return _all({"data"}, fields.values(), seen | {tp})
    if _is_table_type(tp):
        return {"table"}
    if _is_array_type(tp):
        return {"array"}
    return {"live"}


def _all(kinds: set[Kind], parts: Iterable[Any], seen: frozenset[type]) -> set[Kind]:
    for part in parts:
        kinds |= _kinds(part, seen)
    return kinds


def _named_types(tp: Any, found: dict[str, type], seen: set[int]) -> None:
    """Collect the record and enum classes ``tp`` names, by name: what
    agent code needs bound to build a value of ``tp``."""
    if id(tp) in seen:
        return
    seen.add(id(tp))
    for arg in typing.get_args(tp):
        if isinstance(arg, list):  # Callable's parameter list
            for a in arg:
                _named_types(a, found, seen)
        else:
            _named_types(arg, found, seen)
    if not isinstance(tp, type):
        return
    if issubclass(tp, enum.Enum) or _fields(tp) is not None:
        if tp.__module__ != "builtins":
            other = found.setdefault(tp.__name__, tp)
            if other is not tp:
                raise TypeError(
                    f"two types named {tp.__name__!r} ({_qualified(other)} and "
                    f"{_qualified(tp)}): agent code sees types by name"
                )
        for annotation in (_fields(tp) or {}).values():
            _named_types(annotation, found, seen)


def _qualified(tp: type) -> str:
    return f"{tp.__module__}.{tp.__qualname__}"


# -- checking a value against a type ----------------------------------------------------


def _checker(tp: Any) -> Callable[[Any], None]:
    """A strict check of a value against ``tp``, raising pydantic's
    ``ValidationError`` or a ``TypeError``: no coercion, so ``"42"`` is
    not an ``int``. A type pydantic can't build a validator for is
    checked by ``isinstance``, and ``Any`` accepts anything."""
    if tp is Any or tp is object or tp is _EMPTY:
        return lambda value: None
    adapter: TypeAdapter[Any] | None
    try:
        adapter = TypeAdapter(tp)
    except PydanticSchemaGenerationError:
        try:
            adapter = TypeAdapter(tp, config=ConfigDict(arbitrary_types_allowed=True))
        except Exception:  # noqa: BLE001 - checked by isinstance instead
            adapter = None
    except Exception:  # noqa: BLE001
        adapter = None
    if adapter is not None:
        return functools.partial(adapter.validate_python, strict=True)
    if isinstance(tp, type):

        def check(value: Any) -> None:
            if not isinstance(value, tp):
                raise TypeError(
                    f"got {type(value).__name__}, not an instance of {tp.__name__}"
                )

        return check
    return lambda value: None


def _summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(map(str, e['loc'])) or 'the value'}: {e['msg']}"
            for e in error.errors()[:5]
        ]
        more = error.error_count() - len(problems)
        return "; ".join(problems) + (f"; and {more} more" if more > 0 else "")
    return str(error)


def _schema(tp: Any, kinds: frozenset[Kind]) -> Mapping[str, Any] | None:
    if tp is _EMPTY or not kinds or not kinds <= {"data", "bytes"}:
        return None
    try:
        return TypeAdapter(tp).json_schema()
    except Exception:  # noqa: BLE001 - a record holding an opaque type, say
        return None


@dataclass(frozen=True)
class ValueSpec:
    """One input or the return of a task: its annotation, the kinds of
    value it needs carried, and its JSON schema when it is plain data."""

    annotation: Any
    kinds: frozenset[Kind]
    schema: Mapping[str, Any] | None = field(default=None, compare=False)
    _check: Callable[[Any], None] = field(
        default=lambda value: None, repr=False, compare=False
    )

    @classmethod
    def of(cls, annotation: Any) -> ValueSpec:
        kinds = kinds_of(annotation)
        return cls(
            annotation=annotation,
            kinds=kinds,
            schema=_schema(annotation, kinds),
            _check=_checker(annotation),
        )

    def check(self, value: Any, what: str, *, bound: bool = False) -> None:
        """Refuse ``value`` with a ``TypeError`` naming ``what``, unless
        it fits. ``bound`` says the value was built in a world where the
        types are bound by name, which a class of the same name defined
        there does not match."""
        try:
            self._check(value)
        except (ValidationError, TypeError) as error:
            summary = _summary(error)
            if bound and "instance of" in summary:
                summary += (
                    ". Build it from the types already bound by name: a class "
                    "defined in your code is a different type, even with the "
                    "same name"
                )
            raise TypeError(
                f"{what} must be {_fmt(self.annotation)}: {summary}"
            ) from None


# -- the spec --------------------------------------------------------------------------


def _refuse_body(fn: Callable[..., Any], name: str) -> None:
    """A task's body is its docstring; code there would never run."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return
    node = tree.body[0] if tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    for statement in body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        ):
            continue
        raise TypeError(
            f"task {name!r} has code in its body, which would never run: a "
            "task's body is its docstring, and the agent does the work"
        )


@dataclass(frozen=True)
class TaskSpec:
    """What a task is: its name, its instructions (the docstring), its
    inputs and its return, and the record and enum types they name
    (bound by name in the task's world, so agent code can build them)."""

    name: str
    instructions: str
    params: Mapping[str, ValueSpec]
    returns: ValueSpec
    types: Mapping[str, type] = field(default_factory=dict, compare=False)
    signature: inspect.Signature = field(
        default=inspect.Signature(), repr=False, compare=False
    )

    @classmethod
    def of(cls, fn: Callable[..., Any]) -> TaskSpec:
        name = fn.__name__
        _refuse_body(fn, name)
        signature = inspect.signature(fn)
        try:
            hints = typing.get_type_hints(fn, include_extras=True)
        except Exception:  # noqa: BLE001 - names the module can't resolve
            hints = {}
        params: dict[str, ValueSpec] = {}
        for param in signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"task {name!r} takes *{param.name}: a task's inputs are "
                    "named, since agent code sees each by its name"
                )
            if param.name in RESERVED:
                raise TypeError(
                    f"task {name!r} has a parameter named {param.name!r}, which "
                    f"is reserved ({', '.join(sorted(RESERVED))})"
                )
            params[param.name] = ValueSpec.of(hints.get(param.name, param.annotation))
        returns = ValueSpec.of(hints.get("return", signature.return_annotation))
        found: dict[str, type] = {}
        seen: set[int] = set()
        for spec in (*params.values(), returns):
            _named_types(spec.annotation, found, seen)
        clash = sorted(set(found) & set(params))
        if clash:
            raise TypeError(
                f"task {name!r} has a parameter named like a type it uses "
                f"({', '.join(clash)}): agent code sees both by name"
            )
        return cls(
            name=name,
            instructions=inspect.cleandoc(fn.__doc__ or ""),
            params=params,
            returns=returns,
            types=found,
            signature=signature,
        )

    def bind(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """The inputs of a call, checked against their annotations and
        made the values the task receives: a ``TypeError`` for a call
        that doesn't fit."""
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(f"{self.name}(): {error}") from None
        bound.apply_defaults()
        inputs: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            self.params[name].check(value, f"{self.name}()'s argument {name!r}")
            try:
                inputs[name] = _as_input(value)
            except Exception as error:  # noqa: BLE001 - says which input
                raise TypeError(
                    f"{self.name}()'s argument {name!r} can't be copied for the "
                    f"task ({type(error).__name__}: {error}); pass data, or a "
                    "live object the task may use as it is"
                ) from None
        return inputs


# -- inputs as the task receives them ----------------------------------------------------


def _value_kind(value: Any) -> Kind:
    if value is None or isinstance(value, (*_SCALARS, enum.Enum)):
        return "data"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    tp = type(value)
    if _fields(tp) is not None or isinstance(
        value, (list, tuple, set, frozenset, dict)
    ):
        return "data"
    if _is_table_type(tp):
        return "table"
    if _is_array_type(tp):
        return "array"
    return "live"


def _shallow_table(table: Any) -> Any:
    """A copy of ``table`` whose writes don't reach the original, made
    without copying its data where the library allows."""
    if _root(type(table)) == "pandas":
        import pandas

        # copy-on-write is how pandas works from 3.0: a shallow copy is
        # cheap, and a write to either copies what it touches
        deep = int(pandas.__version__.split(".")[0]) < 3
        return table.copy(deep=deep)
    clone = getattr(table, "clone", None)  # polars
    if callable(clone):
        return clone()
    return table  # Arrow tables don't change in place


def _as_input(value: Any) -> Any:
    """``value`` as a task receives it: data copied, a table shallow-copied,
    an array as a read-only view, and a live object as it is."""
    kind = _value_kind(value)
    if kind == "live":
        return value
    if kind == "table":
        return _shallow_table(value)
    if kind == "array":
        view = value.view()
        view.flags.writeable = False
        return view
    if kind == "bytes":
        return value if isinstance(value, bytes) else bytes(value)
    tp = type(value)
    if tp is list:
        return [_as_input(v) for v in value]
    if tp is tuple:
        return tuple(_as_input(v) for v in value)
    if tp is dict:
        return {k: _as_input(v) for k, v in value.items()}
    return copy.deepcopy(value)


# -- the brief the model starts from -------------------------------------------------------


def _fmt(tp: Any) -> str:
    """``tp`` written the way the agent's code would write it."""
    if tp is _EMPTY or tp is Any:
        return "Any"
    if tp is None or tp is type(None):
        return "None"
    origin = typing.get_origin(tp)
    if origin is not None:
        args = typing.get_args(tp)
        if origin is typing.Annotated:
            return _fmt(args[0])
        if origin is Union or origin is types.UnionType:
            return " | ".join(_fmt(a) for a in args)
        if origin is Literal:
            return f"Literal[{', '.join(map(repr, args))}]"
        name = getattr(origin, "__name__", None) or repr(origin).replace("typing.", "")
        parts = [
            "[" + ", ".join(_fmt(x) for x in a) + "]"
            if isinstance(a, list)
            else "..."
            if a is Ellipsis
            else _fmt(a)
            for a in args
        ]
        return f"{name}[{', '.join(parts)}]" if parts else name
    if isinstance(tp, type):
        return tp.__name__
    return repr(tp).replace("typing.", "")


def _source(tp: type) -> str:
    try:
        return textwrap.dedent(inspect.getsource(tp)).strip()
    except (OSError, TypeError):
        fields = _fields(tp) or {}
        body = "\n".join(f"    {n}: {_fmt(t)}" for n, t in fields.items())
        return f"class {tp.__name__}:\n{body or '    ...'}"


def _brief(spec: TaskSpec, inputs: Mapping[str, Any]) -> str:
    """The task's opening message: the job, the inputs, how to finish,
    and the types it names."""
    parts = [spec.instructions or f"Do the task {spec.name!r}."]
    if inputs:
        lines = [
            f"- `{name}: {_fmt(spec.params[name].annotation)}` = "
            f"{reprobate.render(value, budget=PREVIEW)}"
            for name, value in inputs.items()
        ]
        parts.append("Inputs, bound by name:\n" + "\n".join(lines))
    returns = spec.returns.annotation
    if returns is None or returns is type(None):
        parts.append("When it's done, call `task.success()`.")
    else:
        parts.append(
            f"When it's done, call `task.success(value)` with a `{_fmt(returns)}`."
        )
    if spec.types:
        sources = "\n\n".join(_source(tp) for tp in spec.types.values())
        parts.append(
            "The types, already bound by name (use these: a class you define "
            f"yourself is a different type):\n```python\n{sources}\n```"
        )
    return "\n\n".join(parts)


# -- the run ------------------------------------------------------------------------------


class _Finished(BaseException):
    """Stops the script that ended its task. A ``BaseException``, so the
    agent code's ``except Exception`` lets it through."""


class TaskObject:
    """``task`` in agent code: how a task's run hands back its result.

    It records the result and stops the script; the loop reads the
    record, never the script's error, so a script that catches the stop
    still ends the task.
    """

    def __init__(self, returns: ValueSpec) -> None:
        self._returns = returns
        self._result: tuple[Literal["success", "failed"], Any] | None = None

    def __repr__(self) -> str:
        return f"<task returning {_fmt(self._returns.annotation)}>"

    def success(self, value: Any = None) -> None:
        """End the task with ``value``, which must fit its return type;
        one that doesn't raises ``TypeError`` here."""
        self._returns.check(value, "task.success's value", bound=True)
        self._result = ("success", value)
        raise _Finished("task.success")

    def fail(self, reason: str) -> None:
        """End the task without a value, saying why it can't be done."""
        self._result = ("failed", str(reason))
        raise _Finished("task.fail")


class _TaskSession(Session):
    """A session over a task's world: it ends when ``task`` records a
    result, and nudges a model that stops without one."""

    def __init__(self, agent: Agent, ws: Workspace, task: TaskObject) -> None:
        super().__init__(agent, ws)
        self._task = task
        self._nudged = 0

    def _system(self) -> Message:
        base = super()._system()
        return Message(
            id=base.id,
            role=base.role,
            parts=(Text(text=f"{base.text}\n\n{INSTRUCTIONS}"),),
        )

    def _after_call(self, output: str, is_error: bool) -> tuple[str, bool]:
        result = self._task._result
        if result is None:
            return output, is_error
        return (FINISHED if result[0] == "success" else GAVE_UP), False

    def _done(self) -> bool:
        return self._task._result is not None

    def _on_stop(self, reply: Message) -> Message | str | None:
        if self._nudged < NUDGES:
            self._nudged += 1
            return Message(id=new_id(), role="user", parts=(Text(text=NUDGE),))
        return (
            f"the model stopped {self._nudged + 1} times without calling "
            "task.success or task.fail"
        )


class TaskError(Exception):
    """A task that didn't hand back a value; ``outcome`` says how it
    ended."""

    def __init__(self, name: str, outcome: Outcome) -> None:
        detail = f": {outcome.message}" if outcome.message else ""
        super().__init__(f"task {name!r} {outcome.status}{detail}")
        self.outcome = outcome


class TaskFailed(TaskError):
    """The task failed: ``task.fail``, an error, or a model that never
    finished."""


class TaskInterrupted(TaskError):
    """A provider error worth retrying cut the task short."""


def _unwrap(name: str, outcome: Outcome) -> Any:
    if outcome.status == "success":
        return outcome.value
    error = {"failed": TaskFailed, "interrupted": TaskInterrupted}.get(
        outcome.status, TaskError
    )
    raise error(name, outcome)


def _where(ws: Workspace) -> str | None:
    """Where ``ws``'s code runs, when it is not this process."""
    executor = ws.runtime.executor
    if not isinstance(executor, LocalExecutor):
        return f"on {type(executor).__name__}"
    isolation = Profile.of(ws).python.isolation
    return None if isolation == "none" else f"under isolation={isolation!r}"


def _refuse_elsewhere(name: str, ws: Workspace) -> None:
    where = _where(ws)
    if where is not None:
        raise NotSupportedError(
            f"task {name!r} can't run on this world: its code runs {where}, "
            "and task values travel only within this process so far"
        )


def _with_objects(profile: Profile, objects: Mapping[str, Any]) -> Profile:
    held = profile.python.host_objects
    clash = sorted(set(held) & set(objects))
    if clash:
        raise ValueError(
            f"the world already has host objects named {', '.join(clash)}, "
            "which the task binds too; rename the input or the type"
        )
    python = replace(profile.python, host_objects={**held, **objects})
    return replace(profile, python=python)


@dataclass
class _World:
    ws: Workspace
    close: Callable[[], None]
    ref: str | None


def _close_opened(opening: asyncio.Future[_World]) -> None:
    if not opening.cancelled() and opening.exception() is None:
        asyncio.get_running_loop().run_in_executor(None, opening.result().close)


class Task:
    """A task: ``fn``'s signature and docstring, run by ``agent``.

    Call it for the value (``await`` it, for an ``async def`` task):
    ``TaskFailed`` or ``TaskInterrupted`` when there is none. ``run`` and
    ``arun`` return the :class:`~agex.agent.Outcome` instead.

    ``world=`` runs the task on a fork of that workspace's last commit,
    with a fresh conversation; without it, on a scratch world in memory
    built from ``agent.profile``. ``keep=True`` keeps the fork, and the outcome's
    ``ref`` names it; otherwise it is deleted once the task ends.
    """

    def __init__(self, agent: Agent, fn: Callable[..., Any]) -> None:
        self.agent = agent
        self.spec = TaskSpec.of(fn)
        self._async = inspect.iscoroutinefunction(fn)
        functools.update_wrapper(self, fn)

    def __repr__(self) -> str:
        return f"<Task {self.spec.name} of {self.agent!r}>"

    def __call__(
        self, *args: Any, world: Workspace | None = None, **kwargs: Any
    ) -> Any:
        if self._async:
            return self._value(args, kwargs, world)
        return _block(self._value(args, kwargs, world), f"{self.spec.name}.arun")

    async def _value(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], world: Workspace | None
    ) -> Any:
        outcome = await self.arun(*args, world=world, **kwargs)
        return _unwrap(self.spec.name, outcome)

    def run(
        self,
        *args: Any,
        world: Workspace | None = None,
        keep: bool = False,
        **kwargs: Any,
    ) -> Outcome:
        """Run the task to its end; its outcome. From a coroutine,
        ``await arun``."""
        return _block(
            self.arun(*args, world=world, keep=keep, **kwargs), f"{self.spec.name}.arun"
        )

    async def arun(
        self,
        *args: Any,
        world: Workspace | None = None,
        keep: bool = False,
        **kwargs: Any,
    ) -> Outcome:
        """Run the task to its end; its outcome.

        A caller cancelled while waiting stops the task, which ends
        ``cancelled`` (a kept fork stores that run), and the cancellation
        goes on up.
        """
        if keep and world is None:
            raise ValueError(
                "keep= needs world=: a scratch world lives in memory, and is "
                "gone once the task ends"
            )
        inputs = self.spec.bind(args, kwargs)
        task = TaskObject(self.spec.returns)
        objects = {**self.spec.types, **inputs, "task": task}
        opening = asyncio.ensure_future(
            asyncio.to_thread(self._open, world, keep, objects)
        )
        try:
            opened = await asyncio.shield(opening)
        except asyncio.CancelledError:
            # the world opens on its thread regardless; close it once it has
            opening.add_done_callback(_close_opened)
            raise
        try:
            session = _TaskSession(self.agent, opened.ws, task)
            stream = session.stream(_brief(self.spec, inputs))
            try:
                ran = await stream.wait()
            except asyncio.CancelledError:
                session.cancel()
                await asyncio.wait({stream._start()})
                raise
        finally:
            await asyncio.to_thread(opened.close)
        return self._outcome(ran, task, opened.ref)

    def _open(
        self, world: Workspace | None, keep: bool, objects: Mapping[str, Any]
    ) -> _World:
        if world is None:
            profile = _with_objects(self.agent.profile or Profile(), objects)
            store = Store(memory=True)
            try:
                ws = store.open(self.spec.name, profile=profile)
            except BaseException:
                store.close()
                raise

            def discard() -> None:
                ws.close()
                store.close()

            try:
                _refuse_elsewhere(self.spec.name, ws)
            except BaseException:
                discard()
                raise
            return _World(ws, discard, None)

        _refuse_elsewhere(self.spec.name, world)
        store = world.store
        if store is None:
            raise NotSupportedError(
                f"task {self.spec.name!r} can't fork this world: a workspace "
                "opened from a Store can be forked for a task, and this one "
                "wasn't"
            )
        profile = _with_objects(Profile.of(world), objects)
        name = f"{world.session}.{self.spec.name}-{new_id()[:8]}"
        # from the last commit, so the caller's branch gains nothing,
        # not even a commit of the writes it has pending
        world.fork(name, at=world.head, inherit="fresh").close()
        try:
            fork = store.open(name, profile=profile)
        except BaseException:
            store.delete(name)
            raise

        def close() -> None:
            fork.close()
            if not keep:
                store.delete(name)

        return _World(fork, close, name if keep else None)

    def _outcome(self, ran: Outcome, task: TaskObject, ref: str | None) -> Outcome:
        result = task._result
        status: Status = ran.status
        value: Any = None
        message = ran.message
        if result is not None and result[0] == "success":
            status, value, message = "success", result[1], None
        elif result is not None:
            status, message = "failed", result[1]
        elif status == "completed":
            status, message = "failed", "the run ended without a result"
        return replace(ran, status=status, value=value, message=message, ref=ref)
