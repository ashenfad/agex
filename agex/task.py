"""Tasks: a typed call that runs an agent on a world of its own.

``@agent.task`` turns a function's signature and docstring into a task;
the docstring is the job, and the body stays empty. Calling the task
runs the agent on a fresh world, a scratch one built from
``agent.profile`` or a fork of ``world=``, and hands back the value the
agent's code passed to ``task.success``, built as the return type. The
caller's world is never touched.

    @agent.task
    def grade(responses: list[Response]) -> Report:
        \"""Grade the quiz.\"""

    report = grade(rs)                         # the value, or TaskFailed
    out = grade.run(rs, world=ws, keep=True)   # an Outcome; out.ref names the fork

Inside the world, agent code sees each argument bound by its own name,
the record and enum types the signature names, and ``task``:
``task.success(value)`` ends the task with its value,
``task.fail(reason)`` ends it without one, and
``task.needs_input(question)`` stops it to ask something only its
caller can settle. A value that doesn't fit is a ``TypeError`` at the
call, so the agent fixes it in the same script. A model that stops
without any of them is nudged to finish, twice; then the task fails.

A task that asks keeps its world, and carries on there with the answer:

    out = grade.run(rs, world=ws)                # out.status == "needs_input"
    out = out.resume("yes")                      # in this process
    out = grade.resume(ref, "yes", world=ws)     # by ref, after a restart too

The world's ``__task__`` plane (:data:`PLANE`) holds what the task is,
its inputs and how it stands, so a resume finds its inputs there; a
live input can't be stored, and is passed again.

A task runs on any world: in this process, under process isolation or
on a dud machine, the same way. Types are nontainer's
(:mod:`nontainer.values`):

- **Inputs are values.** Each is checked against its annotation and
  reaches the task's code by value, a fresh copy each run, so a task
  can't change what its caller passed. A live input (a client, a
  callable) is a capability: it is bound as a host object of the
  task's world as it is, under that world's host-object policy, and
  reached through a proxy where the world's code runs elsewhere. A
  value mixing the two (``dict[str, Client]``), which only in-process
  can carry, gets one copy of its data for the task, its live objects
  shared.
- **The value comes back built as the return type.** ``task`` is a
  stubbed host object (``nontainer.remote``): agent code holds
  :class:`agex.stubs.TaskStub`, and what it passes to ``task.success``
  is encoded and decoded by the return type, on every rung, so a value
  of the agent's own making arrives as the declared type or is refused.
- **Off in-process only data, bytes, tables and arrays cross.** A task
  whose return type has a live part, or an input that mixes data with a
  live part, is refused on such a world before any model call. Tables
  cross as Arrow, so they need pyarrow.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import functools
import inspect
import sys
import textwrap
import threading
import typing
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import reprobate
from nontainer import HostObject, NotSupportedError, Profile, Store, Workspace, values
from nontainer.executor import LocalExecutor
from nontainer.values import Kind, kinds_of
from pydantic import TypeAdapter

from .agent import Outcome, Session, Status, _block
from .record import Message, Text, new_id
from .stubs import TaskStub

if TYPE_CHECKING:
    from .agent import Agent

__all__ = [
    "NUDGES",
    "RESERVED",
    "Kind",
    "Task",
    "TaskError",
    "TaskFailed",
    "TaskHost",
    "TaskInterrupted",
    "TaskSpec",
    "NeedsInput",
    "PLANE",
    "ValueSpec",
    "kinds_of",
]

RESERVED = frozenset({"task", "host", "world", "keep"})
"""Names a task's parameters can't take: ``task`` and ``host`` are bound
in the world, and ``world`` and ``keep`` are the call's own."""

NUDGES = 2
"""How many times a model that stops without finishing is sent back to
finish, before the task fails."""

PREVIEW = 600
"""How many characters of each input the task's brief shows."""

METHODS = 20
"""How many public methods of a live input's class the brief lists."""

NUDGE = (
    "The task isn't finished. In run_python, call `task.success(value)` "
    "with the result, `task.fail(reason)` if it can't be done, or "
    "`task.needs_input(question)` to ask something only the person who "
    "gave it can settle."
)

FINISHED = "task.success: the task is done."
GAVE_UP = "task.fail: the task is over."
ASKED = "task.needs_input: the question is sent; the task waits for its answer."

ANSWER = "The answer to your question:\n{answer}"
"""The message a resumed task's run starts from."""

INSTRUCTIONS = """\
You are doing a task. Its inputs are bound by name in every run_python \
call, along with the types it names and `task`:
- `task.success(value)` hands back the result and ends the task. The value \
must have the task's return type; one that doesn't raises TypeError right \
there, so fix it and call again.
- `task.fail(reason)` ends the task without a result, saying why it can't \
be done.
- `task.needs_input(question)` asks the person who gave you the task \
something only they can settle (a choice, a permission, a missing fact). \
Their answer comes back as the next message, and you carry on from there.
Each call ends the script it is in: nothing after it runs."""

PLANE = "__task__/"
"""The task's own plane in its world, beside the conversation: what a
resume, or anyone reading the world later, finds there. ``spec`` says
what the task is and which inputs are stored, ``inputs/<name>`` holds
each input sent by value, encoded (:mod:`nontainer.values`), ``state``
how the task stands, and ``value`` the value it handed back, encoded,
when it can be. A live input or value is not stored. Written in the
commits of the task's own runs, so a world at any commit says how its
task stood there."""

_SPEC_KEY = PLANE + "spec"
_STATE_KEY = PLANE + "state"
_VALUE_KEY = PLANE + "value"
_INPUTS = PLANE + "inputs/"

_EMPTY = inspect.Parameter.empty

Names = Mapping[str, Any] | None
"""Names to resolve postponed annotations with, beside a type's own
module: those of the code that defined it, for a type defined in a
function."""


# -- one input or the return ---------------------------------------------------------------


def _schema(tp: Any, kinds: frozenset[Kind]) -> Mapping[str, Any] | None:
    if tp is _EMPTY or not kinds or not kinds <= {"data", "bytes"}:
        return None
    try:
        return TypeAdapter(tp).json_schema()
    except Exception:  # noqa: BLE001 - a record holding an opaque type, say
        return None


@dataclass(frozen=True)
class ValueSpec:
    """One input or the return of a task: its annotation, compiled with
    the names it needed (``spec``), the kinds of value it needs
    carried, and its JSON schema when it is plain data."""

    annotation: Any
    kinds: frozenset[Kind]
    spec: values.Spec = field(repr=False, compare=False)
    schema: Mapping[str, Any] | None = field(default=None, compare=False)

    @classmethod
    def of(cls, annotation: Any, *, names: Names = None) -> ValueSpec:
        """``annotation`` compiled; ``values.Unsupported`` for one no
        check can be built for."""
        spec = values.Spec.of(annotation, names=names)
        return cls(
            annotation=annotation,
            kinds=spec.kinds,
            spec=spec,
            schema=_schema(annotation, spec.kinds),
        )

    def check(self, value: Any, what: str) -> None:
        """Refuse ``value`` with a ``TypeError`` naming ``what``, unless
        it fits, strictly."""
        try:
            self.spec.check(value)
        except values.Mismatch as mismatch:
            raise TypeError(
                f"{what} must be {values.fmt(self.annotation)}: {mismatch}"
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
    def of(cls, fn: Callable[..., Any], *, names: Names = None) -> TaskSpec:
        """The spec of ``fn``. ``names`` resolve annotations that name
        types defined in a function (under postponed annotations); an
        annotation that still can't be resolved is refused, since a type
        the task can't see is one it can neither check nor bind."""
        name = fn.__name__
        _refuse_body(fn, name)
        try:
            return cls._of(fn, name, names)
        except NameError as error:
            raise TypeError(
                f"task {name!r} names a type that can't be resolved ({error}); "
                "define the types its annotations name before the task, or at "
                "module level"
            ) from None
        except values.Unsupported as error:
            raise TypeError(f"task {name!r}: {error}") from None

    @classmethod
    def _of(cls, fn: Callable[..., Any], name: str, names: Names) -> TaskSpec:
        signature = inspect.signature(fn)
        hints = typing.get_type_hints(
            fn, localns=dict(names) if names else None, include_extras=True
        )
        # what the hints still hold as text resolves with these
        names = {**getattr(fn, "__globals__", {}), **(names or {})}
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
            params[param.name] = ValueSpec.of(
                hints.get(param.name, param.annotation), names=names
            )
        returns = ValueSpec.of(
            hints.get("return", signature.return_annotation), names=names
        )
        found: dict[str, type] = {}
        for spec in (*params.values(), returns):
            for tp in spec.spec.types:
                other = found.setdefault(tp.__name__, tp)
                if other is not tp:
                    raise TypeError(
                        f"task {name!r} names two types called {tp.__name__!r} "
                        f"({_qualified(other)} and {_qualified(tp)}): agent "
                        "code sees types by name"
                    )
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
        """The inputs of a call, by name, each checked against its
        annotation: a ``TypeError`` for a call that doesn't fit."""
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(f"{self.name}(): {error}") from None
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            self.params[name].check(value, f"{self.name}()'s argument {name!r}")
        return dict(bound.arguments)


def _qualified(tp: type) -> str:
    return f"{tp.__module__}.{tp.__qualname__}"


# -- inputs as the task receives them ----------------------------------------------------


def _entry(spec: ValueSpec, value: Any) -> Any:
    """The host object entry for one input: data sent by value, a fresh
    copy each run on every rung, or a live object as it is. A value
    mixing the two (under a type with a live part, or a live object
    where the type allows anything), which only in-process can carry
    (:func:`_unsendable`), is one copy of its data for the task,
    sharing its live objects."""
    if spec.spec.travels and not (
        "any" in spec.kinds and values.find_live(value, full=True) is not None
    ):
        return HostObject(value, type=spec.spec)
    return _detach(value)


def _detach(value: Any) -> Any:
    """``value``'s data copied and its live objects shared: built-in
    containers, dataclasses and named tuples are rebuilt down to the
    live objects, and a part holding none is copied whole
    (:func:`nontainer.values.copy`). A live object, or a container of a
    class of its own that holds one, passes as it is."""
    if values.find_live(value, full=True) is None:
        return values.copy(value)
    kind = type(value)
    if kind is list:
        return [_detach(item) for item in value]
    if kind is tuple:
        return tuple(_detach(item) for item in value)
    if kind is dict:
        return {key: _detach(item) for key, item in value.items()}
    if kind is set or kind is frozenset:
        return kind(_detach(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return replace(
            value, **{f.name: _detach(getattr(value, f.name)) for f in fields if f.init}
        )
    if isinstance(value, tuple) and hasattr(kind, "_fields"):
        return kind._make(_detach(item) for item in value)  # type: ignore[attr-defined]
    return value


# -- the brief the model starts from -------------------------------------------------------


def _source(tp: type) -> str:
    try:
        return textwrap.dedent(inspect.getsource(tp)).strip()
    except (OSError, TypeError):
        hints = getattr(tp, "__annotations__", {})
        body = "\n".join(
            f"    {n}: {t if isinstance(t, str) else values.fmt(t)}"
            for n, t in hints.items()
        )
        return f"class {tp.__name__}:\n{body or '    ...'}"


def _live(value: Any, name: str | None, found: list[tuple[Any, str]]) -> None:
    """Add to ``found`` how agent code uses each live object in
    ``value``, an input the task is handed as it is, once each: a
    function (or a bound method) as itself, under ``name`` when it is
    the input itself; a class as itself; any other object as its class.
    What :func:`_detach` looks inside (built-in containers, named tuples
    and dataclasses) is looked inside, and pydantic models too. A part
    is live when it holds a live object; the input itself also when its
    class can't be sent by value (a class of its own acting as a
    mapping, say). Data is left out."""
    kind = type(value)
    if values.find_live(value, full=True) is None and (
        name is None or _sent_by_value(kind)
    ):
        return
    model_fields = getattr(kind, "model_fields", None)  # pydantic's
    if kind is dict:
        parts: Iterable[Any] = [*value.keys(), *value.values()]
    elif kind in (list, tuple, set, frozenset) or (
        isinstance(value, tuple) and hasattr(kind, "_fields")
    ):
        parts = value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        parts = [getattr(value, f.name) for f in dataclasses.fields(value)]
    elif isinstance(model_fields, Mapping):
        parts = [getattr(value, name) for name in model_fields]
    else:
        if inspect.isroutine(value) or isinstance(value, functools.partial):
            key = value
        else:
            key = value if isinstance(value, type) else kind
        if not any(k is key for k, _ in found):
            found.append((key, _usage(key, name)))
        return
    for part in parts:
        _live(part, None, found)


def _sent_by_value(kind: type) -> bool:
    try:
        return values.Spec.of(kind).travels
    except values.Unsupported:
        return False


def _usage(live: Any, name: str | None) -> str:
    if isinstance(live, type):
        return _interface(live)
    fn = live.func if isinstance(live, functools.partial) else live
    return _def(name or fn.__name__, live)


class _Text:
    """An annotation already written out, which ``inspect`` prints as
    it is."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _written(annotation: Any) -> Any:
    if annotation is inspect.Parameter.empty:
        return annotation
    return _Text(annotation if isinstance(annotation, str) else values.fmt(annotation))


def _parameters(fn: Any) -> str:
    """``fn``'s parameters and return, or ``(...)`` when it has no
    signature to read. No annotation is evaluated: one could run code,
    or name what only a type checker imports. One written as text stays
    as written, and one already evaluated is written as code writes it."""
    try:
        if sys.version_info >= (3, 14):
            from annotationlib import Format

            sig = inspect.signature(fn, annotation_format=Format.STRING)
        else:
            sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "(...)"
    params = [
        p.replace(annotation=_written(p.annotation)) for p in sig.parameters.values()
    ]
    return str(
        sig.replace(
            parameters=params, return_annotation=_written(sig.return_annotation)
        )
    )


def _summary(doc: str | None) -> str:
    """The first paragraph of a docstring."""
    return inspect.cleandoc(doc or "").split("\n\n")[0].strip()


def _def(name: str, fn: Any) -> str:
    """``fn`` written as a stub: its signature, and the first paragraph
    of its docstring."""
    summary = _summary(getattr(fn, "__doc__", None))
    body = "\n" + textwrap.indent(f'"""{summary}"""', "    ") if summary else " ..."
    return f"def {name}{_parameters(fn)}:{body}"


def _interface(tp: type) -> str:
    """``tp`` as agent code uses a live object of it: its name, the
    first paragraph of its docstring, and its public methods (the
    class's own first, in the order it defines them, then those it
    inherits), up to :data:`METHODS` of them."""
    methods: dict[str, str] = {}
    for klass in tp.__mro__:
        if klass is object:
            continue
        for name, raw in vars(klass).items():
            if name.startswith("_") or name in methods:
                continue
            if isinstance(raw, (property, type)) or not (
                callable(raw) or isinstance(raw, (staticmethod, classmethod))
            ):
                continue  # a data attribute, or a class, which isn't a call
            fn, decorator = raw, ""
            if isinstance(raw, (staticmethod, classmethod)):
                fn, decorator = raw.__func__, f"@{type(raw).__name__}\n"
            methods[name] = decorator + _def(name, fn)
    blocks = list(methods.values())[:METHODS]
    if len(methods) > len(blocks):
        blocks.append(f"# and {len(methods) - len(blocks)} more public methods")
    doc = _summary(tp.__doc__)
    if doc:
        blocks.insert(0, f'"""{doc}"""')
    body = "\n\n".join(blocks) or "..."
    return f"class {tp.__name__}:\n" + textwrap.indent(body, "    ")


def _brief(spec: TaskSpec, inputs: Mapping[str, Any], live: Mapping[str, Any]) -> str:
    """The task's opening message: the job, the inputs, how to finish,
    the types it names, and how to use the inputs in ``live``, those it
    is handed as they are."""
    parts = [spec.instructions or f"Do the task {spec.name!r}."]
    if inputs:
        lines = [
            f"- `{name}: {values.fmt(spec.params[name].annotation)}` = "
            f"{reprobate.render(value, budget=PREVIEW)}"
            for name, value in inputs.items()
        ]
        parts.append("Inputs, bound by name:\n" + "\n".join(lines))
    returns = spec.returns.annotation
    if returns is None or returns is type(None):
        parts.append("When it's done, call `task.success()`.")
    else:
        parts.append(
            f"When it's done, call `task.success(value)` with a `{values.fmt(returns)}`."
        )
    if spec.types:
        sources = "\n\n".join(_source(tp) for tp in spec.types.values())
        parts.append(
            "The types, already bound by name; build values from these:"
            f"\n```python\n{sources}\n```"
        )
    found: list[tuple[Any, str]] = []
    for name, value in live.items():
        _live(value, name, found)
    if found:
        usage = "\n\n".join(text for _, text in found)
        parts.append(
            "The live inputs are objects on the host: use them as listed here, "
            f"and don't build new ones.\n```python\n{usage}\n```"
        )
    return "\n\n".join(parts)


# -- the run ------------------------------------------------------------------------------


class TaskHost:
    """The host half of ``task``: what :class:`~agex.stubs.TaskStub`'s
    calls reach, wherever the world's code runs, typed by the task's
    return. It records the result; the loop reads the record, never the
    script's error, so a script that catches the stop still ends the
    task."""

    def __init__(self, returns: values.Spec) -> None:
        self._returns = returns
        self._result: tuple[Literal["success", "failed", "needs_input"], Any] | None = (
            None
        )

        def success(value: Any) -> None:
            self._result = ("success", value)

        # nontainer types the call by these: the return type, compiled
        # with the names it needed, so the value arrives built as it
        success.__annotations__ = {"value": returns, "return": None}
        self.success = success

    def __repr__(self) -> str:
        return "<task host>"

    def fail(self, reason: str) -> None:
        self._result = ("failed", reason)

    def needs_input(self, question: str) -> None:
        self._result = ("needs_input", question)


class _TaskSession(Session):
    """A session over a task's world: it ends when ``task`` records a
    result, and nudges a model that stops without one."""

    def __init__(self, agent: Agent, ws: Workspace, task: TaskHost) -> None:
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
        # the turn commits at its end: the plane says how the task stands
        # in the commit of the run that settled it (the task's own world,
        # which its loop alone writes)
        _settle(self.ws.provider.kv, result, self._task._returns)
        return _SAID[result[0]], False

    def _done(self) -> bool:
        return self._task._result is not None

    def _on_stop(self, reply: Message) -> Message | str | None:
        if self._nudged < NUDGES:
            self._nudged += 1
            return Message(id=new_id(), role="user", parts=(Text(text=NUDGE),))
        return (
            f"the model stopped {self._nudged + 1} times without calling "
            "task.success, task.fail or task.needs_input"
        )


_SAID = {"success": FINISHED, "failed": GAVE_UP, "needs_input": ASKED}
"""What the model reads as the result of the call that settled its task."""


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


class NeedsInput(TaskError):
    """The task asked a question (``task.needs_input``) and waits for its
    answer: ``outcome.resume(answer)``, or the task's ``resume`` with
    ``ref``."""

    @property
    def question(self) -> str:
        return self.outcome.message or ""

    @property
    def ref(self) -> str | None:
        return self.outcome.ref


def _unwrap(name: str, outcome: Outcome) -> Any:
    if outcome.status == "success":
        return outcome.value
    error = {
        "failed": TaskFailed,
        "interrupted": TaskInterrupted,
        "needs_input": NeedsInput,
    }.get(outcome.status, TaskError)
    raise error(name, outcome)


def _where(executor: Any, isolation: str) -> str | None:
    """Where code runs on ``executor`` (``None`` for the default, a
    ``LocalExecutor``) under ``isolation``, when it is not this
    process."""
    if executor is not None and not isinstance(executor, LocalExecutor):
        return f"on {type(executor).__name__}"
    return None if isolation == "none" else f"under isolation={isolation!r}"


def _unsendable(spec: TaskSpec, entries: Mapping[str, Any]) -> str | None:
    """Why the task can't run where only data, bytes, tables and arrays
    cross, or ``None`` when it can: a return type with a live part, or
    an input passed as it is that isn't a live object as a whole (which
    a proxy reaches)."""
    if not spec.returns.spec.travels:
        return (
            f"it returns {values.fmt(spec.returns.annotation)}, which has a live part"
        )
    for name, entry in entries.items():
        param = spec.params[name]
        if isinstance(entry, HostObject) or param.kinds == {"live"}:
            continue
        if param.spec.travels:
            return (
                f"its argument {name!r} holds a live object where its type "
                f"allows anything ({values.find_live(entry)})"
            )
        return (
            f"its argument {name!r} is {values.fmt(param.annotation)}, which "
            "mixes data with a live part"
        )
    return None


def _refuse_unsendable(
    spec: TaskSpec, entries: Mapping[str, Any], where: str | None
) -> None:
    why = _unsendable(spec, entries) if where is not None else None
    if why is not None:
        raise NotSupportedError(
            f"task {spec.name!r} can't run on this world: its code runs {where}, "
            f"where only data, bytes, tables and arrays cross, and {why}"
        )


def _with_objects(
    profile: Profile, objects: Mapping[str, Any], classes: tuple[type, ...]
) -> Profile:
    python = profile.python
    held = python.host_objects
    clash = sorted(set(held) & set(objects))
    if clash:
        raise ValueError(
            f"the world already has host objects named {', '.join(clash)}, "
            "which the task binds too; rename the input or the type"
        )
    have = {klass.__name__: klass for klass in python.classes}
    more = tuple(klass for klass in classes if have.get(klass.__name__) is not klass)
    twice = sorted(klass.__name__ for klass in more if klass.__name__ in have)
    if twice:
        raise ValueError(
            f"the world already has a class named {twice[0]!r}, a different "
            "one from the task's; rename one of them"
        )
    python = replace(
        python,
        host_objects={**held, **objects},
        classes=(*python.classes, *more),
    )
    return replace(profile, python=python)


# -- the plane -------------------------------------------------------------------------


def _keys(kv: Any, prefix: str) -> list[str]:
    return [k for k in list(kv.keys()) if isinstance(k, str) and k.startswith(prefix)]


def _signature(name: str, params: Mapping[str, str], returns: str) -> str:
    listed = ", ".join(f"{p}: {t}" for p, t in params.items())
    return f"{name}({listed}) -> {returns}"


def _contract(spec: TaskSpec) -> str:
    """What a task takes and returns, written as its signature: what a
    resume checks a world's task against."""
    return _signature(
        spec.name,
        {n: values.fmt(p.annotation) for n, p in spec.params.items()},
        values.fmt(spec.returns.annotation),
    )


def _begin(
    kv: Any, spec: TaskSpec, entries: Mapping[str, Any], origin: str | None
) -> None:
    """Start the plane for a task about to run: what it is, the world it
    was forked from (``origin``, ``None`` for a scratch world), and each
    input sent by value, encoded. What the world held there before (a
    task's world forked for another task) goes."""
    for key in _keys(kv, PLANE):
        del kv[key]

    def described(value: ValueSpec) -> dict[str, Any]:
        said: dict[str, Any] = {"type": values.fmt(value.annotation)}
        if value.schema is not None:
            said["schema"] = dict(value.schema)
        return said

    params: dict[str, Any] = {}
    for name, param in spec.params.items():
        entry = entries[name]
        stored = isinstance(entry, HostObject)
        if stored:
            kv[_INPUTS + name] = values.encode(entry.obj).to_bytes()
        params[name] = {**described(param), "stored": stored}
    kv[_SPEC_KEY] = {
        "format": 1,
        "name": spec.name,
        "world": origin,
        "instructions": spec.instructions,
        "params": params,
        "returns": described(spec.returns),
    }
    kv[_STATE_KEY] = {"status": "running"}


def _settle(kv: Any, result: tuple[str, Any], returns: values.Spec) -> None:
    """Record how the task stands once its code has called ``task``."""
    status, detail = result
    state: dict[str, Any] = {"status": status}
    if status == "needs_input":
        state["question"] = detail
    elif status == "failed":
        state["reason"] = detail
    kv[_STATE_KEY] = state
    blob = None
    if status == "success" and returns.travels:
        try:
            blob = values.encode(detail).to_bytes()
        except values.Unencodable:
            pass  # a live object where the type allows anything: not stored
    if blob is not None:
        kv[_VALUE_KEY] = blob
    elif kv.get(_VALUE_KEY) is not None:
        del kv[_VALUE_KEY]


def _read(store: Store, ref: str) -> dict[str, Any]:
    """What the plane of the task world ``ref`` holds."""
    if not store.exists(ref):
        raise LookupError(f"no task world {ref!r} in this store")
    ws = store.open(ref)
    try:
        kv = ws.provider.kv
        return {
            "spec": kv.get(_SPEC_KEY),
            "state": kv.get(_STATE_KEY),
            "inputs": {k[len(_INPUTS) :]: kv[k] for k in _keys(kv, _INPUTS)},
        }
    finally:
        ws.close()


# -- worlds -------------------------------------------------------------------------


@dataclass
class _World:
    """A task's world, open: its workspace, its name in its store, and
    how to put it away, kept or not. ``keep_unsettled`` is whether to
    keep it when its task doesn't settle (its caller gone before it
    ran, or its run cut short): a world kept for a resume stays kept
    until its task settles, so a resume cut short can be made again."""

    ws: Workspace
    name: str
    put_away: Callable[[bool], None]
    keep_unsettled: bool = False


_WORLDS = ThreadPoolExecutor(thread_name_prefix="agex-task-world")
"""Where a task's world is opened and closed. Its own pool, so the
cleanup of a world nobody waits for any more hangs on this pool's
future, which settles in its thread whatever becomes of the event loop
(a loop shut down after a cancel would cancel an asyncio-side callback
before it ran)."""

_KEPT: dict[str, tuple[Store, Profile]] = {}
"""Scratch worlds kept for a resume, by ref: the memory store each lives
in, and the profile it was built from. In this process only, which is
why a scratch world resumes only here. One resumed to an end is closed
and dropped; one never resumed lasts as long as the process."""

_KEPT_LOCK = threading.Lock()


def _close_abandoned(opening: Future[_World]) -> None:
    if not opening.cancelled() and opening.exception() is None:
        opened = opening.result()
        opened.put_away(opened.keep_unsettled)


def _store_of(name: str, world: Workspace) -> Store:
    store = world.store
    if store is None:
        raise NotSupportedError(
            f"task {name!r} can't fork this world: a workspace opened from a "
            "Store can be forked for a task, and this one wasn't"
        )
    return store


Resumer = Callable[..., Awaitable[Outcome]]


class Task:
    """A task: ``fn``'s signature and docstring, run by ``agent``.

    Call it for the value (``await`` it, for an ``async def`` task):
    ``TaskFailed``, ``NeedsInput`` or ``TaskInterrupted`` when there is
    none. ``run`` and ``arun`` return the :class:`~agex.agent.Outcome`
    instead.

    ``world=`` runs the task on a fork of that workspace's last commit,
    with a fresh conversation; without it, on a scratch world in memory
    built from ``agent.profile``. ``keep=True`` keeps the fork, and the
    outcome's ``ref`` names it; otherwise it is deleted once the task
    ends.

    A task that asks for input (``needs_input``) keeps its world
    whatever ``keep`` says, and ``resume`` continues it there with the
    answer: by ``ref``, through the same ``world=`` for one kept in a
    store (after a restart, too, on a store that persists), or in this
    process for a scratch world. Its inputs come back from the world,
    except a live one, which can't be stored and is passed again.
    """

    def __init__(
        self, agent: Agent, fn: Callable[..., Any], *, names: Names = None
    ) -> None:
        self.agent = agent
        self.spec = TaskSpec.of(fn, names=names)
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
        entries = {
            name: _entry(self.spec.params[name], v) for name, v in inputs.items()
        }
        live = {
            n: v for n, v in inputs.items() if not isinstance(entries[n], HostObject)
        }
        return await self._drive(
            lambda task: self._open(world, entries, task),
            _brief(self.spec, inputs, live),
            keep=keep,
            again=self._again(world, keep, live),
        )

    def resume(
        self,
        ref: str,
        answer: Any,
        /,
        *,
        world: Workspace | None = None,
        keep: bool = False,
        **live_inputs: Any,
    ) -> Outcome:
        """Continue the task at ``ref``, which asked for input, with
        ``answer``; its outcome. From a coroutine, ``await aresume``."""
        return _block(
            self.aresume(ref, answer, world=world, keep=keep, **live_inputs),
            f"{self.spec.name}.aresume",
        )

    async def aresume(
        self,
        ref: str,
        answer: Any,
        /,
        *,
        world: Workspace | None = None,
        keep: bool = False,
        **live_inputs: Any,
    ) -> Outcome:
        """Continue the task at ``ref``, which asked for input, with
        ``answer`` as the next message its model reads; its outcome.

        ``world=`` is the world the task ran on, whose store keeps
        ``ref``; without it, ``ref`` is a scratch world kept in this
        process. The inputs come back from the task's world, and a live
        one, which can't be stored, is passed again by name. ``keep``
        is as for ``run``: a task that asks again is kept regardless.

        The world's task must be this one, by name, inputs and return,
        and ``world=`` the world it ran on, by session. That world's
        settings are not stored with the task, so it should be opened as
        it was: host objects, isolation and executor are taken from it.
        """
        if keep and world is None:
            raise ValueError(
                "keep= needs world=: a scratch world lives in memory, and is "
                "gone once the task ends"
            )
        if world is None:
            with _KEPT_LOCK:
                kept = _KEPT.get(ref)
            if kept is None:
                raise LookupError(
                    f"no scratch world {ref!r} is kept in this process: a scratch "
                    "world resumes only in the process that ran it, and one kept "
                    "in a store resumes through the world it ran on (world=)"
                )
            store, base = kept
        else:
            store, base = _store_of(self.spec.name, world), Profile.of(world)
        plane = await asyncio.wrap_future(_WORLDS.submit(_read, store, ref))
        self._waiting(ref, plane, world)
        inputs = self._restore(plane["inputs"], live_inputs)
        entries = {
            name: _entry(self.spec.params[name], v) for name, v in inputs.items()
        }
        live = {
            n: v for n, v in inputs.items() if not isinstance(entries[n], HostObject)
        }
        text = (
            answer
            if isinstance(answer, str)
            else reprobate.render(answer, budget=PREVIEW)
        )
        return await self._drive(
            lambda task: self._reopen(ref, store, world, base, entries, task),
            ANSWER.format(answer=text),
            keep=keep,
            again=self._again(world, keep, live),
        )

    def _again(
        self, world: Workspace | None, keep: bool, live: Mapping[str, Any]
    ) -> Callable[[str], Resumer]:
        """How an outcome that asks for input resumes in this process:
        the same world and ``keep``, the same live inputs unless others
        are passed."""

        def at(ref: str) -> Resumer:
            async def again(answer: Any, /, **more: Any) -> Outcome:
                return await self.aresume(
                    ref, answer, world=world, keep=keep, **{**live, **more}
                )

            return again

        return at

    async def _drive(
        self,
        open_world: Callable[[TaskHost], _World],
        prompt: str,
        *,
        keep: bool,
        again: Callable[[str], Resumer],
    ) -> Outcome:
        """Open the task's world, run the model there from ``prompt`` to
        an end, and put the world away: kept if ``keep``, or if the task
        asked for input."""
        task = TaskHost(self.spec.returns.spec)
        opening = _WORLDS.submit(open_world, task)
        try:
            opened = await asyncio.wrap_future(opening)
        except BaseException:
            # a world already opening finishes on its thread: close it then
            opening.add_done_callback(_close_abandoned)
            raise
        finished = False
        try:
            session = _TaskSession(self.agent, opened.ws, task)
            stream = session.stream(prompt)
            try:
                ran = await stream.wait()
            except asyncio.CancelledError:
                session.cancel()
                await asyncio.wait({stream._start()})
                raise
            finished = True
        finally:
            result = task._result
            asked = result is not None and result[0] == "needs_input"
            if finished:
                # still waiting: it asked, or a resume of it was cut short
                waiting = asked or (result is None and opened.keep_unsettled)
            else:
                # the caller is gone, and learns no ref: only a world whose
                # ref it already holds (a resume's) is kept, unsettled
                waiting = opened.keep_unsettled and (result is None or asked)
            kept = keep or waiting
            await asyncio.wrap_future(_WORLDS.submit(opened.put_away, kept))
        outcome = self._outcome(ran, task, opened.name if kept else None)
        if waiting:
            outcome = replace(outcome, _resume=again(opened.name))
        return outcome

    def _profile(
        self, base: Profile, executor: Any, entries: Mapping[str, Any], task: TaskHost
    ) -> Profile:
        """The task world's profile: ``base`` with the task's objects and
        types, refused first when the task can't run where its code
        would."""
        _refuse_unsendable(self.spec, entries, _where(executor, base.python.isolation))
        objects = {**entries, "task": HostObject(task, stub=TaskStub)}
        return _with_objects(base, objects, tuple(self.spec.types.values()))

    def _scratch_profile(
        self, base: Profile, entries: Mapping[str, Any], task: TaskHost
    ) -> Profile:
        # the executor is made here, to see where its code runs before
        # the world opens (opening refuses what can't cross, less clearly)
        made = base.executor_factory() if base.executor_factory else None
        profile = self._profile(base, made, entries, task)
        if made is not None:
            profile = replace(profile, executor_factory=lambda: made)
        return profile

    def _open(
        self, world: Workspace | None, entries: Mapping[str, Any], task: TaskHost
    ) -> _World:
        if world is None:
            base = self.agent.profile or Profile()
            profile = self._scratch_profile(base, entries, task)
            name = f"{self.spec.name}-{new_id()[:8]}"
            store = Store(memory=True)
            try:
                ws = store.open(name, profile=profile)
            except BaseException:
                store.close()
                raise
            try:
                _begin(ws.provider.kv, self.spec, entries, None)
            except BaseException:
                ws.close()
                store.close()
                raise

            def put_away(kept: bool) -> None:
                ws.close()
                if kept:
                    with _KEPT_LOCK:
                        _KEPT[name] = (store, base)
                else:
                    store.close()

            return _World(ws, name, put_away)

        profile = self._profile(
            Profile.of(world), world.runtime.executor, entries, task
        )
        store = _store_of(self.spec.name, world)
        name = f"{world.session}.{self.spec.name}-{new_id()[:8]}"
        # from the last commit, so the caller's branch gains nothing,
        # not even a commit of the writes it has pending
        world.fork(name, at=world.head, inherit="fresh").close()
        try:
            fork = store.open(name, profile=profile)
        except BaseException:
            store.delete(name)
            raise
        try:
            _begin(fork.provider.kv, self.spec, entries, world.session)
        except BaseException:
            fork.close()
            store.delete(name)
            raise

        def close(kept: bool) -> None:
            fork.close()
            if not kept:
                store.delete(name)

        return _World(fork, name, close)

    def _reopen(
        self,
        ref: str,
        store: Store,
        world: Workspace | None,
        base: Profile,
        entries: Mapping[str, Any],
        task: TaskHost,
    ) -> _World:
        if world is None:
            profile = self._scratch_profile(base, entries, task)
        else:
            profile = self._profile(base, world.runtime.executor, entries, task)
        ws = store.open(ref, profile=profile)

        def put_away(kept: bool) -> None:
            ws.close()
            if kept:
                return
            if world is None:
                with _KEPT_LOCK:
                    _KEPT.pop(ref, None)
                store.close()
            else:
                store.delete(ref)

        return _World(ws, ref, put_away, keep_unsettled=True)

    def _waiting(
        self, ref: str, plane: Mapping[str, Any], world: Workspace | None
    ) -> None:
        """Refuse a world that isn't this task's (the same name, inputs
        and return), reached through the world it ran on, and waiting
        for an answer."""
        spec, state = plane["spec"], plane["state"]
        if not isinstance(spec, Mapping):
            raise ValueError(f"{ref!r} isn't the world of a task")
        params = spec.get("params") or {}
        held = _signature(
            str(spec.get("name")),
            {n: str(p.get("type")) for n, p in params.items()},
            str((spec.get("returns") or {}).get("type")),
        )
        own = _contract(self.spec)
        if held != own:
            raise ValueError(
                f"{ref!r} isn't a world of this task: it holds {held}, and this "
                f"task is {own}"
            )
        origin = spec.get("world")
        if world is not None and origin != world.session:
            raise ValueError(
                f"task {self.spec.name!r} at {ref!r} ran on world {origin!r}, "
                f"not {world.session!r}: resume it through the world it ran on"
            )
        status = state.get("status") if isinstance(state, Mapping) else None
        if status != "needs_input":
            raise ValueError(
                f"task {self.spec.name!r} at {ref!r} isn't waiting for an answer: "
                f"it is {status or 'in no known state'}"
            )

    def _restore(
        self, stored: Mapping[str, Any], supplied: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The inputs to resume with: each stored one decoded, and each
        other passed again, checked."""
        name = self.spec.name
        unknown = sorted(set(supplied) - set(self.spec.params))
        if unknown:
            raise TypeError(
                f"{name}.resume() got inputs the task doesn't take: {', '.join(unknown)}"
            )
        inputs: dict[str, Any] = {}
        for param_name, param in self.spec.params.items():
            blob = stored.get(param_name)
            if blob is not None:
                if param_name in supplied:
                    raise TypeError(
                        f"{name}.resume(): {param_name!r} is stored with the task; "
                        "only an input that can't be stored is passed again"
                    )
                try:
                    inputs[param_name] = param.spec.decode(blob)
                except values.Mismatch as mismatch:
                    raise TypeError(
                        f"{name}()'s stored input {param_name!r} no longer fits "
                        f"{values.fmt(param.annotation)}: {mismatch}"
                    ) from None
            elif param_name in supplied:
                value = supplied[param_name]
                param.check(value, f"{name}()'s argument {param_name!r}")
                inputs[param_name] = value
            else:
                raise TypeError(
                    f"resuming task {name!r} needs its input {param_name!r} again: "
                    f"it isn't stored (a live object isn't), so pass it as "
                    f"{param_name}=..."
                )
        return inputs

    def _outcome(self, ran: Outcome, task: TaskHost, ref: str | None) -> Outcome:
        result = task._result
        status: Status = ran.status
        value: Any = None
        message = ran.message
        if result is not None and result[0] == "success":
            status, value, message = "success", result[1], None
        elif result is not None and result[0] == "needs_input":
            status, message = "needs_input", result[1]
        elif result is not None:
            status, message = "failed", result[1]
        elif status == "completed":
            status, message = "failed", "the run ended without a result"
        return replace(ran, status=status, value=value, message=message, ref=ref)
