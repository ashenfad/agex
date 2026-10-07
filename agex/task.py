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
``task.success(value)`` ends the task with its value, and
``task.fail(reason)`` ends it without one. A value that doesn't fit is a
``TypeError`` at the call, so the agent fixes it in the same script. A
model that stops without either call is nudged to finish, twice; then
the task fails.

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
import textwrap
import typing
from collections.abc import Callable, Mapping
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


def _brief(spec: TaskSpec, inputs: Mapping[str, Any]) -> str:
    """The task's opening message: the job, the inputs, how to finish,
    and the types it names."""
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
    return "\n\n".join(parts)


# -- the run ------------------------------------------------------------------------------


class TaskHost:
    """The host half of ``task``: what :class:`~agex.stubs.TaskStub`'s
    calls reach, wherever the world's code runs, typed by the task's
    return. It records the result; the loop reads the record, never the
    script's error, so a script that catches the stop still ends the
    task."""

    def __init__(self, returns: values.Spec) -> None:
        self._result: tuple[Literal["success", "failed"], Any] | None = None

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


@dataclass
class _World:
    ws: Workspace
    close: Callable[[], None]
    ref: str | None


_WORLDS = ThreadPoolExecutor(thread_name_prefix="agex-task-world")
"""Where a task's world is opened and closed. Its own pool, so the
cleanup of a world nobody waits for any more hangs on this pool's
future, which settles in its thread whatever becomes of the event loop
(a loop shut down after a cancel would cancel an asyncio-side callback
before it ran)."""


def _close_abandoned(opening: Future[_World]) -> None:
    if not opening.cancelled() and opening.exception() is None:
        opening.result().close()


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
        task = TaskHost(self.spec.returns.spec)
        opening = _WORLDS.submit(self._open, world, keep, entries, task)
        try:
            opened = await asyncio.wrap_future(opening)
        except BaseException:
            # a world already opening finishes on its thread: close it then
            opening.add_done_callback(_close_abandoned)
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
            await asyncio.wrap_future(_WORLDS.submit(opened.close))
        return self._outcome(ran, task, opened.ref)

    def _open(
        self,
        world: Workspace | None,
        keep: bool,
        entries: Mapping[str, Any],
        task: TaskHost,
    ) -> _World:
        objects = {**entries, "task": HostObject(task, stub=TaskStub)}
        classes = tuple(self.spec.types.values())
        if world is None:
            profile = self.agent.profile or Profile()
            # made here, to see where its code runs before the world opens
            # (opening refuses what can't cross, less clearly than this)
            made = profile.executor_factory() if profile.executor_factory else None
            _refuse_unsendable(
                self.spec, entries, _where(made, profile.python.isolation)
            )
            if made is not None:
                profile = replace(profile, executor_factory=lambda: made)
            profile = _with_objects(profile, objects, classes)
            store = Store(memory=True)
            try:
                ws = store.open(self.spec.name, profile=profile)
            except BaseException:
                store.close()
                raise

            def discard() -> None:
                ws.close()
                store.close()

            return _World(ws, discard, None)

        _refuse_unsendable(
            self.spec,
            entries,
            _where(world.runtime.executor, Profile.of(world).python.isolation),
        )
        store = world.store
        if store is None:
            raise NotSupportedError(
                f"task {self.spec.name!r} can't fork this world: a workspace "
                "opened from a Store can be forked for a task, and this one "
                "wasn't"
            )
        profile = _with_objects(Profile.of(world), objects, classes)
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

    def _outcome(self, ran: Outcome, task: TaskHost, ref: str | None) -> Outcome:
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
