"""The shape corpus's format: a task, the calls made to it, and what must
come of them.

The shape corpus pins agex's task API the way nontainer's harness
corpus pins the loop's contract, and in the same format: frozen
dataclasses with builders, exported as one JSON file per scenario (by
``agex.conformance.export``) for agex-ts to run. It extends that format
with what tasks need, and reuses its model steps (``says``, ``fails``,
``calls``) and its world.

**A task is written as JSON Schemas**, so another language can build
it: each parameter's type, the return type, and the record and live
classes they name, under ``defs``. Kinds JSON has no type for are marked
``"x-kind"``: ``"table"``, ``"array"`` and ``"bytes"``, and ``"live"``
for a class whose methods agent code calls on the host (each with
canned replies, and its calls recorded).

**The model's script is neutral.** A task finishes in agent code, so a
scenario can't avoid code; rather than write it in one language, it
says what the code does (:class:`TaskSuccess`, :class:`TaskFail`,
:class:`TaskNeedsInput`), and each harness writes that in its own
language as one call of its code tool.

**Values are JSON**, read by the type they are given as: a record is an
object, a table its columns (``{"name": ["ada", "bo"]}``), an array its
nested lists, bytes base64 text. A live input is an object of its class
whatever value is given (``{}`` by convention).

**Where it runs.** A harness runs a scenario on each of its rungs that
``where`` allows: in the process, or off it (a worker process, a
machine). The values a task takes and returns must come back the same
on every rung; what only the process can carry is refused off it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nontainer.conformance.corpus import ModelStep, World

__all__ = [
    "CAPABILITIES",
    "Act",
    "Expr",
    "Input",
    "LiveCall",
    "MethodCall",
    "OutcomeExp",
    "Param",
    "PlaneExp",
    "Restart",
    "Step",
    "TaskCall",
    "TaskDef",
    "TaskFail",
    "TaskNeedsInput",
    "TaskResume",
    "TaskScenario",
    "TaskSuccess",
    "Value",
    "call",
    "failure",
    "question",
    "refusal",
    "restart",
    "resume",
    "success",
    "task_fail",
    "task_needs_input",
    "task_success",
]

CAPABILITIES: dict[str, str] = {
    "tables": "carries tables (Arrow), as a task's inputs and its value",
    "arrays": "carries arrays (numpy's), as a task's inputs and its value",
}
"""What a scenario may need beyond the base API, by name. A harness
declares the ones it has; a scenario that needs one it lacks does not
apply to it."""

Where = Literal["anywhere", "in-process", "off-in-process"]
Status = Literal["success", "failed", "needs_input", "interrupted", "refused"]


# -- the task -------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Param:
    """One input of the task: its name and its type, as a JSON Schema."""

    name: str
    type: Any


@dataclass(frozen=True, kw_only=True)
class TaskDef:
    """The task: its name, its instructions (a docstring in Python), its
    parameters in order, its return type (``{"type": "null"}`` for
    none), and in ``defs`` the classes those name, each by
    ``{"$ref": "#/$defs/Name"}`` as JSON Schema refers to its ``$defs``.

    A record class is an object schema; its fields are its ``required``
    properties, in that order. A live class is
    ``{"x-kind": "live", "description": ..., "methods": {name: {"params":
    [{"name": ..., "type": ...}], "returns": ..., "replies": [{"args":
    [...], "returns": ...}]}}}``: a method answers the call whose
    arguments match a reply's, and agent code's calls are recorded."""

    name: str
    instructions: str
    params: tuple[Param, ...] = ()
    returns: Any = field(default_factory=lambda: {"type": "null"})
    defs: dict[str, Any] = field(default_factory=dict)


# -- the model's script ---------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Value:
    """A value written out, as JSON, read by the type it is given as."""

    kind: Literal["value"] = "value"
    value: Any


@dataclass(frozen=True, kw_only=True)
class Input:
    """One of the task's inputs, as agent code sees it."""

    kind: Literal["input"] = "input"
    name: str


@dataclass(frozen=True, kw_only=True)
class MethodCall:
    """What a method of a live input returns, called with ``args``."""

    kind: Literal["method_call"] = "method_call"
    input: str
    method: str
    args: tuple[Any, ...] = ()


Expr = Value | Input | MethodCall


@dataclass(frozen=True, kw_only=True)
class TaskSuccess:
    """Agent code calls ``task.success`` with ``value`` (``None``: with
    no value)."""

    kind: Literal["task_success"] = "task_success"
    value: Expr | None = None


@dataclass(frozen=True, kw_only=True)
class TaskFail:
    """Agent code calls ``task.fail(reason)``."""

    kind: Literal["task_fail"] = "task_fail"
    reason: str


@dataclass(frozen=True, kw_only=True)
class TaskNeedsInput:
    """Agent code calls ``task.needs_input(question)``."""

    kind: Literal["task_needs_input"] = "task_needs_input"
    question: str


Step = ModelStep | TaskSuccess | TaskFail | TaskNeedsInput
"""One reply of the model: a task step is one call of the harness's code
tool, written in its language."""


# -- the acts -------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TaskCall:
    """A call of the task with ``inputs`` (JSON by parameter name), on a
    scratch world, or with ``world`` on a fork of the scenario's world;
    ``keep`` keeps that fork when the task is done."""

    kind: Literal["call"] = "call"
    inputs: dict[str, Any] = field(default_factory=dict)
    steps: tuple[Step, ...]
    world: bool = False
    keep: bool = False


@dataclass(frozen=True, kw_only=True)
class TaskResume:
    """The answer to the last outcome's question. ``by="outcome"``
    resumes through the outcome itself, in this process;
    ``by="ref"`` resumes the task by its world's ref (through the
    scenario's world when the call was on a fork of it), passing again
    the live inputs ``live`` names, a new object of each, and ``keep``
    keeps that world when the task is done. A resume by outcome keeps
    what its call kept."""

    kind: Literal["resume"] = "resume"
    answer: str
    steps: tuple[Step, ...]
    by: Literal["outcome", "ref"] = "outcome"
    live: tuple[str, ...] = ()
    keep: bool = False


@dataclass(frozen=True, kw_only=True)
class Restart:
    """A restart, as far as the task can tell: the scenario's store is
    opened again and the task defined again, and the outcomes held so
    far are gone (their refs are still known)."""

    kind: Literal["restart"] = "restart"


Act = TaskCall | TaskResume | Restart


# -- expectations ---------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class LiveCall:
    """A call agent code made on a live input."""

    input: str
    method: str
    args: tuple[Any, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlaneExp:
    """The ``__task__`` plane of the world an outcome keeps: the status
    of its state, the inputs stored (by name, sorted), and whether the
    value is."""

    status: Literal["running", "success", "failed", "needs_input"]
    stored: tuple[str, ...] = ()
    value: bool = False


@dataclass(frozen=True, kw_only=True)
class OutcomeExp:
    """How one call or resume came out.

    - ``status``: ``refused`` is a call refused before it runs (an
      error, where other statuses are an outcome);
    - ``value``: on ``success``, the value as JSON, of the return type;
    - ``message``: the reason a task failed, or its question;
    - ``names``: words a refusal's error must name;
    - ``asked``: how many replies the model was asked for (0 for a
      refusal);
    - ``kept``: whether the outcome keeps a world, named by its ref;
    - ``plane``: what that world's plane holds;
    - ``calls``: the calls agent code made on live inputs, in order.

    ``None`` leaves a part unchecked."""

    status: Status
    value: Any = None
    message: str | None = None
    names: tuple[str, ...] = ()
    asked: int | None = None
    kept: bool | None = None
    plane: PlaneExp | None = None
    calls: tuple[LiveCall, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskScenario:
    """One scenario: a task, the acts that call it, and one expectation
    per call or resume. ``where`` limits the rungs it runs on; ``needs``
    are the :data:`CAPABILITIES` a harness must have for it to apply."""

    name: str
    summary: str
    task: TaskDef
    where: Where = "anywhere"
    needs: tuple[str, ...] = ()
    world: World = field(default_factory=World)
    acts: tuple[Act, ...]
    expect: tuple[OutcomeExp, ...]

    def __post_init__(self) -> None:
        unknown = set(self.needs) - set(CAPABILITIES)
        if unknown:
            raise ValueError(f"{self.name}: unknown capabilities {sorted(unknown)}")
        outcomes = sum(1 for a in self.acts if not isinstance(a, Restart))
        if len(self.expect) != outcomes:
            raise ValueError(
                f"{self.name}: {outcomes} call(s) and resume(s) but "
                f"{len(self.expect)} expectation(s)"
            )
        if not self.acts or not isinstance(self.acts[0], TaskCall):
            raise ValueError(f"{self.name}: the first act must be a call")
        restarted = False  # with no outcome since: a restart drops them
        for act in self.acts:
            if isinstance(act, Restart):
                restarted = True
            elif isinstance(act, TaskCall) or act.by == "ref":
                restarted = False
            elif restarted:
                raise ValueError(
                    f"{self.name}: a resume by outcome after a restart, with no "
                    "outcome since (a restart drops them); resume by ref"
                )


# -- builders ---------------------------------------------------------------------------


def _expr(value: Any) -> Expr:
    return (
        value if isinstance(value, (Value, Input, MethodCall)) else Value(value=value)
    )


def task_success(value: Any = None, /) -> TaskSuccess:
    """``task.success(value)``: ``value`` written out as JSON, or an
    :class:`Input` or :class:`MethodCall`; with nothing, no value."""
    return TaskSuccess(value=None if value is None else _expr(value))


def task_fail(reason: str) -> TaskFail:
    return TaskFail(reason=reason)


def task_needs_input(question: str) -> TaskNeedsInput:
    return TaskNeedsInput(question=question)


def call(
    *steps: Step, world: bool = False, keep: bool = False, **inputs: Any
) -> TaskCall:
    return TaskCall(inputs=inputs, steps=steps, world=world, keep=keep)


def resume(
    answer: str,
    *steps: Step,
    by: Literal["outcome", "ref"] = "outcome",
    live: tuple[str, ...] = (),
    keep: bool = False,
) -> TaskResume:
    return TaskResume(answer=answer, steps=steps, by=by, live=live, keep=keep)


def restart() -> Restart:
    return Restart()


def success(value: Any = None, /, **kw: Any) -> OutcomeExp:
    """An outcome that succeeds with ``value`` (JSON)."""
    return OutcomeExp(status="success", value=value, **kw)


def failure(reason: str, /, **kw: Any) -> OutcomeExp:
    """An outcome that fails with ``reason``."""
    return OutcomeExp(status="failed", message=reason, **kw)


def question(text: str, /, **kw: Any) -> OutcomeExp:
    """An outcome that asks ``text``, keeping its world for the answer."""
    return OutcomeExp(status="needs_input", message=text, kept=True, **kw)


def refusal(*names: str, **kw: Any) -> OutcomeExp:
    """A call refused before the model is asked anything, its error
    naming ``names``."""
    return OutcomeExp(status="refused", names=names, asked=0, **kw)
