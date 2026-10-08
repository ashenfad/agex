"""Running a shape scenario on a harness, and checking what came of it.

The runner owns the script: a harness's model asks it for each reply
(:meth:`Script.next`), and writes a task step in its own language. So
the runner counts what the model was asked for, the same way for every
harness, and a harness that asks for more than the script holds is
caught. The harness owns the rest: its world, its rungs, and the values
it builds from the scenario's JSON.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nontainer.conformance.corpus import ModelStep

from .shape import (
    CAPABILITIES,
    LiveCall,
    OutcomeExp,
    PlaneExp,
    Restart,
    Step,
    TaskCall,
    TaskResume,
    TaskScenario,
)

__all__ = [
    "OutcomeView",
    "PlaneView",
    "Script",
    "TaskHarness",
    "TaskRun",
    "applies",
    "check",
    "run",
]

RAN_OUT = ModelStep(text="(the script has nothing more)")
"""The reply past the end of a script: text with no call, which a task
takes for a model that stopped without finishing."""


class Script:
    """The model's replies for the act being run, in order."""

    def __init__(self) -> None:
        self._steps: list[Step] = []
        self.asked = 0
        self.exhausted = False

    def load(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self.asked = 0

    def next(self) -> Step:
        """The model's next reply: the script's next step, or
        :data:`RAN_OUT` past its end."""
        self.asked += 1
        if not self._steps:
            self.exhausted = True
            return RAN_OUT
        return self._steps.pop(0)


@dataclass(frozen=True, kw_only=True)
class PlaneView:
    """What a kept world's ``__task__`` plane holds, as
    :class:`~agex.conformance.shape.PlaneExp` describes it."""

    status: str
    stored: tuple[str, ...] = ()
    value: bool = False


@dataclass(frozen=True, kw_only=True)
class OutcomeView:
    """How a call or resume came out, as a harness saw it: its status, the
    value as JSON (on ``success``), its message (the reason, or the
    question), the error of a refusal, whether it keeps a world and what
    that world's plane holds, and the calls agent code made on live
    inputs."""

    status: str
    value: Any = None
    message: str | None = None
    error: str | None = None
    kept: bool = False
    plane: PlaneView | None = None
    calls: tuple[LiveCall, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Observed:
    """What came of a scenario: an outcome per call and resume, how many
    replies the model was asked for in each, and whether a harness asked
    past the end of a script."""

    outcomes: tuple[OutcomeView, ...]
    asked: tuple[int, ...]
    exhausted: bool = False


class TaskRun(Protocol):
    """One scenario, open on one rung of a harness."""

    def call(self, act: TaskCall) -> OutcomeView: ...

    def resume(self, act: TaskResume) -> OutcomeView: ...

    def restart(self) -> None: ...

    def close(self) -> None: ...


class TaskHarness(Protocol):
    """A harness the shape corpus runs: the rungs it has, by name, which
    of them is in the process, and the :data:`CAPABILITIES` it has."""

    name: str
    capabilities: frozenset[str]
    rungs: tuple[str, ...]

    def in_process(self, rung: str) -> bool: ...

    def open(self, scenario: TaskScenario, rung: str, script: Script) -> TaskRun: ...


def applies(scenario: TaskScenario, harness: TaskHarness, rung: str) -> bool:
    """Whether the harness has what the scenario needs, and the rung is
    one it runs on."""
    if not set(scenario.needs) <= harness.capabilities:
        return False
    if scenario.where == "anywhere":
        return True
    return harness.in_process(rung) == (scenario.where == "in-process")


def run(scenario: TaskScenario, harness: TaskHarness, rung: str) -> Observed:
    """Run ``scenario`` on one rung of ``harness``."""
    unknown = set(harness.capabilities) - set(CAPABILITIES)
    if unknown:
        raise ValueError(f"{harness.name}: unknown capabilities {sorted(unknown)}")
    script = Script()
    opened = harness.open(scenario, rung, script)
    outcomes: list[OutcomeView] = []
    asked: list[int] = []
    exhausted = False
    try:
        for act in scenario.acts:
            if isinstance(act, Restart):
                opened.restart()
                continue
            script.load(act.steps)
            if isinstance(act, TaskCall):
                outcomes.append(opened.call(act))
            else:
                outcomes.append(opened.resume(act))
            asked.append(script.asked)
            exhausted = exhausted or script.exhausted
    finally:
        opened.close()
    return Observed(outcomes=tuple(outcomes), asked=tuple(asked), exhausted=exhausted)


def check(scenario: TaskScenario, observed: Observed) -> dict[str, str]:
    """Every way ``observed`` differs from the scenario's expectations,
    by check name: ``outcome<N>.<part>`` (N from 1, a part named as in
    :class:`~agex.conformance.shape.OutcomeExp`), ``outcomes`` and
    ``script``. Empty when everything holds."""
    failures: dict[str, str] = {}
    if len(observed.outcomes) != len(scenario.expect):
        failures["outcomes"] = f"{len(observed.outcomes)} outcome(s) came"
    for n, (want, got, asked) in enumerate(
        zip(scenario.expect, observed.outcomes, observed.asked), start=1
    ):
        failures.update(
            (f"outcome{n}.{part}", problem)
            for part, problem in _differences(want, got, asked).items()
        )
    if observed.exhausted:
        failures["script"] = (
            "the harness asked the model for more than the script holds"
        )
    return failures


def _differences(want: OutcomeExp, got: OutcomeView, asked: int) -> dict[str, str]:
    found: dict[str, str] = {}
    if got.status != want.status:
        why = got.error or got.message
        found["status"] = f"{got.status!r}, not {want.status!r}" + (
            f" ({why})" if why else ""
        )
        return found
    if want.status == "success" and not _same(got.value, want.value):
        found["value"] = f"{got.value!r} != {want.value!r}"
    if want.message is not None and got.message != want.message:
        found["message"] = f"{got.message!r} != {want.message!r}"
    missing = [name for name in want.names if name not in (got.error or "")]
    if missing:
        found["names"] = f"{got.error!r} doesn't name {missing}"
    if want.asked is not None and asked != want.asked:
        found["asked"] = f"the model was asked {asked} time(s), not {want.asked}"
    if want.kept is not None and got.kept != want.kept:
        found["kept"] = f"{'kept' if got.kept else 'kept nothing'}, not as expected"
    if want.plane is not None and not _plane_fits(got.plane, want.plane):
        found["plane"] = f"{got.plane} != {want.plane}"
    if want.calls is not None and got.calls != want.calls:
        found["calls"] = f"{list(got.calls)} != {list(want.calls)}"
    return found


def _plane_fits(got: PlaneView | None, want: PlaneExp) -> bool:
    return got is not None and (got.status, tuple(got.stored), got.value) == (
        want.status,
        tuple(want.stored),
        want.value,
    )


def _same(a: Any, b: Any) -> bool:
    """JSON equality: a boolean is no number, and lists and tuples are
    both arrays."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b
