"""agex under the shape corpus's code scenarios (tasks agent code
defines, :mod:`agex.conformance.code_scenarios`).

:class:`AgexCodeTasks` runs each act as one turn of a lead session on a
world granted ``agex`` (:func:`agex.agent_tasks.agent_tasks`), on each
rung it has. The lead's model writes the calling code, in Python, from
the task's schemas:

- the task's record classes as dataclasses, and its live classes;
- the task as a ``def`` decorated with ``@agex.task``, its docstring
  the task's instructions;
- the call (or ``.map``), and a line reporting what came of it: the
  value written out with its classes named, or the error raised.

Each helper follows the act's ``helper`` steps, written as a task's are
(:mod:`agex.conformance.tasks`), and its value is built of the classes
its world is given.

    from agex.conformance.code_tasks import AgexCodeTasks, check_code, run_code
    from agex.conformance.code_scenarios import CODE_SCENARIOS

    harness = AgexCodeTasks()
    for scenario in CODE_SCENARIOS:
        for rung in harness.rungs:
            print(scenario.name, rung, check_code(scenario, run_code(scenario, harness, rung)))
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nontainer import Profile, PythonConfig, Store
from nontainer.conformance.corpus import ModelStep, calls, says
from pydantic_ai import messages as pai
from pydantic_ai.models.function import AgentInfo

from ..agent import Agent
from ..agent_tasks import agent_tasks
from ..providers.scripted import ScriptedProvider
from .runner import RAN_OUT, _same
from .shape import (
    CodeCall,
    CodeExp,
    CodeScenario,
    Input,
    Step,
    TaskFail,
    TaskNeedsInput,
    TaskSuccess,
)
from .tasks import _IMPORTS, _annotation, _has, _live, _python, _record, _ref

__all__ = ["AgexCodeTasks", "CodeView", "check_code", "run_code"]

LEAD = "LEAD"
"""What a lead turn's prompt opens with: how the model tells the lead's
requests from a helper's."""

MARK = "@@agex-code@@"
"""What opens the line the calling code reports on."""

_REPORT = '''
def _tag(v):
    """A value as JSON, saying what it was built as."""
    if dataclasses.is_dataclass(v) and not hasattr(v, "__mro__"):  # not a class
        name = type(v).__name__
        return {"__record__": name, "own": _OWN.get(name) is type(v),
                "fields": {f.name: _tag(getattr(v, f.name))
                           for f in dataclasses.fields(v)}}
    kind = type(v).__name__
    if kind == "DataFrame":
        return {"__table__": {str(c): v[c].tolist() for c in v.columns}}
    if kind == "ndarray":
        return {"__array__": v.tolist()}
    if isinstance(v, bytes):
        return {"__bytes__": base64.b64encode(v).decode()}
    if isinstance(v, (list, tuple)):
        return [_tag(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _tag(x) for k, x in v.items()}
    return v


def _report(**what):
    print(MARK + json.dumps(what))
'''


@dataclass(frozen=True, kw_only=True)
class CodeView:
    """What the calling code reported: how its act came out, the value
    (or a ``.map``'s values) as corpus JSON, and the text of the error
    it saw."""

    status: str
    value: Any = None
    values: tuple[Any, ...] | None = None
    message: str | None = None


class _Routed(ScriptedProvider):
    """A scripted model whose every request is answered by ``route(first,
    turn)``: the text the conversation opens with, and how many replies
    it already holds. Concurrent helpers each follow their own script."""

    def __init__(self, route: Callable[[str, int], ModelStep]) -> None:
        super().__init__(lambda: RAN_OUT)
        self._route = route

    async def _reply(
        self, messages: list[pai.ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        turns = sum(isinstance(m, pai.ModelResponse) for m in messages)
        first = next(
            (
                str(part.content)
                for m in messages
                if isinstance(m, pai.ModelRequest)
                for part in m.parts
                if isinstance(part, pai.UserPromptPart)
            ),
            "",
        )
        step = self._route(first, turns)
        self._next = lambda sent: step  # read before the reply's first await
        async for event in super()._reply(messages, info):
            yield event


def _step_code(step: Step, scenario: CodeScenario) -> ModelStep:
    """A helper's step as its model's reply."""
    if isinstance(step, ModelStep):
        return step
    if isinstance(step, TaskFail):
        return calls("run_python", code=f"task.fail({step.reason!r})")
    if isinstance(step, TaskNeedsInput):
        return calls("run_python", code=f"task.needs_input({step.question!r})")
    assert isinstance(step, TaskSuccess)
    value = step.value
    if value is None:
        return calls("run_python", code="task.success()")
    if isinstance(value, Input):
        return calls("run_python", code=f"task.success({value.name})")
    uses: set[str] = set()
    task = scenario.task
    expr = _python(getattr(value, "value", value), task.returns, task.defs, uses)
    code = "\n".join([*(_IMPORTS[u] for u in sorted(uses)), f"task.success({expr})"])
    return calls("run_python", code=code)


def _ordered(defs: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """``defs`` with each class after the classes its fields name, so
    each is defined before a class that uses it."""
    done: dict[str, Any] = {}

    def named(schema: Any) -> list[str]:
        if isinstance(schema, Mapping):
            ref = _ref(schema)
            found = [ref] if ref is not None else []
            return found + [n for v in schema.values() for n in named(v)]
        if isinstance(schema, list):
            return [n for v in schema for n in named(v)]
        return []

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in done or name in path or name not in defs:
            return
        for used in named(defs[name]):
            visit(used, (*path, name))
        done[name] = defs[name]

    for name in sorted(defs):
        visit(name, ())
    return list(done.items())


def _lead_code(scenario: CodeScenario, act: CodeCall) -> str:
    """The calling code for one act: the task's classes, the task, the
    call, and the report of what came of it."""
    task = scenario.task
    uses: set[str] = set()
    blocks = []
    for name, schema in _ordered(task.defs):
        if schema.get("x-kind") == "live":
            blocks.append(_live(name, schema, uses))
        else:
            blocks.append(_record(name, schema, uses))
    params = ", ".join(
        p.name if p.type is None else f"{p.name}: {_annotation(p.type, uses)}"
        for p in task.params
    )
    returns = _annotation(task.returns, uses)
    doc = f"        {json.dumps(task.instructions)}\n" if task.instructions else ""
    defined = (
        "try:\n"
        "    @agex.task\n"
        f"    def {task.name}({params}) -> {returns}:\n"
        f"{doc}"
        "        ...\n"
        "except TypeError as error:\n"
        '    _report(status="refused", message=str(error))\n'
        "else:\n"
    )
    types = {p.name: p.type or {} for p in task.params}
    if act.items is None:
        args = ", ".join(
            f"{k}={_python(v, types.get(k, {}), task.defs, uses)}"
            for k, v in act.inputs.items()
        )
        made = f"{task.name}({args})"
        reported = '_report(status="value", value=_tag(result))'
    else:
        items = ", ".join(
            "{"
            + ", ".join(
                f"{k!r}: {_python(v, types.get(k, {}), task.defs, uses)}"
                for k, v in item.items()
            )
            + "}"
            for item in act.items
        )
        one = len(task.params) == 1
        made = (
            f"{task.name}.map([i[{task.params[0].name!r}] for i in [{items}]])"
            if one
            else f"{task.name}.map([{items}])"
        )
        reported = '_report(status="value", values=_tag(result))'
    called = (
        "    try:\n"
        f"        result = {made}\n"
        "    except agex.TaskNeedsInput as asked:\n"
        '        _report(status="needs_input", message=asked.question)\n'
        "    except agex.TaskFailed as failed:\n"
        '        _report(status="failed", message=str(failed))\n'
        "    else:\n"
        f"        {reported}\n"
    )
    head = [
        "import base64",
        "import dataclasses",
        "import json",
        "from dataclasses import dataclass",
        *(_IMPORTS[u] for u in sorted(uses)),
        f"MARK = {MARK!r}",
        _REPORT,
    ]
    own = (
        "_OWN = {"
        + ", ".join(
            f"{n!r}: {n}" for n, sc in task.defs.items() if sc.get("x-kind") != "live"
        )
        + "}\n"
    )
    return (
        "\n".join(head)
        + "\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + own
        + "\n"
        + defined
        + called
    )


def _untag(value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Any:
    """What the calling code reported, as corpus JSON read by ``schema``;
    a value not built as its declared type (another class, say) as text
    no expectation equals."""
    name = _ref(schema)
    if name is not None:
        if not (
            isinstance(value, dict)
            and value.get("__record__") == name
            and value.get("own") is True
        ):
            return f"!not the code's own {name}: {value!r}"
        props = defs[name].get("properties", {})
        return {
            k: _untag(v, props.get(k, {}), defs) for k, v in value["fields"].items()
        }
    kind = schema.get("x-kind")
    marks = {"table": "__table__", "array": "__array__", "bytes": "__bytes__"}
    if kind in marks:
        if not (isinstance(value, dict) and marks[kind] in value):
            return f"!not {kind}: {value!r}"
        return value[marks[kind]]
    if "anyOf" in schema:
        for member in schema["anyOf"]:
            found = _untag(value, member, defs)
            if not (isinstance(found, str) and found.startswith("!not ")):
                return found
        return f"!not any of its types: {value!r}"
    if schema.get("type") == "array" and isinstance(value, list):
        return [_untag(v, schema.get("items", {}), defs) for v in value]
    if (
        schema.get("type") == "object"
        and isinstance(value, dict)
        and "properties" not in schema
    ):
        each = schema.get("additionalProperties", {})
        return {k: _untag(v, each, defs) for k, v in value.items()}
    return value


class AgexCodeTasks:
    """agex as a harness for the code scenarios, on the rungs it has
    here: in this process (``"none"``), under process isolation
    (``"process"``) and on a dud machine (``"dud"``)."""

    name = "agex"

    def __init__(self, *, dud_backend: str = "subprocess") -> None:
        self.dud_backend = dud_backend
        rungs = ["none", "process"]
        if sys.version_info >= (3, 11) and _has("dud"):
            rungs.append("dud")
        self.rungs: tuple[str, ...] = tuple(rungs)
        has = {"tables": _has("pandas", "pyarrow"), "arrays": _has("numpy")}
        self.capabilities = frozenset(n for n, ok in has.items() if ok)

    def in_process(self, rung: str) -> bool:
        return rung == "none"

    def profile(self, rung: str, scenario: CodeScenario, agent: Agent) -> Profile:
        modules = []
        if {"tables", "arrays"} & set(scenario.needs):
            from nontainer.presets import dataframes

            modules.append(dataframes())
        python = PythonConfig(
            modules=modules, host_objects={"agex": agent_tasks(agent)}
        )
        if rung == "dud":
            from nontainer.executor_dud import DudExecutor

            backend = self.dud_backend
            return Profile(
                python=python, executor_factory=lambda: DudExecutor(backend=backend)
            )
        return Profile(
            python=PythonConfig(
                isolation="process" if rung == "process" else "none",
                modules=modules,
                host_objects=python.host_objects,
            )
        )


def applies_code(scenario: CodeScenario, harness: AgexCodeTasks, rung: str) -> bool:
    """Whether the harness has what the scenario needs, and the rung is
    one it runs on."""
    if not set(scenario.needs) <= harness.capabilities:
        return False
    if scenario.where == "anywhere":
        return True
    return harness.in_process(rung) == (scenario.where == "in-process")


def run_code(
    scenario: CodeScenario, harness: AgexCodeTasks, rung: str
) -> list[CodeView]:
    """Run ``scenario`` on one rung: what the calling code reported, per
    act."""
    lead: list[ModelStep] = []
    for act in scenario.acts:
        lead += [
            calls("run_python", code=_lead_code(scenario, act)),
            says("done"),
        ]
    current: list[CodeCall] = [scenario.acts[0]]

    def route(first: str, turn: int) -> ModelStep:
        if first.startswith(LEAD):
            # the lead's turns are one conversation: its replies so far
            # count across the acts
            return lead[turn] if turn < len(lead) else RAN_OUT
        helper = current[0].helper
        return _step_code(helper[turn], scenario) if turn < len(helper) else RAN_OUT

    agent = Agent(_Routed(route))
    store = Store(memory=True)
    ws = store.open("lead", profile=harness.profile(rung, scenario, agent))
    chat = agent.session(ws, sessions=True)
    views: list[CodeView] = []
    try:
        for n, act in enumerate(scenario.acts):
            current[0] = act
            outcome = chat.say(f"{LEAD} act {n + 1}")
            views.append(_view(outcome, chat, scenario))
    finally:
        chat.close()
        ws.close()
        store.close()
    return views


def _view(outcome: Any, chat: Any, scenario: CodeScenario) -> CodeView:
    if outcome.status != "completed":
        return CodeView(status=f"turn {outcome.status}", message=outcome.message)
    run = chat.runs[-1]
    text = "\n".join(
        str(getattr(part, "content", ""))
        for message in run.messages
        for part in message.parts
    )
    line = next((ln for ln in text.splitlines() if ln.startswith(MARK)), None)
    if line is None:
        return CodeView(status="unreported", message=text[-2000:])
    said = json.loads(line[len(MARK) :])
    task = scenario.task
    if said["status"] != "value":
        return CodeView(status=said["status"], message=said.get("message"))
    if "values" in said:
        return CodeView(
            status="value",
            values=tuple(_untag(v, task.returns, task.defs) for v in said["values"]),
        )
    return CodeView(
        status="value", value=_untag(said["value"], task.returns, task.defs)
    )


def check_code(scenario: CodeScenario, views: list[CodeView]) -> dict[str, str]:
    """Every way ``views`` differ from the scenario's expectations, by
    check name (``act<N>.<part>``); empty when everything holds."""
    failures: dict[str, str] = {}
    if len(views) != len(scenario.expect):
        failures["acts"] = f"{len(views)} act(s) reported"
    for n, (want, got) in enumerate(zip(scenario.expect, views), start=1):
        failures.update((f"act{n}.{k}", v) for k, v in _differences(want, got).items())
    return failures


def _differences(want: CodeExp, got: CodeView) -> dict[str, str]:
    if got.status != want.status:
        why = f" ({got.message})" if got.message else ""
        return {"status": f"{got.status!r}, not {want.status!r}{why}"}
    found: dict[str, str] = {}
    if want.status == "value":
        if want.values is not None:
            if not _same(list(got.values or ()), list(want.values)):
                found["values"] = f"{got.values!r} != {want.values!r}"
        elif not _same(got.value, want.value):
            found["value"] = f"{got.value!r} != {want.value!r}"
    if want.message is not None and want.message not in (got.message or ""):
        found["message"] = f"{got.message!r} doesn't say {want.message!r}"
    missing = [n for n in want.names if n not in (got.message or "")]
    if missing:
        found["names"] = f"{got.message!r} doesn't name {missing}"
    return found
