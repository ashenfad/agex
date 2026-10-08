"""agex under the shape corpus.

:class:`AgexTasks` runs a scenario's task as an agex task, on each rung
it has: in this process (``"none"``), under process isolation
(``"process"``) and on a dud machine (``"dud"``, a local guest):

    from agex.conformance import AgexTasks
    from agex.conformance.runner import applies, check, run
    from agex.conformance.scenarios import SCENARIOS

    harness = AgexTasks()
    for scenario in SCENARIOS:
        for rung in harness.rungs:
            if applies(scenario, harness, rung):
                print(scenario.name, rung, check(scenario, run(scenario, harness, rung)))

**The task becomes Python source**: its record classes, its live
classes and the task function, written from the scenario's schemas
into a module of its own and imported. A worker process imports those
classes by name, and a dud guest rebuilds the module from its source,
as either would an embedder's. The module's directory goes on
``sys.path``.

**A task step becomes one ``run_python`` call** that does what it says:
``task.success(...)`` with the value written as Python, ``task.fail``
or ``task.needs_input``.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import importlib.util
import itertools
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from nontainer import Profile, PythonConfig, Store
from nontainer.conformance.corpus import ModelStep, calls

from ..agent import Agent, Outcome
from ..providers.scripted import ScriptedProvider
from ..task import _KEPT, _KEPT_LOCK, PLANE
from .runner import OutcomeView, PlaneView, Script
from .shape import (
    Input,
    LiveCall,
    MethodCall,
    Step,
    TaskCall,
    TaskFail,
    TaskNeedsInput,
    TaskResume,
    TaskScenario,
    TaskSuccess,
)

__all__ = ["AgexTasks"]

_IMPORTS = {
    "Any": "from typing import Any",
    "numpy": "import numpy as np",
    "pandas": "import pandas as pd",
}

_SIMPLE = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}

_ANSWER = '''

_CALLS: list[tuple[Any, str, list[Any]]] = []
"""Each call agent code made on a live object: the object, the method
and its arguments."""


def _answer(obj: Any, cls: str, method: str, args: list[Any]) -> Any:
    _CALLS.append((obj, method, args))
    for reply in _REPLIES[cls][method]:
        if reply["args"] == args:
            return reply["returns"]
    raise LookupError(f"{cls}.{method} has no reply for {args!r}")
'''


# -- the task as Python source ------------------------------------------------------


def _ref(schema: Any) -> str | None:
    """The class a schema names, by ``{"$ref": "#/$defs/Name"}``."""
    ref = schema.get("$ref") if isinstance(schema, Mapping) else None
    return ref.rsplit("/", 1)[-1] if isinstance(ref, str) else None


def _annotation(schema: Mapping[str, Any], uses: set[str]) -> str:
    """The Python annotation for ``schema``, adding to ``uses`` what it
    needs imported."""
    name = _ref(schema)
    if name is not None:
        return name
    kind = schema.get("x-kind")
    if kind == "table":
        uses.add("pandas")
        return "pd.DataFrame"
    if kind == "array":
        uses.add("numpy")
        return "np.ndarray"
    if kind == "bytes":
        return "bytes"
    if "anyOf" in schema:
        return " | ".join(_annotation(s, uses) for s in schema["anyOf"])
    tp = schema.get("type")
    if tp in _SIMPLE:
        return _SIMPLE[tp]
    if tp == "array":
        return f"list[{_annotation(schema.get('items', {}), uses)}]"
    if tp == "object" and "properties" not in schema:
        return f"dict[str, {_annotation(schema.get('additionalProperties', {}), uses)}]"
    if not schema:
        uses.add("Any")
        return "Any"
    raise ValueError(f"no Python type for the schema {dict(schema)!r}")


def _fields(schema: Mapping[str, Any]) -> list[str]:
    """A record's fields in order: its required properties as listed,
    then the others by name."""
    required = list(schema.get("required", ()))
    return required + sorted(set(schema.get("properties", {})) - set(required))


def _doc(text: str | None, indent: str) -> list[str]:
    return [f"{indent}{json.dumps(text)}"] if text else []


def _record(name: str, schema: Mapping[str, Any], uses: set[str]) -> str:
    props = schema.get("properties", {})
    lines = ["@dataclass", f"class {name}:", *_doc(schema.get("description"), "    ")]
    lines += [f"    {f}: {_annotation(props[f], uses)}" for f in _fields(schema)]
    if len(lines) == 2:
        lines.append("    pass")
    return "\n".join(lines)


def _live(name: str, schema: Mapping[str, Any], uses: set[str]) -> str:
    lines = [f"class {name}:", *_doc(schema.get("description"), "    ")]
    for method, spec in schema.get("methods", {}).items():
        params = [
            (p["name"], _annotation(p["type"], uses)) for p in spec.get("params", ())
        ]
        signature = ", ".join(["self", *(f"{n}: {t}" for n, t in params)])
        returns = _annotation(spec.get("returns", {}), uses)
        args = ", ".join(n for n, _ in params)
        lines += [
            "",
            f"    def {method}({signature}) -> {returns}:",
            *_doc(spec.get("description"), "        "),
            f"        return _answer(self, {name!r}, {method!r}, [{args}])",
        ]
    return "\n".join(lines)


def _source(scenario: TaskScenario) -> str:
    """The scenario's task as a module: its classes and the task."""
    task = scenario.task
    uses: set[str] = set()
    blocks: list[str] = []
    replies: dict[str, dict[str, Any]] = {}
    for name, schema in task.defs.items():
        if schema.get("x-kind") == "live":
            uses.add("Any")
            replies[name] = {
                m: spec.get("replies", [])
                for m, spec in schema.get("methods", {}).items()
            }
            blocks.append(_live(name, schema, uses))
        else:
            blocks.append(_record(name, schema, uses))
    params = ", ".join(f"{p.name}: {_annotation(p.type, uses)}" for p in task.params)
    returns = _annotation(task.returns, uses)
    blocks.append(
        f"def {task.name}({params}) -> {returns}: ...\n\n\n"
        f"{task.name}.__doc__ = {json.dumps(task.instructions)}"
    )
    head = [
        f'"""The task of the shape scenario {scenario.name!r}, generated by '
        'agex.conformance.tasks."""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "from dataclasses import dataclass",
        *(_IMPORTS[u] for u in sorted(uses)),
    ]
    if replies:
        head.append(f"\n_REPLIES = json.loads({json.dumps(json.dumps(replies))})")
        head.append(_ANSWER.rstrip("\n"))
    return "\n".join(head) + "\n\n\n" + "\n\n\n".join(blocks) + "\n"


# -- values to and from JSON ----------------------------------------------------------


def _fits(value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any]) -> bool:
    """Whether a JSON value can be read as ``schema``'s type."""
    name = _ref(schema)
    if name is not None:
        target = defs[name]
        if target.get("x-kind") == "live":
            return isinstance(value, dict)
        props = target.get("properties", {})
        return (
            isinstance(value, dict)
            and set(value) <= set(props)
            and set(target.get("required", ())) <= set(value)
        )
    kind = schema.get("x-kind")
    if kind == "table":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "bytes":
        return isinstance(value, str)
    if "anyOf" in schema:
        return any(_fits(value, s, defs) for s in schema["anyOf"])
    tp = schema.get("type")
    if tp in ("integer", "number", "boolean") and isinstance(value, bool):
        return tp == "boolean"
    checks: dict[Any, Any] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    if tp == "null":
        return value is None
    return tp not in checks or isinstance(value, checks[tp])


def _member(value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Any:
    """The member of a union a JSON value is read as: the first it fits."""
    return next((s for s in schema["anyOf"] if _fits(value, s, defs)), None)


def _python(
    value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any], uses: set[str]
) -> str:
    """A JSON value written as Python that builds it as ``schema``'s
    type; a value that doesn't fit is written as the plain data it is."""
    name = _ref(schema)
    if name is not None and _fits(value, schema, defs):
        props = defs[name].get("properties", {})
        args = ", ".join(
            f"{k}={_python(v, props[k], defs, uses)}" for k, v in value.items()
        )
        return f"{name}({args})"
    kind = schema.get("x-kind")
    if kind == "table" and isinstance(value, dict):
        uses.add("pandas")
        return f"pd.DataFrame({value!r})"
    if kind == "array" and isinstance(value, list):
        uses.add("numpy")
        return f"np.array({value!r})"
    if kind == "bytes" and isinstance(value, str):
        return repr(base64.b64decode(value))
    if "anyOf" in schema:
        member = _member(value, schema, defs)
        return repr(value) if member is None else _python(value, member, defs, uses)
    if schema.get("type") == "array" and isinstance(value, list):
        items = schema.get("items", {})
        return "[" + ", ".join(_python(v, items, defs, uses) for v in value) + "]"
    if (
        schema.get("type") == "object"
        and isinstance(value, dict)
        and "properties" not in schema
    ):
        each = schema.get("additionalProperties", {})
        pairs = (f"{k!r}: {_python(v, each, defs, uses)}" for k, v in value.items())
        return "{" + ", ".join(pairs) + "}"
    return repr(value)


def _value(
    value: Any,
    schema: Mapping[str, Any],
    defs: Mapping[str, Any],
    module: ModuleType,
    lives: list[Any],
) -> Any:
    """A JSON value as ``schema``'s type, in this process; a live object
    is a new one of its class, added to ``lives``."""
    name = _ref(schema)
    if name is not None:
        target, cls = defs[name], getattr(module, name)
        if target.get("x-kind") == "live":
            obj = cls()
            lives.append(obj)
            return obj
        if not isinstance(value, dict):
            return value
        props = target.get("properties", {})
        return cls(
            **{
                k: _value(v, props.get(k, {}), defs, module, lives)
                for k, v in value.items()
            }
        )
    kind = schema.get("x-kind")
    if kind == "table":
        import pandas as pd

        return pd.DataFrame(value)
    if kind == "array":
        import numpy as np

        return np.array(value)
    if kind == "bytes":
        return base64.b64decode(value)
    if "anyOf" in schema:
        member = _member(value, schema, defs)
        return value if member is None else _value(value, member, defs, module, lives)
    if schema.get("type") == "array" and isinstance(value, list):
        return [_value(v, schema.get("items", {}), defs, module, lives) for v in value]
    if (
        schema.get("type") == "object"
        and isinstance(value, dict)
        and "properties" not in schema
    ):
        each = schema.get("additionalProperties", {})
        return {k: _value(v, each, defs, module, lives) for k, v in value.items()}
    return value


def _wrong(expected: str, value: Any) -> str:
    """What stands in for a value that isn't of its type: text no JSON
    expectation equals, saying what it is."""
    return f"!not {expected}: {type(value).__name__} {value!r}"


def _json(value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Any:
    """A value as JSON, read by ``schema``: what a scenario's expected
    value is compared with."""
    name = _ref(schema)
    if name is not None:
        target = defs[name]
        if (
            target.get("x-kind") == "live"
            or not dataclasses.is_dataclass(value)
            or type(value).__name__ != name
        ):
            return _wrong(name, value)
        props = target.get("properties", {})
        return {
            f.name: _json(getattr(value, f.name), props.get(f.name, {}), defs)
            for f in dataclasses.fields(value)
        }
    kind = schema.get("x-kind")
    if kind == "table":
        if type(value).__name__ != "DataFrame":
            return _wrong("a table", value)
        return {str(c): value[c].tolist() for c in value.columns}
    if kind == "array":
        if type(value).__name__ != "ndarray":
            return _wrong("an array", value)
        return value.tolist()
    if kind == "bytes":
        if not isinstance(value, bytes):
            return _wrong("bytes", value)
        return base64.b64encode(value).decode()
    if "anyOf" in schema:
        for member in schema["anyOf"]:
            found = _json(value, member, defs)
            if not (isinstance(found, str) and found.startswith("!not ")):
                return found
        return _wrong("any of its types", value)
    tp = schema.get("type")
    if tp == "array":
        if not isinstance(value, list):
            return _wrong("a list", value)
        return [_json(v, schema.get("items", {}), defs) for v in value]
    if tp == "object":
        if not isinstance(value, dict):
            return _wrong("a dict", value)
        each = schema.get("additionalProperties", {})
        return {k: _json(v, each, defs) for k, v in value.items()}
    if tp in _SIMPLE and not _fits(value, schema, defs):
        return _wrong(_SIMPLE[tp], value)
    return value


# -- the harness ----------------------------------------------------------------------


def _has(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


class AgexTasks:
    """agex as a shape-corpus harness. ``root`` holds the modules it
    writes and the stores of its worlds (a temporary directory by
    default); ``dud_backend`` is the dud backend of the ``"dud"`` rung."""

    name = "agex"

    def __init__(
        self, root: str | Path | None = None, *, dud_backend: str = "subprocess"
    ) -> None:
        self.root = (
            Path(root)
            if root is not None
            else Path(tempfile.mkdtemp(prefix="agex-shape-"))
        )
        self.modules = self.root / "modules"
        self.modules.mkdir(parents=True, exist_ok=True)
        if str(self.modules) not in sys.path:
            sys.path.insert(0, str(self.modules))
        self.dud_backend = dud_backend
        rungs = ["none", "process"]
        if sys.version_info >= (3, 11) and _has("dud"):
            rungs.append("dud")
        self.rungs: tuple[str, ...] = tuple(rungs)
        has = {"tables": _has("pandas", "pyarrow"), "arrays": _has("numpy")}
        self.capabilities = frozenset(name for name, ok in has.items() if ok)
        self._count = itertools.count(1)

    def in_process(self, rung: str) -> bool:
        return rung == "none"

    def profile(self, rung: str, scenario: TaskScenario) -> Profile:
        """The environment ``scenario`` runs in on ``rung``."""
        if rung not in self.rungs:
            raise ValueError(f"agex has no rung {rung!r} here, only {list(self.rungs)}")
        modules = []
        if {"tables", "arrays"} & set(scenario.needs):
            from nontainer.presets import dataframes

            modules.append(dataframes())
        if rung == "dud":
            from nontainer.executor_dud import DudExecutor

            backend = self.dud_backend
            return Profile(
                python=PythonConfig(modules=modules),
                executor_factory=lambda: DudExecutor(backend=backend),
            )
        isolation = "process" if rung == "process" else "none"
        return Profile(python=PythonConfig(isolation=isolation, modules=modules))

    def open(self, scenario: TaskScenario, rung: str, script: Script) -> _Opened:
        slug = re.sub(r"\W", "_", scenario.name)
        name = f"agex_shape_{slug}_{next(self._count)}"
        (self.modules / f"{name}.py").write_text(_source(scenario))
        importlib.invalidate_caches()
        module = importlib.import_module(name)
        world = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.root))
        return _Opened(scenario, self.profile(rung, scenario), script, module, world)


class _Opened:
    """A scenario open on one rung: its store and world, the agent and
    task, and the last outcome."""

    def __init__(
        self,
        scenario: TaskScenario,
        profile: Profile,
        script: Script,
        module: ModuleType,
        directory: Path,
    ) -> None:
        self.scenario = scenario
        self.task_def = scenario.task
        self.params = {p.name: p.type for p in scenario.task.params}
        self.profile = profile
        self.script = script
        self.module = module
        self.directory = directory
        self.lives: dict[int, str] = {}
        # the live objects made for inputs, held so their ids name them
        self.objects: list[Any] = []
        self.outcome: Outcome | None = None
        self.ref: str | None = None
        self.on_world = False
        self.scratch: set[str] = set()
        self._open(seed=True)

    def _open(self, *, seed: bool) -> None:
        self.store = Store(self.directory / "store")
        self.ws = self.store.open("main", profile=self.profile)
        if seed:
            for path, text in self.scenario.world.files.items():
                self.ws.files.write(path, text)
            if self.ws.uncommitted:
                self.ws.commit(info={"tool": "world"})
        agent = Agent(ScriptedProvider(self._reply), profile=self.profile)
        self.task = agent.task(getattr(self.module, self.task_def.name))

    def _shut(self) -> None:
        self.ws.close()
        self.store.close()

    # -- the model --

    def _reply(self) -> ModelStep:
        step = self.script.next()
        if isinstance(step, ModelStep):
            return step
        return calls("run_python", code=self._code(step))

    def _code(self, step: Step) -> str:
        if isinstance(step, TaskFail):
            return f"task.fail({step.reason!r})"
        if isinstance(step, TaskNeedsInput):
            return f"task.needs_input({step.question!r})"
        assert isinstance(step, TaskSuccess)
        value = step.value
        if value is None:
            return "task.success()"
        if isinstance(value, Input):
            return f"task.success({value.name})"
        if isinstance(value, MethodCall):
            args = ", ".join(map(repr, value.args))
            return f"task.success({value.input}.{value.method}({args}))"
        uses: set[str] = set()
        expr = _python(value.value, self.task_def.returns, self.task_def.defs, uses)
        return "\n".join(
            [*(_IMPORTS[u] for u in sorted(uses)), f"task.success({expr})"]
        )

    # -- the acts --

    def _input(self, name: str, value: Any) -> Any:
        if name not in self.params:
            return value  # a name the task doesn't take: the call refuses it
        found: list[Any] = []
        built = _value(value, self.params[name], self.task_def.defs, self.module, found)
        for obj in found:
            self.lives[id(obj)] = name
            self.objects.append(obj)
        return built

    def _calls(self) -> list[tuple[Any, str, list[Any]]]:
        return getattr(self.module, "_CALLS", [])

    def call(self, act: TaskCall) -> OutcomeView:
        """Call the task. A call that can't be made is refused: an input
        that can't even be built (a record missing a field), or one the
        task refuses."""
        self._calls().clear()
        try:
            inputs = {name: self._input(name, v) for name, v in act.inputs.items()}
            out = self.task.run(
                world=self.ws if act.world else None, keep=act.keep, **inputs
            )
        except Exception as error:
            return OutcomeView(
                status="refused", error=f"{type(error).__name__}: {error}"
            )
        self.outcome, self.ref, self.on_world = out, out.ref, act.world
        return self._view(out)

    def resume(self, act: TaskResume) -> OutcomeView:
        self._calls().clear()
        try:
            live = {name: self._input(name, {}) for name in act.live}
            if act.by == "outcome":
                if self.outcome is None:
                    raise LookupError("no outcome to resume")
                out = self.outcome.resume(act.answer, **live)
            else:
                if self.ref is None:
                    raise LookupError("no task world to resume")
                out = self.task.resume(
                    self.ref,
                    act.answer,
                    world=self.ws if self.on_world else None,
                    keep=act.keep,
                    **live,
                )
        except Exception as error:
            return OutcomeView(
                status="refused", error=f"{type(error).__name__}: {error}"
            )
        self.outcome = out
        self.ref = out.ref or self.ref
        return self._view(out)

    def restart(self) -> None:
        self._shut()
        self.outcome = None
        self._open(seed=False)

    def close(self) -> None:
        try:
            self._shut()
        finally:
            with _KEPT_LOCK:
                kept = [_KEPT.pop(ref) for ref in self.scratch if ref in _KEPT]
            for store, _ in kept:
                store.close()
            shutil.rmtree(self.directory, ignore_errors=True)

    # -- what came of it --

    def _view(self, out: Outcome) -> OutcomeView:
        if out.ref is not None and not self.store.exists(out.ref):
            self.scratch.add(out.ref)
        value = None
        if out.status == "success":
            value = _json(out.value, self.task_def.returns, self.task_def.defs)
        return OutcomeView(
            status=out.status,
            value=value,
            message=out.message,
            kept=out.ref is not None,
            plane=None if out.ref is None else self._plane(out.ref),
            calls=tuple(
                LiveCall(
                    input=self.lives.get(id(obj), "?"), method=method, args=tuple(args)
                )
                for obj, method, args in self._calls()
            ),
        )

    def _plane(self, ref: str) -> PlaneView | None:
        store = (
            self.store if self.store.exists(ref) else _KEPT.get(ref, (None, None))[0]
        )
        if store is None:
            return None
        kept = store.open(ref)
        try:
            kv = kept.provider.kv
            state = kv.get(PLANE + "state") or {}
            prefix = PLANE + "inputs/"
            return PlaneView(
                status=str(state.get("status")),
                stored=tuple(
                    sorted(k[len(prefix) :] for k in kv if k.startswith(prefix))
                ),
                value=PLANE + "value" in kv,
            )
        finally:
            kept.close()
