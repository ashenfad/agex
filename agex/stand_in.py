"""A stand-in for ``agex``: agent-defined tasks answered without a model.

A page check or a handler test shouldn't spend model calls, or make a
branch per click. Grant a stand-in beside the real thing and bind it for
the run:

    PythonConfig(host_objects={
        "agex": agent_tasks(agent),
        "agex_stand_in": stand_in_tasks(),
    })

    test_app(actions, bind={"agex": "agex_stand_in"})   # the page
    call("summary", "POST", json={...}, agex=agex_stand_in)  # ws-pytest

Code defines and calls tasks exactly as it does against ``agex``. Each
call is answered here, at once: by ``answer(name, inputs)`` when the
embedder gives one, or with the simplest value of the task's return
type (an empty list, ``0``, ``""``, the first member of an enum, a record
of such fields). Either way the value crosses as a real one does, so the
caller decodes it into its own classes.
"""

from __future__ import annotations

import datetime
import decimal
import pathlib
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from nontainer import HostObject, Workspace, values

from .agent_tasks import _Called, _reply
from .stubs import AgexStub

__all__ = ["StandInTasks", "stand_in_tasks"]

Answer = Callable[[str, Mapping[str, Any]], Any]
"""``answer(name, inputs)``: the value for one call of the task ``name``,
its inputs by parameter name. Records may be given as dicts of their
fields."""


def stand_in_tasks(answer: Answer | None = None) -> HostObject:
    """``agex`` without a model: each task call is answered at once, by
    ``answer(name, inputs)`` or with the simplest value of the task's
    return type (see :mod:`agex.stand_in`)."""
    return HostObject(factory=lambda ws: StandInTasks(ws, answer), stub=AgexStub)


class StandInTasks:
    """The host half of a stand-in ``agex`` for one world: the same
    calls as :class:`agex.agent_tasks.AgentTasks`, answered here."""

    def __init__(self, ws: Workspace, answer: Answer | None = None) -> None:
        self._ws = ws
        self._answer = answer

    def __repr__(self) -> str:
        return f"<agex stand-in for {self._ws.session!r}>"

    def held(self) -> list[str]:
        """The world's host objects, by name."""
        return sorted(self._ws.runtime.python_config.host_objects)

    def call(
        self, spec: dict[str, Any], inputs: dict[str, bytes], primer: str
    ) -> dict[str, Any]:
        """Answer the task ``spec`` once on ``inputs``."""
        return self._one(spec, inputs, primer)

    def map(
        self, spec: dict[str, Any], calls: list[dict[str, bytes]], primer: str
    ) -> list[dict[str, Any]]:
        """Answer the task ``spec`` once per item of ``calls``, in order."""
        return [self._one(spec, inputs, primer) for inputs in calls]

    def _one(
        self, spec: dict[str, Any], inputs: dict[str, bytes], primer: str
    ) -> dict[str, Any]:
        try:
            called = _Called(spec, inputs, primer)
        except ValueError as error:
            return _reply("failed", str(error))
        returns = called.spec.returns.spec
        name = called.spec.name
        try:
            if self._answer is not None:
                value = self._answer(name, dict(called.inputs))
            else:
                tree = spec["specs"]["specs"]["return"]
                value = _sample(tree, called.spec.types, spec["specs"]["types"])
            # the answer as the return type: a record given as a dict
            # becomes one, and a value that doesn't fit is refused here
            value = returns.decode(values.encode(value))
            return _reply("success", value=values.encode(value).to_bytes())
        except Exception as error:  # noqa: BLE001 - the calling code reads it
            return _reply("failed", f"the stand-in couldn't answer {name}: {error}")


_SCALARS: dict[str, Any] = {
    "any": None,
    "none": None,
    "bool": False,
    "int": 0,
    "float": 0.0,
    "str": "",
    "bytes": b"",
    "timedelta": datetime.timedelta(0),
    "datetime": datetime.datetime(1970, 1, 1),
    "date": datetime.date(1970, 1, 1),
    "time": datetime.time(0),
    "decimal": decimal.Decimal(0),
    "uuid": uuid.UUID(int=0),
    "path": pathlib.PurePosixPath("."),
}


def _sample(
    tree: Mapping[str, Any],
    classes: Mapping[str, Any],
    types: Mapping[str, Any],
    seen: frozenset[str] = frozenset(),
) -> Any:
    """The simplest value of the exported type ``tree``: what a stand-in
    answers with when the embedder gives no ``answer``."""
    kind = tree["t"]
    if kind in _SCALARS:
        return _SCALARS[kind]
    if kind == "literal":
        return tree["values"][0]
    if kind == "union":
        options = tree["of"]
        if any(o.get("t") == "none" for o in options):
            return None
        return _sample(options[0], classes, types, seen)
    if kind == "list":
        return []
    if kind == "set":
        return frozenset() if tree.get("frozen") else set()
    if kind == "dict":
        return {}
    if kind == "tuple":
        if "rest" in tree:
            return ()
        return tuple(_sample(t, classes, types, seen) for t in tree["items"])
    if kind == "ref":
        name = tree["name"]
        cls, entry = classes[name], types[name]
        if entry["kind"] == "enum":
            return next(iter(cls))
        if name in seen:
            raise ValueError(f"{name} holds itself, so it has no simplest value")
        return cls(
            **{
                f["name"]: _sample(f["type"], classes, types, seen | {name})
                for f in entry["fields"]
                if f["required"]
            }
        )
    raise ValueError(
        f"a {kind} has no simplest value here; give the stand-in an answer="
    )
