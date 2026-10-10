"""Agent-defined tasks in apps: a handler calls a task defined in a
module, its helpers are the session's delegates, and a stand-in
(``agex.stand_in``) answers without a model for a page check or a
handler test."""

import sys
import textwrap
from dataclasses import dataclass, replace
from typing import Literal, Optional

import pytest
from nontainer import Profile, PythonConfig, Store, values
from nontainer.apps import enable_apps, request
from test_agent_tasks import SQUARE, routed, runs, says

from agex.agent_tasks import agent_tasks
from agex.stand_in import StandInTasks, stand_in_tasks

TASKS = textwrap.dedent('''
    from dataclasses import dataclass
    from host import agex

    @dataclass
    class Point2D:
        x: float
        y: float

    @agex.task
    def corners(shape: str) -> list[Point2D]:
        """HELPER The corner points of the named shape."""
''')

HANDLER = textwrap.dedent("""
    from app.tasks import corners

    def post(req):
        pts = corners(req.require("shape"))
        return {"n": len(pts), "first": [pts[0].x, pts[0].y] if pts else None}
""")


@pytest.fixture(params=["none", "process", "dud"])
def rung(request):
    return request.param


def world(store, objects, rung="none"):
    python = PythonConfig(host_objects=objects)
    if rung == "dud":
        if sys.version_info < (3, 11):
            pytest.skip("dud needs Python 3.11+")
        pytest.importorskip("dud")
        from nontainer.executor_dud import DudExecutor

        profile = Profile(
            python=python, executor_factory=lambda: DudExecutor(backend="subprocess")
        )
    else:
        profile = Profile(python=replace(python, isolation=rung))
    ws = store.open("lead", profile=profile)
    apps = enable_apps(ws)
    ws.files.fs.makedirs("/workspace/app/api", exist_ok=True)
    ws.files.fs.write("/workspace/app/tasks.py", TASKS.encode())
    ws.files.fs.write("/workspace/app/api/corners.py", HANDLER.encode())
    return ws, apps


POST = request("POST", "/api/corners", body=b'{"shape": "square"}')


def test_a_handler_calls_a_task_its_helpers_the_sessions_delegates(rung):
    """The live preview runs against the session's own world, so a task a
    handler calls is one more of the session's delegates; the module that
    defines it is imported per request, and defining it again is fine."""
    store = Store(memory=True)
    agent = routed(LEAD=[says("hi")], HELPER=[runs(SQUARE)] * 2)
    ws, apps = world(store, {"agex": agent_tasks(agent)}, rung)
    chat = agent.session(ws, sessions=True)
    try:
        chat.say("LEAD hello")  # the session's first turn builds its helper
        for _ in range(2):
            r = apps.dispatch(POST)
            assert (r.status, r.content) == (200, b'{"n": 4, "first": [0.0, 0.0]}')
        assert [(j.name, j.status) for j in chat.sessions.list()] == [
            ("lead.corners-1", "answered"),
            ("lead.corners-2", "answered"),
        ]
    finally:
        chat.close()
        ws.close()
        store.close()


def test_a_bound_stand_in_answers_without_a_model(rung):
    """``bind={"agex": "agex_stand_in"}``, as test_app and ws-curl take
    it: the request's tasks are answered by the stand-in, and the next
    request's by the real thing."""
    store = Store(memory=True)
    agent = routed(HELPER=[runs(SQUARE)])
    objects = {
        "agex": agent_tasks(agent),
        "agex_stand_in": stand_in_tasks(
            lambda name, inputs: [{"x": 9.0, "y": 9.0}] if name == "corners" else None
        ),
    }
    ws, apps = world(store, objects, rung)
    try:
        bound = apps.dispatch(POST, bind={"agex": "agex_stand_in"})
        assert bound.content == b'{"n": 1, "first": [9.0, 9.0]}'
        real = apps.dispatch(POST)
        assert real.content == b'{"n": 4, "first": [0.0, 0.0]}'
    finally:
        ws.close()
        store.close()


HANDLER_WITH_TASK = textwrap.dedent('''
    from host import agex

    @agex.task
    def shout(word: str) -> str:
        """Shout the word."""

    def post(req):
        return {"said": shout(req.json["word"])}
''')

TEST = textwrap.dedent("""
    from host import call

    def test_shout():
        resp = call("shout", "POST", json={"word": "hi"}, agex=agex_stand_in)
        assert resp.json == {"said": "HI!"}, resp.json
""")


def test_a_handler_test_substitutes_the_stand_in():
    """ws-pytest's ``call(..., agex=...)`` substitutes in the handler's
    own module, so a handler that defines its task there is tested
    against the stand-in."""
    store = Store(memory=True)
    objects = {
        "agex": agent_tasks(routed()),
        "agex_stand_in": stand_in_tasks(
            lambda name, inputs: inputs["word"].upper() + "!"
        ),
    }
    ws, _ = world(store, objects)
    try:
        ws.files.fs.write("/workspace/app/api/shout.py", HANDLER_WITH_TASK.encode())
        ws.files.fs.makedirs("/workspace/tests", exist_ok=True)
        ws.files.fs.write("/workspace/tests/test_shout.py", TEST.encode())
        r = ws.terminal("ws-pytest tests/test_shout.py")
        assert r.exit_code == 0, r.stdout
    finally:
        ws.close()
        store.close()


# -- what a stand-in answers --------------------------------------------------------


class Kind(__import__("enum").Enum):
    FIRST = "first"
    SECOND = "second"


@dataclass
class Shape:
    name: str
    kind: Kind
    sides: list[int]
    note: Optional[str] = None
    pair: tuple[int, str] = (1, "a")


@dataclass
class Node:
    label: str
    child: "Node"


def answer(ret, answer=None):
    specs = {"word": values.Spec.of(str), "return": values.Spec.of(ret)}
    data = {
        "format": "agex-task/1",
        "name": "make",
        "doc": "Make it.",
        "params": ["word"],
        "specs": values.export_specs(specs),
    }

    class World:
        session = "s"

    reply = StandInTasks(World(), answer).call(  # type: ignore[arg-type]
        data, {"word": values.encode("hi").to_bytes()}, ""
    )
    if reply["status"] != "success":
        return reply
    return values.Spec.of(ret).decode(reply["value"])


@pytest.mark.parametrize(
    "ret, made",
    [
        (int, 0),
        (str, ""),
        (list[Shape], []),
        (dict[str, int], {}),
        (Optional[int], None),
        (Literal["b", "c"], "b"),
        (tuple[int, str], (0, "")),
        (Kind, Kind.FIRST),
        (Shape, Shape("", Kind.FIRST, [])),
    ],
)
def test_without_an_answer_a_stand_in_makes_the_simplest_value(ret, made):
    assert answer(ret) == made


def test_an_answer_may_give_a_record_as_a_dict():
    assert answer(
        Shape,
        lambda name, inputs: {"name": inputs["word"], "kind": "second", "sides": [3]},
    ) == Shape("hi", Kind.SECOND, [3])


def test_a_stand_in_says_when_it_cant_answer():
    assert "expected Shape" in answer(Shape, lambda name, inputs: 3)["message"]
    assert "holds itself" in answer(Node)["message"]
