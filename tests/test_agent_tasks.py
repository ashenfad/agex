"""Agent-defined tasks: agent code defines a task with ``agex``, and a
helper agent runs each call as a delegate of the calling world
(``agex.agent_tasks``)."""

import asyncio
import enum
import sys
import textwrap
from dataclasses import replace

import pytest
from nontainer import Profile, PythonConfig, Store
from nontainer.conformance.corpus import calls, says

from agex import Agent
from agex.agent_tasks import agent_tasks
from agex.providers.scripted import ScriptedProvider


class Slow(ScriptedProvider):
    """A scripted model that takes ``delay`` seconds a reply, and counts
    how many replies were being made at once."""

    def __init__(self, steps, delay):
        super().__init__(steps)
        self.delay = delay
        self.now = 0
        self.most = 0

    async def stream(self, *args, **kwargs):
        self.now += 1
        self.most = max(self.most, self.now)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.now -= 1
        async for event in super().stream(*args, **kwargs):
            yield event


def routed(*, delay=0.0, **scripts):
    """One scripted model for every session: a request goes to the
    script whose key opens the session's first message."""
    queues = {key: list(steps) for key, steps in scripts.items()}

    def next_step(sent):
        first = sent[1] if len(sent) > 1 else ""
        for key, queue in queues.items():
            if first.startswith(key):
                assert queue, f"{key}'s script ran out"
                return queue.pop(0)
        raise AssertionError(f"no script for a session opening {first[:60]!r}")

    provider = Slow(next_step, delay) if delay else ScriptedProvider(next_step)
    return Agent(provider)


def runs(code):
    return calls("run_python", code=textwrap.dedent(code))


class Lookup:
    """A host object of the world's own."""

    def find(self, key: str) -> str:
        return key.upper()


@pytest.fixture
def store():
    st = Store(memory=True)
    yield st
    st.close()


@pytest.fixture(params=["none", "process", "dud"])
def rung(request):
    return request.param


def world(store, agent, *, rung="none", scratch=False, objects=None, **python):
    python = PythonConfig(
        host_objects={"agex": agent_tasks(agent, scratch=scratch), **(objects or {})},
        **python,
    )
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
    return store.open("lead", profile=profile)


class Lead:
    """One lead turn, run: what its script printed, and its session and
    world for a closer look."""

    def __init__(self, store, agent, *, sessions=True, **world_kw):
        self.ws = world(store, agent, **world_kw)
        self.chat = agent.session(self.ws, sessions=sessions)
        self.commits = len(self.ws.log())
        try:
            self.outcome = self.chat.say("LEAD go")
        except BaseException:
            self.close()
            raise
        assert self.outcome.status == "completed", self.outcome.message

    @property
    def printed(self):
        [run] = self.chat.runs
        return run.messages[2].parts[0].content

    def close(self):
        self.chat.close()
        self.ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -- the value comes back as the caller's own classes -------------------------------


CORNERS = """\
    from dataclasses import dataclass

    @dataclass
    class Point2D:
        x: float
        y: float

        def norm(self):
            return (self.x ** 2 + self.y ** 2) ** 0.5

    @agex.task
    def corners(shape: str) -> list[Point2D]:
        \"""HELPER The corner points of the named shape, counterclockwise.\"""

    pts = corners("unit square")
    print(type(pts[0]) is Point2D, [p.norm() for p in pts])
"""

SQUARE = """\
    task.success([Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)])
"""


def test_a_task_the_code_defines_returns_its_own_classes(store, rung):
    agent = routed(LEAD=[runs(CORNERS), says("done")], HELPER=[runs(SQUARE)])
    with Lead(store, agent, rung=rung) as lead:
        assert "True [0.0, 1.0, 1.4142135623730951, 1.0]" in lead.printed, lead.printed
        [job] = lead.chat.sessions.list()
        assert (job.name, job.status) == ("lead.corners-1", "answered")
        assert job.task == "corners(shape='unit square')"


RICH = """\
    import enum
    from dataclasses import dataclass, field

    import numpy as np
    import pandas as pd

    class Color(enum.Flag):
        RED = 1
        BLUE = 2

    @dataclass
    class Leaf:
        name: str
        color: Color

    @dataclass
    class Bundle:
        leaves: list[Leaf]
        table: pd.DataFrame
        grid: np.ndarray
        blob: bytes
        tags: dict[str, int] = field(default_factory=dict)

    @agex.task
    def bundle(n: int) -> Bundle:
        \"""HELPER Make a bundle of n leaves.\"""

    b = bundle(2)
    print(type(b) is Bundle, [type(l) is Leaf for l in b.leaves],
          b.leaves[1].color == Color.RED | Color.BLUE, type(b.table).__name__,
          b.table["x"].tolist(), b.grid.tolist(), b.blob, b.tags)
"""

BUNDLE = """\
    import numpy as np
    import pandas as pd
    task.success(Bundle(
        leaves=[Leaf("a", Color.RED), Leaf("b", Color.RED | Color.BLUE)],
        table=pd.DataFrame({"x": list(range(n))}),
        grid=np.arange(3),
        blob=b"hi",
        tags={"k": n},
    ))
"""


def test_records_enums_tables_and_arrays_come_back_as_declared(store, rung):
    from nontainer.presets import dataframes

    agent = routed(LEAD=[runs(RICH), says("done")], HELPER=[runs(BUNDLE)])
    with Lead(store, agent, rung=rung, modules=[dataframes(), enum]) as lead:
        assert (
            "True [True, True] True DataFrame [0, 1] [0, 1, 2] b'hi' {'k': 2}"
            in lead.printed
        ), lead.printed


# -- what the decorator refuses ------------------------------------------------------


REFUSED = """\
    from typing import Callable

    def attempt(make):
        try:
            make()
        except TypeError as error:
            print("REFUSED", str(error).splitlines()[0])
        else:
            print("ACCEPTED")

    attempt(lambda: agex.task(lambda x: x))

    def undocumented(x: int) -> int:
        pass
    attempt(lambda: agex.task(undocumented))

    def unannotated(x) -> int:
        \"""Do it.\"""
    attempt(lambda: agex.task(unannotated))

    def no_return(x: int):
        \"""Do it.\"""
    attempt(lambda: agex.task(no_return))

    def live(x: int) -> Callable[[int], int]:
        \"""Do it.\"""
    attempt(lambda: agex.task(live))

    def clash(db: str) -> str:
        \"""Do it.\"""
    attempt(lambda: agex.task(clash))

    def reserved(task: str) -> str:
        \"""Do it.\"""
    attempt(lambda: agex.task(reserved))
"""


def test_the_decorator_refuses_saying_what_to_write(store, rung):
    agent = routed(LEAD=[runs(REFUSED), says("done")])
    with Lead(store, agent, rung=rung, objects={"db": Lookup()}) as lead:
        lines = [line for line in lead.printed.splitlines() if "REFUSED" in line]
        assert "ACCEPTED" not in lead.printed, lead.printed
        assert len(lines) == 7, lead.printed
        for line, said in zip(
            lines,
            [
                "needs a def, not a lambda",
                "has no docstring",
                "parameter 'x' has no annotation",
                "has no return annotation",
                "has a live part",
                "a host object of this world",
                "'task', a name the helper's world binds itself",
            ],
        ):
            assert said in line, line


def test_code_in_the_body_is_refused_where_its_source_is_readable():
    from agex.stubs import _compile

    def busy(x: int) -> int:
        """Do it."""
        return x + 1

    with pytest.raises(TypeError, match="has code in its body"):
        _compile(busy, {}, set())


# -- a helper that hands back no value ---------------------------------------------------


ENDINGS = """\
    @agex.task
    def pick(options: list[str]) -> str:
        \"""HELPER Pick the best option.\"""

    for _ in range(2):
        try:
            pick(["a", "b"])
        except agex.TaskNeedsInput as asked:
            print("ASKED", asked.question)
        except agex.TaskFailed as failed:
            print("FAILED", failed)
"""


def test_a_helper_that_fails_or_asks_raises_in_the_calling_code(store, rung):
    agent = routed(
        LEAD=[runs(ENDINGS), says("done")],
        HELPER=[
            runs("task.fail('no way to tell')"),
            runs("task.needs_input('best by which measure?')"),
        ],
    )
    with Lead(store, agent, rung=rung) as lead:
        assert "FAILED task 'pick' failed: no way to tell" in lead.printed, lead.printed
        assert "ASKED best by which measure?" in lead.printed, lead.printed


STRICT = """\
    from dataclasses import dataclass

    @dataclass
    class Positive:
        n: int

        def __post_init__(self):
            if self.n <= 0:
                raise ValueError("n must be positive")
            assert self.n < 100

    @dataclass
    class Small:
        n: int

        def __post_init__(self):
            assert self.n < 100

    @agex.task
    def count(word: str) -> Positive:
        \"""HELPER Count the vowels.\"""

    try:
        count("xyz")
    except agex.TaskFailed as failed:
        print("FAILED", failed)

    @agex.task
    def big(word: str) -> Small:
        \"""HELPER Make it big.\"""

    try:
        big("xyz")
    except agex.TaskFailed as failed:
        print("ALSO FAILED", failed)
"""


def test_a_value_the_callers_class_rejects_fails_the_call(store):
    agent = routed(
        LEAD=[runs(STRICT), says("done")],
        HELPER=[runs("task.success(Positive(0))"), runs("task.success(Small(500))")],
    )
    with Lead(store, agent) as lead:
        assert "FAILED task 'count' handed back a value that isn't" in lead.printed
        assert "n must be positive" in lead.printed, lead.printed
        assert "ALSO FAILED task 'big' handed back a value that isn't Small" in (
            lead.printed
        ), lead.printed
        assert "AssertionError" in lead.printed, lead.printed


def test_any_error_the_callers_class_raises_is_task_failed():
    """A record's own check is a mismatch already; a pydantic model's
    ``model_post_init`` can raise anything, and that fails the call too."""
    from pydantic import BaseModel

    from agex.stubs import AgentTask, TaskFailed

    class Capped(BaseModel):
        n: int

        def model_post_init(self, context):
            if self.n > 100:
                raise RuntimeError("too big")

    def cap(word: str) -> Capped:
        """Cap it."""

    made = AgentTask(None, cap, None, {"Capped": Capped}, set())
    from nontainer import values

    blob = values.encode(Capped(n=1)).to_bytes()
    assert made._value({"status": "success", "value": blob}) == Capped(n=1)
    big = values.encode({"n": 500}).to_bytes()
    with pytest.raises(TaskFailed, match="RuntimeError|too big") as failed:
        made._value({"status": "success", "value": big})
    assert isinstance(failed.value.__cause__, RuntimeError)


# -- .map ---------------------------------------------------------------------------


MAP = """\
    @agex.task
    def shout(word: str) -> str:
        \"""HELPER Shout the word.\"""

    print(shout.map(["a", "b", "c"]))
"""


def test_map_runs_at_once_and_returns_in_order(store, rung):
    agent = routed(
        delay=0.2,
        LEAD=[runs(MAP), says("done")],
        HELPER=[runs("task.success(word.upper() * 2)")] * 3,
    )
    with Lead(store, agent, rung=rung) as lead:
        assert "['AA', 'BB', 'CC']" in lead.printed, lead.printed
        assert agent.provider.most >= 2
        assert sorted(j.name for j in lead.chat.sessions.list()) == [
            "lead.shout-1",
            "lead.shout-2",
            "lead.shout-3",
        ]


# -- the helper and its world ----------------------------------------------------------------


SLOW = """\
    @agex.task
    def wait(seconds: float) -> str:
        \"""HELPER Take your time.\"""

    print(wait(1.0))
"""


def test_helper_time_doesnt_end_the_callers_script(store, rung):
    """The lead's script allows 1 second, and its helper's two model
    calls take 1.2: a host call is the host's time."""
    agent = routed(
        delay=0.6,
        LEAD=[runs(SLOW), says("done")],
        HELPER=[runs("x = 1"), runs("task.success('took it')")],
    )
    with Lead(store, agent, rung=rung, timeout=1.0) as lead:
        assert "took it" in lead.printed, lead.printed


WORLD = """\
    with open("/workspace/notes.md", "w") as f:
        f.write("mine")

    @agex.task
    def look(word: str) -> list[str]:
        \"""HELPER Look around.\"""

    print(look("w"))
"""

LOOK = """\
    import os
    task.success([
        str(os.path.exists("/workspace/seed.md")),
        str("agex" in dir(__import__("host"))),
        str(db.find(word)),
    ])
"""


def test_a_helper_holds_its_callers_world_but_agex_and_sees_no_files(store):
    agent = routed(LEAD=[runs(WORLD), says("done")], HELPER=[runs(LOOK)])
    ws = world(store, agent, objects={"db": Lookup()})
    ws.files.write("/workspace/seed.md", "seed")
    ws.index.commit("seed")
    ws.close()
    with Lead(store, agent, objects={"db": Lookup()}) as lead:
        assert "['False', 'False', 'W']" in lead.printed, lead.printed


def test_the_callers_script_lands_in_one_commit_and_its_model_gets_no_note(store):
    agent = routed(LEAD=[runs(WORLD), says("done")], HELPER=[runs(LOOK)])
    with Lead(store, agent, objects={"db": Lookup()}) as lead:
        made = lead.ws.log()[: len(lead.ws.log()) - lead.commits]
        # the script's commit and the turn's, as for a script with no task:
        # the helper's fork landed nothing of the script's for it
        assert [c.info.get("tool") for c in made] == ["turn", "run_python"]
        script = made[1]
        assert "/workspace/notes.md" in lead.ws.diff(script.parents[0], script.id).added
        assert lead.ws.files.read("/workspace/notes.md") == b"mine"
        [run] = lead.chat.runs
        assert all("delegate" not in str(m) for m in run.messages)
        assert lead.chat.sessions.take() == []
        assert lead.chat.sessions.outstanding() == []


# -- scratch worlds ------------------------------------------------------------------


def test_without_a_helper_open_a_task_runs_on_a_scratch_world(store, rung):
    agent = routed(LEAD=[runs(CORNERS), says("done")], HELPER=[runs(SQUARE)])
    with Lead(store, agent, rung=rung, sessions=False) as lead:
        assert "True [0.0, 1.0, 1.4142135623730951, 1.0]" in lead.printed
        assert sorted(store.sessions()) == ["lead"]


def test_an_embedder_can_put_every_helper_on_a_scratch_world(store):
    agent = routed(LEAD=[runs(CORNERS), says("done")], HELPER=[runs(SQUARE)])
    with Lead(store, agent, scratch=True) as lead:
        assert "True [0.0, 1.0, 1.4142135623730951, 1.0]" in lead.printed
        assert lead.chat.sessions.list() == []
        assert sorted(store.sessions()) == ["lead"]


class Looping(ScriptedProvider):
    """A scripted model that notes which loop each reply is made on."""

    def __init__(self, steps):
        super().__init__(steps)
        self.loops = []

    async def stream(self, *args, **kwargs):
        self.loops.append(asyncio.get_running_loop())
        async for event in super().stream(*args, **kwargs):
            yield event


MAP_CORNERS = CORNERS.replace(
    'pts = corners("unit square")', 'pts = corners.map(["unit square"] * 2)[0]'
)


@pytest.mark.parametrize("code", [CORNERS, MAP_CORNERS], ids=["call", "map"])
def test_a_scratch_helper_runs_on_the_turns_own_loop(store, code):
    """A turn driven from a loop of the caller's own: its scratch helpers'
    model calls are made there too, where the model's client opened its
    connections, and not on agex's own loop."""
    routes = routed(LEAD=[runs(code), says("done")], HELPER=[runs(SQUARE)] * 2)
    agent = Agent(Looping(routes.provider._next))
    ws = world(store, agent)
    chat = agent.session(ws)

    async def turn():
        outcome = await chat.asay("LEAD go")
        assert outcome.status == "completed", outcome.message
        return asyncio.get_running_loop()

    try:
        loop = asyncio.run(turn())
        [run] = chat.runs
        assert "True [0.0, 1.0" in run.messages[2].parts[0].content
        helpers = len(agent.provider.loops) - 2  # the lead made two calls
        assert helpers >= 1
        assert all(seen is loop for seen in agent.provider.loops)
    finally:
        chat.close()
        ws.close()


PROFILED = """\
    @agex.task
    def rank(names: list[str]) -> Ranking:
        \"""HELPER Rank the students.\"""

    r = rank(["ann", "bob"])
    print(type(r) is Ranking, [type(s) is Score for s in r.scores], r.best)
"""

RANKED = """\
    task.success(Ranking(best=names[0], scores=[Score(n, 1) for n in names]))
"""


def test_a_task_can_name_the_classes_its_world_provides(store):
    from task_types import Ranking, Score

    agent = routed(LEAD=[runs(PROFILED), says("done")], HELPER=[runs(RANKED)])
    with Lead(store, agent, classes=(Ranking, Score)) as lead:
        assert "True [True, True] ann" in lead.printed, lead.printed
