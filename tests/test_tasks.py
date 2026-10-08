"""Tasks: a typed call that runs an agent on a world of its own, and
hands back the value its code passed to ``task.success``."""

import asyncio
import collections
import enum
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Union

import pytest
from nontainer import Profile, PythonConfig, Store
from nontainer.conformance.corpus import ModelStep, ToolCall, calls, fails, says
from nontainer.turns import ToolEnded
from pydantic import BaseModel
from pydantic_ai import messages as pai

from agex import Agent, Outcome, TaskFailed, TaskInterrupted
from agex.agent import Session
from agex.providers.scripted import ScriptedProvider
from agex.task import (
    FINISHED,
    INSTRUCTIONS,
    NUDGE,
    NUDGES,
    Task,
    TaskSpec,
    kinds_of,
)


@dataclass
class Response:
    student: str
    answers: list[int]


@dataclass
class Report:
    best: str
    total: int


class Grade(enum.Enum):
    PASS = "pass"
    FAIL = "fail"


class Note(BaseModel):
    text: str
    grade: Grade


@dataclass
class Tree:
    name: str
    children: list["Tree"]


class Frame:
    """Stands in for a table type: it speaks the Arrow stream protocol."""

    def __arrow_c_stream__(self, requested_schema=None):  # pragma: no cover
        raise NotImplementedError


class Client:
    """Stands in for a live resource."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def lookup(self, key: str) -> str:
        self.calls.append(key)
        return key.upper()


RESPONSES = [Response("ada", [3, 4]), Response("bo", [1, 1])]


def python(code: str) -> ModelStep:
    return calls("run_python", code=code)


def agent(*steps: ModelStep, **kw: Any) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(list(steps))
    return Agent(provider, **kw), provider


def best_of(agent: Agent):
    @agent.task
    def best(responses: list[Response]) -> Report:
        """Find the student with the highest total."""

    return best


FIND_BEST = python(
    "top = max(responses, key=lambda r: sum(r.answers))\n"
    "task.success(Report(best=top.student, total=sum(top.answers)))"
)


@pytest.fixture
def store():
    with Store(memory=True) as store:
        yield store


@pytest.fixture
def ws(store):
    ws = store.open("main")
    ws.files.write("/workspace/notes.txt", b"keep me")
    ws.commit(info={"tool": "test"})
    yield ws
    ws.close()


def user_text(request: pai.ModelRequest) -> str:
    return "\n".join(
        p.content for p in request.parts if isinstance(p, pai.UserPromptPart)
    )


# -- calling a task --------------------------------------------------------------------


def test_a_task_hands_back_the_value_its_code_made():
    a, provider = agent(FIND_BEST)
    report = best_of(a)(RESPONSES)
    assert report == Report(best="ada", total=7)
    assert isinstance(report, Report)
    (request,) = provider.seen[0]
    system, user = request.parts
    assert isinstance(system, pai.SystemPromptPart)
    assert system.content.endswith(INSTRUCTIONS)
    assert user.content.startswith("Find the student with the highest total.")
    assert "`responses: list[Response]` = " in user.content and "ada" in user.content
    assert "call `task.success(value)` with a `Report`" in user.content
    assert "class Report:" in user.content and "class Response:" in user.content


def test_the_brief_previews_a_table_and_an_array_on_one_line_each():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    from nontainer.presets import dataframes

    a, provider = agent(
        python("task.success(int(values.sum()))"),
        profile=Profile(python=PythonConfig(modules=[dataframes()])),
    )

    @a.task
    def total(frame: pd.DataFrame, values: np.ndarray) -> int:
        """Sum the values."""

    frame = pd.DataFrame({"name": ["ada", "bo"], "score": [3, 1]})
    assert total(frame, np.array([1, 2, 3], dtype=np.int64)) == 6
    (request,) = provider.seen[0]
    lines = request.parts[1].content.splitlines()
    assert (
        "- `frame: DataFrame` = DataFrame(2x2, {'name': str, 'score': int64}, "
        "[('ada', 3), ('bo', 1)])" in lines
    )
    assert "- `values: ndarray` = ndarray(3, int64, [1, 2, 3])" in lines


def test_run_returns_the_outcome():
    a, _ = agent(FIND_BEST)
    out = best_of(a).run(RESPONSES)
    assert isinstance(out, Outcome)
    assert (out.status, out.value, out.message, out.ref) == (
        "success",
        Report(best="ada", total=7),
        None,
        None,
    )
    ended = [e for e in out.events if isinstance(e, ToolEnded)]
    assert [(e.result, e.is_error) for e in ended] == [(FINISHED, False)]


def test_an_async_task_is_awaited():
    a, _ = agent(FIND_BEST)

    @a.task
    async def best(responses: list[Response]) -> Report:
        """Find the student with the highest total."""

    async def go():
        return await best(RESPONSES)

    assert asyncio.run(go()) == Report(best="ada", total=7)


def test_arun_from_a_running_loop_and_the_blocking_call_refused_there():
    a, _ = agent(FIND_BEST)
    best = best_of(a)

    async def go():
        with pytest.raises(RuntimeError, match=r"best\.arun"):
            best(RESPONSES)
        return await best.arun(RESPONSES)

    assert asyncio.run(go()).value == Report(best="ada", total=7)


def test_a_live_value_comes_back_as_it_is():
    a, _ = agent(python("def double(x):\n    return 2 * x\ntask.success(double)"))

    @a.task
    def doubler() -> Callable[[int], int]:
        """Make a function that doubles a number."""

    double = doubler()
    assert double(21) == 42


def test_a_task_returning_none_finishes_with_no_value():
    a, _ = agent(python("task.success()"))

    @a.task
    def noop() -> None:
        """Do nothing."""

    assert noop() is None


# -- the value must fit ----------------------------------------------------------------


def test_a_value_that_does_not_fit_is_a_type_error_the_script_can_fix():
    a, _ = agent(
        python(
            "try:\n"
            "    task.success(Report(best='ada', total='7'))\n"
            "except TypeError as e:\n"
            "    print('refused:', e)\n"
            "task.success(Report(best='ada', total=7))"
        )
    )
    assert best_of(a)(RESPONSES) == Report(best="ada", total=7)


def test_an_uncaught_type_error_is_a_result_the_model_reads_and_fixes():
    a, provider = agent(python("task.success('ada')"), FIND_BEST)
    out = best_of(a).run(RESPONSES)
    assert out.status == "success"
    first, second = [e for e in out.events if isinstance(e, ToolEnded)]
    assert first.is_error
    assert first.result.endswith(
        "TypeError: task.success(): at value: expected Report, got str 'ada'"
    )
    assert (second.result, second.is_error) == (FINISHED, False)


def test_a_record_of_the_agent_own_making_arrives_as_the_declared_type():
    """A value crosses by its shape, decoded as the return type, so a
    class the agent defined with the same fields comes back as the one
    the task declared."""
    a, _ = agent(
        python(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Report:\n"
            "    best: str\n"
            "    total: int\n"
            "task.success(Report(best='ada', total=7))"
        )
    )
    report = best_of(a)(RESPONSES)
    assert type(report) is Report
    assert report == Report(best="ada", total=7)


def test_the_check_is_strict():
    a, _ = agent(python("task.success('42')"), python("task.success(42)"))

    @a.task
    def answer() -> int:
        """The answer."""

    out = answer.run()
    first = next(e for e in out.events if isinstance(e, ToolEnded))
    assert first.is_error and "at value: expected int, got str '42'" in first.result
    assert out.value == 42


def test_a_script_that_swallows_the_stop_still_ends_the_task():
    a, _ = agent(
        python(
            "try:\n"
            "    task.success(Report(best='ada', total=7))\n"
            "except BaseException:\n"
            "    pass\n"
            "print('ran on')"
        )
    )
    assert best_of(a)(RESPONSES) == Report(best="ada", total=7)


def test_calls_after_the_one_that_finished_are_not_run(store):
    finish = ToolCall(
        name="run_python",
        args={"code": "task.success(Report(best='ada', total=7))"},
    )
    write = ToolCall(
        name="file_write", args={"path": "/workspace/a.txt", "content": "A"}
    )
    a, _ = agent(ModelStep(tool_calls=(finish, write)))
    out = best_of(a).run(RESPONSES)
    assert out.status == "success"
    ended = [e for e in out.events if isinstance(e, ToolEnded)]
    assert [e.name for e in ended] == ["run_python"]


# -- endings without a value -----------------------------------------------------------


def test_task_fail_ends_it_failed():
    a, _ = agent(python("task.fail('no responses to grade')"))
    out = best_of(a).run([])
    assert (out.status, out.message, out.value) == (
        "failed",
        "no responses to grade",
        None,
    )
    again, _ = agent(python("task.fail('no responses to grade')"))
    with pytest.raises(TaskFailed, match="no responses to grade") as caught:
        best_of(again)([])
    assert caught.value.outcome.status == "failed"


def test_a_model_that_stops_is_nudged_then_the_task_fails():
    a, provider = agent(*[says("done")] * (NUDGES + 1))
    with pytest.raises(TaskFailed, match="without calling task.success") as caught:
        best_of(a)(RESPONSES)
    assert caught.value.outcome.status == "failed"
    nudges = [
        part
        for message in provider.seen[-1]
        if isinstance(message, pai.ModelRequest)
        for part in message.parts
        if isinstance(part, pai.UserPromptPart) and part.content == NUDGE
    ]
    assert len(nudges) == NUDGES


def test_a_nudge_brings_the_model_back():
    a, _ = agent(says("all done"), FIND_BEST)
    assert best_of(a)(RESPONSES) == Report(best="ada", total=7)


def test_a_transient_provider_error_interrupts_the_task():
    a, _ = agent(fails("provider"))
    out = best_of(a).run(RESPONSES)
    assert out.status == "interrupted"
    again, _ = agent(fails("provider"))
    with pytest.raises(TaskInterrupted):
        best_of(again)(RESPONSES)


def test_an_error_fails_the_task():
    a, _ = agent(fails("error"))
    assert best_of(a).run(RESPONSES).status == "failed"


def test_a_cancelled_caller_cancels_the_task_and_a_kept_fork_keeps_the_run(store, ws):
    a, _ = agent(python("import time\ntime.sleep(0.5)"), FIND_BEST)
    best = best_of(a)

    async def go():
        running = asyncio.ensure_future(best.arun(RESPONSES, world=ws, keep=True))
        await asyncio.sleep(0.2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(go())
    (fork,) = [s for s in store.sessions() if s.startswith("main.best-")]
    kept = store.open(fork)
    try:
        assert [r.status for r in Session(a, kept).runs] == ["cancelled"]
    finally:
        kept.close()


def test_a_caller_cancelled_while_the_world_opens_leaves_no_fork(store, ws):
    a, provider = agent(FIND_BEST)
    best = best_of(a)

    async def go():
        running = asyncio.ensure_future(best.arun(RESPONSES, world=ws))
        await asyncio.sleep(0)  # into the open, on its thread
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        for _ in range(200):
            if store.sessions() == ["main"]:
                break
            await asyncio.sleep(0.01)

    asyncio.run(go())
    assert store.sessions() == ["main"]
    assert provider.seen == []


def test_a_cancel_then_the_loop_shutting_down_leaves_no_fork(store, ws, monkeypatch):
    """The world finishes opening on its thread after the caller has
    gone and the event loop with it; it is closed and deleted anyway."""
    opening = Task._open

    def slow_open(self, *args):
        time.sleep(0.3)
        return opening(self, *args)

    monkeypatch.setattr(Task, "_open", slow_open)
    a, provider = agent(FIND_BEST)
    best = best_of(a)

    async def go():
        running = asyncio.ensure_future(best.arun(RESPONSES, world=ws))
        await asyncio.sleep(0.05)  # the world is opening on its thread
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(go())  # and the loop is gone, mid-open
    for _ in range(200):
        if store.sessions() == ["main"]:
            break
        time.sleep(0.01)
    assert store.sessions() == ["main"]
    assert provider.seen == []


# -- worlds ----------------------------------------------------------------------------


def test_a_task_on_a_world_leaves_it_untouched_and_its_fork_is_gone(store, ws):
    head = ws.head
    a, _ = agent(
        python(
            "open('/workspace/notes.txt', 'w').write('changed')\n"
            "open('/workspace/new.txt', 'w').write('new')\n"
            "print(open('/workspace/notes.txt').read())"
        ),
        FIND_BEST,
    )
    out = best_of(a).run(RESPONSES, world=ws)
    assert out.status == "success" and out.ref is None
    assert ws.head == head
    assert ws.files.read("/workspace/notes.txt") == b"keep me"
    assert not ws.files.exists("/workspace/new.txt")
    assert store.sessions() == ["main"]


def test_keep_keeps_the_fork_with_the_task_work_and_its_run(store, ws):
    a, _ = agent(
        python("open('/workspace/new.txt', 'w').write('new')"),
        FIND_BEST,
    )
    out = best_of(a).run(RESPONSES, world=ws, keep=True)
    assert out.ref is not None and out.ref.startswith("main.best-")
    assert out.ref in store.sessions()
    kept = store.open(out.ref)
    try:
        assert kept.files.read("/workspace/new.txt") == b"new"
        assert kept.files.read("/workspace/notes.txt") == b"keep me"
        (run,) = Session(a, kept).runs
        assert run.status == "completed" and run.run_id == out.run_id
    finally:
        kept.close()


def test_the_fork_starts_from_the_last_commit_and_commits_nothing_here(store):
    ws = store.open("main", autocommit=False)
    try:
        ws.files.write("/workspace/committed.txt", b"c")
        ws.commit(info={"tool": "test"})
        ws.files.write("/workspace/pending.txt", b"p")
        head = ws.head
        a, _ = agent(
            python(
                "import os\nprint(sorted(os.listdir('/workspace')))\n"
                "task.success(sorted(os.listdir('/workspace')))"
            )
        )

        @a.task
        def listing() -> list[str]:
            """List the workspace."""

        assert listing(world=ws) == ["committed.txt"]
        assert ws.head == head
        assert ws.files.read("/workspace/pending.txt") == b"p"
    finally:
        ws.close()


def test_the_task_world_has_the_world_profile_and_its_own_objects(store):
    ws = store.open(
        "main", profile=Profile(python=PythonConfig(host_objects={"limit": 5}))
    )
    try:
        a, _ = agent(python("task.success(limit + len(responses))"))

        @a.task
        def count(responses: list[Response]) -> int:
            """Add the limit to the number of responses."""

        assert count(RESPONSES, world=ws) == 7
    finally:
        ws.close()


def test_an_input_shadowing_a_host_object_of_the_world_is_refused(store):
    ws = store.open(
        "main", profile=Profile(python=PythonConfig(host_objects={"responses": 1}))
    )
    try:
        a, provider = agent(FIND_BEST)
        with pytest.raises(
            ValueError, match="already has host objects named responses"
        ):
            best_of(a)(RESPONSES, world=ws)
        assert provider.seen == []
        assert store.sessions() == ["main"]
    finally:
        ws.close()


def test_keep_needs_a_world():
    a, provider = agent(FIND_BEST)
    with pytest.raises(ValueError, match="keep= needs world="):
        best_of(a).run(RESPONSES, keep=True)
    assert provider.seen == []


def test_the_scratch_world_is_built_from_the_agent_profile():
    a, _ = agent(
        python("task.success(greeting)"),
        profile=Profile(python=PythonConfig(host_objects={"greeting": "hi"})),
    )

    @a.task
    def greet() -> str:
        """Say the greeting."""

    assert greet() == "hi"


# -- inputs ----------------------------------------------------------------------------


def test_an_argument_that_does_not_fit_is_refused_before_the_model():
    a, provider = agent(FIND_BEST)
    with pytest.raises(TypeError, match="argument 'responses' must be list") as caught:
        best_of(a)([{"student": "ada", "answers": [1]}])
    # what the call raised, not chained to the blocking front door's own
    # check for a running loop
    assert not isinstance(caught.value.__context__, RuntimeError)
    with pytest.raises(TypeError, match="missing a required argument"):
        best_of(a)()
    assert provider.seen == []


def test_a_task_that_changes_its_inputs_leaves_the_caller_values_alone():
    a, _ = agent(
        python(
            "responses.append(Response('cy', [9]))\n"
            "responses[0].answers.append(100)\n"
            "task.success(len(responses))"
        )
    )

    @a.task
    def tamper(responses: list[Response]) -> int:
        """Change the responses."""

    mine = [Response("ada", [3, 4])]
    assert tamper(mine) == 2
    assert mine == [Response("ada", [3, 4])]


def test_a_table_input_changed_in_place_leaves_the_caller_frame_alone():
    pd = pytest.importorskip("pandas")
    from nontainer.presets import dataframes

    a, _ = agent(
        python(
            "frame.loc[0, 'score'] = 99\n"
            "frame.drop(columns=['name'], inplace=True)\n"
            "task.success(int(frame['score'].sum()))"
        ),
        profile=Profile(python=PythonConfig(modules=[dataframes()])),
    )

    @a.task
    def tamper(frame: pd.DataFrame) -> int:
        """Change the table."""

    mine = pd.DataFrame({"name": ["ada", "bo"], "score": [1, 2]})
    assert tamper(mine) == 101
    assert mine.to_dict("list") == {"name": ["ada", "bo"], "score": [1, 2]}


def test_an_array_input_changed_in_place_leaves_the_caller_array_alone():
    """Each run gets its own copy, as it would in a worker or on a dud
    machine."""
    np = pytest.importorskip("numpy")
    from nontainer.presets import dataframes

    a, _ = agent(
        python("values[0] = 99\nprint(int(values.sum()))"),
        python("task.success(int(values.sum()))"),
        profile=Profile(python=PythonConfig(modules=[dataframes()])),
    )

    @a.task
    def total(values: np.ndarray) -> int:
        """Sum the values."""

    mine = np.array([1, 2, 3])
    out = total.run(mine)
    first = next(e for e in out.events if isinstance(e, ToolEnded))
    assert not first.is_error and first.result.strip() == "104"
    assert out.value == 6
    assert mine.tolist() == [1, 2, 3]


def test_a_user_dict_input_is_copied():
    a, _ = agent(python("counts.update(ada=99)\ntask.success(counts['ada'])"))

    @a.task
    def bump(counts: MutableMapping[str, int]) -> int:
        """Bump ada."""

    mine = collections.UserDict(ada=1)
    assert bump(mine) == 99
    assert dict(mine) == {"ada": 1}


class Ledger(Mapping[str, int]):
    """A mapping of a class of its own, holding its entries in a dict."""

    def __init__(self, **entries: int) -> None:
        self.entries = dict(entries)

    def __getitem__(self, key: str) -> int:
        return self.entries[key]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def test_an_input_declared_as_data_is_copied_whatever_its_class():
    a, _ = agent(python("ledger.entries['ada'] = 99\ntask.success(ledger['ada'])"))

    @a.task
    def peek(ledger: Mapping[str, int]) -> int:
        """Peek at ada."""

    mine = Ledger(ada=1)
    assert peek(mine) == 99
    assert mine.entries == {"ada": 1}


def test_a_live_input_is_bound_as_it_is():
    a, _ = agent(python("task.success(client.lookup('ada'))"))

    @a.task
    def look(client: Client) -> str:
        """Look up ada."""

    client = Client()
    assert look(client) == "ADA"
    assert client.calls == ["ada"]


def test_a_container_of_live_objects_is_the_task_own_sharing_them():
    """Its data is copied, so the caller's container is untouched; its
    live objects are the caller's own."""
    a, _ = agent(
        python(
            "clients['b'].lookup('x')\n"
            "clients.clear()\n"
            "things.append(3)\n"
            "things[0].lookup('y')\n"
            "task.success(len(clients) + len(things))"
        )
    )

    @a.task
    def use(clients: dict[str, Client], things: list) -> int:
        """Use the clients."""

    first, second = Client(), Client()
    clients = {"a": first, "b": second}
    things = [first, {"n": [1, 2]}]
    assert use(clients, things) == 3
    assert clients == {"a": first, "b": second}
    assert things == [first, {"n": [1, 2]}]
    assert second.calls == ["x"] and first.calls == ["y"]


# -- the spec --------------------------------------------------------------------------


def test_kinds_say_what_a_type_needs_carried():
    assert kinds_of(list[Response]) == {"data"}
    assert kinds_of(Note) == {"data"}
    assert kinds_of(Tree) == {"data"}
    assert kinds_of(dict[str, Frame]) == {"data", "table"}
    assert kinds_of(bytes) == {"bytes"}
    assert kinds_of(Callable[[int], int]) == {"live"}
    assert kinds_of(Client) == {"live"}
    assert kinds_of(Union[Report, Client]) == {"data", "live"}
    assert kinds_of(Any) == {"any"}
    assert kinds_of(list) == {"data", "any"}
    np = pytest.importorskip("numpy")
    assert kinds_of(np.ndarray) == {"array"}


def test_a_schema_only_for_data():
    def spec_of(annotation):
        def fn(x: annotation) -> None:  # type: ignore[valid-type]
            """Spec."""

        return TaskSpec.of(fn).params["x"]

    assert spec_of(list[Response]).schema is not None
    assert spec_of(Note).schema["properties"]["grade"]
    # a union with an opaque branch would lose the branch in a schema
    assert spec_of(Union[Report, Client]).schema is None
    assert spec_of(Frame).schema is None
    assert spec_of(Any).schema is None


def test_the_spec_names_the_types_agent_code_builds():
    a, _ = agent()

    @a.task
    def judge(notes: list[Note], tree: Tree) -> Report:
        """Judge."""

    assert set(judge.spec.types) == {"Note", "Grade", "Tree", "Report"}
    assert judge.spec.instructions == "Judge."


def test_a_task_body_is_its_docstring():
    a, _ = agent()
    with pytest.raises(TypeError, match="code in its body"):

        @a.task
        def busy() -> int:
            """Busy."""
            return 1

    @a.task
    def fine() -> int:
        """Fine."""
        ...


@pytest.mark.parametrize("name", ["task", "host", "world", "keep"])
def test_reserved_parameter_names_are_refused(name):
    a, _ = agent()
    namespace: dict[str, Any] = {}
    exec(f'def fn({name}: int) -> int:\n    """Doc."""', namespace)
    with pytest.raises(TypeError, match="reserved"):
        a.task(namespace["fn"])


def test_inputs_are_named():
    a, _ = agent()
    with pytest.raises(TypeError, match=r"takes \*rest"):

        @a.task
        def many(*rest: int) -> int:
            """Many."""


def test_a_parameter_named_like_a_type_is_refused():
    a, _ = agent()
    with pytest.raises(TypeError, match="named like a type"):

        @a.task
        def odd(Report: int) -> Report:  # noqa: N803
            """Odd."""
