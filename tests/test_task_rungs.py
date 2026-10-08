"""Tasks on every rung: in this process, under process isolation and on
a dud machine, the same value comes back for data, tables, arrays and
bytes, and what can't cross is refused before any model call."""

import sys
from collections.abc import Callable

import pytest
from nontainer import NotSupportedError, Profile, PythonConfig, Store
from nontainer.conformance.corpus import ModelStep, calls
from nontainer.turns import ToolEnded
from task_types import Directory, Ranking, Score

from agex import Agent
from agex.providers.scripted import ScriptedProvider

SCORES = [Score("ada", 7), Score("bo", 2)]


def python(code: str) -> ModelStep:
    return calls("run_python", code=code)


def profile(rung: str, **python_kw) -> Profile:
    if rung == "dud":
        if sys.version_info < (3, 11):
            pytest.skip("dud needs Python 3.11+")
        pytest.importorskip("dud")
        from nontainer.executor_dud import DudExecutor

        return Profile(
            python=PythonConfig(**python_kw),
            executor_factory=lambda: DudExecutor(backend="subprocess"),
        )
    return Profile(python=PythonConfig(isolation=rung, **python_kw))


def frames() -> list:
    from nontainer.presets import dataframes

    return [dataframes()]


@pytest.fixture(params=["none", "process", "dud"])
def rung(request):
    return request.param


def agent(rung: str, *steps: ModelStep, **python_kw) -> Agent:
    return Agent(ScriptedProvider(list(steps)), profile=profile(rung, **python_kw))


# -- the same value back, on every rung -------------------------------------------------


def test_data(rung):
    a = agent(
        rung,
        python(
            "top = max(scores, key=lambda s: s.total)\n"
            "task.success(Ranking(best=top.student, scores=scores))"
        ),
    )

    @a.task
    def rank(scores: list[Score]) -> Ranking:
        """Rank the scores."""

    ranking = rank(SCORES)
    assert ranking == Ranking(best="ada", scores=SCORES)
    assert type(ranking) is Ranking and type(ranking.scores[0]) is Score


def test_a_table(rung):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    a = agent(
        rung,
        python("frame['double'] = frame['score'] * 2\ntask.success(frame)"),
        modules=frames(),
    )

    @a.task
    def double(frame: pd.DataFrame) -> pd.DataFrame:
        """Add a column with each score doubled."""

    mine = pd.DataFrame({"name": ["ada", "bo"], "score": [3, 1]})
    out = double(mine)
    pd.testing.assert_frame_equal(
        out, pd.DataFrame({"name": ["ada", "bo"], "score": [3, 1], "double": [6, 2]})
    )
    assert list(mine.columns) == ["name", "score"]


def test_an_array(rung):
    np = pytest.importorskip("numpy")
    a = agent(rung, python("task.success(values * 2)"), modules=frames())

    @a.task
    def double(values: np.ndarray) -> np.ndarray:
        """Double the values."""

    out = double(np.array([1, 2, 3]))
    assert isinstance(out, np.ndarray) and out.tolist() == [2, 4, 6]


def test_bytes(rung):
    a = agent(rung, python("task.success(data[::-1])"))

    @a.task
    def reverse(data: bytes) -> bytes:
        """Reverse the bytes."""

    assert reverse(b"\x00\x01\xff") == b"\xff\x01\x00"


def test_a_value_that_does_not_fit_is_refused_at_the_call_the_same_way(rung):
    a = agent(
        rung,
        python("x = 1\ntask.success(Ranking(best='ada', scores=[Score('ada', '7')]))"),
        python("task.success(Ranking(best='ada', scores=scores))"),
    )

    @a.task
    def rank(scores: list[Score]) -> Ranking:
        """Rank the scores."""

    out = rank.run(SCORES)
    assert out.status == "success"
    first = next(e for e in out.events if isinstance(e, ToolEnded))
    assert first.is_error and "line 2" in first.result
    assert first.result.endswith(
        "TypeError: task.success(): at value.scores[0].total: expected int, got str '7'"
    )


def test_a_live_input_is_a_capability(rung):
    a = agent(rung, python("task.success(directory.lookup('ada'))"))

    @a.task
    def look(directory: Directory) -> str:
        """Look up ada."""

    directory = Directory()
    assert look(directory) == "ADA"
    assert directory.looked_up == ["ada"]


def test_needs_input_then_resume(rung):
    a = agent(
        rung,
        python("task.needs_input('By total?')"),
        python("task.success(Ranking(best=scores[0].student, scores=scores))"),
    )

    @a.task
    def rank(scores: list[Score]) -> Ranking:
        """Rank the scores."""

    out = rank.run(SCORES)
    assert (out.status, out.message) == ("needs_input", "By total?")
    done = out.resume("yes")
    assert done.value == Ranking(best="ada", scores=SCORES)


def test_task_fail_and_the_world_untouched(rung):
    with Store(memory=True) as store:
        ws = store.open("main", profile=profile(rung))
        try:
            ws.files.write("/workspace/notes.txt", b"keep me")
            ws.commit(info={"tool": "test"})
            head = ws.head
            a = Agent(
                ScriptedProvider(
                    [
                        python(
                            "open('notes.txt', 'w').write('changed')\n"
                            "task.fail('nothing to rank')"
                        )
                    ]
                )
            )

            @a.task
            def rank(scores: list[Score]) -> Ranking:
                """Rank the scores."""

            out = rank.run([], world=ws)
            assert (out.status, out.message) == ("failed", "nothing to rank")
            assert ws.head == head
            assert ws.files.read("/workspace/notes.txt") == b"keep me"
        finally:
            ws.close()


# -- what can't cross ------------------------------------------------------------------


@pytest.mark.parametrize("rung_off", ["process", "dud"])
def test_a_live_return_type_is_refused_off_in_process_before_the_model(rung_off):
    provider = ScriptedProvider([python("task.success(lambda x: 2 * x)")])
    a = Agent(provider, profile=profile(rung_off))

    @a.task
    def doubler() -> Callable[[int], int]:
        """Make a function that doubles a number."""

    with pytest.raises(NotSupportedError, match=r"returns Callable\[\[int\], int\]"):
        doubler()
    assert provider.seen == []


def test_an_input_mixing_data_and_a_live_part_is_refused_off_in_process():
    provider = ScriptedProvider([])
    a = Agent(provider, profile=profile("process"))

    @a.task
    def use(directories: dict[str, Directory]) -> str:
        """Use the directories."""

    with pytest.raises(NotSupportedError, match="'directories' is dict"):
        use({"main": Directory()})
    assert provider.seen == []


def test_a_live_object_where_anything_goes_is_refused_off_in_process():
    provider = ScriptedProvider([])
    a = Agent(provider, profile=profile("process"))

    @a.task
    def use(things: list) -> int:
        """Count the things."""

    with pytest.raises(NotSupportedError, match="'things' holds a live object"):
        use([Directory()])
    assert provider.seen == []
