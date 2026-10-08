"""A task that asks for input: its world is kept, and a resume with the
answer carries on there, from this process or after a restart, its
inputs coming back from the task's own plane."""

import pytest
from nontainer import Store, values
from nontainer.conformance.corpus import ModelStep, calls, fails
from nontainer.turns import ToolEnded
from pydantic_ai import messages as pai
from task_types import Directory, Ranking, Score

from agex import Agent, NeedsInput
from agex.providers.scripted import ScriptedProvider
from agex.task import ANSWER, ASKED, PLANE

SCORES = [Score("ada", 7), Score("bo", 2)]
QUESTION = "Rank by total, or by the best single score?"


def python(code: str) -> ModelStep:
    return calls("run_python", code=code)


ASK = python(f"task.needs_input({QUESTION!r})")
RANK = python("task.success(Ranking(best=scores[0].student, scores=scores))")


def agent(*steps: ModelStep) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(list(steps))
    return Agent(provider), provider


def rank_of(a: Agent):
    @a.task
    def rank(scores: list[Score]) -> Ranking:
        """Rank the scores."""

    return rank


def prompts(provider: ScriptedProvider) -> list[str]:
    return [
        part.content
        for message in provider.seen[-1]
        if isinstance(message, pai.ModelRequest)
        for part in message.parts
        if isinstance(part, pai.UserPromptPart)
    ]


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


# -- asking ----------------------------------------------------------------------------


def test_a_task_that_asks_ends_needs_input_with_its_question():
    a, _ = agent(ASK)
    out = rank_of(a).run(SCORES)
    assert (out.status, out.message, out.value) == ("needs_input", QUESTION, None)
    assert out.ref is not None
    ended = [e for e in out.events if isinstance(e, ToolEnded)]
    assert [(e.result, e.is_error) for e in ended] == [(ASKED, False)]


def test_a_plain_call_raises_needs_input_with_what_resumes_it():
    a, provider = agent(ASK, RANK)
    with pytest.raises(NeedsInput, match="needs_input") as caught:
        rank_of(a)(SCORES)
    asked = caught.value
    assert asked.question == QUESTION and asked.ref == asked.outcome.ref
    out = asked.outcome.resume("by total")
    assert (out.status, out.value) == ("success", Ranking("ada", SCORES))
    assert out.ref is None


# -- resuming in this process ----------------------------------------------------------


def test_resume_carries_on_from_the_answer_with_the_conversation():
    a, provider = agent(ASK, RANK)
    rank = rank_of(a)
    out = rank.run(SCORES)
    done = rank.resume(out.ref, "by total")
    assert (done.status, done.value) == ("success", Ranking("ada", SCORES))
    said = prompts(provider)
    assert said[0].startswith("Rank the scores.")  # the brief, still there
    assert said[-1] == ANSWER.format(answer="by total")


def test_an_answer_that_is_not_text_is_shown_as_a_value():
    a, provider = agent(ASK, RANK)
    out = rank_of(a).run(SCORES)
    out.resume({"by": "total", "ties": "first"})
    assert prompts(provider)[-1] == ANSWER.format(
        answer="{'by': 'total', 'ties': 'first'}"
    )


def test_a_task_can_ask_again():
    a, _ = agent(ASK, python("task.needs_input('And ties?')"), RANK)
    out = rank_of(a).run(SCORES)
    again = out.resume("by total")
    assert (again.status, again.message) == ("needs_input", "And ties?")
    assert again.ref == out.ref
    assert again.resume("first wins").value == Ranking("ada", SCORES)


def test_a_scratch_world_resumes_only_where_it_ran():
    a, _ = agent()
    with pytest.raises(LookupError, match="resumes only in the process that ran it"):
        rank_of(a).resume("rank-nowhere", "by total")


def test_a_scratch_world_resumed_to_its_end_is_gone():
    a, _ = agent(ASK, RANK)
    rank = rank_of(a)
    out = rank.run(SCORES)
    out.resume("by total")
    with pytest.raises(LookupError):
        rank.resume(out.ref, "again")


def test_only_a_task_outcome_that_asked_resumes():
    a, _ = agent(RANK)
    out = rank_of(a).run(SCORES)
    with pytest.raises(ValueError, match="only a task's needs_input outcome"):
        out.resume("anything")


# -- worlds kept in a store ------------------------------------------------------------


def test_a_task_that_asks_keeps_its_fork_and_a_finished_resume_drops_it(store, ws):
    head = ws.head
    a, _ = agent(
        python(
            "open('/workspace/draft.txt', 'w').write('half done')\n"
            f"task.needs_input({QUESTION!r})"
        ),
        python("print(open('/workspace/draft.txt').read())"),
        RANK,
    )
    rank = rank_of(a)
    out = rank.run(SCORES, world=ws)
    assert out.status == "needs_input" and out.ref in store.sessions()
    kept = store.open(out.ref)
    try:
        assert kept.files.read("/workspace/draft.txt") == b"half done"
    finally:
        kept.close()
    done = rank.resume(out.ref, "by total", world=ws)
    assert done.status == "success" and done.ref is None
    first = next(e for e in done.events if isinstance(e, ToolEnded))
    assert first.result.startswith("half done")  # the world it left
    assert store.sessions() == ["main"]
    assert ws.head == head
    assert ws.files.read("/workspace/notes.txt") == b"keep me"


def test_keep_keeps_the_fork_after_a_resume(store, ws):
    a, _ = agent(ASK, RANK)
    rank = rank_of(a)
    out = rank.run(SCORES, world=ws)
    done = rank.resume(out.ref, "by total", world=ws, keep=True)
    assert done.ref == out.ref and out.ref in store.sessions()
    with pytest.raises(ValueError, match="isn't waiting for an answer: it is success"):
        rank.resume(out.ref, "again", world=ws)


def test_a_resume_cut_short_keeps_the_world_waiting(store, ws):
    a, _ = agent(ASK, fails("provider"), RANK)
    rank = rank_of(a)
    out = rank.run(SCORES, world=ws)
    cut = rank.resume(out.ref, "by total", world=ws)
    assert (cut.status, cut.ref) == ("interrupted", out.ref)
    assert out.ref in store.sessions()
    done = cut.resume("by total")
    assert (done.status, done.value) == ("success", Ranking("ada", SCORES))
    assert store.sessions() == ["main"]


def test_the_plane_holds_the_spec_inputs_state_and_value(store, ws):
    a, _ = agent(ASK, RANK)
    rank = rank_of(a)
    out = rank.run(SCORES, world=ws)

    def plane():
        kept = store.open(out.ref)
        try:
            kv = kept.provider.kv
            return {k[len(PLANE) :]: kv[k] for k in kv if k.startswith(PLANE)}
        finally:
            kept.close()

    asked = plane()
    spec = asked["spec"]
    assert (spec["name"], spec["returns"]["type"]) == ("rank", "Ranking")
    assert spec["params"]["scores"]["type"] == "list[Score]"
    assert spec["params"]["scores"]["stored"] is True
    assert spec["params"]["scores"]["schema"]["type"] == "array"
    assert values.decode(asked["inputs/scores"], list[Score]) == SCORES
    assert asked["state"] == {"status": "needs_input", "question": QUESTION}
    assert "value" not in asked
    rank.resume(out.ref, "by total", world=ws, keep=True)
    done = plane()
    assert done["state"] == {"status": "success"}
    assert values.decode(done["value"], Ranking) == Ranking("ada", SCORES)


def test_resume_after_a_restart_with_a_live_input_passed_again(tmp_path):
    def look(a: Agent):
        @a.task
        def look(scores: list[Score], directory: Directory) -> str:
            """Look up the best student."""

        return look

    with Store(tmp_path / "store") as store:
        ws = store.open("main")
        try:
            first, _ = agent(ASK)
            out = look(first).run(SCORES, Directory(), world=ws)
            assert out.status == "needs_input"
            ref = out.ref
        finally:
            ws.close()

    # a new process, as far as the task can tell: a new store, agent and task
    with Store(tmp_path / "store") as store:
        ws = store.open("main")
        try:
            later, provider = agent(
                python("task.success(directory.lookup(scores[0].student))")
            )
            task = look(later)
            with pytest.raises(TypeError, match="needs its input 'directory' again"):
                task.resume(ref, "by total", world=ws)
            assert provider.seen == []
            directory = Directory()
            done = task.resume(ref, "by total", world=ws, directory=directory)
            assert (done.status, done.value) == ("success", "ADA")
            assert directory.looked_up == ["ada"]
        finally:
            ws.close()


def test_what_a_resume_refuses(store, ws):
    a, provider = agent(ASK)
    rank = rank_of(a)
    out = rank.run(SCORES, world=ws)
    with pytest.raises(TypeError, match="'scores' is stored with the task"):
        rank.resume(out.ref, "by total", world=ws, scores=SCORES)
    with pytest.raises(TypeError, match="doesn't take: extra"):
        rank.resume(out.ref, "by total", world=ws, extra=1)
    other, _ = agent()

    @other.task
    def summarize(scores: list[Score]) -> str:
        """Summarize the scores."""

    with pytest.raises(ValueError, match="isn't a world of task 'summarize'"):
        summarize.resume(out.ref, "by total", world=ws)
    with pytest.raises(LookupError, match="no task world 'main.nowhere'"):
        rank.resume("main.nowhere", "by total", world=ws)
    assert len(provider.seen) == 1  # only the run that asked
