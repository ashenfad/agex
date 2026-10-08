"""agex's own folding: past its budget, a conversation is folded into a
summary in what the model is sent, within a run as well, and recorded
as nontainer's fold records. nontainer's harness corpus checks the
record and its rules (tests/test_conformance.py); these check what agex
decides."""

from dataclasses import replace

import pytest
from nontainer import Store
from nontainer.compaction import Policy, folds
from nontainer.conformance.corpus import ModelStep, calls, says, summarizes
from pydantic_ai import messages as pai

from agex import Agent
from agex.providers.scripted import ScriptedProvider

BUDGET = 1000
SUMMARY = "SUMMARY: the first of two steps has run."


@pytest.fixture
def ws():
    with Store(memory=True) as store:
        ws = store.open("main")
        yield ws
        ws.close()


def python(code: str, tokens: int = 0) -> ModelStep:
    return replace(calls("run_python", code=code), input_tokens=tokens)


def texts(request: list[pai.ModelMessage]) -> str:
    return "\n".join(
        str(part.content)
        for message in request
        for part in message.parts
        if getattr(part, "content", None) is not None
    )


def agent(*steps: ModelStep, budget: int | None = BUDGET):
    provider = ScriptedProvider(list(steps))
    policy = None if budget is None else Policy(budget=budget)
    return Agent(provider, compaction=policy), provider


# the run's first two steps; the second's request reports it over budget
STEPS = (
    python("print('FIRST-OUTPUT')", tokens=100),
    python("print('SECOND-OUTPUT')", tokens=5 * BUDGET),
)


def test_a_run_over_budget_folds_its_earlier_steps(ws):
    """The fold covers the run's steps before its latest one; the
    request after it carries the summary, the run's opening message
    again, and the latest step as it is."""
    a, provider = agent(*STEPS, summarizes(SUMMARY), says("done"))
    out = a.session(ws).say("Do the work in two steps.")
    assert out.status == "completed"
    assert [e.kind for e in out.events].count("Compacted") == 1
    (fold,) = folds(ws)
    run = a.session(ws).runs[-1]
    assert fold.through == run.messages[2].id  # the first step's tool result
    assert (fold.runs, fold.first, fold.summary) == (1, None, SUMMARY)
    sent = texts(provider.seen[-1])
    assert SUMMARY in sent and "Do the work in two steps." in sent
    assert "SECOND-OUTPUT" in sent and "FIRST-OUTPUT" not in sent
    # the summary request: the run so far up to the fold, then the ask
    asked = texts(provider.seen[-2])
    assert "FIRST-OUTPUT" in asked and "SECOND-OUTPUT" not in asked
    assert "Reply with the summary alone" in asked


def test_the_stored_run_keeps_every_message_and_the_reply_names_its_fold(ws):
    a, _ = agent(*STEPS, summarizes(SUMMARY), says("done"))
    session = a.session(ws)
    session.say("Do the work in two steps.")
    (run,) = session.runs
    assert [m.role for m in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert all(SUMMARY not in m.text for m in run.messages)
    (fold,) = folds(ws)
    assert run.messages[-1].fold == fold.through
    assert run.messages[1].fold is None


def test_the_fold_lands_with_the_turn_and_holds_in_the_next(ws):
    a, provider = agent(*STEPS, summarizes(SUMMARY), says("done"), says("again"))
    session = a.session(ws)
    session.say("Do the work in two steps.")
    assert folds(ws) and not ws.uncommitted
    out = session.say("And now?")
    assert "Compacted" not in [e.kind for e in out.events]
    sent = texts(provider.seen[-1])
    assert SUMMARY in sent and "Do the work in two steps." in sent
    assert "FIRST-OUTPUT" not in sent and "And now?" in sent


def test_a_run_too_small_to_fold_within_folds_the_earlier_runs(ws):
    """A run with one step has nothing of its own to fold before its
    latest step: the fold ends at the earlier runs' end."""
    a, provider = agent(
        says("noted", input_tokens=5 * BUDGET),
        summarizes(SUMMARY),
        says("done"),
    )
    session = a.session(ws)
    session.say("Remember the plan.")
    out = session.say("Now act on it.")
    assert "Compacted" in [e.kind for e in out.events]
    (fold,) = folds(ws)
    assert fold.through == session.runs[0].messages[-1].id
    sent = texts(provider.seen[-1])
    assert SUMMARY in sent and "Remember the plan." not in sent


def test_a_summary_with_no_text_falls_back_to_a_transcript(ws):
    """A summary request answered with a tool call (or that fails) is
    made again from a transcript, sent without tools."""
    a, provider = agent(
        *STEPS, calls("run_python", code="1"), summarizes(SUMMARY), says("done")
    )
    a.session(ws).say("Do the work in two steps.")
    assert folds(ws)[0].summary == SUMMARY
    transcript = texts(provider.seen[-2])
    assert "<transcript>" in transcript and "FIRST-OUTPUT" in transcript


def test_without_a_budget_nothing_is_folded_but_a_recorded_fold_is_sent(ws):
    folding, _ = agent(*STEPS, summarizes(SUMMARY), says("done"))
    folding.session(ws).say("Do the work in two steps.")
    plain, provider = agent(
        says("again", input_tokens=50 * BUDGET), says("more"), budget=None
    )
    session = plain.session(ws)
    session.say("And now?")
    session.say("And then?")
    assert len(folds(ws)) == 1
    assert SUMMARY in texts(provider.seen[-1])


def test_a_reply_under_a_fold_is_measured_as_its_request_was(ws):
    """A reply's usage counts the request it answered, which had the
    fold in force then: what that fold had already taken out isn't taken
    off again. Here the folded turn is large, and the reply after the
    fold still reports the request over budget, so it folds again."""
    a, _ = agent(
        python("print('x' * 20_000)", tokens=100),
        says("printed", input_tokens=6 * BUDGET),
        summarizes(SUMMARY),
        says("next", input_tokens=2 * BUDGET),
        summarizes("SUMMARY: both turns."),
        says("done"),
    )
    session = a.session(ws)
    for prompt in ("Print a lot.", "Go on.", "And on."):
        assert session.say(prompt).status == "completed"
    assert [f.runs for f in folds(ws)] == [1, 2]


def test_summary_calls_count_against_max_steps(ws):
    """A fold is made only with a call to spare for the request after
    it: with no call left over, the request goes unfolded, and no run
    makes more model calls than max_steps."""
    for limit, folded in ((3, False), (4, True)):
        store_ws = ws.fork(f"limit{limit}", inherit="fresh")
        provider = ScriptedProvider([*STEPS, summarizes(SUMMARY), says("done")])
        a = Agent(provider, compaction=Policy(budget=BUDGET), max_steps=limit)
        a.session(store_ws).say("Do the work in two steps.")
        assert len(provider.seen) == limit
        assert bool(folds(store_ws)) is folded
        store_ws.close()


def test_summary_calls_are_in_the_usage_streamed_and_stored(ws):
    a, _ = agent(*STEPS, replace(summarizes(SUMMARY), input_tokens=777), says("done"))
    session = a.session(ws)
    out = session.say("Do the work in two steps.")
    reported = [e.input_tokens for e in out.events if e.kind == "Usage"]
    assert 777 in reported
    (run,) = session.runs
    assert run.compaction is not None and run.compaction.input_tokens == 777
    replies = sum(m.usage.input_tokens for m in run.messages if m.usage)
    assert run.usage.input_tokens == replies + 777
