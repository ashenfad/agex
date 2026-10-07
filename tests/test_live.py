"""agex against real models: OpenRouter, Anthropic, OpenAI and Google.

Opt-in: deselected unless asked for with ``-m live``. Each provider's
tests run when its key is set and skip otherwise; ``-k anthropic`` (and
so on) runs one provider. The default models are small ones; override
them with ``AGEX_LIVE_MODEL`` (OpenRouter), ``AGEX_LIVE_ANTHROPIC``,
``AGEX_LIVE_OPENAI`` and ``AGEX_LIVE_GOOGLE``, keeping in mind that the
expectations below (whether a model shows its reasoning, streams it,
caches without being asked) are the default models'.

    uv run --all-extras pytest -m live
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from nontainer import Store
from nontainer.turns import ThinkingDelta, ToolEnded, Usage

from agex import Agent
from agex.providers import Reply, Settings
from agex.providers.pydanticai import PydanticAIProvider
from agex.record import Message, Text, Thinking, ToolCall

pytestmark = pytest.mark.live


@dataclass(frozen=True)
class Live:
    name: str
    keys: tuple[str, ...]
    model: str
    thinking: str
    """The effort that makes the model reason (the small ones skip it
    on easy questions at "low")."""
    shows_reasoning: bool
    """Whether a reply carries its reasoning (as text, or encrypted)."""
    streams_reasoning: bool
    """Whether readable reasoning streams as ThinkingDelta."""
    bad_model: str
    cache: dict[str, Any] = field(default_factory=dict)
    """What it takes for the provider to cache a prompt."""

    def settings(self, **kw: Any) -> Settings:
        return Settings(thinking=self.thinking, **kw)  # type: ignore[arg-type]


PROVIDERS = [
    Live(
        "openrouter",
        ("OPENROUTER_API_KEY",),
        os.environ.get("AGEX_LIVE_MODEL", "openrouter:meta/muse-spark-1.3-contributor"),
        thinking="low",
        shows_reasoning=True,
        streams_reasoning=False,
        bad_model="openrouter:meta/no-such-model",
    ),
    Live(
        "anthropic",
        ("ANTHROPIC_API_KEY",),
        os.environ.get("AGEX_LIVE_ANTHROPIC", "anthropic:claude-haiku-4-5-20251001"),
        thinking="low",
        shows_reasoning=True,
        streams_reasoning=True,
        bad_model="anthropic:claude-no-such-model",
        cache={"anthropic_cache_instructions": True},
    ),
    Live(
        "openai",
        ("OPENAI_API_KEY",),
        os.environ.get("AGEX_LIVE_OPENAI", "openai-responses:gpt-6-luna"),
        thinking="high",
        shows_reasoning=True,
        streams_reasoning=False,
        bad_model="openai-responses:gpt-no-such-model",
    ),
    Live(
        "google",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        os.environ.get("AGEX_LIVE_GOOGLE", "google:gemini-3.5-flash-lite"),
        thinking="high",
        shows_reasoning=False,
        streams_reasoning=False,
        bad_model="google:gemini-no-such-model",
    ),
]


@pytest.fixture(params=PROVIDERS, ids=lambda live: live.name)
def live(request):
    live = request.param
    if not any(os.environ.get(key) for key in live.keys):
        pytest.skip(f"needs {' or '.join(live.keys)}")
    return live


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("live")
    yield ws
    ws.close()
    store.close()


def say(chat, prompt):
    """A turn, resumed once if a provider's failure interrupted it: an
    OpenRouter backend sometimes relays an error (a 404 for a model it
    serves) that the next request gets past."""
    outcome = chat.say(prompt)
    if outcome.status == "interrupted":
        outcome = chat.resume()
    return outcome


def test_a_turn_that_calls_a_tool(live, ws):
    outcome = say(
        Agent(live.model, primer="You are terse.").session(ws),
        "Create /workspace/hello.txt containing the word hi, then reply done.",
    )
    assert outcome.status == "completed", outcome.message
    assert ws.files.read("/workspace/hello.txt").strip() == b"hi"
    assert any(isinstance(e, ToolEnded) and not e.is_error for e in outcome.events)
    assert all(e.input_tokens > 0 for e in outcome.events if isinstance(e, Usage))


def test_turn_after_turn_with_reasoning_and_tools(live, ws):
    """Two turns from blocking code, each reasoning through a tool call:
    what a reply carried (reasoning, signatures, item ids) goes back with
    the history, and the second turn's model call reuses the client the
    first one opened."""
    chat = Agent(live.model, settings=live.settings()).session(ws)
    for path, product in (("/workspace/n.txt", "7*6"), ("/workspace/m.txt", "8*9")):
        outcome = say(
            chat,
            f"Work out {product}, write it as digits to {path} with file_write, "
            "then reply done.",
        )
        assert outcome.status == "completed", outcome.message
    assert ws.files.read("/workspace/n.txt").strip() == b"42"
    assert ws.files.read("/workspace/m.txt").strip() == b"72"


def test_reasoning_survives_a_round_trip(live):
    """The reasoning a reply carried is kept in the run and goes back
    with the history. A model thinking adaptively (Anthropic's at a low
    effort) may answer without reasoning, so the first question gets a
    few fresh tries to show some."""
    agent = Agent(
        live.model, primer="Answer with a number only.", settings=live.settings()
    )
    question = (
        "A bat and a ball cost 1.10 in total, and the bat costs 1.00 more "
        "than the ball. What does the ball cost?"
    )
    for _ in range(3):
        store = Store(memory=True)
        chat = agent.session(store.open("live"))
        first = say(chat, question)
        assert first.status == "completed", first.message
        # anywhere in the run: a model may reason, call a tool, and then
        # answer in a reply of its own with no reasoning left to do
        shown = any(
            isinstance(p, Thinking) and (p.text or p.signature)
            for m in chat.runs[0].messages
            for p in m.parts
        )
        if shown or not live.shows_reasoning:
            break
        chat.ws.close()
        store.close()
    else:
        pytest.fail("three replies in a row carried no reasoning")
    try:
        second = say(chat, "And the bat? Number only.")
        assert second.status == "completed", second.message
        assert "1.05" in second.text
        if live.streams_reasoning:
            assert any(isinstance(e, ThinkingDelta) for e in first.events)
    finally:
        chat.ws.close()
        store.close()


def test_a_signed_tool_call_keeps_its_signature(live, ws):
    """What a provider attaches to a tool call (Gemini's thought
    signature, OpenAI's item id) is kept in the run, to be sent back."""
    chat = Agent(live.model, settings=live.settings()).session(ws)
    outcome = say(
        chat, "Work out 6*7, write it as digits to /workspace/k.txt, then reply done."
    )
    assert outcome.status == "completed", outcome.message
    calls = [
        p for m in chat.runs[0].messages for p in m.parts if isinstance(p, ToolCall)
    ]
    assert calls
    if live.name == "google":
        assert any((c.details or {}).get("thought_signature") for c in calls)
    if live.name == "openai":
        assert all(c.id for c in calls)


def test_a_repeated_prompt_reads_from_the_cache(live):
    """Cache reads reach the usage agex reports. Whether a call is
    served from cache is the provider's choice, so a run that never hits
    it skips rather than fails."""
    rules = " ".join(
        f"Rule {i}: answer briefly and precisely about topic {i}." for i in range(400)
    )
    system = Message(id="s", role="system", parts=(Text(text=rules),))
    user = Message(id="u", role="user", parts=(Text(text="Say ok."),))
    provider = PydanticAIProvider(live.model)
    brief = Settings(max_tokens=64, extra=live.cache)

    async def usage():
        for attempt in range(2):
            try:
                async for event in provider.stream([system, user], settings=brief):
                    if isinstance(event, Reply) and event.message.usage:
                        return event.message.usage
            except Exception as error:
                if attempt or not provider.transient(error):
                    raise
        raise AssertionError("no reply")

    async def calls():
        reads = []
        for _ in range(4):
            reads.append((await usage()).cache_read_tokens)
            if len(reads) > 1 and reads[-1] > 0:
                break
        return reads

    reads = asyncio.run(calls())
    if max(reads[1:]) == 0:
        pytest.skip(f"the provider served no cache read in {len(reads)} calls")
    assert max(reads[1:]) > 0


@dataclass
class Score:
    student: str
    total: int


@dataclass
class Ranking:
    best: str
    scores: list[Score]


def test_a_task_hands_back_a_typed_value(live):
    """A task's brief is enough for the model to build the value with
    the types bound in its world, and finish with task.success."""
    agent = Agent(live.model, settings=live.settings())

    @agent.task
    def rank(answers: dict[str, list[int]]) -> Ranking:
        """Total each student's answers, and name the student with the
        highest total."""

    out = rank.run({"ada": [3, 4], "bo": [1, 1], "cy": [5, 0]})
    if out.status == "interrupted":
        out = rank.run({"ada": [3, 4], "bo": [1, 1], "cy": [5, 0]})
    assert out.status == "success", out.message
    assert isinstance(out.value, Ranking)
    assert out.value.best == "ada"
    totals = {s.student: s.total for s in out.value.scores}
    assert totals == {"ada": 7, "bo": 2, "cy": 5}


def test_a_refused_request_fails_the_turn(live, ws):
    outcome = Agent(live.bad_model).session(ws).say("hi")
    assert outcome.status == "failed", outcome.message
