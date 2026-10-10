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
from nontainer import Profile, PythonConfig, Store
from nontainer.compaction import Policy
from nontainer.turns import Compacted, ThinkingDelta, ToolEnded, Usage

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


def test_a_model_sees_an_image_a_tool_returns(live, ws):
    """view_image's image reaches the model in its tool result: it names
    the colour of a picture only the image shows."""
    import struct
    import zlib

    from nontainer.adapters.tools import Toolset

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    side = 64
    row = b"\x00" + b"\xff\x00\x00" * side  # a solid red 64x64 image
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * side))
        + chunk(b"IEND", b"")
    )
    ws.files.write("/workspace/swatch.png", png)
    chat = Agent(live.model, primer="You are terse.").session(
        ws, toolset=Toolset(ws, vision=True)
    )
    outcome = say(
        chat,
        "Look at /workspace/swatch.png with view_image and reply with only the "
        "colour it shows, in one lowercase word.",
    )
    assert outcome.status == "completed", outcome.message
    assert any(
        e.name == "view_image" for e in outcome.events if isinstance(e, ToolEnded)
    )
    assert "red" in (outcome.text or "").lower()


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


@pytest.mark.parametrize("isolation", ["none", "process"])
def test_a_task_hands_back_a_typed_value(live, isolation):
    """A task's brief is enough for the model to build the value with
    the types bound in its world, and finish with task.success, in this
    process or in a worker."""
    agent = Agent(
        live.model,
        settings=live.settings(),
        profile=Profile(python=PythonConfig(isolation=isolation)),
    )

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


class Registry:
    """The school's records, kept on the host."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def enrolled_in(self, course: str) -> list[str]:
        """The students enrolled in a course."""
        self.asked.append(course)
        return ["ada", "bo", "cy"] if course == "math" else []

    def final_mark(self, student: str, course: str) -> int:
        """A student's final mark in a course, out of 100."""
        return {"ada": 78, "bo": 91, "cy": 64}[student] if course == "math" else 0


def test_a_task_uses_a_live_input_by_the_methods_its_brief_lists(live):
    """The task's docstring says what to do, not how: the methods to
    call come from the brief's description of the live input's class."""
    agent = Agent(live.model, settings=live.settings())

    @agent.task
    def top_student(registry: Registry, course: str) -> str:
        """Name the student with the highest final mark in the course."""

    registry = Registry()
    out = top_student.run(registry, "math")
    if out.status == "interrupted":
        out = top_student.run(registry, "math")
    assert out.status == "success", out.message
    assert out.value == "bo"
    assert "math" in registry.asked


def test_a_task_asks_then_carries_on_with_the_answer(live):
    """A model that is told it must ask does, with task.needs_input, and
    finishes from the answer in the same world."""
    agent = Agent(live.model, settings=live.settings())

    @agent.task
    def rank(answers: dict[str, list[int]]) -> Ranking:
        """Rank the students. You don't know whether to rank them by each
        student's total or by their best single answer, so ask with
        task.needs_input before you rank, then rank as the answer says."""

    answers = {"ada": [3, 4], "bo": [1, 1], "cy": [5, 0]}
    out = rank.run(answers)
    if out.status == "interrupted":
        out = rank.run(answers)
    assert out.status == "needs_input", out.message
    done = out.resume("By total.")
    if done.status == "interrupted":
        pytest.skip(f"the provider was interrupted on resume: {done.message}")
    assert done.status == "success", done.message
    assert done.value.best == "ada"


def test_a_task_over_budget_folds_within_its_run_and_finishes(live):
    """A budget a little above the task's opening request: partway
    through, the run's earlier steps fold into a summary its own model
    writes, and the task still finishes with the right value."""
    agent = Agent(live.model, settings=live.settings(), compaction=Policy(budget=1800))

    @agent.task
    def total() -> int:
        """In four separate run_python calls, print the integers 1-100,
        then 101-200, then 201-300, then 301-400, one per line (one call
        per hundred, nothing else in each call). Then call task.success
        with the sum of all four hundred integers."""

    out = total.run()
    if out.status == "interrupted":
        pytest.skip(f"the provider was interrupted: {out.message}")
    assert out.status == "success", out.message
    assert out.value == 80200
    assert any(isinstance(e, Compacted) for e in out.events)


def test_a_session_delegates_through_its_sessions_tool(live):
    """The model asks a delegate, which is the same agent on a fork, and
    reads its answer; the work is on the delegate's branch."""
    store = Store(memory=True)
    ws = store.open("lead")
    chat = Agent(live.model, primer="You are terse.").session(ws, sessions=True)
    try:
        outcome = say(
            chat,
            "Use the sessions tool to ask a delegate named scout to create "
            "/workspace/note.txt containing the word ok. Wait for its answer "
            "(wait=true), then reply done.",
        )
        assert outcome.status == "completed", outcome.message
        jobs = {job.name: job.status for job in chat.sessions.list()}
        assert jobs.get("lead.scout") == "answered", jobs
        scout = store.open("lead.scout")
        try:
            assert scout.files.read("/workspace/note.txt").strip() == b"ok"
        finally:
            scout.close()
        assert not ws.files.exists("/workspace/note.txt")
    finally:
        chat.close()
        ws.close()
        store.close()


def test_a_refused_request_fails_the_turn(live, ws):
    outcome = Agent(live.bad_model).session(ws).say("hi")
    assert outcome.status == "failed", outcome.message
