"""agex against a real model, over OpenRouter.

Opt-in: deselected unless asked for with ``-m live``, and skipped
without ``OPENROUTER_API_KEY``. ``AGEX_LIVE_MODEL`` picks the model.

    uv run --extra openrouter pytest -m live
"""

import asyncio
import os

import pytest
from nontainer import Store
from nontainer.turns import ToolEnded, Usage

from agex import Agent
from agex.providers import Reply, Settings
from agex.providers.pydanticai import PydanticAIProvider
from agex.record import Message, Text, Thinking

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY"
    ),
]

MODEL = os.environ.get("AGEX_LIVE_MODEL", "openrouter:meta/muse-spark-1.3-contributor")
REASONING = Settings(extra={"openrouter_reasoning": {"effort": "low"}})


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("live")
    yield ws
    ws.close()
    store.close()


def test_a_turn_that_calls_a_tool(ws):
    outcome = (
        Agent(MODEL, primer="You are terse.")
        .session(ws)
        .say("Create /workspace/hello.txt containing the word hi, then reply done.")
    )
    assert outcome.status == "completed", outcome.message
    assert ws.files.read("/workspace/hello.txt").strip() == b"hi"
    assert any(isinstance(e, ToolEnded) and not e.is_error for e in outcome.events)
    assert all(e.input_tokens > 0 for e in outcome.events if isinstance(e, Usage))


def test_reasoning_survives_a_round_trip(ws):
    """The reasoning a reply carried (with its signature) goes back with
    the history, and the model takes it."""
    agent = Agent(MODEL, primer="Answer with a number only.", settings=REASONING)
    chat = agent.session(ws)
    first = chat.say("What is 17 * 23?")
    assert first.status == "completed", first.message
    thinking = [p for p in chat.runs[0].messages[-1].parts if isinstance(p, Thinking)]
    assert thinking and (thinking[0].text or thinking[0].signature)
    second = chat.say("Add 9 to that.")
    assert second.status == "completed", second.message
    assert "400" in second.text


def test_reasoning_across_a_tool_loop(ws):
    outcome = (
        Agent(MODEL, settings=REASONING)
        .session(ws)
        .say("Write /workspace/n.txt containing 7*6 as digits, then reply done.")
    )
    assert outcome.status == "completed", outcome.message
    assert ws.files.read("/workspace/n.txt").strip() == b"42"


def test_a_repeated_prompt_reads_from_the_cache():
    """The provider's cache reads reach the usage agex reports. Whether
    a call is served from cache is the provider's choice (it depends on
    where the request is routed), so a run that never hits it skips
    rather than fails."""
    rules = " ".join(
        f"Rule {i}: answer briefly and precisely about topic {i}." for i in range(400)
    )
    system = Message(id="s", role="system", parts=(Text(text=rules),))
    user = Message(id="u", role="user", parts=(Text(text="Say ok."),))
    provider = PydanticAIProvider(MODEL)
    brief = Settings(max_tokens=64)

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

    reads = []
    for _ in range(4):
        reads.append(asyncio.run(usage()).cache_read_tokens)
        if len(reads) > 1 and reads[-1] > 0:
            break
    if max(reads[1:]) == 0:
        pytest.skip(f"the provider served no cache read in {len(reads)} calls")
    assert reads[0] >= 0 and max(reads[1:]) > 0


def test_a_refused_request_fails_the_turn(ws):
    outcome = Agent("openrouter:meta/no-such-model").session(ws).say("hi")
    assert outcome.status == "failed"
    assert "400" in (outcome.message or "")
