import time

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.datatypes import LLMFail
from agex.llm.core import ResponseParseError
from agex.llm.dummy_client import Dummy
from tests.agex._emissions import make_response


def test_llm_retries_then_success(monkeypatch):
    """First two attempts fail, then succeed; ensure only one completion of task and no LLMFail."""
    clear_agent_registry()

    # Make backoff fast
    monkeypatch.setattr(time, "sleep", lambda s: None)

    responses = [
        ResponseParseError("network hiccup 1"),
        ResponseParseError("network hiccup 2"),
        make_response(thinking="ok", code="task_success(42)"),
    ]
    client = Dummy(responses=responses)

    agent = Agent(
        name="retry-success",
        llm=client,
        llm_max_retries=2,
    )

    @agent.task("simple task")
    def t() -> int:  # type: ignore[return-value]
        pass

    result = t()
    assert result == 42


def test_llm_retries_exhaust_and_fail(monkeypatch):
    """All attempts fail; expect an LLMFail to propagate."""
    clear_agent_registry()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    responses = [
        ResponseParseError("down 1"),
        ResponseParseError("down 2"),
        ResponseParseError("down 3"),
    ]
    client = Dummy(responses=responses)

    agent = Agent(name="retry-fail", llm=client, llm_max_retries=2)

    @agent.task("simple task")
    def t() -> int:  # type: ignore[return-value]
        pass

    with pytest.raises(LLMFail):
        t()
