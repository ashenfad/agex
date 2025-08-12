import time

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.datatypes import LLMFail
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient


def test_llm_retries_then_success(monkeypatch):
    """First two attempts fail, then succeed; ensure only one completion of task and no LLMFail."""
    clear_agent_registry()

    # Make backoff fast
    monkeypatch.setattr(time, "sleep", lambda s: None)

    responses = [
        RuntimeError("network hiccup 1"),
        RuntimeError("network hiccup 2"),
        LLMResponse(thinking="ok", code="task_success(42)"),
    ]
    client = DummyLLMClient(responses=responses)

    agent = Agent(
        name="retry-success",
        llm_client=client,
        llm_max_retries=2,
        llm_retry_backoff=0.0,
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

    responses = [RuntimeError("down 1"), RuntimeError("down 2"), RuntimeError("down 3")]
    client = DummyLLMClient(responses=responses)

    agent = Agent(
        name="retry-fail", llm_client=client, llm_max_retries=2, llm_retry_backoff=0.0
    )

    @agent.task("simple task")
    def t() -> int:  # type: ignore[return-value]
        pass

    with pytest.raises(LLMFail):
        t()
