from dataclasses import dataclass

import pytest

from agex import Agent, TaskTimeout
from agex.llm import DummyLLMClient
from agex.llm.core import LLMResponse


def test_pydantic_strict_mode():
    """
    Tests that strict mode prevents coercion from dictionary to dataclass.

    This is a regression test for a bug where Pydantic would silently coerce
    dictionaries to dataclasses, hiding type mismatches. With strict mode,
    the agent should fail validation and timeout trying the same response repeatedly.

    Background: Before this fix, an agent task returning {"msg": "value"} when
    expecting a Thing dataclass would be incorrectly accepted due to Pydantic's
    default coercion behavior. This test ensures that such coercion is prevented.
    """
    responses = [
        LLMResponse(
            thinking="I should return a dict.",
            code='task_success({"msg": "a shiny ring"})',
        )
    ]
    llm_client = DummyLLMClient(responses=responses)
    agent = Agent(
        llm_client=llm_client, max_iterations=2
    )  # Limit iterations to fail faster

    @agent.cls
    @dataclass
    class Thing:
        msg: str

    @agent.task
    def make_something(prompt: str) -> Thing:  # type: ignore[return-value]
        "Make a Thing"
        pass

    # Should timeout because validation fails and agent retries with same invalid response
    with pytest.raises(TaskTimeout):
        make_something("a shiny thing")
