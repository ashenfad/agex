import numpy as np
import pytest

from agex import Agent, clear_agent_registry
from agex.agent.datatypes import TaskTimeout


def test_numpy_random_normal_without_recursive_fails():
    """
    Without recursive registration, accessing numpy.random (a submodule)
    via attribute access should be blocked by the policy.
    """
    from agex.llm.dummy_client import Dummy, LLMResponse

    clear_agent_registry()
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="import numpy and call numpy.random.normal",
                code=(
                    "import numpy\n"
                    "noise = numpy.random.normal(0, 5, size=12)\n"
                    "task_success(True)"
                ),
            )
        ]
    )
    agent = Agent(name="test_agent_np_no_recursive", max_iterations=2, llm=llm)
    # Register only the top-level module, no recursive
    agent.module(np, name="numpy")

    @agent.task("Return True if noise can be generated")
    def make_noise() -> bool:  # type: ignore[return-value]
        pass

    # Should fail because numpy.random (a submodule) is blocked without recursive=True
    with pytest.raises(TaskTimeout):
        make_noise()


def test_numpy_random_normal_with_recursive_succeeds_dummy():
    """
    With recursive registration, attribute resolution across submodules should work
    and the task should complete successfully.
    """
    from agex.llm.dummy_client import Dummy, LLMResponse

    clear_agent_registry()
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="import numpy and call numpy.random.normal",
                code=(
                    "import numpy\n"
                    "noise = numpy.random.normal(0, 5, size=12)\n"
                    "task_success(True)"
                ),
            )
        ]
    )
    agent = Agent(name="test_agent_np_recursive", max_iterations=2, llm=llm)
    agent.module(np, name="numpy", recursive=True)

    @agent.task("Return True if noise can be generated")
    def make_noise() -> bool:  # type: ignore[return-value]
        pass

    assert make_noise() is True
