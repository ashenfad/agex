import numpy as np
import pytest

from agex import Agent, clear_agent_registry
from agex.eval.error import EvalError
from tests.agex.eval.helpers import eval_and_get_state


def test_module_import_name_collision():
    """
    Tests that registering `numpy.random` does not allow `import random`.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent")

    # Register numpy.random. This should NOT create an importable module named 'random'.
    agent.module(np.random)

    # Policy: module namespace should be present under its full name
    assert "numpy.random" in agent._policy.namespaces
    # Short name should not exist
    assert "random" not in agent._policy.namespaces

    # Now, try to import 'random' in the evaluator. This should fail.
    with pytest.raises(EvalError) as exc_info:
        eval_and_get_state("import random", agent=agent)

    assert "Module 'random' is not registered or whitelisted" in str(exc_info.value)

    # Verify that importing the correct, full name works.
    try:
        eval_and_get_state("import numpy.random", agent=agent)
    except EvalError as e:
        pytest.fail(
            f"Importing 'numpy.random' should have succeeded, but failed with: {e}"
        )


def test_module_import_with_alias():
    """
    Tests that registering a module with an explicit alias works correctly.
    """
    clear_agent_registry()
    agent = Agent(name="test_agent")

    # Register numpy.random with a specific alias 'rand'.
    agent.module(np.random, name="rand")

    # Policy: alias namespace exists; original full name does not
    assert "rand" in agent._policy.namespaces
    assert "numpy.random" not in agent._policy.namespaces

    # Importing the alias should work.
    try:
        eval_and_get_state("import rand", agent=agent)
        eval_and_get_state("import rand as r", agent=agent)
    except EvalError as e:
        pytest.fail(f"Importing alias 'rand' failed: {e}")

    # Importing the original name should fail.
    with pytest.raises(EvalError):
        eval_and_get_state("import numpy.random", agent=agent)

    # Importing a similarly named module should also fail.
    with pytest.raises(EvalError):
        eval_and_get_state("import random", agent=agent)


def test_numpy_random_normal_resolution_with_alias_and_full_path():
    """
    Reproduces attribute resolution for submodules: np.random.normal and numpy.random.normal.
    Previously failed with "module has no attribute 'normal'" when submodule wasn't wrapped as AgexModule.
    """
    from agex.llm.dummy_client import DummyLLMClient, LLMResponse

    clear_agent_registry()
    # 1) Alias path: import np; np.random.normal
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="Will import np and call np.random.normal",
                code=(
                    "import np\n"
                    "noise = np.random.normal(0, 5, size=12)\n"
                    "task_success(True)"
                ),
            )
        ]
    )
    agent = Agent(name="test_agent", max_iterations=2, llm_client=llm_client)
    agent.module(np, name="np")
    agent.module(np.random)

    @agent.task("Return True if noise vector can be generated")
    def make_noise_alias() -> bool:  # type: ignore[return-value]
        pass

    assert make_noise_alias() is True

    # 2) Full path: import numpy; numpy.random.normal
    llm_client2 = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="Will import numpy and call numpy.random.normal",
                code=(
                    "import numpy\n"
                    "noise = numpy.random.normal(0, 5, size=12)\n"
                    "task_success(True)"
                ),
            )
        ]
    )
    agent2 = Agent(name="test_agent2", max_iterations=2, llm_client=llm_client2)
    agent2.module(np, name="numpy", recursive=True)

    @agent2.task("Return True if noise vector can be generated")
    def make_noise_full() -> bool:  # type: ignore[return-value]
        pass

    assert make_noise_full() is True
