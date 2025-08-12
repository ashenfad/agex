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
