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

    # This should be in the registry under its full name.
    assert "numpy.random" in agent.importable_modules

    # This should NOT be in the registry under the short name.
    assert "random" not in agent.importable_modules

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

    # The alias should be in the registry.
    assert "rand" in agent.importable_modules
    # The original full name should NOT be.
    assert "numpy.random" not in agent.importable_modules

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
