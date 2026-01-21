import types

import numpy as np
import pytest

from agex import Agent, clear_agent_registry
from agex.eval.error import EvalError
from agex.eval.user_errors import AgexError
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
    from agex.llm.dummy_client import Dummy, LLMResponse

    clear_agent_registry()
    # 1) Alias path: import np; np.random.normal
    llm = Dummy(
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
    agent = Agent(name="test_agent", max_iterations=2, llm=llm)
    agent.module(np, name="np")
    agent.module(np.random)

    @agent.task("Return True if noise vector can be generated")
    def make_noise_alias() -> bool:  # type: ignore[return-value]
        pass

    assert make_noise_alias() is True

    # 2) Full path: import numpy; numpy.random.normal
    llm2 = Dummy(
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
    agent2 = Agent(name="test_agent2", max_iterations=2, llm=llm2)
    agent2.module(np, name="numpy", recursive=True)

    @agent2.task("Return True if noise vector can be generated")
    def make_noise_full() -> bool:  # type: ignore[return-value]
        pass

    assert make_noise_full() is True


def test_dunder_import_registered_module():
    clear_agent_registry()
    import math

    agent = Agent(name="dunder_import")
    agent.module(math, name="math")

    state = eval_and_get_state(
        "mod = __import__('math')\nresult = mod.sqrt(25)\n", agent=agent
    )
    assert state.get("result") == pytest.approx(5.0)


def test_dunder_import_fromlist_attaches_members():
    clear_agent_registry()
    pkg = types.ModuleType("pkg_for_dunder")
    pkg.value = 42

    agent = Agent(name="dunder_import_fromlist")
    agent.module(pkg, name="pkg_for_dunder")

    state = eval_and_get_state(
        "pkg = __import__('pkg_for_dunder', fromlist=['value'])\n"
        "result = pkg.value\n",
        agent=agent,
    )
    assert state.get("result") == 42


def test_dunder_import_unregistered_module_raises():
    clear_agent_registry()
    agent = Agent(name="dunder_import_fail")
    with pytest.raises(EvalError):
        eval_and_get_state("__import__('not_registered')", agent=agent)


def test_dunder_import_relative_level_error():
    clear_agent_registry()
    agent = Agent(name="dunder_import_relative")
    with pytest.raises(AgexError):
        eval_and_get_state("__import__('anything', level=1)", agent=agent)


def test_import_nonexistent_member_from_module():
    """Test that importing a non-existent name from a module raises an error."""
    clear_agent_registry()

    # Create a simple module with one attribute
    testmod = types.ModuleType("testmod")
    testmod.ExistingClass = lambda: "exists"

    agent = Agent(name="test_import_validation")
    agent.module(testmod, name="testmod")

    # Test 1: Import existing class - should work
    state = eval_and_get_state("from testmod import ExistingClass", agent=agent)
    assert "ExistingClass" in state

    # Test 2: Import non-existent class - should raise error
    with pytest.raises(EvalError) as exc_info:
        eval_and_get_state("from testmod import NonExistentClass", agent=agent)

    assert "NonExistentClass" in str(exc_info.value)
    assert "testmod" in str(exc_info.value)


def test_import_nonexistent_member_from_recursive_module():
    """Test that importing a non-existent name from a RECURSIVE module raises an error."""
    clear_agent_registry()

    # Create a simple module with one attribute
    testmod = types.ModuleType("testmod")
    testmod.ExistingClass = lambda: "exists"

    agent = Agent(name="test_recursive_import_validation")
    # Register with recursive=True - this is the key difference
    agent.module(testmod, name="testmod", recursive=True)

    # Test 1: Import existing class - should work
    state = eval_and_get_state("from testmod import ExistingClass", agent=agent)
    assert "ExistingClass" in state

    # Test 2: Import non-existent class from recursive module - should raise error
    with pytest.raises(EvalError) as exc_info:
        eval_and_get_state("from testmod import NonExistentClass", agent=agent)

    assert "NonExistentClass" in str(exc_info.value)
    assert "testmod" in str(exc_info.value)
