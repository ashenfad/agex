import numpy as np

from tic.agent import Agent
from tic.eval.core import evaluate_program
from tic.render.view import view
from tic.state.ephemeral import Ephemeral
from tic.state.kv import Memory
from tic.state.versioned import Versioned


def test_numpy_ephemeral_ops():
    """Tests that numpy operator overloading works with a simple state."""
    agent = Agent()
    agent.module(np, name="np")

    program = "import np\nresult = np.array([1, 2]) + np.array([3, 4])"
    state = Ephemeral()

    evaluate_program(program, agent, state)

    assert np.array_equal(state.get("result"), np.array([4, 6]))


def test_numpy_versioned_view():
    """
    Tests numpy integration, including `view(agent)` and `view(state)`.
    """
    # 1. Set up the agent and register the numpy module
    agent = Agent()
    agent.module(np, name="np")

    # 2. Define the program to be run
    program = """
import np

# Basic operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result_mul = (arr1 + arr2) * 10
"""
    # 3. Initialize a versioned state with an in-memory store
    store = Memory(as_bytes=True)
    state = Versioned(store)

    # 4. Execute the program and save the state
    evaluate_program(program, agent, state)
    state.snapshot()

    # 5. Check the operator overloading results
    result = state.get("result_mul")
    expected = np.array([50, 70, 90])
    assert np.array_equal(result, expected)

    # 6. Check the rendered view of the AGENT's API
    agent_view = view(agent)
    assert isinstance(agent_view, str)
    assert "module np:" in agent_view
    # Check for a couple of representative numpy functions
    assert "def array(...)" in agent_view
    assert "def mean(...)" in agent_view

    # 7. Check the rendered view of the STATE
    state_view = view(state, focus="full")
    assert isinstance(state_view, dict)
    assert "result_mul" in state_view
    assert np.array_equal(state_view["result_mul"], expected)


def test_numpy_state_continuity():
    """
    Tests that an imported module persists across state snapshots and can be
    used in a subsequent execution session.
    """
    # 1. Set up the agent and a versioned state
    agent = Agent()
    agent.module(np, name="np")
    store = Memory(as_bytes=True)
    state1 = Versioned(store)

    # 2. Run a program that just imports the module, and snapshot the state
    evaluate_program("import np", agent, state1)
    commit_hash = state1.snapshot()
    assert commit_hash is not None

    # 3. Create a new state from the snapshot and run code that uses the module
    state2 = Versioned(store, commit_hash=commit_hash)
    program = "result = np.array([1, 2, 3]) + 10"
    evaluate_program(program, agent, state2)

    # 4. Assert that the code ran successfully, using the rehydrated module
    assert np.array_equal(state2.get("result"), np.array([11, 12, 13]))
