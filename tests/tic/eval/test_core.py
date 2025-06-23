from tic.agent import Agent
from tic.eval.call import PrintTuple

from .helpers import eval_and_get_state


def test_eval_assignment():
    state = eval_and_get_state("x = 1\ny = x")
    assert state.get("x") == 1
    assert state.get("y") == 1


def test_stateful_print_builtin():
    """
    Tests that the stateful `print` function correctly appends items to the
    `__stdout__` list in the state.
    """
    agent = Agent()
    state = eval_and_get_state("1+1", agent)
    assert state.get("__stdout__") is None

    # First print call with multiple arguments becomes a single PrintTuple
    state = eval_and_get_state('print(1, "hello")', agent, state)
    assert state.get("__stdout__") == [PrintTuple((1, "hello"))]

    # Second print call appends another PrintTuple
    state = eval_and_get_state("print(True, None)", agent, state)
    assert state.get("__stdout__") == [
        PrintTuple((1, "hello")),
        PrintTuple((True, None)),
    ]

    # Printing a variable
    state = eval_and_get_state("x = [10, 20]\nprint(x)", agent, state)
    stdout = state.get("__stdout__")
    assert stdout[-1] == PrintTuple(([10, 20],))
    assert stdout == [
        PrintTuple((1, "hello")),
        PrintTuple((True, None)),
        PrintTuple(([10, 20],)),
    ]
