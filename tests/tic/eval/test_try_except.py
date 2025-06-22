"""
Tests for try...except...else...finally blocks.
"""

import pytest

from tests.tic.eval.helpers import eval_and_get_state
from tic.agent import ExitSuccess
from tic.eval.user_errors import TicKeyError


def test_simple_try_except():
    program = """
x = 1
try:
    raise ValueError("Something went wrong")
    x = 2 # This should not run
except ValueError:
    x = 3
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3


def test_except_as_name():
    program = """
err_msg = ""
try:
    raise ValueError("The message")
except ValueError as e:
    err_msg = str(e)
"""
    state = eval_and_get_state(program)
    assert state.get("err_msg") == "The message"


def test_try_else_finally():
    program = """
x = 1
y = 1
try:
    x = 2 # No exception
except ValueError:
    x = 99 # Should not run
else:
    x = 3
finally:
    y = 2
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3
    assert state.get("y") == 2


def test_try_finally_with_exception():
    program = """
x = 1
y = 1
try:
    try:
        raise ValueError("Error")
        x = 2
    finally:
        y = 2
except ValueError:
    x = 3
"""
    state = eval_and_get_state(program)
    assert state.get("y") == 2
    assert state.get("x") == 3


def test_uncaught_exception():
    program = """
try:
    raise KeyError("Wrong key")
except ValueError:
    pass # This should not catch KeyError
"""
    with pytest.raises(TicKeyError):
        eval_and_get_state(program)


def test_bare_except():
    program = """
x = 1
try:
    raise ValueError("Something went wrong")
    x = 2
except:
    x = 3
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3


def test_agent_exit_is_not_caught():
    """
    Ensures that our internal _AgentExit signals are NOT caught by user code.
    """
    program = """
try:
    exit_success(result="Finished")
except:
    pass # This must not catch the exit signal
"""
    with pytest.raises(ExitSuccess) as e:
        eval_and_get_state(program)
    assert e.value.result == "Finished"
