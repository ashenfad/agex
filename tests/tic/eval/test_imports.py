import math

import pytest

from tic.agent import Agent
from tic.eval.core import evaluate_program
from tic.eval.error import EvalError
from tic.state.ephemeral import Ephemeral


def test_import_module():
    agent = Agent()
    agent.module(math)

    program = """
import math
x = math.sqrt(4)
pi = math.pi
"""
    state = Ephemeral()
    evaluate_program(program, agent, state)

    assert state.get("x") == 2.0
    assert state.get("pi") == math.pi


def test_import_module_with_alias():
    agent = Agent()
    agent.module(math, name="m")

    program = """
import m
x = m.sqrt(9)
"""
    state = Ephemeral()
    evaluate_program(program, agent, state)

    assert state.get("x") == 3.0


def test_import_from_module():
    agent = Agent()
    agent.module(math)

    program = """
from math import sqrt, pi
x = sqrt(16)
y = pi
"""
    state = Ephemeral()
    evaluate_program(program, agent, state)

    assert state.get("x") == 4.0
    assert state.get("y") == math.pi


def test_import_from_module_with_alias():
    agent = Agent()
    agent.module(math)

    program = """
from math import sqrt as square_root
x = square_root(25)
"""
    state = Ephemeral()
    evaluate_program(program, agent, state)

    assert state.get("x") == 5.0


def test_unregistered_module_error():
    agent = Agent()
    # math is NOT registered

    program = "import math"
    state = Ephemeral()

    with pytest.raises(EvalError, match="Module 'math' is not registered"):
        evaluate_program(program, agent, state)


def test_unregistered_from_import_error():
    agent = Agent()
    agent.module(math)

    program = "from math import non_existent_function"
    state = Ephemeral()

    with pytest.raises(EvalError, match="Cannot import name 'non_existent_function'"):
        evaluate_program(program, agent, state)
