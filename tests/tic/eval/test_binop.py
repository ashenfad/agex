import pytest

from .helpers import eval_and_get_state


def test_eval_binary_ops():
    program = """
x = 10 + 5
y = x - 2
z = y * 3
a = z / 9
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 15
    assert state.get("y") == 13
    assert state.get("z") == 39
    assert state.get("a") == pytest.approx(4.3333333)


def test_eval_membership_and_identity_ops():
    program = """
l = [1, 2, 3]
x = 1 in l
y = 4 not in l
z = l is l
w = l is not [1, 2, 3]
"""
    state = eval_and_get_state(program)
    assert state.get("x") is True
    assert state.get("y") is True
    assert state.get("z") is True
    assert state.get("w") is True
