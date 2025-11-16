import pytest

from agex.eval.error import EvalError
from agex.state import Live

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


def test_compare_preserves_non_boolean_results():
    class FilterResult:
        def __init__(self, label: str):
            self.label = label

    class Dummy:
        def __ge__(self, other):
            return FilterResult(f">={other}")

    host_state = Live()
    host_state.set("dummy", Dummy())

    program = "result = dummy >= 2"
    state = eval_and_get_state(program, state=host_state)
    result = state.get("result")
    assert isinstance(result, FilterResult)
    assert result.label == ">=2"


def test_chained_compare_with_non_boolean_raises():
    class FilterResult:
        pass

    class Dummy:
        def __ge__(self, other):
            return FilterResult()

    host_state = Live()
    host_state.set("dummy", Dummy())

    program = "dummy >= 2 >= 1"
    with pytest.raises(EvalError) as excinfo:
        eval_and_get_state(program, state=host_state)

    assert "non-boolean results" in str(excinfo.value)
