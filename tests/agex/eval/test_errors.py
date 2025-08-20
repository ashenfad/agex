import pytest

from agex.eval.error import EvalError
from agex.eval.user_errors import AgexNameError, AgexTypeError

from .helpers import eval_and_get_state


def test_error_on_undefined_variable():
    with pytest.raises(AgexNameError) as e:
        eval_and_get_state("x = y")
    assert "name 'y' is not defined" in str(e.value)
    assert e.value.node.lineno == 1  # type: ignore


def test_error_on_unsupported_op():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = 1 << 2")
    assert "Operator LShift not supported" in str(e.value)
    assert e.value.node.lineno == 1  # type: ignore


def test_error_on_type_mismatch():
    with pytest.raises(AgexTypeError) as e:
        eval_and_get_state("x = 1 + 'a'")
    assert "unsupported operand type" in str(e.value)
