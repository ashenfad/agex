import pytest

from tic.eval.error import EvalError

from .helpers import eval_and_get_state


def test_error_on_undefined_variable():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = y")
    assert "Name 'y' is not defined" in str(e.value)
    assert e.value.node.lineno == 1


def test_error_on_unsupported_op():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = 1 << 2")
    assert "Operator LShift not supported" in str(e.value)
    assert e.value.node.lineno == 1


def test_error_on_type_mismatch():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = 1 + 'a'")
    assert "Failed to execute operation" in str(e.value)
    assert isinstance(e.value.cause, TypeError)
