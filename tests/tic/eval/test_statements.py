import pytest

from tic.eval.error import EvalError
from tic.eval.user_errors import TicNameError, TicTypeError

from .helpers import eval_and_get_state


def test_destructuring_assignment_tuple():
    program = "a, b = (1, 2)"
    state = eval_and_get_state(program)
    assert state.get("a") == 1
    assert state.get("b") == 2


def test_destructuring_assignment_list():
    program = "x, y, z = [True, 'hello', 3.14]"
    state = eval_and_get_state(program)
    assert state.get("x") is True
    assert state.get("y") == "hello"
    assert state.get("z") == 3.14


def test_destructuring_assignment_implicit_tuple():
    program = "c, d = 3, 4"
    state = eval_and_get_state(program)
    assert state.get("c") == 3
    assert state.get("d") == 4


def test_destructuring_mismatch_error_too_many():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("a, b = (1, 2, 3)")
    assert "Expected 2 values to unpack, but got 3" in str(e.value)


def test_destructuring_mismatch_error_too_few():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("a, b, c = (1, 2)")
    assert "Expected 3 values to unpack, but got 2" in str(e.value)


def test_destructuring_non_iterable_error():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("a, b = 1")
    assert "Cannot unpack non-iterable value" in str(e.value)


def test_chained_assignment():
    program = "a = b = 10"
    state = eval_and_get_state(program)
    assert state.get("a") == 10
    assert state.get("b") == 10


def test_augmented_assignment():
    program = """
x = 10
x += 5
y = 2
y *= 3
z = 10.0
z /= 4
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 15
    assert state.get("y") == 6
    assert state.get("z") == 2.5


def test_aug_assign_undefined_error():
    with pytest.raises(TicNameError) as e:
        eval_and_get_state("a += 1")
    assert "name 'a' is not defined" in str(e.value)


def test_dict_indexed_assignment():
    program = """
d = {'a': 1}
d['b'] = 2
d['a'] = 100
"""
    state = eval_and_get_state(program)
    assert state.get("d") == {"a": 100, "b": 2}


def test_dict_indexed_assignment_on_non_dict_error():
    with pytest.raises(TicTypeError) as e:
        eval_and_get_state("x = 1\nx[0] = 2")
    assert "Indexed assignment is only supported" in str(e.value)


def test_list_indexed_assignment():
    program = """
l = [1, 2, 99]
l[1] = 20
"""
    state = eval_and_get_state(program)
    assert state.get("l") == [1, 20, 99]


def test_nested_assignment_dict():
    program = """
d = {"a": {"b": 1}, "c": [10, 20]}
d["a"]["b"] = 100
d["c"][1] = 200
"""
    state = eval_and_get_state(program)
    assert state.get("d") == {"a": {"b": 100}, "c": [10, 200]}


def test_nested_assignment_augassign():
    program = """
d = {"a": {"b": 1}, "c": [10, 20]}
d["a"]["b"] += 99
d["c"][1] *= 10
"""
    state = eval_and_get_state(program)
    assert state.get("d") == {"a": {"b": 100}, "c": [10, 200]}


def test_mutation_side_effects_are_pythonic():
    """
    This test confirms that mutations have side effects on shared objects
    between commits, which is the desired Pythonic behavior.
    """
    program = """
l = [1, 2, 3]
d = {"a": l}
d["a"][0] = 99
"""
    state = eval_and_get_state(program)
    # The key assertion: the original `l` variable should also be mutated.
    assert state.get("l") == [99, 2, 3]
    assert state.get("d")["a"] is state.get("l")  # They should be the same object


def test_del_statement():
    """Tests the 'del' statement for deleting variables."""
    program = """
x = 10
y = 20
del x
"""
    state = eval_and_get_state(program)
    assert "x" not in state
    assert state.get("y") == 20


def test_del_statement_nested():
    """Tests the 'del' statement for nested data structures."""
    program = """
my_list = [1, 2, 3, 4]
my_dict = {"a": 1, "b": 2, "c": 3}
nested = {"list": [10, 20, 30]}

del my_list[1]
del my_dict["b"]
del nested["list"][0]
"""
    state = eval_and_get_state(program)

    assert state.get("my_list") == [1, 3, 4]
    assert state.get("my_dict") == {"a": 1, "c": 3}
    assert state.get("nested")["list"] == [20, 30]


def test_del_statement_attributes():
    """Tests the 'del' statement for object attributes."""
    program = """
class MyClass:
    def __init__(self):
        self.a = 1
        self.b = 2

inst = MyClass()
del inst.a
"""
    state = eval_and_get_state(program)
    inst = state.get("inst")
    assert "a" not in inst.attributes
    assert "b" in inst.attributes
    assert inst.attributes["b"] == 2
