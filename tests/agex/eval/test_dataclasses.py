"""
Tests for our limited, dataclass-only class implementation.
"""

import pytest

from agex.eval.error import EvalError
from agex.eval.user_errors import AgexAttributeError, AgexTypeError
from tests.agex.eval.helpers import eval_and_get_state


def test_dataclass_definition_and_instantiation():
    program = """
@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)
x_val = p.x
y_val = p.y
"""
    state = eval_and_get_state(program)
    assert state.get("x_val") == 10
    assert state.get("y_val") == 20
    p = state.get("p")
    assert p is not None
    assert p.cls.name == "Point"
    assert p.attributes == {"x": 10, "y": 20}


def test_dataclass_attribute_assignment():
    program = """
@dataclass
class Box:
    value: int

b = Box(1)
b.value = 100
"""
    state = eval_and_get_state(program)
    b = state.get("b")
    assert b.attributes["value"] == 100


def test_isinstance_with_dataclass():
    program = """
@dataclass
class A:
    x: int
@dataclass
class B:
    x: int

a_inst = A(1)
b_inst = B(1)

is_a_A = isinstance(a_inst, A)
is_a_B = isinstance(a_inst, B)
"""
    state = eval_and_get_state(program)
    assert state.get("is_a_A") is True
    assert state.get("is_a_B") is False


def test_error_on_inheritance():
    """Tests that an error is raised for dataclass inheritance."""
    program = """
@dataclass
class Parent:
    x: int

@dataclass
class Child(Parent):
    y: int
"""
    with pytest.raises(EvalError) as e:
        eval_and_get_state(program)
    assert "Inheritance and other advanced class features are not supported" in str(
        e.value
    )


def test_error_on_methods():
    program = """
@dataclass
class MyClass:
    x: int
    def my_method(self):
        return self.x
"""
    with pytest.raises(EvalError) as e:
        eval_and_get_state(program)
    assert "Methods are not supported" in str(e.value)


def test_error_on_adding_new_attribute():
    program = """
@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
p.z = 3 # Should fail
"""
    with pytest.raises(AgexAttributeError) as e:
        eval_and_get_state(program)
    assert "object has no attribute 'z'" in str(e.value)


def test_constructor_missing_arg_error():
    program = """
@dataclass
class Point:
    x: int
    y: int
p = Point(1)
"""
    with pytest.raises(AgexTypeError) as e:
        eval_and_get_state(program)
    assert "missing required positional argument: 'y'" in str(e.value)


def test_constructor_extra_arg_error():
    program = """
@dataclass
class Point:
    x: int
p = Point(1, 2)
"""
    # Our simple constructor doesn't give a great error message here,
    # but it should fail because 'y' will be an unexpected keyword.
    # This is an area for future improvement.
    with pytest.raises(AgexTypeError):
        eval_and_get_state(program)


def test_constructor_unexpected_kw_error():
    program = """
@dataclass
class Point:
    x: int
p = Point(x=1, z=3)
"""
    with pytest.raises(AgexTypeError) as e:
        eval_and_get_state(program)
    assert "got an unexpected keyword argument 'z'" in str(e.value)


def test_import_dataclass_is_allowed():
    """
    Tests that `from dataclasses import dataclass` is silently ignored and works.
    """
    program = """
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
x_val = p.x
"""
    state = eval_and_get_state(program)
    assert state.get("x_val") == 1


def test_import_other_from_dataclasses_fails():
    """
    Tests that importing anything other than `dataclass` from the `dataclasses`
    module fails, as the module is not actually registered.
    """
    program = """
from dataclasses import field
"""
    with pytest.raises(EvalError) as e:
        eval_and_get_state(program)
    assert "No module named 'dataclasses' is registered" in str(e.value)
