import pytest

from tic.eval.error import EvalError
from tic.eval.user_errors import TicAttributeError, TicTypeError

from .helpers import eval_and_get_state


def test_class_definition_and_instantiation():
    """Tests that a simple class can be defined and an instance created."""
    program = """
class MyClass:
    def get_val(self):
        return 42

inst = MyClass()
x = inst.get_val()
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 42


def test_init_method_with_args():
    """Tests that the __init__ method is called with arguments."""
    program = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

p = Point(10, 20)
x_val = p.get_x()
y_val = p.y
"""
    state = eval_and_get_state(program)
    assert state.get("x_val") == 10
    assert state.get("y_val") == 20


def test_instance_methods_modify_state():
    """Tests that methods can modify the instance's state."""
    program = """
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count = self.count + 1

c = Counter()
c.increment()
c.increment()
x = c.count
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 2


def test_method_calling_another_method():
    """Tests that one method can call another on the same instance."""
    program = """
class Greeter:
    def __init__(self, name):
        self.name = name

    def get_greeting(self):
        return "Hello, " + self.name

    def greet(self):
        return self.get_greeting() + "!"

g = Greeter("world")
message = g.greet()
"""
    state = eval_and_get_state(program)
    assert state.get("message") == "Hello, world!"


def test_instances_have_separate_state():
    """Tests that two instances of the same class have isolated state."""
    program = """
class Box:
    def __init__(self, value):
        self.value = value

b1 = Box(1)
b2 = Box(2)
b1.value = 3

val1 = b1.value
val2 = b2.value
"""
    state = eval_and_get_state(program)
    assert state.get("val1") == 3
    assert state.get("val2") == 2


def test_inheritance_is_not_supported():
    """Tests that defining a class with a base class raises an error."""
    program = """
class Parent:
    pass
class Child(Parent):
    pass
"""
    with pytest.raises(EvalError) as excinfo:
        eval_and_get_state(program)
    assert "Inheritance and other advanced class features are not supported" in str(
        excinfo.value
    )


def test_accessing_nonexistent_attribute_fails():
    """Tests that getting a non-existent attribute raises an error."""
    program = """
class MyClass:
    pass
inst = MyClass()
x = inst.non_existent
"""
    with pytest.raises(TicAttributeError) as excinfo:
        eval_and_get_state(program)
    assert "'MyClass' object has no attribute 'non_existent'" in str(excinfo.value)


def test_calling_nonexistent_method_fails():
    """Tests that calling a non-existent method raises an error."""
    program = """
class MyClass:
    pass
inst = MyClass()
inst.non_existent()
"""
    with pytest.raises(TicAttributeError) as excinfo:
        eval_and_get_state(program)
    assert "'MyClass' object has no attribute 'non_existent'" in str(excinfo.value)


def test_init_with_wrong_arg_count_fails():
    """Tests that calling __init__ with wrong number of args raises an error."""
    program = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(10)
"""
    with pytest.raises(TicTypeError) as excinfo:
        eval_and_get_state(program)
    assert "missing required positional argument: 'y'" in str(excinfo.value)
