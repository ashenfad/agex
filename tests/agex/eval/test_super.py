"""
Test super() functionality for class inheritance.
"""

import pytest

from agex import Agent
from tests.agex.eval.helpers import eval_and_get_state


def test_super_init_chaining():
    """Test that super().__init__() works for constructor chaining."""

    class Parent:
        def __init__(self, x):
            self.parent_x = x

    agent = Agent()
    # Use constructable=True to whitelist __init__
    agent.cls(Parent, constructable=True, include=["parent_x"])

    program = """
class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)
        self.child_y = y

c = Child(10, 20)
result_x = c.parent_x
result_y = c.child_y
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result_x") == 10
    assert state.get("result_y") == 20


def test_super_method_calling():
    """Test that super().method() works for method chaining."""

    class Parent:
        def greet(self):
            return "Hello from Parent"

    agent = Agent()
    agent.cls(Parent, include=["greet"])

    program = """
class Child(Parent):
    def greet(self):
        parent_greeting = super().greet()
        return parent_greeting + " and Child"

c = Child()
result = c.greet()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "Hello from Parent and Child"


def test_super_outside_method_fails():
    """Test that super() raises error when called outside a method."""

    from agex.eval.user_errors import AgexError

    agent = Agent()

    program = """
class MyClass:
    pass

# Try to call super() outside a method
s = super()
"""

    with pytest.raises(AgexError) as e:
        eval_and_get_state(program, agent)
    assert "super() can only be called from within a method" in str(e.value)
