"""
Test that attribute access properly respects policy and local_instance_attrs.
"""

import pytest

from agex import Agent
from agex.eval.user_errors import AgexAttributeError
from tests.agex.eval.helpers import eval_and_get_state


def test_dynamic_attrs_on_simple_class():
    """Test that dynamic attributes work on classes without inheritance."""

    agent = Agent()

    program = """
class Box:
    pass

box = Box()
box.a = 5
box.b = 10
result_a = box.a
result_b = box.b
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result_a") == 5
    assert state.get("result_b") == 10


def test_local_instance_attrs_accessible():
    """Test that attributes declared in __init__ are accessible."""

    agent = Agent()

    program = """
class MyClass:
    def __init__(self):
        self.declared_attr = "value"

obj = MyClass()
result = obj.declared_attr
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "value"

    # Verify it's in local_instance_attrs
    my_class = state.get("MyClass")
    assert "declared_attr" in my_class.local_instance_attrs


def test_write_to_nonwhitelisted_parent_attr_blocked():
    """Test that writes to non-whitelisted parent attributes are blocked."""

    class Parent:
        def __init__(self):
            self.public_attr = "public"

        def sensitive_method(self):
            return "sensitive"

    agent = Agent()
    # Only whitelist public_attr, block sensitive_method
    agent.cls(Parent, include=["public_attr"], exclude=["sensitive_*"])

    program = """
class Child(Parent):
    def __init__(self):
        self.child_attr = "child"

c = Child()
# Try to set an attribute that shadows a non-whitelisted parent method
c.sensitive_method = "hacked"
"""

    with pytest.raises(AgexAttributeError) as e:
        eval_and_get_state(program, agent)
    assert "sensitive_method" in str(e.value)
    assert "parent class" in str(e.value).lower()
