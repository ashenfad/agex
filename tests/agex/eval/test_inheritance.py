"""
Tests for agex class inheritance support.
"""

import pytest

from agex import Agent
from agex.eval.error import EvalError
from tests.agex.eval.helpers import eval_and_get_state


def test_basic_inheritance_from_registered_class():
    """Test that AgexClass can inherit from a registered host class."""

    class Parent:
        def __init__(self):
            self.parent_attr = "parent_val"

        def parent_method(self):
            return "parent_method_called"

    agent = Agent()
    agent.cls(Parent, include=["parent_attr", "parent_method"])

    program = """
class Child(Parent):
    def __init__(self):
        self.child_attr = "child_val"
        
c = Child()
result = c.child_attr
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "child_val"

    # Verify the class was created with correct fields
    child_cls = state.get("Child")
    assert child_cls.name == "Child"
    assert len(child_cls.bases) == 1
    assert child_cls.bases[0] is Parent
    assert "child_attr" in child_cls.local_instance_attrs


def test_mro_computation():
    """Test that MRO is computed correctly for mixed hierarchies."""

    class HostBase:
        pass

    agent = Agent()
    agent.cls(HostBase)

    program = """
class A(HostBase):
    pass
    
class B(A):
    pass
"""

    state = eval_and_get_state(program, agent)

    a_cls = state.get("A")
    b_cls = state.get("B")

    # A's MRO should be: A, HostBase, object
    assert len(a_cls.mro) == 3
    assert a_cls.mro[0] is a_cls
    assert a_cls.mro[1] is HostBase
    assert a_cls.mro[2] is object

    # B's MRO should be: B, A, HostBase, object
    assert len(b_cls.mro) == 4
    assert b_cls.mro[0] is b_cls
    assert b_cls.mro[1] is a_cls
    assert b_cls.mro[2] is HostBase
    assert b_cls.mro[3] is object


def test_unregistered_base_rejected():
    """Test that cannot inherit from unregistered class."""

    agent = Agent()
    # Don't register any classes - test with a string import that will fail

    program = """
# This should fail because string is not a registered class
# even though it's a built-in type
class MyClass(str):
    pass
"""

    with pytest.raises(EvalError) as e:
        eval_and_get_state(program, agent)
    assert "Cannot inherit from unregistered class" in str(e.value)
