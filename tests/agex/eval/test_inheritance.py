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


def test_multi_level_inheritance():
    """Test inheritance across multiple levels (grandparent -> parent -> child)."""

    class Grandparent:
        def __init__(self):
            self.gp_attr = "grandparent"

        def gp_method(self):
            return "gp"

    agent = Agent()
    agent.cls(Grandparent, include=["gp_attr", "gp_method"])

    program = """
class Parent(Grandparent):
    def __init__(self):
        super().__init__()
        self.p_attr = "parent"

    def p_method(self):
        return "p"

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.c_attr = "child"

    def c_method(self):
        return "c"

c = Child()
result_gp = c.gp_attr
result_p = c.p_attr
result_c = c.c_attr
method_gp = c.gp_method()
method_p = c.p_method()
method_c = c.c_method()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result_gp") == "grandparent"
    assert state.get("result_p") == "parent"
    assert state.get("result_c") == "child"
    assert state.get("method_gp") == "gp"
    assert state.get("method_p") == "p"
    assert state.get("method_c") == "c"


def test_agex_class_to_agex_class_inheritance():
    """Test that AgexClass can inherit from another AgexClass."""

    agent = Agent()

    program = """
class Parent:
    def __init__(self, x):
        self.x = x

    def get_x(self):
        return self.x

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y

    def get_sum(self):
        return self.x + self.y

c = Child(10, 20)
result_x = c.get_x()
result_sum = c.get_sum()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result_x") == 10
    assert state.get("result_sum") == 30


def test_method_override():
    """Test that child methods can override parent methods."""

    class Parent:
        def greet(self):
            return "Hello from Parent"

    agent = Agent()
    agent.cls(Parent, include=["greet"])

    program = """
class Child(Parent):
    def greet(self):
        return "Hello from Child"

c = Child()
result = c.greet()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "Hello from Child"


def test_method_override_with_super():
    """Test that overridden methods can call parent via super()."""

    class Parent:
        def compute(self, x):
            return x * 2

    agent = Agent()
    agent.cls(Parent, include=["compute"])

    program = """
class Child(Parent):
    def compute(self, x):
        parent_result = super().compute(x)
        return parent_result + 1

c = Child()
result = c.compute(5)
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == 11  # (5 * 2) + 1


def test_attribute_shadowing():
    """Test that child can set attributes with same name as parent."""

    agent = Agent()

    program = """
class Parent:
    def __init__(self):
        self.value = "parent"

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.value = "child"  # Shadow parent's value

c = Child()
result = c.value
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "child"


def test_diamond_inheritance():
    """Test diamond inheritance pattern (C3 linearization)."""

    class A:
        def method(self):
            return "A"

    class B:
        def method(self):
            return "B"

    agent = Agent()
    agent.cls(A, include=["method"])
    agent.cls(B, include=["method"])

    program = """
class C(A, B):
    pass

c = C()
result = c.method()  # Should resolve to A.method (first in bases)
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "A"


def test_mro_preserves_order():
    """Test that MRO preserves local precedence order."""

    class A:
        pass

    class B:
        pass

    agent = Agent()
    agent.cls(A)
    agent.cls(B)

    program = """
class C(A, B):
    pass
"""

    state = eval_and_get_state(program, agent)
    c_cls = state.get("C")

    # MRO should be: C, A, B, object
    assert c_cls.mro[0] is c_cls
    assert c_cls.mro[1] is A
    assert c_cls.mro[2] is B
    assert c_cls.mro[3] is object


def test_isinstance_with_inheritance():
    """Test that isinstance works correctly with inherited classes."""

    class Parent:
        pass

    agent = Agent()
    agent.cls(Parent)

    program = """
class Child(Parent):
    pass

c = Child()
is_child = isinstance(c, Child)
is_parent = isinstance(c, Parent)
"""

    state = eval_and_get_state(program, agent)
    assert state.get("is_child") is True
    assert state.get("is_parent") is True
