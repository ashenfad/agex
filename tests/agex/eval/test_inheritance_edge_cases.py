"""
Additional edge case and security tests for inheritance.
"""

import pytest

from agex import Agent
from agex.eval.error import EvalError
from agex.eval.user_errors import AgexAttributeError
from tests.agex.eval.helpers import eval_and_get_state


def test_constructable_false_blocks_inheritance():
    """Test that constructable=False prevents class from being inherited."""

    class NotConstructable:
        def __init__(self):
            self.x = 10

    agent = Agent()
    agent.cls(NotConstructable, constructable=False, include=["x"])

    program = """
class Child(NotConstructable):
    pass
"""

    with pytest.raises(EvalError) as e:
        eval_and_get_state(program, agent)
    assert "not constructable" in str(e.value).lower()
    assert "NotConstructable" in str(e.value)


def test_read_nonwhitelisted_parent_attr_blocked():
    """Test that reading non-whitelisted parent attributes is blocked."""

    class Parent:
        def __init__(self):
            self.public = "ok"
            self._private = "secret"

    agent = Agent()
    agent.cls(Parent, include=["public"], exclude=["_*"])

    program = """
class Child(Parent):
    def __init__(self):
        super().__init__()

c = Child()
public_val = c.public  # Should work
"""

    state = eval_and_get_state(program, agent)
    assert state.get("public_val") == "ok"

    # Now try to read the non-whitelisted attribute
    program2 = """
private_val = c._private  # Should fail
"""

    with pytest.raises(AgexAttributeError) as e:
        eval_and_get_state(program2, agent, state)
    assert "_private" in str(e.value)


def test_host_method_accesses_synced_attribute():
    """Test that host methods can access attributes set by agex code via host proxy."""

    class Parent:
        def __init__(self):
            self.value = 0

        def get_double(self):
            # This host method should be able to access self.value
            # set by the child's __init__
            return self.value * 2

    agent = Agent()
    agent.cls(Parent, include=["value", "get_double"])

    program = """
class Child(Parent):
    def __init__(self):
        super().__init__()
        self.value = 5  # Set via agex, should sync to host proxy

c = Child()
result = c.get_double()  # Host method should see value=5
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == 10


def test_super_with_no_parent_method():
    """Test that super().method() fails if method doesn't exist in parent."""

    class Parent:
        pass

    agent = Agent()
    agent.cls(Parent)

    program = """
class Child(Parent):
    def call_parent(self):
        return super().nonexistent()

c = Child()
c.call_parent()
"""

    with pytest.raises(AgexAttributeError) as e:
        eval_and_get_state(program, agent)
    assert "nonexistent" in str(e.value)


def test_multiple_inheritance_with_host_and_agex():
    """Test multiple inheritance mixing host and AgexClass bases."""

    class HostBase:
        def host_method(self):
            return "host"

    agent = Agent()
    agent.cls(HostBase, include=["host_method"])

    program = """
class AgexBase:
    def agex_method(self):
        return "agex"

class Child(AgexBase, HostBase):
    def combined(self):
        return self.agex_method() + " " + self.host_method()

c = Child()
result = c.combined()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "agex host"


def test_super_in_deeply_nested_hierarchy():
    """Test super() works correctly in deeply nested class hierarchies."""

    agent = Agent()

    program = """
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return super().method() + "B"

class C(B):
    def method(self):
        return super().method() + "C"

class D(C):
    def method(self):
        return super().method() + "D"

d = D()
result = d.method()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == "ABCD"


def test_attribute_access_respects_mro():
    """Test that attribute access follows MRO correctly."""

    class A:
        def __init__(self):
            self.value = "A"

    class B:
        def __init__(self):
            self.value = "B"

    agent = Agent()
    agent.cls(A, include=["value"])
    agent.cls(B, include=["value"])

    program = """
class C(A, B):
    def __init__(self):
        super().__init__()

c = C()
result = c.value
"""

    state = eval_and_get_state(program, agent)
    # Should get A's value since A comes first in MRO
    assert state.get("result") == "A"


def test_super_context_isolation():
    """Test that super() context is properly isolated per method call."""

    agent = Agent()

    program = """
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return super().method() + "B"

    def other_method(self):
        # This should have its own super() context
        return super().method() + "X"

b = B()
result1 = b.method()
result2 = b.other_method()
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result1") == "AB"
    assert state.get("result2") == "AX"
