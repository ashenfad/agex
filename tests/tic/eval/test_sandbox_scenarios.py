import pytest

from tic.agent import Agent
from tic.eval.error import EvalError

from .helpers import eval_and_get_state


def test_inheritance_sandboxing():
    """
    Tests that whitelisted attributes are correctly inherited
    and accessible in the sandbox.
    """
    agent = Agent()

    class Parent:
        def __init__(self):
            self.parent_attr = "parent_val"
            self.parent_unincluded_attr = "secret"

        def parent_method(self):
            return "parent_method_called"

    class Child(Parent):
        def __init__(self):
            super().__init__()
            self.child_attr = "child_val"

        def child_method(self):
            return "child_method_called"

    # Register parent and child, including some attributes/methods from each
    agent.cls(Parent, include=["parent_attr", "parent_method"])
    agent.cls(Child, include=["child_attr", "child_method"])

    program = """
c = Child()
# Access attributes from both
p_attr = c.parent_attr
c_attr = c.child_attr

# Call methods from both
p_method_res = c.parent_method()
c_method_res = c.child_method()

# dir the object
dir(c)
"""
    state = eval_and_get_state(program, agent)

    # Check returned values
    assert state.get("p_attr") == "parent_val"
    assert state.get("c_attr") == "child_val"
    assert state.get("p_method_res") == "parent_method_called"
    assert state.get("c_method_res") == "child_method_called"

    # Check dir() output
    stdout = state.get("__stdout__")
    assert isinstance(stdout, list)
    dir_result = stdout[-1][0]
    assert "parent_attr" in dir_result
    assert "parent_method" in dir_result
    assert "child_attr" in dir_result
    assert "child_method" in dir_result
    assert "parent_unincluded_attr" not in dir_result

    # Verify direct access to unincluded parent attribute fails
    with pytest.raises(
        EvalError, match="'Child' object has no attribute 'parent_unincluded_attr'"
    ):
        eval_and_get_state("c.parent_unincluded_attr", agent, state)


def test_mutation_on_unregistered_datastructures():
    """
    Tests that even if a container (list, dict) is exposed on a registered
    class, methods on that container that are not explicitly whitelisted
    cannot be called.
    """
    agent = Agent()

    class Container:
        def __init__(self):
            self.my_list = [1, 2]
            self.my_dict = {"a": 1}

    # Expose the list and dict attributes themselves
    agent.cls(Container, include=["my_list", "my_dict"])

    # 1. Try to call a non-whitelisted method on the list
    program1 = """
c = Container()
c.my_list.append(3)  # This should work
c.my_list.__sizeof__()  # This should fail
"""
    with pytest.raises(EvalError, match="'list' object has no attribute '__sizeof__'"):
        eval_and_get_state(program1, agent)

    # 2. Try to call a non-whitelisted method on the dict
    program2 = """
c = Container()
c.my_dict["b"] = 2  # This should work
c.my_dict.popitem() # This should fail as popitem is not in WHITELISTED_METHODS
"""
    with pytest.raises(EvalError, match="'dict' object has no attribute 'popitem'"):
        eval_and_get_state(program2, agent)

    # 3. Verify the successful mutation worked before the failure
    program3 = "c = Container(); c.my_list.append(3); c.my_dict['b'] = 2"
    final_state = eval_and_get_state(program3, agent)
    container_instance = final_state.get("c")
    assert container_instance.my_list == [1, 2, 3]
    assert container_instance.my_dict == {"a": 1, "b": 2}


def test_str_format_sandbox_escape_is_blocked():
    """
    Tests that the str.format escape vector is neutralized by the sandboxed
    type() built-in, which returns a placeholder instead of a raw type.
    """
    # This program would have been able to read __subclasses__ if `int` referred
    # to the real type object.
    program = "subclasses_str = '{0.__subclasses__}'.format(int)"

    # We expect this to fail because our sandboxed `int` is a placeholder,
    # and the format mini-language will fail to find `__subclasses__` on it.
    # The error comes from inside Python's format logic, hence the AttributeError,
    # which is wrapped in our EvalError.
    with pytest.raises(EvalError, match="object has no attribute '__subclasses__'"):
        eval_and_get_state(program)
