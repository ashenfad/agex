from types import ModuleType

import pytest

from tic.agent import Agent
from tic.eval.objects import PrintTuple
from tic.eval.user_errors import TicAttributeError

from .helpers import eval_and_get_state


def test_hasattr():
    """Tests the hasattr() builtin on various object types."""
    mod = ModuleType("my_mod")
    mod.my_fn = lambda: 1  # type: ignore

    agent = Agent()
    agent.module(mod, name="my_mod")

    program = """
import my_mod

list_val = []

# Test cases
t1 = hasattr(list_val, "append")  # True
t2 = hasattr(list_val, "non_existent") # False
t3 = hasattr(my_mod, "my_fn") # True
t4 = hasattr(my_mod, "other") # False
"""
    state = eval_and_get_state(program, agent)

    assert state.get("t1") is True
    assert state.get("t2") is False
    assert state.get("t3") is True
    assert state.get("t4") is False


def test_dir_and_hasattr_sandboxing():
    """
    Tests that dir() and hasattr() respect the agent's registration rules,
    only exposing members that are explicitly included.
    """
    agent = Agent()

    class MySandboxClass:
        def included_method(self):
            return 1

        def excluded_method(self):
            return 2

    # Register the class, but only include one of the methods
    agent.cls(MySandboxClass, include=["included_method"])

    program = """
inst = MySandboxClass()

# Test hasattr
has_included = hasattr(inst, "included_method")
has_excluded = hasattr(inst, "excluded_method")

# Test dir()
# This should also be a side-effect now, need to check stdout
dir()
"""
    state = eval_and_get_state(program, agent)

    assert state.get("has_included") is True
    assert state.get("has_excluded") is False

    # Check that dir() on the object respects the sandbox
    program2 = "dir(inst)"
    state2 = eval_and_get_state(program2, agent, state)
    stdout = state2.get("__stdout__")
    assert isinstance(stdout, list)
    # The last printed item should be the result of dir(inst)
    dir_result_tuple = stdout[-1]
    assert isinstance(dir_result_tuple, PrintTuple)
    dir_result_list = dir_result_tuple[0]
    assert "included_method" in dir_result_list
    assert "excluded_method" not in dir_result_list


def test_dir_hasattr_on_unregistered_nested_object():
    """
    Tests that dir() and hasattr() respect the sandbox even on attributes
    of a registered class that are themselves instances of unregistered classes.
    """
    agent = Agent()

    # This class is NOT registered with the agent
    class UnregisteredContainer:
        def __init__(self):
            self.safe_attr = 1
            self.unsafe_attr = 2

        def safe_method(self):
            return "safe"

    # This class IS registered
    class RegisteredHost:
        def __init__(self):
            # It holds an instance of a class that the agent knows nothing about
            self.native_list = [1, 2, 3]
            self.unregistered_obj = UnregisteredContainer()

    agent.cls(RegisteredHost, include=["native_list", "unregistered_obj"])

    program = """
# Run hasattr on a whitelisted method of a native type
has_append = hasattr(host.native_list, "append")
has_sizeof = hasattr(host.native_list, "__sizeof__")

# Run hasattr on an unregistered object's attributes
has_safe_attr = hasattr(host.unregistered_obj, "safe_attr")
has_safe_method = hasattr(host.unregistered_obj, "safe_method")

# Run dir() on the native list and the unregistered object
dir(host.native_list)
dir(host.unregistered_obj)
"""
    full_program = f"""
host = RegisteredHost()
{program}
"""
    state = eval_and_get_state(full_program, agent)

    # Check hasattr results
    assert state.get("has_append") is True
    assert state.get("has_sizeof") is False
    assert state.get("has_safe_attr") is False
    assert state.get("has_safe_method") is False

    # Check dir() output
    stdout = state.get("__stdout__")
    assert isinstance(stdout, list)

    # First dir() call was on the list
    dir_list_result = stdout[-2][0]
    assert "append" in dir_list_result
    assert "__sizeof__" not in dir_list_result

    # Second dir() call was on the unregistered object
    dir_unregistered_result = stdout[-1][0]
    assert dir_unregistered_result == []

    # Verify that direct access fails for unregistered attributes/methods
    with pytest.raises(
        TicAttributeError,
        match="'UnregisteredContainer' object has no attribute 'safe_attr'",
    ):
        eval_and_get_state("host.unregistered_obj.safe_attr", agent, state)

    with pytest.raises(
        TicAttributeError,
        match="'UnregisteredContainer' object has no attribute 'safe_method'",
    ):
        eval_and_get_state("host.unregistered_obj.safe_method()", agent, state)


def test_interaction_with_returned_unregistered_object():
    """
    Tests that if a registered function returns an instance of an unregistered
    class, the sandbox still prevents interaction with that object.
    """
    agent = Agent()

    # This class is NOT registered with the agent
    class UnregisteredResult:
        def __init__(self):
            self.value = 42

        def get_value(self):
            return self.value

    # This function IS registered, and it returns an unregistered object
    def get_unregistered_object():
        return UnregisteredResult()

    agent.fn(get_unregistered_object)

    program = """
x = get_unregistered_object()
dir_x = dir(x)
has_value = hasattr(x, "value")
has_get_value = hasattr(x, "get_value")
"""
    state = eval_and_get_state(program, agent)

    # dir() should be empty because no attributes/methods are whitelisted
    stdout = state.get("__stdout__")
    assert isinstance(stdout, list)
    dir_result = stdout[-1][0]
    assert dir_result == []

    # hasattr() should always be False
    assert state.get("has_value") is False
    assert state.get("has_get_value") is False

    # Direct access should fail
    with pytest.raises(
        TicAttributeError, match="'UnregisteredResult' object has no attribute 'value'"
    ):
        eval_and_get_state("get_unregistered_object().value", agent)

    with pytest.raises(
        TicAttributeError,
        match="'UnregisteredResult' object has no attribute 'get_value'",
    ):
        eval_and_get_state("get_unregistered_object().get_value()", agent)
