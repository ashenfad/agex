from types import ModuleType

from agex import events
from agex.agent import Agent
from agex.agent.events import OutputEvent

from .helpers import eval_and_get_state


def test_dir_no_args():
    """Tests that dir() with no arguments lists names in the current scope."""
    program = """
x = 1
y = "hello"
def my_func():
    z = 3
dir()
"""
    state = eval_and_get_state(program)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 1
    dir_result = output_events[0].parts[0]
    # It should not include 'z' from the function's inner scope
    assert sorted(dir_result) == ["my_func", "x", "y"]


def test_dir_with_module():
    """Tests that dir() on a module lists its members."""
    mod = ModuleType("my_mod")
    mod.my_public_fn = lambda: 1  # type: ignore
    mod._my_private_fn = lambda: 2  # type: ignore

    agent = Agent()
    agent.module(mod, name="my_mod")

    program = """
import my_mod
dir(my_mod)
"""
    state = eval_and_get_state(program, agent)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 1
    dir_result = output_events[0].parts[0]
    assert dir_result == ["my_public_fn"]


def test_dir_on_native_object():
    """Tests that dir() on a native object like a list works."""
    program = """
my_list = [1, 2]
dir(my_list)
"""
    state = eval_and_get_state(program)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 1
    dir_result = output_events[0].parts[0]
    # Check for a few common, public list methods
    assert "append" in dir_result
    assert "pop" in dir_result
    assert "clear" in dir_result
    assert "sort" in dir_result
    assert "__sizeof__" not in dir_result
