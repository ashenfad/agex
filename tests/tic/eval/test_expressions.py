from types import ModuleType

import pytest

from tic.agent import Agent
from tic.eval.error import EvalError
from tic.eval.functions import NativeFunction

from .helpers import eval_and_get_state


def test_eval_list_literal():
    program = """
x = 3
y = [1, 2, x]
z = []
"""
    state = eval_and_get_state(program)
    assert state.get("y") == [1, 2, 3]
    assert state.get("z") == []


def test_eval_collections():
    """Tests tuples, sets, and dicts."""
    program = """
x = (1, "a", True)
y = {1, "a", True, 1}
z = {"a": 1, "b": x}
w = [1, (2, 3), {"a": {4, 5}}]
"""
    state = eval_and_get_state(program)
    assert state.get("x") == (1, "a", True)
    assert state.get("y") == {1, "a", True}
    assert state.get("z") == {"a": 1, "b": (1, "a", True)}
    assert state.get("w") == [1, (2, 3), {"a": {4, 5}}]


def test_eval_subscript():
    """Tests getting values via subscript."""
    program = """
data = {"a": [10, 20], "b": {"c": 100}}
x = data["a"][1]
y = data["b"]["c"]
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 20
    assert state.get("y") == 100


def test_registered_fn_is_first_class():
    """Tests that a registered function can be accessed directly by name."""
    agent = Agent()

    @agent.fn
    def my_registered_fn(a, b):
        return a + b

    program = """
x = my_registered_fn
y = x(1, 2)
"""
    state = eval_and_get_state(program, agent)
    assert isinstance(state.get("x"), NativeFunction)
    assert state.get("y") == 3


def test_registered_cls_is_first_class():
    """Tests that a registered class can be accessed directly by name."""
    agent = Agent()

    class MyRegisteredCls:
        def __init__(self, val: int):
            self.val = val

    agent.cls(MyRegisteredCls)

    program = """
MyCls = MyRegisteredCls
inst = MyCls(42)
is_inst = isinstance(inst, MyCls)
"""
    state = eval_and_get_state(program, agent)
    assert state.get("MyCls") == MyRegisteredCls
    assert isinstance(state.get("inst"), MyRegisteredCls)
    assert state.get("inst").val == 42
    assert state.get("is_inst") is True


def test_registered_module_is_first_class():
    """Tests that a registered module and its members are accessible."""

    # Create a dummy module and class for testing
    class DummyDataFrame:
        pass

    dummy_module = ModuleType("pd")
    dummy_module.DataFrame = DummyDataFrame

    agent = Agent()
    agent.module(dummy_module, name="pd")

    program = """
import pd
df = pd.DataFrame()
is_inst = isinstance(df, pd.DataFrame)
"""
    state = eval_and_get_state(program, agent)
    assert isinstance(state.get("df"), DummyDataFrame)
    assert state.get("is_inst") is True


def test_help_on_registered_module():
    """Tests that help() works on a registered module."""
    dummy_module = ModuleType("my_mod")
    dummy_module.my_fn = lambda: 1
    agent = Agent()
    agent.module(dummy_module, name="my_mod")

    program = """
import my_mod
help(my_mod)
"""
    state = eval_and_get_state(program, agent)
    stdout = state.get("__stdout__")
    assert len(stdout) == 1
    help_text = stdout[0][0]
    assert "Help on module my_mod" in help_text
    assert "my_fn" in help_text


def test_help_on_registered_fn():
    """Tests that help() works on a registered fn not otherwise in scope."""
    agent = Agent()

    @agent.fn
    def my_reg_fn_for_help():
        "A docstring for help."
        pass

    # Note that `my_reg_fn_for_help` is NOT defined in the program string.
    # The evaluator should find it in the agent's registry via `visit_Name`.
    program = "help(my_reg_fn_for_help)"

    state = eval_and_get_state(program, agent)
    stdout = state.get("__stdout__")
    assert len(stdout) == 1
    help_text = stdout[0][0]
    assert "A docstring for help." in help_text


def test_f_string_formatting():
    """Tests that f-string formatting works as expected."""
    program = """
name = "world"
s = f"hello {name.upper()}"
"""
    state = eval_and_get_state(program)
    assert state.get("s") == "hello WORLD"


def test_percent_string_formatting():
    """Tests that %-string formatting works as expected."""
    program = """
name = "world"
s = "hello %s" % name.upper()
"""
    state = eval_and_get_state(program)
    assert state.get("s") == "hello WORLD"


def test_f_string_sandbox():
    """Tests that f-string expressions cannot call un-whitelisted methods."""
    agent = Agent()

    class MyHostClass:
        def safe_method(self):
            return "SAFE"

        def unsafe_method(self):
            return "UNSAFE"

    agent.cls(MyHostClass, include=["safe_method"])

    # This program tries to call the un-whitelisted `unsafe_method` inside an f-string.
    program = """
inst = MyHostClass()
s = f"hello {inst.unsafe_method()}"
"""

    with pytest.raises(EvalError) as excinfo:
        eval_and_get_state(program, agent)

    assert "object has no attribute 'unsafe_method'" in str(excinfo.value)

    # Also test that the safe method works.
    program_safe = """
inst = MyHostClass()
s = f"hello {inst.safe_method()}"
"""
    state = eval_and_get_state(program_safe, agent)
    assert state.get("s") == "hello SAFE"


def test_multiline_strings():
    """Tests that various ways of defining multi-line strings work."""
    expected = "hello\nworld"
    programs = {
        "triple_double": 's = """hello\nworld"""',
        "triple_single": "s = '''hello\nworld'''",
        "implicit_concat_parens": 's = ("hello" "\\n" "world")',
        "explicit_backslash": 's = "hello\\n" \\\n    "world"',
        "list_join": 's = "\\n".join(["hello", "world"])',
    }

    for name, program in programs.items():
        state = eval_and_get_state(program)
        assert state.get("s") == expected, f"Test case '{name}' failed"
