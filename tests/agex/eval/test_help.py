from agex.agent import Agent

from .helpers import eval_and_get_state


def test_stateful_help_builtin_general():
    """
    Tests that the stateful `help()` function, when called with no arguments,
    writes its output to __stdout__.
    """
    agent = Agent()

    # Register some items to test against
    @agent.fn
    def my_test_func():
        pass

    @agent.cls
    class MyTestClass:
        pass

    state = eval_and_get_state("help()", agent)
    stdout = state.get("__stdout__")
    assert len(stdout) == 1
    help_str = stdout[0][0]  # It's a PrintTuple with one string inside

    assert "Available items:" in help_str
    assert "Functions:" in help_str
    assert "- my_test_func" in help_str
    # help() is a builtin, so it shouldn't be in the *agent's* fn_registry
    assert "help" not in help_str
    assert "Classes:" in help_str
    assert "- MyTestClass" in help_str


def test_help_with_simple_name_argument():
    """
    Tests that `help(x)` where `x` is a simple variable (a name) correctly
    triggers the help lookup for the object referred to by `x`.
    """
    agent = Agent()
    code = """
x = 1
help(x)
"""
    state = eval_and_get_state(code, agent)
    stdout = state.get("__stdout__")
    assert len(stdout) == 1
    help_str = stdout[0][0]
    assert "Convert a number or string to an integer" in help_str
