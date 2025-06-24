from tic.agent import Agent

from .helpers import eval_and_get_state


def test_stateful_help_builtin_general():
    """
    Tests that the stateful `help()` function, when called with no arguments,
    returns a string listing available items from the agent's registry.
    """
    agent = Agent()

    # Register some items to test against
    @agent.fn
    def my_test_func():
        pass

    @agent.cls
    class MyTestClass:
        pass

    state = eval_and_get_state("help_text = help()", agent)
    help_str = state.get("help_text")

    assert "Available items:" in help_str
    assert "Functions:" in help_str
    assert "- my_test_func" in help_str
    # help() is a builtin, so it shouldn't be in the *agent's* fn_registry
    assert "help" not in help_str
    assert "Classes:" in help_str
    assert "- MyTestClass" in help_str
