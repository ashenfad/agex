from agex import events
from agex.agent import Agent
from agex.agent.events import OutputEvent

from .helpers import eval_and_get_state


def test_stateful_help_builtin_general():
    """
    Tests that the stateful `help()` function, when called with no arguments,
    writes its output to the event log.
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
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 1
    help_str = output_events[0].parts[0]

    assert "Functions:" in help_str
    assert "my_test_func" in help_str
    # help() is a builtin, so it shouldn't be in the *agent's* fn_registry
    assert "help" not in help_str
    assert "Classes:" in help_str
    assert "MyTestClass" in help_str


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
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 1
    help_str = output_events[0].parts[0]
    assert "Convert a number or string to an integer" in help_str
