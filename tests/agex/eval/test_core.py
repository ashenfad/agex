from agex import events
from agex.agent import Agent
from agex.agent.events import OutputEvent

from .helpers import eval_and_get_state


def test_eval_assignment():
    state = eval_and_get_state("x = 1\ny = x")
    assert state.get("x") == 1
    assert state.get("y") == 1


def test_stateful_print_builtin():
    """
    Tests that the stateful `print` function correctly appends items to the
    event log.
    """
    agent = Agent()
    state = eval_and_get_state("1+1", agent)
    all_events = events(state)
    # The standalone expression "1+1" should now produce an OutputEvent.
    assert len(all_events) == 1
    assert isinstance(all_events[0], OutputEvent)
    assert all_events[0].parts == [2]

    # First print call with multiple arguments becomes a single OutputEvent
    state = eval_and_get_state('print(1, "hello")', agent, state)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 2
    assert output_events[1].parts == [1, "hello"]

    # Second print call appends another OutputEvent
    state = eval_and_get_state("print(True, None)", agent, state)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 3
    assert output_events[2].parts == [True, None]

    # Printing a variable
    state = eval_and_get_state("x = [10, 20]\nprint(x)", agent, state)
    output_events = [e for e in events(state) if isinstance(e, OutputEvent)]
    assert len(output_events) == 4
    assert output_events[3].parts == [[10, 20]]
    assert [e.parts for e in output_events] == [
        [2],
        [1, "hello"],
        [True, None],
        [[10, 20]],
    ]
