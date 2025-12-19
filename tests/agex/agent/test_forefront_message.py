"""Tests for transient forefront message injection."""

from unittest.mock import MagicMock

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.events import SystemNoteEvent
from agex.llm.dummy_client import DummyLLMClient


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_transient_message_injection():
    """Test that forefront message is injected into LLM context but not event log."""
    # 1. Setup Agent with Dummy Client
    client = DummyLLMClient()
    # Mock the complete method to inspect arguments
    client.complete = MagicMock(
        return_value=MagicMock(thinking="ok", code="task_success()", title="done")
    )

    # 5 iterations. Threshold is max(0, 5-3) = 2.
    # Iteration 0 (silent), 1 (silent), 2 (warn), 3 (warn), 4 (warn)
    # The agent loop runs for range(5).
    # We want to verify that `complete` is called 5 times, and check arg on the LAST one.

    # We need the task to NOT finish immediately so the loop runs multiple times.
    # But DummyLLMClient returns "task_success()", so it would finish immediately.
    # We need to change the loop behavior or mock client responses to be "continue" for a few steps.

    # EASIER: Mock `_get_forefront_message` to behave predictably for a simple single-step test,
    # OR just unit test `_get_forefront_message` logic separately?
    # Let's rely on the integration test but force the loop to run once at a high iteration index?
    # No, can't easily jump ahead.

    # Cleanest way: Test with 1 iteration (max=1).
    # Threshold = max(0, 1-3) = 0. So Iteration 0 >= 0 -> Message SHOULD appear.
    agent = Agent(name="tester", llm_client=client, max_iterations=1)

    # 2. Run a simple task
    @agent.task
    def simple_task():
        """Do nothing."""
        pass

    simple_task()

    # 3. Verify complete call arguments
    call_args = client.complete.call_args
    assert call_args is not None
    events_passed_to_llm = call_args[0][1]

    # 4. Assert the LAST event passed to LLM is our transient message
    last_event = events_passed_to_llm[-1]
    assert isinstance(last_event, SystemNoteEvent)
    assert "iteration 1 of 1" in last_event.message

    # 5. Assert the transient message is NOT in the agent's actual event log
    transient_count = sum(
        1 for e in events_passed_to_llm if isinstance(e, SystemNoteEvent)
    )
    assert transient_count == 1


def test_forefront_logic_thresholds():
    """Unit test for the threshold logic."""
    agent = Agent(name="tester", max_iterations=10)
    # Mock state
    from agex.state import Live, Namespaced

    state = Namespaced(Live(), "test")

    # Iteration 7: No message
    assert agent._get_forefront_message(7, state) is None

    # Iteration 8: Message
    msg = agent._get_forefront_message(8, state)
    assert msg is not None
    assert "iteration 9 of 10" in msg

    # Small max_iterations logic
    agent.max_iterations = 5
    # Threshold = max(0, 5-3) = 2.

    assert agent._get_forefront_message(1, state) is None
    assert "iteration 3 of 5" in agent._get_forefront_message(2, state)


def test_forefront_user_functions():
    """Test that user defined functions appear in the system note."""
    from unittest.mock import Mock

    from agex.eval.functions import UserFunction
    from agex.state import Live, Namespaced

    agent = Agent(name="tester", max_iterations=10)
    state = Namespaced(Live(), "test_namespace")

    # Mock a UserFunction in the agent's state
    # We need to set it in both the object store AND the shadow set
    mock_func = Mock(spec=UserFunction)
    mock_func.__class__ = UserFunction  # Hack to pass isinstance check
    mock_func.__signature__ = "(x, y=1)"

    # Set in state
    state.set("my_func", mock_func)
    state.set("__sys_user_fn_names__", {"my_func"})

    # Run _get_forefront_message for an early iteration (0)
    # Should contain user functions even if iteration warning is not triggered
    msg = agent._get_forefront_message(0, state)

    assert msg is not None
    assert "User Defined Functions" in msg
    assert "GUARANTEE" in msg
    assert "PERFORMANCE" in msg
    assert "DO NOT" in msg
    assert "my_func(x, y=1)" in msg

    # Test lazy cleanup
    # Remove function from state but keep in shadow set
    state.remove("my_func")
    # shadow set still has "my_func"
    assert "my_func" in state.get("__sys_user_fn_names__")

    # Call again -> should trigger cleanup
    msg = agent._get_forefront_message(0, state)

    # Should be None because no functions found and iteration is early
    assert msg is None

    # Shadow set should be updated
    assert "my_func" not in state.get("__sys_user_fn_names__")
