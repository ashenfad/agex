"""Test VFS module persistence across agent turns."""

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.console import pprint_events
from agex.llm import Dummy, LLMResponse


def test_vfs_module_survives_across_turns():
    """VFS module imported in turn 1 should be accessible in turn 2."""
    clear_agent_registry()

    agent = Agent(
        name="test",
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
        max_iterations=3,
    )

    agent.fs().write("utils.py", b"CONST = 42\ndef get_val(): return CONST")

    agent.llm.responses = [
        LLMResponse(
            thinking="Turn 1: import and use module",
            code='import utils\nresult = utils.get_val()\ntask_continue("got", result)',
        ),
        LLMResponse(
            thinking="Turn 2: use module again without re-importing",
            code="result2 = utils.get_val()\ntask_success(result2)",
        ),
    ]

    @agent.task
    def multi_turn() -> int:
        """Test multi-turn VFS module access."""
        pass

    result = multi_turn(on_event=pprint_events)
    assert result == 42


def test_vfs_closure_survives_across_turns():
    """Closure over VFS module should work across turns."""
    clear_agent_registry()

    agent = Agent(
        name="test",
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
        max_iterations=3,
    )

    agent.fs().write("utils.py", b"CONST = 42\ndef get_val(): return CONST")

    agent.llm.responses = [
        LLMResponse(
            thinking="Turn 1: define closure over module",
            code="import utils\ndef my_fn():\n    return utils.get_val()\ntask_continue(my_fn())",
        ),
        LLMResponse(
            thinking="Turn 2: call closure again",
            code="task_success(my_fn())",
        ),
    ]

    @agent.task
    def closure_turn() -> int:
        """Test closure persistence."""
        pass

    result = closure_turn(on_event=pprint_events)
    assert result == 42
