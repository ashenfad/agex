"""Test VFS persistence across agent turns under the stateless contract.

VFS files survive across turns and tasks; Python namespace state does not.
The agent must re-import its VFS modules each turn it wants to use them.
"""

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.console import pprint_events
from agex.llm import Dummy
from tests.agex._emissions import make_response


def test_vfs_module_reimport_works_across_turns():
    """A VFS module written once is importable on any subsequent turn.

    Each turn must import freshly — the import binding does not survive
    between python_action emissions.
    """
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
        make_response(
            thinking="Turn 1: import and probe the module, then continue",
            code="import utils\nprint(utils.get_val())",
        ),
        make_response(
            thinking="Turn 2: re-import (namespace doesn't carry) and finish",
            code="import utils\ntask_success(utils.get_val())",
        ),
    ]

    @agent.task
    def multi_turn() -> int:
        """Test multi-turn VFS module access."""
        pass

    result = multi_turn(on_event=pprint_events)
    assert result == 42


def test_namespace_does_not_persist_across_turns():
    """Imports and definitions from turn 1 must NOT carry into turn 2.

    Turn 2 references ``utils`` without importing — should fail with
    NameError, surfaced back to the agent as an OutputEvent.  The third
    turn re-imports cleanly and finishes.
    """
    clear_agent_registry()

    agent = Agent(
        name="test",
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
        max_iterations=4,
    )

    agent.fs().write("utils.py", b"CONST = 42\ndef get_val(): return CONST")

    agent.llm.responses = [
        make_response(
            thinking="Turn 1: import",
            code="import utils\nprint(utils.get_val())",
        ),
        make_response(
            thinking="Turn 2: try to use without importing — should fail",
            code="print(utils.get_val())",
        ),
        make_response(
            thinking="Turn 3: re-import and finish",
            code="import utils\ntask_success(utils.get_val())",
        ),
    ]

    @agent.task
    def t() -> int:
        """Test stateless namespace contract."""
        pass

    result = t(on_event=pprint_events)
    assert result == 42
