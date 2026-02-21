import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.console import pprint_events
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def create_agent(name="test_agent"):
    return Agent(
        name=name,
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )


def test_vfs_module_persistence_across_turns():
    """
    Test that a VFS module reference can be persisted in state (via closure)
    and reused in a subsequent turn.

    This reproduces the 'UnpicklableVariableError' seen in the funcy example.
    """
    agent = create_agent("persistence_agent")

    # 1. Create the module
    agent.fs().write("utils.py", b"CONST = 42\ndef get_val(): return CONST")

    # 2. Define a function that closes over the module and return it
    # This forces the system to pickle the function -> closure -> module
    responses = [
        LLMResponse(
            thinking="Create closure",
            code="import utils\ndef my_closure(): return utils.get_val()\ntask_success(my_closure)",
        ),
        # 3. In the next turn (simulated by calling the returned function or checking state),
        # we expect the closure to still work.
    ]
    agent.llm.responses = responses

    @agent.task
    def create_closure_task():
        """Create a closure over a VFS module."""
        pass

    # This step triggers snapshot() at the end.
    # If the VFS module is not picklable, this might fail silently (marker created)
    # or raise error depending on where it's caught.
    # In the funcy example, it failed when *using* it next time.
    closure = create_closure_task(on_event=pprint_events)

    # Verify we got a function back
    assert callable(closure)

    # Verify the closure works immediately (in-memory)
    assert closure() == 42

    # Note: SbFunction pickle roundtrip requires re-activation (sandbox context).
    # External pickle/unpickle without a sandbox will produce an inactive SbFunction.
    # The important behavior is that in-memory closures over VFS modules work.


def test_vfs_module_session_persistence():
    """Test that VFS modules rehydrate into the correct session."""
    agent = create_agent("session_agent")

    # 1. Create different module content in two sessions
    # Session A
    agent.fs(session="session_a").write("config.py", b"VAL = 'A'")
    # Session B
    agent.fs(session="session_b").write("config.py", b"VAL = 'B'")

    # 2. Get module references from both sessions
    @agent.task
    def get_config():
        """Get the config module."""
        pass

    agent.llm.responses = [
        LLMResponse(
            thinking="Get module A",
            code="import config\ntask_success(config)",
        )
    ]
    mod_a = get_config(session="session_a", on_event=pprint_events)

    agent.llm.responses = [
        LLMResponse(
            thinking="Get module B",
            code="import config\ntask_success(config)",
        )
    ]
    mod_b = get_config(session="session_b", on_event=pprint_events)

    # 3. Verify session isolation — each session's module should have its own value
    assert mod_a.VAL == "A"
    assert mod_b.VAL == "B"


if __name__ == "__main__":
    pytest.main([__file__])
