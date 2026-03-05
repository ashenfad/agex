"""Tests for Agent.state() method."""

import pytest

from agex import Agent, connect_state, events, view
from agex.llm import Dummy
from agex.llm.core import LLMResponse


class TestAgentState:
    def test_state_local_memory(self):
        """state() works with local host and memory storage."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Incrementing", code="task_success(inputs.x + 1)")
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def increment(x: int) -> int:
            """Increment a number."""
            pass

        # Execute task
        increment(5)

        # Get state
        state = agent.state()
        assert state is not None

    def test_state_with_custom_session(self):
        """state() retrieves the correct session."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Storing", code="task_success()"),
                LLMResponse(thinking="Storing", code="task_success()"),
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def store(value: str) -> None:
            """Store a value."""
            pass

        # Execute on different sessions
        store("alice_data", session="alice")
        store("bob_data", session="bob")

        # Should get different state objects (memory storage caches per session)
        alice_state = agent.state("alice")
        bob_state = agent.state("bob")

        # They should be different instances
        assert alice_state is not bob_state

    def test_state_default_session(self):
        """state() defaults to 'default' session."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Processing", code="task_success(inputs.x)")
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def process(x: int) -> int:
            """Process a number."""
            pass

        # Execute on default session
        process(42)

        # Both should return the same state object
        state1 = agent.state()
        state2 = agent.state("default")
        assert state1 is state2

    def test_state_raises_for_http_host(self):
        """state() raises NotImplementedError for HTTP host."""
        from agex.host import connect_host

        llm = Dummy(responses=[])
        agent = Agent(
            host=connect_host(provider="http", url="http://localhost:8000/execute"),
            llm=llm,
        )

        with pytest.raises(
            NotImplementedError, match="doesn't support client-side state access"
        ):
            agent.state()

    def test_state_with_view(self):
        """state() works with view() utility."""
        llm = Dummy(
            responses=[
                LLMResponse(thinking="Calculating", code="task_success(inputs.x * 2)")
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def calculate(x: int) -> int:
            """Calculate something."""
            pass

        calculate(42)

        state = agent.state()
        output = view(state, focus="recent")
        assert isinstance(output, str)

    def test_state_with_events(self):
        """state() works with events() utility."""
        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Processing", code="task_success(inputs.data.upper())"
                )
            ]
        )
        agent = Agent(state=connect_state(type="versioned", storage="memory"), llm=llm)

        @agent.task
        def process(data: str) -> str:
            """Process data."""
            pass

        process("test")

        state = agent.state()
        event_list = events(state)
        assert len(event_list) > 0

    def test_state_with_live_type(self):
        """state() works with live state type."""
        from agex.state.live import Live

        llm = Dummy(responses=[LLMResponse(thinking="Working", code="task_success()")])
        agent = Agent(state=connect_state(type="live", storage="memory"), llm=llm)

        @agent.task
        def work() -> None:
            """Do some work."""
            pass

        work()

        # Should return a state object
        state = agent.state()
        assert state is not None
        assert isinstance(state, Live)

    def test_state_without_state_config(self):
        """state() works when agent has no state config (ephemeral)."""
        from agex.state.live import Live

        llm = Dummy(
            responses=[LLMResponse(thinking="Working", code='task_success("result")')]
        )
        agent = Agent(llm=llm)  # No state config

        @agent.task
        def ephemeral_task() -> str:
            """Do ephemeral work."""
            pass

        ephemeral_task()

        # Should return ephemeral Live state
        state = agent.state()
        assert state is not None
        assert isinstance(state, Live)


def test_hierarchical_session_inheritance():
    """Verify that sub-agents inherit the session from the parent."""
    from agex import clear_agent_registry

    clear_agent_registry()

    specialist = Agent(
        name="specialist",
        llm=Dummy(),
        state=connect_state(type="versioned", storage="memory"),
    )
    orchestrator = Agent(
        name="orchestrator",
        llm=Dummy(),
        state=connect_state(type="versioned", storage="memory"),
    )

    @orchestrator.fn
    @specialist.task
    def specialist_task():
        """Specialist task."""
        pass

    @orchestrator.task
    def main_task():
        """Main task."""
        pass

    # Specialist: Verify its state is isolated by session even when called via orchestrator
    # Turn 1: Run in session_a
    orchestrator.llm.responses = [
        LLMResponse(thinking="call spec", code="specialist_task()\ntask_success()")
    ]
    specialist.llm.responses = [
        LLMResponse(thinking="set spec", code="Y='Session A'\ntask_success()")
    ]
    main_task(session="session_a")

    # Turn 2: Run in session_b
    orchestrator.llm.responses = [
        LLMResponse(thinking="call spec", code="specialist_task()\ntask_success()")
    ]
    specialist.llm.responses = [
        LLMResponse(thinking="set spec", code="Y='Session B'\ntask_success()")
    ]
    main_task(session="session_b")

    # Turn 3: Verify specialist session_a state
    orchestrator.llm.responses = [
        LLMResponse(thinking="call spec", code="task_success(specialist_task())")
    ]
    specialist.llm.responses = [
        LLMResponse(thinking="get spec", code="task_success(Y)")
    ]
    assert main_task(session="session_a") == "Session A"

    # Turn 4: Verify specialist session_b state
    orchestrator.llm.responses = [
        LLMResponse(thinking="call spec", code="task_success(specialist_task())")
    ]
    specialist.llm.responses = [
        LLMResponse(thinking="get spec", code="task_success(Y)")
    ]
    assert main_task(session="session_b") == "Session B"


def test_session_vfs_isolation():
    """Verify that VFS modules are isolated by session."""
    from agex import clear_agent_registry, connect_fs

    clear_agent_registry()

    agent = Agent(
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )

    @agent.task
    def get_config_val():
        """Import config and return VAL."""
        pass

    # Session A: config.py has VAL=42
    agent.fs(session="session_a").write("config.py", b"VAL = 42")
    # Session B: config.py has VAL=99
    agent.fs(session="session_b").write("config.py", b"VAL = 99")

    # Session A: verify VAL is 42
    agent.llm.responses = [
        LLMResponse(thinking="import", code="import config\ntask_success(config.VAL)")
    ]
    assert get_config_val(session="session_a") == 42

    # Session B: verify VAL is 99
    agent.llm.responses = [
        LLMResponse(thinking="import", code="import config\ntask_success(config.VAL)")
    ]
    assert get_config_val(session="session_b") == 99


def test_vfs_module_rehydration_with_session():
    """Verify that VFS modules work correctly in-memory.

    Note: With sandtrap, VFS modules are real Python module objects,
    so pickle roundtrip is not supported.
    The VFS module cache is also global, not per-session.
    """
    from agex import clear_agent_registry, connect_fs

    clear_agent_registry()

    agent = Agent(
        name="rehydrate_agent",
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )

    @agent.task
    def get_val():
        """Get value from lib module."""
        pass

    # Setup and import from a session
    agent.fs(session="session_a").write("mylib.py", b"X = 'Alpha'")
    agent.llm.responses = [
        LLMResponse(thinking="get", code="import mylib\ntask_success(mylib.X)")
    ]
    val = get_val(session="session_a")

    # Verify the VFS module was loaded and the value is correct
    assert val == "Alpha"
