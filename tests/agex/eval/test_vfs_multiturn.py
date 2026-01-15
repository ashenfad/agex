import pytest

from agex import (
    Agent,
    clear_agent_registry,
    connect_fs,
    connect_state,
    pprint_events,
)
from agex.llm import Dummy, LLMResponse


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_vfs_import_multiturn(tmp_path):
    """Verify that a VFS module can be imported across multiple tasks with fresh agent instances."""
    state_path = str(tmp_path / "state")
    state_config = connect_state(type="versioned", storage="disk", path=state_path)
    fs = connect_fs(type="virtual")

    agent1 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")
    print(f"\nDEBUG: agent1 fingerprint: {agent1.fingerprint}")

    # Task 1: Create and import a module
    agent1.llm.responses = [
        LLMResponse(
            thinking="I will create utils.py",
            files={"utils.py": "VAL = 42"},
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent1.task
    def task1():
        """Create utils."""
        pass

    assert task1(on_event=pprint_events) == 42

    # Task 2: Create a NEW agent instance pointing to the SAME state/fs
    clear_agent_registry()
    agent2 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")
    print(f"DEBUG: agent2 fingerprint: {agent2.fingerprint}")

    agent2.llm.responses = [
        LLMResponse(
            thinking="I will import utils again",
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent2.task
    def task2():
        """Import utils."""
        pass

    assert task2(on_event=pprint_events) == 42


def test_vfs_package_import_multiturn(tmp_path):
    """Verify that a VFS package can be imported across multiple tasks with fresh agent instances."""
    state_path = str(tmp_path / "state_pkg")
    state_config = connect_state(type="versioned", storage="disk", path=state_path)
    fs = connect_fs(type="virtual")

    agent1 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    # Task 1: Create a package and import it
    agent1.llm.responses = [
        LLMResponse(
            thinking="I will create a package",
            files={"pkg/__init__.py": "X = 1", "pkg/mod.py": "Y = 2"},
            code="import pkg.mod\ntask_success((pkg.X, pkg.mod.Y))",
        )
    ]

    @agent1.task
    def task1():
        """Create package."""
        pass

    assert task1(on_event=pprint_events) == (1, 2)

    # Task 2: Create a NEW agent instance pointing to the SAME state/fs
    clear_agent_registry()
    agent2 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    agent2.llm.responses = [
        LLMResponse(
            thinking="Import again",
            code="import pkg.mod\ntask_success((pkg.X, pkg.mod.Y))",
        )
    ]

    @agent2.task
    def task2():
        """Import package."""
        pass

    assert task2(on_event=pprint_events) == (1, 2)


def test_vfs_import_multiturn_live():
    """Verify that a VFS module can be imported across multiple tasks with Live state."""

    state = connect_state(type="live", storage="memory")

    fs = connect_fs(type="virtual")

    agent = Agent(llm=Dummy(), fs=fs, state=state, name="my_agent")

    # Task 1: Create and import a module

    agent.llm.responses = [
        LLMResponse(
            thinking="I will create utils.py",
            files={"utils.py": "VAL = 42"},
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent.task
    def task1():
        """Create utils."""

        pass

    assert task1(on_event=pprint_events) == 42

    # Task 2: Import it again in the same session

    agent.llm.responses = [
        LLMResponse(
            thinking="I will import utils again",
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent.task
    def task2():
        """Import utils."""

        pass

    assert task2(on_event=pprint_events) == 42
