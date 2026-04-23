import pytest

from agex import (
    Agent,
    FileAction,
    clear_agent_registry,
    connect_fs,
    connect_state,
    pprint_events,
)
from agex.llm import Dummy
from tests.agex._emissions import make_response


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
        make_response(
            thinking="I will create utils.py",
            file_actions=[FileAction(path="utils.py", content="VAL = 42")],
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
        make_response(
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
        make_response(
            thinking="I will create a package",
            file_actions=[
                FileAction(path="pkg/__init__.py", content="X = 1"),
                FileAction(path="pkg/mod.py", content="Y = 2"),
            ],
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
        make_response(
            thinking="Import again",
            code="import pkg.mod\ntask_success((pkg.X, pkg.mod.Y))",
        )
    ]

    @agent2.task
    def task2():
        """Import package."""
        pass

    assert task2(on_event=pprint_events) == (1, 2)


def test_vfs_from_parent_import_submodule_with_function(tmp_path):
    """Test `from parent import submodule` where submodule has a function.

    This tests the specific case where:
    1. A package has a submodule with a function
    2. Turn 1: Creates the files and imports via `import pkg.sub`
    3. Turn 2: Imports via `from pkg import sub` and calls the function

    The function can't be pickled, so an UnpicklableMarker is stored.
    On Turn 2, the submodule must be reloaded fresh (not reuse stale reference).
    """
    state_path = str(tmp_path / "state")
    state_config = connect_state(type="versioned", storage="disk", path=state_path)
    fs = connect_fs(type="virtual")

    agent1 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    # Task 1: Create package with submodule containing a function, import it
    agent1.llm.responses = [
        make_response(
            thinking="Creating package structure",
            file_actions=[
                FileAction(path="app/__init__.py", content=""),
                FileAction(path="app/logic/__init__.py", content=""),
                FileAction(
                    path="app/logic/providers.py",
                    content="def get_data():\n    return 'hello from providers'",
                ),
            ],
            code="import app.logic.providers\ntask_success(app.logic.providers.get_data())",
        )
    ]

    @agent1.task
    def task1():
        """Create package."""
        pass

    assert task1(on_event=pprint_events) == "hello from providers"

    # Task 2: Fresh agent, use `from app.logic import providers` syntax
    clear_agent_registry()
    agent2 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    agent2.llm.responses = [
        make_response(
            thinking="Import using from...import syntax",
            code="from app.logic import providers\ntask_success(providers.get_data())",
        )
    ]

    @agent2.task
    def task2():
        """Import using from...import."""
        pass

    # This was failing with: "Variable 'get_data' is not available"
    assert task2(on_event=pprint_events) == "hello from providers"


def test_vfs_both_import_forms_equivalent(tmp_path):
    """Test that `import a.b.c` and `from a.b import c` are equivalent after state restore."""
    state_path = str(tmp_path / "state")
    state_config = connect_state(type="versioned", storage="disk", path=state_path)
    fs = connect_fs(type="virtual")

    agent1 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    # Task 1: Create the module structure
    agent1.llm.responses = [
        make_response(
            thinking="Creating module",
            file_actions=[
                FileAction(path="myapp/__init__.py", content=""),
                FileAction(
                    path="myapp/utils.py",
                    content="def helper():\n    return 42",
                ),
            ],
            code="import myapp.utils\ntask_success(myapp.utils.helper())",
        )
    ]

    @agent1.task
    def task1():
        """Create module."""
        pass

    assert task1(on_event=pprint_events) == 42

    # Task 2: Test `import X.Y as Z` form
    clear_agent_registry()
    agent2 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    agent2.llm.responses = [
        make_response(
            thinking="Import with alias",
            code="import myapp.utils as utils\ntask_success(utils.helper())",
        )
    ]

    @agent2.task
    def task2():
        """Import with alias."""
        pass

    assert task2(on_event=pprint_events) == 42

    # Task 3: Test `from X import Y` form
    clear_agent_registry()
    agent3 = Agent(llm=Dummy(), fs=fs, state=state_config, name="my_agent")

    agent3.llm.responses = [
        make_response(
            thinking="Import with from...import",
            code="from myapp import utils\ntask_success(utils.helper())",
        )
    ]

    @agent3.task
    def task3():
        """Import with from...import."""
        pass

    assert task3(on_event=pprint_events) == 42


def test_vfs_import_multiturn_live():
    """Verify that a VFS module can be imported across multiple tasks with Live state."""

    state = connect_state(type="live", storage="memory")

    fs = connect_fs(type="virtual")

    agent = Agent(llm=Dummy(), fs=fs, state=state, name="my_agent")

    # Task 1: Create and import a module

    agent.llm.responses = [
        make_response(
            thinking="I will create utils.py",
            file_actions=[FileAction(path="utils.py", content="VAL = 42")],
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
        make_response(
            thinking="I will import utils again",
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent.task
    def task2():
        """Import utils."""

        pass

    assert task2(on_event=pprint_events) == 42
