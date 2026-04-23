import pytest

from agex import Agent, clear_agent_registry, connect_fs, connect_state, pprint_events
from agex.llm import Dummy
from tests.agex._emissions import make_response


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def create_agent():
    return Agent(
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )


def test_import_nested_module():
    """Test 'import pkg.mod' where pkg is a directory."""
    agent = create_agent()

    # 1. Setup pkg/mod.py
    agent.fs().write("pkg/mod.py", b"VAL = 42\ndef get_val(): return VAL")

    # 2. Test import pkg.mod
    agent.llm.responses = [
        make_response(
            thinking="import", code="import pkg.mod\ntask_success(pkg.mod.get_val())"
        )
    ]

    @agent.task
    def task():
        """Test task."""
        pass

    assert task(on_event=pprint_events) == 42


def test_import_package_with_init():
    """Test 'import pkg' where pkg has __init__.py."""
    agent = create_agent()

    agent.fs().write("pkg/__init__.py", b"INIT_VAL = 1")
    agent.fs().write("pkg/mod.py", b"MOD_VAL = 2")

    agent.llm.responses = [
        make_response(
            thinking="import",
            code="import pkg.mod\ntask_success((pkg.INIT_VAL, pkg.mod.MOD_VAL))",
        )
    ]

    @agent.task
    def task():
        """Test task."""
        pass

    assert task(on_event=pprint_events) == (1, 2)


def test_from_import_submodule():
    """Test 'from pkg import mod'."""
    agent = create_agent()

    agent.fs().write("pkg/mod.py", b"VAL = 100")

    agent.llm.responses = [
        make_response(
            thinking="import", code="from pkg import mod\ntask_success(mod.VAL)"
        )
    ]

    @agent.task
    def task():
        """Test task."""
        pass

    assert task(on_event=pprint_events) == 100


def test_from_import_function_from_submodule():
    """Test 'from pkg.mod import func'."""
    agent = create_agent()

    agent.fs().write("pkg/mod.py", b"def func(): return 'hello'")

    agent.llm.responses = [
        make_response(
            thinking="import", code="from pkg.mod import func\ntask_success(func())"
        )
    ]

    @agent.task
    def task():
        """Test task."""
        pass

    assert task(on_event=pprint_events) == "hello"


def test_namespace_package():
    """Test importing from a directory without __init__.py."""
    agent = create_agent()

    # No __init__.py in 'ns'
    agent.fs().write("ns/mod.py", b"X = 1")

    agent.llm.responses = [
        make_response(thinking="import", code="import ns.mod\ntask_success(ns.mod.X)")
    ]

    @agent.task
    def task():
        """Test task."""
        pass

    assert task(on_event=pprint_events) == 1


if __name__ == "__main__":
    pytest.main([__file__])
