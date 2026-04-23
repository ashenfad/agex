import pytest

from agex import (
    Agent,
    clear_agent_registry,
    connect_fs,
    connect_state,
    pprint_events,
)
from agex.agent.emissions import FileWriteEmission
from agex.llm import Dummy
from tests.agex._emissions import make_response


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_vfs_module_reloading():
    """Verify that updating a VFS file causes the module to be reloaded on next import."""
    state = connect_state(type="versioned", storage="memory")
    fs = connect_fs(type="virtual")

    agent = Agent(llm=Dummy(), fs=fs, state=state)

    # Task 1: Create and import utils v1
    agent.llm.responses = [
        make_response(
            thinking="Create utils v1",
            file_actions=[FileWriteEmission(path="utils.py", content="VAL = 1")],
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent.task
    def task1():
        """Task 1."""
        pass

    assert task1() == 1

    # Task 2: Update utils to v2 and import again
    agent.llm.responses = [
        make_response(
            thinking="Update utils to v2",
            file_actions=[FileWriteEmission(path="utils.py", content="VAL = 2")],
            code="import utils\ntask_success(utils.VAL)",
        )
    ]

    @agent.task
    def task2():
        """Task 2."""
        pass

    # Should be 2 if reloading works
    assert task2(on_event=pprint_events) == 2


if __name__ == "__main__":
    pytest.main([__file__])
