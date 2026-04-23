import pytest

from agex import Agent, clear_agent_registry, connect_fs, connect_state
from agex.agent.emissions import FileWriteEmission
from agex.llm import Dummy
from tests.agex._emissions import make_response


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_workspace_recap_in_forefront():
    """Verify that the forefront message includes the workspace inventory."""
    state = connect_state(type="versioned", storage="memory")
    fs = connect_fs(type="virtual")

    agent = Agent(llm=Dummy(), fs=fs, state=state, name="recap_agent")

    # Task 1: Create a module. We need 2 iterations to see the forefront message in the second one.
    agent.llm.responses = [
        make_response(
            thinking="Iteration 1: Create utils.py",
            file_actions=[
                FileWriteEmission(
                    path="utils.py",
                    content="""
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

class Processor:
    \"\"\"Process data.\"\"\"
    def process(self, data: list):
        pass
""",
                )
            ],
            code="pass",  # Continue to iteration 2
        ),
        make_response(thinking="Iteration 2: Check context", code="task_success(True)"),
    ]

    @agent.task
    def task():
        """Do something."""
        pass

    task()

    # Verify what was sent to the LLM in iteration 2
    # index 0: complete() for Iteration 1
    # index 1: complete() for Iteration 2

    events = agent.llm.all_events[1]

    # The transient message is the last one in the list
    # Forefront message is injected as a transient event (SystemNoteEvent)
    forefront = events[-1].message

    assert "## Workspace Module Inventory" in forefront
    assert "utils.py" in forefront
    assert "def add(a: int, b: int) -> int:" in forefront
    assert '"""Add two numbers."""' in forefront
    assert "class Processor:" in forefront
    assert '"""Process data."""' in forefront
    assert "def process(self, data: list):" in forefront


if __name__ == "__main__":
    pytest.main([__file__])
