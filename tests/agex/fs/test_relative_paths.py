"""Test relative path access in VFS."""

from agex import Agent, connect_fs, connect_state
from agex.helpers import register_stdlib
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


def test_relative_path_with_os_exists_and_open():
    """Test exact agent code pattern: os.path.exists + open with relative path."""
    llm = Dummy(
        [
            LLMResponse(
                thinking="I'll check and read the file.",
                code="""import os
dom_path = 'debug/dom.html'
if os.path.exists(dom_path):
    with open(dom_path, 'r') as f:
        dom_content = f.read()
    task_success(f'Success: {dom_content}')
else:
    task_fail(f'File not found at {dom_path}')""",
            )
        ]
    )

    agent = Agent(
        name="test",
        fs=connect_fs(type="virtual"),
        state=connect_state(type="ephemeral"),
        llm=llm,
    )

    register_stdlib(agent)

    # Write file
    fs = agent.fs("session")
    fs.write("debug/dom.html", b"<html>test</html>")

    # Verify it exists
    assert fs.exists("debug/dom.html")
    print(f"File exists in VFS: {fs.list(recursive=True)}")

    @agent.task
    def read_file() -> str:
        """Read debug/dom.html"""
        pass

    result = read_file(session="session")
    assert "<html>test</html>" in result
    print(f"Result: {result}")
