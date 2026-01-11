"""Integration tests for VirtualFS with agex agents."""

import shutil

import pytest

from agex import Agent, clear_agent_registry, connect_fs, connect_state, pprint_events
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


@pytest.fixture
def anyio_backend():
    return "asyncio"


class TestAgentVFSIntegration:
    """Test VirtualFS integration with Agent class."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_agent_without_fs(self):
        """Test that agent without fs raises error when accessing fs()."""
        agent = Agent(llm=Dummy())

        with pytest.raises(ValueError, match="not configured with filesystem"):
            agent.fs()

    def test_agent_with_fs_accessor(self):
        """Test agent.fs() accessor returns working VFS."""
        agent = Agent(
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=Dummy(),
        )

        fs = agent.fs()

        # Write a file
        fs.write("test.txt", b"Hello from fs()!")

        # Read it back
        assert fs.read("test.txt") == b"Hello from fs()!"

    def test_agent_task_can_access_uploaded_files(self):
        """Test that agent task can access files uploaded via agent.fs()."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="I'll read the CSV file.",
                    code="""with open("data.csv", "r") as f:
    content = f.read()
task_success(content)""",
                )
            ]
        )

        agent = Agent(
            primer="You help with file operations.",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )

        # Upload a file via external API
        fs = agent.fs()
        fs.write("data.csv", b"a,b,c\n1,2,3")

        @agent.task
        def read_csv() -> str:
            """Read the CSV file and return its content."""
            pass

        # Execute task
        result = read_csv()

        # Agent should have read the file
        assert "a,b,c" in result
        assert "1,2,3" in result

    def test_agent_can_write_files(self):
        """Test that agent can write files that persist in VFS."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="I'll create the file.",
                    code="""with open("output.txt", "w") as f:
    f.write("Hello from agent!")
task_success("file created")""",
                )
            ]
        )

        agent = Agent(
            primer="You write files.",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )
        agent.fn(open)  # Register open for file operations

        @agent.task
        def create_file() -> str:
            """Create a file called output.txt with some content."""
            pass

        # Execute task
        create_file()

        # Check that file was created
        fs = agent.fs()
        assert fs.exists("output.txt") is True
        content = fs.read("output.txt")
        assert content == b"Hello from agent!"

    def test_vfs_persists_across_task_calls(self):
        """Test that VFS state persists across multiple task calls."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="I'll append to the log.",
                    code="""with open("log.txt", "a") as f:
    f.write("First message\\n")
task_success("appended")""",
                ),
                LLMResponse(
                    thinking="I'll append another message.",
                    code="""with open("log.txt", "a") as f:
    f.write("Second message\\n")
task_success("appended")""",
                ),
            ]
        )

        agent = Agent(
            primer="You manage files.",
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )
        agent.fn(open)  # Register open for file operations

        @agent.task
        def write_log(message: str) -> str:
            """Append a message to log.txt."""
            pass

        # Task 1: Write first message
        write_log("First message")

        # Task 2: Append second message
        write_log("Second message")

        # Both messages should be in the file
        fs = agent.fs()
        content = fs.read("log.txt").decode()
        assert "First message" in content
        assert "Second message" in content

    def test_vfs_with_versioned_state(self):
        """Test that VFS works correctly with Versioned state."""
        agent = Agent(
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=Dummy(),
        )

        fs = agent.fs()

        # Write initial file
        fs.write("file.txt", b"version 1")

        # Modify file
        fs.write("file.txt", b"version 2")

        # File should have latest version
        assert fs.read("file.txt") == b"version 2"

    def test_session_isolation(self):
        """Test that different sessions have isolated VFS."""
        agent = Agent(
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=Dummy(),
        )

        fs_session1 = agent.fs(session="session1")
        fs_session2 = agent.fs(session="session2")

        # Write to session 1
        fs_session1.write("file.txt", b"session 1 data")

        # Session 2 should not see it
        assert fs_session2.exists("file.txt") is False

        # Write to session 2
        fs_session2.write("file.txt", b"session 2 data")

        # Each session has its own file
        assert fs_session1.read("file.txt") == b"session 1 data"
        assert fs_session2.read("file.txt") == b"session 2 data"


class TestVFSWithRealLibraries:
    """Test VFS works with real Python libraries."""

    def test_pandas_read_csv(self):
        """Test that pandas can read CSV from VFS."""
        pd = pytest.importorskip("pandas")

        llm = Dummy(
            [
                LLMResponse(
                    thinking="I'll read the CSV with pandas.",
                    code="""import pandas as pd
df = pd.read_csv("data.csv")
age_sum = df["age"].sum()
task_success(str(age_sum))""",
                )
            ]
        )

        agent = Agent(
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )
        agent.module(pd)
        agent.fn(open)  # Register open for file operations

        # Upload CSV via fs()
        fs = agent.fs()
        csv_content = b"name,age\nAlice,30\nBob,25"
        fs.write("data.csv", csv_content)

        @agent.task
        def analyze_csv() -> str:
            """Read data.csv with pandas and return the sum of the age column."""
            pass

        result = analyze_csv()

        # Agent should successfully read and process the CSV
        assert "55" in result

    def test_json_module(self):
        """Test that json module works with VFS."""
        import json

        llm = Dummy(
            [
                LLMResponse(
                    thinking="I'll create a JSON file.",
                    code="""import json
data = {"key": "value", "number": 42}
with open("config.json", "w") as f:
    json.dump(data, f)
task_success("created")""",
                )
            ]
        )

        agent = Agent(
            primer="You work with JSON files.",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )
        agent.module(json)
        agent.fn(open)  # Register open for file operations

        @agent.task
        def create_json() -> str:
            """Create a JSON file called config.json with some data."""
            pass

        create_json()

        # Verify JSON file was created
        fs = agent.fs()
        assert fs.exists("config.json") is True

        # Should be valid JSON
        content = fs.read("config.json").decode()
        data = json.loads(content)
        assert isinstance(data, dict)
        assert data["key"] == "value"
        assert data["number"] == 42


class TestRemoval:
    """Test VFS works with real Python libraries."""

    def test_removal_sync(self):
        """Test that pandas can read CSV from VFS."""

        llm = Dummy(
            [
                LLMResponse(
                    thinking="",
                    code="""
import os;
os.remove("data.csv");
task_success("removed")
""",
                )
            ]
        )

        shutil.rmtree("/tmp/agex/fs-test", ignore_errors=True)
        agent = Agent(
            state=connect_state(
                type="versioned", storage="disk", path="/tmp/agex/fs-test"
            ),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )

        # Upload CSV via fs()
        fs = agent.fs()
        csv_content = b"name,age\nAlice,30\nBob,25"
        fs.write("data.csv", csv_content)

        @agent.task
        def do_a_thing() -> str:
            """Do a thing."""
            pass

        fs = agent.fs()
        print("Before:")
        print(fs.list())

        do_a_thing(on_event=pprint_events)

        fs = agent.fs()
        print("After:")
        print(fs.list())
        assert not fs.list()  # should be empty

    @pytest.mark.anyio(backend="asyncio")
    async def test_removal_async(self):
        """Test that pandas can read CSV from VFS."""

        llm = Dummy(
            [
                LLMResponse(
                    thinking="",
                    code="""
import os;
os.remove("data.csv");
task_success("removed")
""",
                )
            ]
        )

        shutil.rmtree("/tmp/agex/fs-test", ignore_errors=True)
        agent = Agent(
            state=connect_state(
                type="versioned", storage="disk", path="/tmp/agex/fs-test"
            ),
            fs=connect_fs(type="virtual"),
            llm=llm,
        )

        # Upload CSV via fs()
        fs = agent.fs()
        csv_content = b"name,age\nAlice,30\nBob,25"
        fs.write("data.csv", csv_content)

        @agent.task
        async def do_a_thing() -> str:
            """Do a thing."""
            pass

        fs = agent.fs()
        print("Before:")
        print(fs.list())

        await do_a_thing(on_event=pprint_events)

        fs = agent.fs()
        print("After:")
        print(fs.list())
        assert not fs.list()  # should be empty
