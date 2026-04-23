"""Integration tests for agent-side FileEvent emission."""

from agex import Agent
from agex.agent.events import FileEvent
from agex.fs import connect_fs
from agex.llm import Dummy
from agex.state import connect_state
from agex.state.log import get_events_from_log
from tests.agex._emissions import make_response


class TestAgentFileEvents:
    """Test FileEvent emission during agent task execution."""

    def test_agent_write_emits_file_event(self):
        """Test that agent writing a file emits FileEvent with source='agent'."""
        llm = Dummy(
            responses=[
                make_response(
                    thinking="I'll create a file",
                    code="""with open('output.txt', 'w') as f:
    f.write('Hello from agent!')
task_success('done')""",
                )
            ]
        )

        agent = Agent(
            llm=llm,
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        @agent.task
        def write_file() -> str:
            """Write a file."""
            pass

        result = write_file()
        assert result == "done"

        # Check that FileEvent was emitted
        events = get_events_from_log(agent.state())
        file_events = [e for e in events if isinstance(e, FileEvent)]

        assert len(file_events) == 1
        assert file_events[0].file_source == "agent"
        assert file_events[0].added == ["output.txt"]
        assert file_events[0].modified == []
        assert file_events[0].removed == []

    def test_agent_modify_emits_file_event(self):
        """Test that agent modifying a file emits FileEvent with modified."""
        agent = Agent(
            llm=Dummy(
                responses=[
                    make_response(
                        thinking="Update file",
                        code="""with open('data.txt', 'w') as f:
    f.write('updated')
task_success('done')""",
                    )
                ]
            ),
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        # Create file first
        agent.fs().write("data.txt", b"original")

        @agent.task
        def update_file() -> str:
            """Update a file."""

        update_file()

        # Check for FileEvent - should have 2: one from user, one from agent
        events = get_events_from_log(agent.state())
        file_events = [e for e in events if isinstance(e, FileEvent)]

        assert len(file_events) == 2
        assert file_events[0].file_source == "user"  # Initial write
        assert file_events[0].added == ["data.txt"]

        assert file_events[1].file_source == "agent"  # Agent modification
        assert file_events[1].added == []
        assert file_events[1].modified == ["data.txt"]

    def test_agent_multiple_file_operations_single_event(self):
        """Test that multiple file ops in one turn = 1 FileEvent."""
        agent = Agent(
            llm=Dummy(
                responses=[
                    make_response(
                        thinking="Create files",
                        code="""with open('file1.txt', 'w') as f:
    f.write('one')
with open('file2.txt', 'w') as f:
    f.write('two')
with open('file3.txt', 'w') as f:
    f.write('three')
task_success('done')""",
                    )
                ]
            ),
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )
        agent.fn(open)  # Register open for VFS patching

        @agent.task
        def create_files() -> str:
            """Create multiple files."""

        create_files()

        events = get_events_from_log(agent.state())
        file_events = [e for e in events if isinstance(e, FileEvent)]

        # Should be just 1 batched FileEvent
        assert len(file_events) == 1
        assert file_events[0].file_source == "agent"
        assert set(file_events[0].added) == {"file1.txt", "file2.txt", "file3.txt"}

    def test_no_file_event_when_no_changes(self):
        """Test that no FileEvent is emitted when agent doesn't touch files."""
        agent = Agent(
            llm=Dummy(
                responses=[
                    make_response(
                        thinking="Just compute",
                        code="""result = 1 + 1
task_success(result)""",
                    )
                ]
            ),
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )
        agent.fn(open)  # Register open for VFS patching

        @agent.task
        def compute() -> int:
            """Do computation."""

        result = compute()
        assert result == 2

        events = get_events_from_log(agent.state())
        file_events = [e for e in events if isinstance(e, FileEvent)]

        # No file operations = no FileEvent
        assert len(file_events) == 0
