"""
End-to-end tests for terminal integration.

Tests the full flow from LLM response through execution to event generation.
"""

import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.datatypes import FileAction
from agex.agent.events import ActionEvent, OutputEvent
from agex.llm import Dummy
from tests.agex._emissions import (
    event_code,
    event_terminal,
    event_thinking,
    make_response,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear agent registry before each test."""
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestTerminalBasicExecution:
    """Tests for basic terminal command execution."""

    def test_terminal_ls_command(self):
        """Test basic ls command execution."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Let me list the files.",
                    terminal="ls",
                ),
                make_response(
                    thinking="Done exploring.",
                    code="task_success('explored')",
                ),
            ]
        )
        agent = Agent(name="explorer", llm=llm, fs=fs, state=state)

        # Pre-populate VFS with some files
        vfs = agent.fs()
        vfs.write("/workspace/file1.txt", b"content1")
        vfs.write("/workspace/file2.py", b"content2")

        @agent.task
        def explore() -> str:
            """Explore the filesystem."""
            pass

        result = explore()
        assert result == "explored"

    def test_terminal_cat_command(self):
        """Test cat command to read file contents."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Let me read the file.",
                    terminal="cat /workspace/readme.txt",
                ),
                make_response(
                    thinking="Got the content.",
                    code="task_success('read')",
                ),
            ]
        )
        agent = Agent(name="reader", llm=llm, fs=fs, state=state)

        # Create a file to read
        vfs = agent.fs()
        vfs.write("/workspace/readme.txt", b"Hello, World!")

        @agent.task
        def read_file() -> str:
            """Read a file."""
            pass

        events = []
        result = read_file(on_event=events.append)
        assert result == "read"

        # Verify OutputEvent contains the file content
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert "Hello, World!" in output_text

    def test_terminal_grep_command(self):
        """Test grep command for pattern matching."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Let me search for the pattern.",
                    terminal="grep -r 'TODO' /workspace",
                ),
                make_response(
                    thinking="Found the results.",
                    code="task_success('searched')",
                ),
            ]
        )
        agent = Agent(name="searcher", llm=llm, fs=fs, state=state)

        # Create files with content
        vfs = agent.fs()
        vfs.write("/workspace/main.py", b"# TODO: implement this\nprint('hello')")
        vfs.write("/workspace/lib.py", b"def helper():\n    pass")

        @agent.task
        def search_todos() -> str:
            """Search for TODOs."""
            pass

        events = []
        result = search_todos(on_event=events.append)
        assert result == "searched"

        # Verify grep output contains the match
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert "TODO" in output_text

    def test_terminal_find_command(self):
        """Test find command to locate files."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Let me find Python files.",
                    terminal="find /workspace -name '*.py'",
                ),
                make_response(
                    thinking="Found them.",
                    code="task_success('found')",
                ),
            ]
        )
        agent = Agent(name="finder", llm=llm, fs=fs, state=state)

        # Create files
        vfs = agent.fs()
        vfs.write("/workspace/main.py", b"main")
        vfs.write("/workspace/lib/utils.py", b"utils")
        vfs.write("/workspace/readme.md", b"readme")

        @agent.task
        def find_python() -> str:
            """Find Python files."""
            pass

        events = []
        result = find_python(on_event=events.append)
        assert result == "found"

        # Verify find output
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert ".py" in output_text


class TestTerminalWithPipes:
    """Tests for terminal commands with pipes."""

    def test_terminal_pipe_grep_wc(self):
        """Test piped commands: grep | wc."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Count matching lines.",
                    terminal="grep 'def ' /workspace/code.py | wc -l",
                ),
                make_response(
                    thinking="Got the count.",
                    code="task_success('counted')",
                ),
            ]
        )
        agent = Agent(name="counter", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write(
            "/workspace/code.py",
            b"def foo():\n    pass\n\ndef bar():\n    pass\n\ndef baz():\n    pass\n",
        )

        @agent.task
        def count_functions() -> str:
            """Count function definitions."""
            pass

        events = []
        result = count_functions(on_event=events.append)
        assert result == "counted"

        # Verify wc output shows 3 lines
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert "3" in output_text

    def test_terminal_pipe_cat_sort_uniq(self):
        """Test piped commands: cat | sort | uniq."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Get unique sorted lines.",
                    terminal="cat /workspace/data.txt | sort | uniq",
                ),
                make_response(
                    thinking="Got unique lines.",
                    code="task_success('done')",
                ),
            ]
        )
        agent = Agent(name="deduper", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/data.txt", b"banana\napple\nbanana\ncherry\napple\n")

        @agent.task
        def dedupe() -> str:
            """Get unique sorted lines."""
            pass

        events = []
        result = dedupe(on_event=events.append)
        assert result == "done"

        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        # Should have apple, banana, cherry in sorted order
        assert "apple" in output_text
        assert "banana" in output_text
        assert "cherry" in output_text


class TestTerminalWithFileActions:
    """Tests for terminal commands combined with file actions."""

    def test_terminal_with_file_write(self):
        """Test terminal command after file write."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Write a file then list it.",
                    file_actions=[
                        FileAction(path="/workspace/new.txt", content="new content")
                    ],
                    terminal="ls /workspace",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('wrote and listed')",
                ),
            ]
        )
        agent = Agent(name="writer", llm=llm, fs=fs, state=state)

        @agent.task
        def write_and_list() -> str:
            """Write a file and list directory."""
            pass

        events = []
        result = write_and_list(on_event=events.append)
        assert result == "wrote and listed"

        # Verify the file was created and appears in ls output
        vfs = agent.fs()
        assert vfs.exists("/workspace/new.txt")

        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert "new.txt" in output_text


class TestTerminalMultiTurn:
    """Tests for multi-turn terminal interactions."""

    def test_terminal_continues_loop(self):
        """Test that terminal implicitly continues the loop."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="First, list files.",
                    terminal="ls /workspace",
                ),
                make_response(
                    thinking="Now read one.",
                    terminal="cat /workspace/file.txt",
                ),
                make_response(
                    thinking="Now finish.",
                    code="task_success('multi-turn complete')",
                ),
            ]
        )
        agent = Agent(name="multi", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/file.txt", b"file content")

        @agent.task
        def multi_turn() -> str:
            """Do multiple terminal operations."""
            pass

        events = []
        result = multi_turn(on_event=events.append)
        assert result == "multi-turn complete"

        # Verify we had 3 action events (2 terminal, 1 python)
        action_events = [e for e in events if isinstance(e, ActionEvent)]
        assert len(action_events) == 3

        # First two should have terminal, last one should have code
        assert event_terminal(action_events[0]) == "ls /workspace"
        assert event_code(action_events[0]) is None
        assert event_terminal(action_events[1]) == "cat /workspace/file.txt"
        assert event_code(action_events[1]) is None
        assert event_terminal(action_events[2]) is None
        assert event_code(action_events[2]) == "task_success('multi-turn complete')"

    def test_terminal_then_python(self):
        """Test terminal exploration followed by Python action."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Explore the structure.",
                    terminal="find /workspace -type f",
                ),
                make_response(
                    thinking="Now process with Python.",
                    code="result = 'processed'\ntask_success(result)",
                ),
            ]
        )
        agent = Agent(name="hybrid", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/a.txt", b"a")
        vfs.write("/workspace/b.txt", b"b")

        @agent.task
        def hybrid_approach() -> str:
            """Use terminal then Python."""
            pass

        result = hybrid_approach()
        assert result == "processed"


class TestTerminalErrorHandling:
    """Tests for terminal error handling."""

    def test_terminal_file_not_found(self):
        """Test error when file doesn't exist."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Try to read nonexistent file.",
                    terminal="cat /workspace/nonexistent.txt",
                ),
                make_response(
                    thinking="Handle the error.",
                    code="task_success('handled error')",
                ),
            ]
        )
        agent = Agent(name="error_handler", llm=llm, fs=fs, state=state)

        @agent.task
        def handle_missing() -> str:
            """Handle missing file."""
            pass

        events = []
        result = handle_missing(on_event=events.append)
        assert result == "handled error"

        # Verify error was reported in output
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        # Should have an error output
        assert any("error" in str(e.parts).lower() for e in output_events)

    def test_terminal_parse_error_recovery(self):
        """Test recovery from terminal parse error."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Try invalid syntax.",
                    terminal="ls | | grep",  # Invalid pipe syntax
                ),
                make_response(
                    thinking="Fix and retry.",
                    terminal="ls",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('recovered')",
                ),
            ]
        )
        agent = Agent(name="recoverer", llm=llm, fs=fs, state=state)

        @agent.task
        def recover_from_error() -> str:
            """Recover from parse error."""
            pass

        events = []
        result = recover_from_error(on_event=events.append)
        assert result == "recovered"

        # Verify we got an error message (could be parse error or terminal error)
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        # The parser might succeed but the command execution will fail
        # Check for either 'parse' or 'error' in the output
        assert any(
            "error" in str(e.parts).lower() or "💥" in str(e.parts)
            for e in output_events
        )


class TestTerminalActionEvent:
    """Tests for ActionEvent terminal field."""

    def test_action_event_has_terminal(self):
        """Test that ActionEvent correctly captures terminal."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Run a terminal command.",
                    terminal="echo 'hello'",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('done')",
                ),
            ]
        )
        agent = Agent(name="action_test", llm=llm, fs=fs, state=state)

        @agent.task
        def test_action() -> str:
            """Test action events."""
            pass

        events = []
        test_action(on_event=events.append)

        action_events = [e for e in events if isinstance(e, ActionEvent)]
        assert len(action_events) == 2

        # First action should have terminal
        first_action = action_events[0]
        assert event_terminal(first_action) == "echo 'hello'"
        assert event_code(first_action) is None
        assert event_thinking(first_action) == "Run a terminal command."

        # Second action should have code
        second_action = action_events[1]
        assert event_terminal(second_action) is None
        assert event_code(second_action) == "task_success('done')"

    def test_action_event_str_repr(self):
        """Test ActionEvent string representation with terminal."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Multi-line terminal.",
                    terminal="ls -la\ngrep pattern\nwc -l",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('done')",
                ),
            ]
        )
        agent = Agent(name="repr_test", llm=llm, fs=fs, state=state)

        @agent.task
        def test_repr() -> str:
            """Test repr."""
            pass

        events = []
        test_repr(on_event=events.append)

        action_events = [e for e in events if isinstance(e, ActionEvent)]
        first_action = action_events[0]

        # String repr should mention that this turn included a Terminal
        # emission.
        str_repr = str(first_action)
        assert "Terminal" in str_repr


class TestTerminalAsync:
    """Tests for async terminal execution."""

    @pytest.mark.asyncio
    async def test_async_terminal_execution(self):
        """Test terminal execution in async context."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Async terminal.",
                    terminal="ls /workspace",
                ),
                make_response(
                    thinking="Done async.",
                    code="task_success('async done')",
                ),
            ]
        )
        agent = Agent(name="async_terminal", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/async_file.txt", b"async content")

        @agent.task
        async def async_explore() -> str:
            """Async exploration."""
            pass

        result = await async_explore()
        assert result == "async done"

    @pytest.mark.asyncio
    async def test_async_terminal_multi_turn(self):
        """Test multi-turn terminal in async context."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="First async terminal.",
                    terminal="echo 'step1'",
                ),
                make_response(
                    thinking="Second async terminal.",
                    terminal="echo 'step2'",
                ),
                make_response(
                    thinking="Finish.",
                    code="task_success('async multi done')",
                ),
            ]
        )
        agent = Agent(name="async_multi", llm=llm, fs=fs, state=state)

        @agent.task
        async def async_multi() -> str:
            """Async multi-turn."""
            pass

        events = []
        result = await async_multi(on_event=events.append)
        assert result == "async multi done"

        action_events = [e for e in events if isinstance(e, ActionEvent)]
        assert len(action_events) == 3
        assert event_terminal(action_events[0]) == "echo 'step1'"
        assert event_terminal(action_events[1]) == "echo 'step2'"


class TestTerminalJQ:
    """Tests for jq command in terminal."""

    def test_terminal_jq_query(self):
        """Test jq command for JSON processing."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Parse JSON with jq.",
                    terminal="cat /workspace/data.json | jq '.name'",
                ),
                make_response(
                    thinking="Got the name.",
                    code="task_success('jq done')",
                ),
            ]
        )
        agent = Agent(name="jq_user", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/data.json", b'{"name": "Alice", "age": 30}')

        @agent.task
        def parse_json() -> str:
            """Parse JSON."""
            pass

        events = []
        result = parse_json(on_event=events.append)
        assert result == "jq done"

        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) >= 1
        output_text = str(output_events[0].parts[0])
        assert "Alice" in output_text


class TestTerminalFilesystemOperations:
    """Tests for filesystem-modifying terminal commands."""

    def test_terminal_mkdir_touch(self):
        """Test mkdir and touch commands."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Create directory structure.",
                    terminal="mkdir /workspace/newdir",
                ),
                make_response(
                    thinking="Create a file.",
                    terminal="touch /workspace/newdir/file.txt",
                ),
                make_response(
                    thinking="Verify.",
                    terminal="ls /workspace/newdir",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('created')",
                ),
            ]
        )
        agent = Agent(name="creator", llm=llm, fs=fs, state=state)

        @agent.task
        def create_structure() -> str:
            """Create directory structure."""
            pass

        result = create_structure()
        assert result == "created"

        # Verify the structure was created
        vfs = agent.fs()
        assert vfs.exists("/workspace/newdir")
        assert vfs.exists("/workspace/newdir/file.txt")

    def test_terminal_cp_mv_rm(self):
        """Test cp, mv, and rm commands."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Copy file.",
                    terminal="cp /workspace/original.txt /workspace/copy.txt",
                ),
                make_response(
                    thinking="Move file.",
                    terminal="mv /workspace/copy.txt /workspace/moved.txt",
                ),
                make_response(
                    thinking="Remove original.",
                    terminal="rm /workspace/original.txt",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('modified')",
                ),
            ]
        )
        agent = Agent(name="modifier", llm=llm, fs=fs, state=state)

        vfs = agent.fs()
        vfs.write("/workspace/original.txt", b"original content")

        @agent.task
        def modify_files() -> str:
            """Modify files."""
            pass

        result = modify_files()
        assert result == "modified"

        # Verify the operations
        vfs = agent.fs()
        assert not vfs.exists("/workspace/original.txt")
        assert not vfs.exists("/workspace/copy.txt")
        assert vfs.exists("/workspace/moved.txt")
        assert vfs.read("/workspace/moved.txt") == b"original content"


class TestTerminalTokenStreaming:
    """Tests for terminal token streaming."""

    def test_pprint_tokens_handles_terminal(self):
        """Test that pprint_tokens correctly handles terminal tokens."""
        import io

        from agex.agent.console import pprint_tokens
        from agex.llm.core import StreamToken

        output = io.StringIO()

        # Simulate terminal token (using StreamToken which pprint_tokens expects)
        token = StreamToken(
            type="terminal",
            content="ls -la",
            done=False,
            agent_name="test",
            start=False,
        )
        pprint_tokens(token, color="never", stream=output)

        result = output.getvalue()
        assert "ls -la" in result

    def test_terminal_tokens_emitted_during_streaming(self):
        """Test that terminal tokens are emitted during streaming."""
        from agex.llm.formats.xml import tokenize_xml_stream

        xml = "<THINKING>Exploring</THINKING><TERMINAL>find . -name '*.py'</TERMINAL>"
        chunks = list(tokenize_xml_stream([xml]))

        # Should have thinking and terminal tokens
        token_types = [c.type for c in chunks]
        assert "thinking" in token_types
        assert "terminal" in token_types

        # Find the terminal token
        terminal_tokens = [c for c in chunks if c.type == "terminal" and c.content]
        assert len(terminal_tokens) == 1
        assert "find" in terminal_tokens[0].content


class TestTerminalStreaming:
    """Tests for terminal event streaming."""

    def test_terminal_events_stream_correctly(self):
        """Test that terminal events stream in correct order."""
        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Terminal action.",
                    terminal="echo 'streamed'",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('stream test')",
                ),
            ]
        )
        agent = Agent(name="streamer", llm=llm, fs=fs, state=state)

        @agent.task
        def stream_test() -> str:
            """Test streaming."""
            pass

        events = []
        stream_test(on_event=events.append)

        # Verify event order
        event_types = [type(e).__name__ for e in events]

        # Should have: TaskStart, Action (terminal), Output, Action (code), Success
        assert "TaskStartEvent" in event_types
        assert "ActionEvent" in event_types
        assert "OutputEvent" in event_types
        assert "SuccessEvent" in event_types

        # TaskStart should be first
        assert event_types[0] == "TaskStartEvent"

        # Success should be last
        assert event_types[-1] == "SuccessEvent"


class TestTerminalFileEvents:
    """Tests for FileEvent emission during terminal execution."""

    def test_multiple_file_writes_emit_single_agent_file_event(self):
        """Test that multiple file writes emit one FileEvent with file_source='agent'."""
        from agex.agent.events import FileEvent

        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Create multiple files.",
                    terminal=(
                        "echo 'file1' > /workspace/file1.txt\n"
                        "echo 'file2' > /workspace/file2.txt\n"
                        "echo 'file3' > /workspace/file3.txt"
                    ),
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('created')",
                ),
            ]
        )
        agent = Agent(name="multi_writer", llm=llm, fs=fs, state=state)

        @agent.task
        def create_files() -> str:
            """Create multiple files."""
            pass

        events = []
        result = create_files(on_event=events.append)
        assert result == "created"

        # Get all FileEvents
        file_events = [e for e in events if isinstance(e, FileEvent)]

        # Should have exactly one FileEvent (aggregated at task end)
        assert len(file_events) == 1

        # Should be from agent, not user
        assert file_events[0].file_source == "agent"

        # Should contain all created files (VFS paths may or may not have leading slash)
        added_files = set(file_events[0].added)
        assert any("file1.txt" in f for f in added_files)
        assert any("file2.txt" in f for f in added_files)
        assert any("file3.txt" in f for f in added_files)

    def test_terminal_file_modifications_in_same_task(self):
        """Test that creating and modifying a file in the same task is tracked."""
        from agex.agent.events import FileEvent

        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Create a file then modify it.",
                    terminal=(
                        "echo 'original' > /workspace/myfile.txt\n"
                        "echo 'modified' >> /workspace/myfile.txt"
                    ),
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('modified')",
                ),
            ]
        )
        agent = Agent(name="modifier", llm=llm, fs=fs, state=state)

        @agent.task
        def create_and_modify() -> str:
            """Create and modify files."""
            pass

        events = []
        result = create_and_modify(on_event=events.append)
        assert result == "modified"

        # Get FileEvents from agent
        file_events = [
            e for e in events if isinstance(e, FileEvent) and e.file_source == "agent"
        ]

        # Should have exactly one FileEvent
        assert len(file_events) == 1
        event = file_events[0]

        # The file should appear in either added or modified
        # (since it was created and modified in the same task, it shows as added)
        all_changed = event.added + event.modified
        assert any("myfile.txt" in f for f in all_changed)

    def test_no_file_event_when_no_files_changed(self):
        """Test that no FileEvent is emitted when no files are changed."""
        from agex.agent.events import FileEvent

        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Just read files, don't modify.",
                    terminal="ls /workspace",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('read only')",
                ),
            ]
        )
        agent = Agent(name="reader", llm=llm, fs=fs, state=state)

        # Pre-create a file
        vfs = agent.fs()
        vfs.write("/workspace/data.txt", b"some data")

        @agent.task
        def read_only() -> str:
            """Read-only operation."""
            pass

        events = []
        result = read_only(on_event=events.append)
        assert result == "read only"

        # Should have no agent FileEvents (only user event from pre-creation)
        agent_file_events = [
            e for e in events if isinstance(e, FileEvent) and e.file_source == "agent"
        ]
        assert len(agent_file_events) == 0

    @pytest.mark.asyncio
    async def test_async_terminal_emits_single_file_event(self):
        """Test that async terminal also emits single aggregated FileEvent."""
        from agex.agent.events import FileEvent

        fs = connect_fs(type="virtual")
        state = connect_state(type="versioned", storage="memory")
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Create files async.",
                    terminal=(
                        "echo 'async1' > /workspace/async1.txt\n"
                        "echo 'async2' > /workspace/async2.txt"
                    ),
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('async created')",
                ),
            ]
        )
        agent = Agent(name="async_writer", llm=llm, fs=fs, state=state)

        @agent.task
        async def async_create_files() -> str:
            """Create files asynchronously."""
            pass

        events = []
        result = await async_create_files(on_event=events.append)
        assert result == "async created"

        # Get FileEvents
        file_events = [e for e in events if isinstance(e, FileEvent)]

        # Should have exactly one FileEvent
        assert len(file_events) == 1
        assert file_events[0].file_source == "agent"
        # VFS paths may or may not have leading slash
        assert any("async1.txt" in f for f in file_events[0].added)
        assert any("async2.txt" in f for f in file_events[0].added)
