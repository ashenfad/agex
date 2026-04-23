"""Tests for per-session isolation in isolated filesystem."""

from agex import Agent, connect_fs, connect_state
from agex.llm import Dummy
from tests.agex._emissions import make_response


class TestIsolatedFSPerSession:
    """Test per_session parameter for isolated filesystem."""

    def test_per_session_false_shares_root(self, tmp_path):
        """Multiple sessions share same root when per_session=False."""
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Write to file",
                    code="with open('shared.txt', 'w') as f: f.write('data')\ntask_success('done')",
                )
            ]
        )

        agent = Agent(
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="isolated", root=str(tmp_path), per_session=False),
        )

        @agent.task
        def write_file() -> str:
            """Write file."""
            pass

        # Write in session1
        write_file(session="session1")

        # Check file exists in root (not session1 subdir)
        assert (tmp_path / "shared.txt").exists()
        assert not (tmp_path / "session1").exists()

        # Session2 can see the same file
        fs = agent.fs(session="session2")
        assert fs.exists("shared.txt")

    def test_per_session_true_isolates_sessions(self, tmp_path):
        """Each session gets isolated subdirectory when per_session=True."""
        llm = Dummy(
            responses=[
                make_response(
                    thinking="Write to file",
                    code="with open('data.txt', 'w') as f: f.write('session1')\ntask_success('done')",
                ),
                make_response(
                    thinking="Write to file",
                    code="with open('data.txt', 'w') as f: f.write('session2')\ntask_success('done')",
                ),
            ]
        )

        agent = Agent(
            llm=llm,
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="isolated", root=str(tmp_path), per_session=True),
        )

        @agent.task
        def write_data() -> str:
            """Write data."""
            pass

        # Write in session1
        write_data(session="session1")

        # Write in session2
        write_data(session="session2")

        # Check separate directories created
        assert (tmp_path / "session1" / "data.txt").exists()
        assert (tmp_path / "session2" / "data.txt").exists()

        # Check content is different
        assert (tmp_path / "session1" / "data.txt").read_text() == "session1"
        assert (tmp_path / "session2" / "data.txt").read_text() == "session2"

    def test_agent_fs_with_per_session(self, tmp_path):
        """agent.fs(session) respects per_session parameter."""
        agent = Agent(
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="isolated", root=str(tmp_path), per_session=True),
        )

        # Write via fs accessor for session1
        fs1 = agent.fs(session="session1")
        fs1.write("test.txt", b"from fs1")

        # Write via fs accessor for session2
        fs2 = agent.fs(session="session2")
        fs2.write("test.txt", b"from fs2")

        # Check isolation
        assert (tmp_path / "session1" / "test.txt").read_text() == "from fs1"
        assert (tmp_path / "session2" / "test.txt").read_text() == "from fs2"

        # Check fs accessors are isolated
        assert fs1.read("test.txt") == b"from fs1"
        assert fs2.read("test.txt") == b"from fs2"

    def test_per_session_creates_subdirs_automatically(self, tmp_path):
        """Session subdirectories created automatically on first access."""
        agent = Agent(
            fs=connect_fs(type="isolated", root=str(tmp_path), per_session=True),
        )

        # Access fs for new session
        _fs = agent.fs(session="new_session")

        # Subdir should be created
        assert (tmp_path / "new_session").exists()
        assert (tmp_path / "new_session").is_dir()
