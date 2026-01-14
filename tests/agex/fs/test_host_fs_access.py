"""
Tests for host_fs_access feature that allows registered functions/classes/modules
to access the host filesystem even when VirtualFS or IsolatedFS is active.
"""

import pytest

from agex import Agent, connect_fs, connect_state
from agex.llm import Dummy, LLMResponse


@pytest.fixture
def host_test_file(tmp_path):
    """Create a temporary file on the host filesystem for testing."""
    test_file = tmp_path / "test_host_file.txt"
    test_file.write_text("host filesystem content")
    return str(test_file)


# ============================================================================
# Function-level registration tests
# ============================================================================


def test_function_with_host_fs_access_can_read_host(host_test_file):
    """Function registered with host_fs_access=True can read from host filesystem."""

    def read_host_file(filepath):
        with open(filepath, "r") as f:
            return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Reading the host file",
                code=f"""content = read_host_file("{host_test_file}")
task_success(content)""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    # Register with host_fs_access=True
    agent.fn(read_host_file, host_fs_access=True)

    @agent.task
    def test_read() -> str:
        """Read the host file."""
        pass

    result = test_read()
    assert result == "host filesystem content"


def test_function_without_host_fs_access_uses_vfs(host_test_file):
    """Function registered without host_fs_access uses VFS, not host filesystem."""

    def read_file(filepath):
        with open(filepath, "r") as f:
            return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Writing and reading from VFS",
                code="""# Write to VFS
with open("/test.txt", "w") as f:
    f.write("VFS content")

# Read from VFS
content = read_file("/test.txt")
task_success(content)""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    # Register WITHOUT host_fs_access - should use VFS
    agent.fn(read_file)

    @agent.task
    def test_read() -> str:
        """Read from VFS."""
        pass

    result = test_read()
    assert result == "VFS content"


# ============================================================================
# Class-level registration tests
# ============================================================================


def test_class_with_host_fs_access_methods_can_read_host(host_test_file):
    """Class registered with host_fs_access=True allows methods to read from host."""

    class HostFileReader:
        def __init__(self, filepath):
            self.filepath = filepath

        def read_content(self):
            with open(self.filepath, "r") as f:
                return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Reading via class method",
                code=f"""reader = HostFileReader("{host_test_file}")
content = reader.read_content()
task_success(content)""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    # Register class with host_fs_access=True
    agent.cls(HostFileReader, host_fs_access=True)

    @agent.task
    def test_read() -> str:
        """Read via class."""
        pass

    result = test_read()
    assert result == "host filesystem content"


def test_class_without_host_fs_access_uses_vfs(host_test_file):
    """Class registered without host_fs_access uses VFS, not host filesystem."""

    class FileManager:
        def write_and_read(self, path, content):
            with open(path, "w") as f:
                f.write(content)
            with open(path, "r") as f:
                return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Using VFS",
                code="""fm = FileManager()
result = fm.write_and_read("/vfs.txt", "VFS content")
task_success(result)""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    # Register WITHOUT host_fs_access
    agent.cls(FileManager)

    @agent.task
    def test_vfs() -> str:
        """Use VFS."""
        pass

    result = test_vfs()
    assert result == "VFS content"


# ============================================================================
# Agent code isolation test
# ============================================================================


def test_agent_code_uses_vfs_not_host(host_test_file):
    """Agent code uses VFS, not host filesystem."""

    def read_host_file(filepath):
        with open(filepath, "r") as f:
            return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Writing to and reading from VFS",
                code="""# Agent writes to VFS
with open("/agent_file.txt", "w") as f:
    f.write("Agent VFS content")

# Agent reads from VFS
with open("/agent_file.txt", "r") as f:
    content = f.read()

task_success(content)""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    # Register a function with host_fs_access, but agent code should still use VFS
    agent.fn(read_host_file, host_fs_access=True)

    @agent.task
    def test_vfs_access() -> str:
        """Agent uses VFS."""
        pass

    result = test_vfs_access()
    assert result == "Agent VFS content"


# ============================================================================
# VirtualFS files still work
# ============================================================================


def test_vfs_virtual_files_still_work(host_test_file):
    """VirtualFS files still work alongside host_fs_access functions."""

    def read_any_file(filepath):
        with open(filepath, "r") as f:
            return f.read()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Testing both VFS and host access",
                code=f"""# Create a virtual file
with open("/vfs_file.txt", "w") as f:
    f.write("virtual content")

# Read virtual file directly
with open("/vfs_file.txt", "r") as f:
    vfs_content = f.read()

# Read host file through registered function
host_content = read_any_file("{host_test_file}")

task_success(f"{{vfs_content}}|{{host_content}}")""",
            )
        ]
    )

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
        llm=llm,
    )

    agent.fn(read_any_file, host_fs_access=True)

    @agent.task
    def test_both() -> str:
        """Test both VFS and host access."""
        pass

    result = test_both()
    assert "virtual content" in result
    assert "host filesystem content" in result


# ============================================================================
# Recursive module registration tests
# ============================================================================


def test_module_function_host_fs_access_check():
    """Verify _should_suspend_fs_interception correctly identifies module functions."""
    import types

    # Create a function attached to a module
    def read_file(path):
        with open(path) as f:
            return f.read()

    test_mod = types.ModuleType("mylib")
    test_mod.read_file = read_file
    read_file.__module__ = "mylib"

    from agex import Agent, connect_fs, connect_state

    agent = Agent(
        fs=connect_fs(type="virtual"),
        state=connect_state(type="live", storage="memory"),
    )

    # Register module with host_fs_access=True
    agent.module(test_mod, name="mylib", host_fs_access=True)

    # Verify the check function works correctly
    from agex.eval.call import CallEvaluator

    class MockAgent:
        _policy = agent._policy

    evaluator = CallEvaluator.__new__(CallEvaluator)
    evaluator.agent = MockAgent()

    # The function should be identified as needing host fs access
    assert evaluator._should_suspend_fs_interception(test_mod.read_file) is True

    # A random function not in a registered module should not get host access
    def unregistered():
        pass

    assert evaluator._should_suspend_fs_interception(unregistered) is False
