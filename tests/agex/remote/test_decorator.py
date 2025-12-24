import base64
import json
from unittest.mock import patch

import cloudpickle
import pytest

from agex import Agent
from agex.remote import RemoteExecutionError, remote


# Mock SSE response generator
def mock_sse_stream(events):
    for event_type, payload in events:
        # payload can be any object, we pickle and b64 encode it
        pickled = base64.b64encode(cloudpickle.dumps(payload)).decode("utf-8")
        data = json.dumps({"type": event_type, "payload": pickled})
        yield f"data: {data}\n\n"


@pytest.fixture
def mock_client():
    with patch("httpx.Client") as mock:
        yield mock


def test_remote_decorator_structure():
    """Test that @remote wraps the function correctly."""
    agent = Agent()

    @remote("http://example.com")
    @agent.task
    def my_task(x: int):
        """Docstring required."""
        pass

    # Check if wrapper allows access to metadata
    assert my_task.__name__ == "my_task"
    # The wrapper should be callable
    assert callable(my_task)


def test_remote_execution_flow(mock_client):
    """Test full execution flow with mocked HTTP response."""
    agent = Agent()

    @remote("http://test.server/exec")
    @agent.task
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        pass

    # Mock response
    mock_instance = mock_client.return_value.__enter__.return_value
    mock_stream = mock_instance.stream.return_value.__enter__.return_value

    expected_result = 42
    events = [("token", "thinking..."), ("result", expected_result)]
    mock_stream.iter_lines.return_value = mock_sse_stream(events)

    # Call the task
    tokens = []

    def on_token(t):
        tokens.append(t)

    result = add(20, 22, on_token=on_token)

    # Verify result
    assert result == 42
    assert tokens == ["thinking..."]

    # Verify request
    mock_instance.stream.assert_called_once()
    args, kwargs = mock_instance.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "http://test.server/exec"

    payload = kwargs["json"]
    assert payload["task_name"] == "add"
    assert payload["args"] == (20, 22)
    assert "agent_payload" in payload


def test_remote_state_override(mock_client):
    """Test state URI override."""
    agent = Agent()

    @remote("http://test.server/exec", state="disk://default")
    @agent.task
    def task():
        """Do something."""
        pass

    mock_instance = mock_client.return_value.__enter__.return_value
    mock_stream = mock_instance.stream.return_value.__enter__.return_value
    mock_stream.iter_lines.side_effect = lambda: mock_sse_stream([("result", "ok")])

    # 1. Default state
    task()
    payload = mock_instance.stream.call_args_list[0].kwargs["json"]
    assert payload["state_uri"] == "disk://default"

    # 2. Override state
    task(state="disk://new_session")
    payload = mock_instance.stream.call_args_list[1].kwargs["json"]
    assert payload["state_uri"] == "disk://new_session"

    # Ensure 'state' was removed from kwargs sent to server
    assert "state" not in payload["kwargs"]


def test_remote_error_handling(mock_client):
    """Test server-side error propagation."""
    agent = Agent()

    @remote("http://fail.com")
    @agent.task
    def fail_task():
        """Fail."""
        pass

    mock_instance = mock_client.return_value.__enter__.return_value
    mock_stream = mock_instance.stream.return_value.__enter__.return_value

    # Simulate error event
    error_data = json.dumps({"type": "error", "message": "Boom!", "traceback": "..."})
    mock_stream.iter_lines.return_value = [f"data: {error_data}\n\n"]

    with pytest.raises(RemoteExecutionError, match="Boom!"):
        fail_task()


def test_remote_missing_result(mock_client):
    """Test missing result event."""
    agent = Agent()

    @remote("http://test.com")
    @agent.task
    def no_result():
        """No result."""
        pass

    mock_instance = mock_client.return_value.__enter__.return_value
    mock_stream = mock_instance.stream.return_value.__enter__.return_value
    mock_stream.iter_lines.return_value = []  # Empty stream

    with pytest.raises(RemoteExecutionError, match="Connection closed"):
        no_result()


def test_remote_async_task():
    """Test async task wrapper."""
    agent = Agent()

    @remote("http://test.com")
    @agent.task
    async def async_add(a: int, b: int) -> int:
        """Add async."""
        pass

    # Verify that the wrapper is a coroutine function
    import inspect

    assert inspect.iscoroutinefunction(async_add)


def test_remote_url_validation_missing_scheme():
    """Test that URLs without scheme are rejected."""
    agent = Agent()

    with pytest.raises(ValueError, match="must start with"):

        @remote("example.com:8000")
        @agent.task
        def task():
            """Test."""
            pass


def test_remote_url_validation_missing_host():
    """Test that URLs without host are rejected."""
    agent = Agent()

    with pytest.raises(ValueError, match="missing host"):

        @remote("http://")
        @agent.task
        def task():
            """Test."""
            pass


def test_remote_url_validation_valid_urls():
    """Test that valid URLs are accepted."""
    agent = Agent()

    # These should not raise
    @remote("http://localhost:8000")
    @agent.task
    def task1():
        """Test."""
        pass

    @remote("https://api.example.com/execute")
    @agent.task
    def task2():
        """Test."""
        pass

    assert callable(task1)
    assert callable(task2)


def test_remote_execution_error_str():
    """Test that RemoteExecutionError includes traceback in str output."""
    # Without traceback
    e1 = RemoteExecutionError("Task failed")
    assert str(e1) == "Task failed"
    assert "Remote Traceback" not in str(e1)

    # With traceback
    tb = '  File "/server/app.py", line 42\n    raise ValueError("x")\nValueError: x'
    e2 = RemoteExecutionError("Task failed", remote_traceback=tb)
    s = str(e2)
    assert "Task failed" in s
    assert "Remote Traceback:" in s
    assert 'File "/server/app.py"' in s
    assert "ValueError: x" in s


def test_remote_state_type_validation():
    """Test that passing a state object instead of URI raises helpful error."""
    from agex import Versioned
    from agex.remote.decorator import _extract_remote_kwargs

    # String URI should work
    kwargs = {"state": "disk://session", "foo": "bar"}
    state_uri, _, _ = _extract_remote_kwargs(kwargs, default_state=None)
    assert state_uri == "disk://session"
    assert "foo" in kwargs  # Other kwargs preserved

    # Versioned object should raise TypeError
    kwargs = {"state": Versioned()}
    with pytest.raises(TypeError, match="state URI string"):
        _extract_remote_kwargs(kwargs, default_state=None)

    # Any non-string should raise
    kwargs = {"state": {"invalid": "dict"}}
    with pytest.raises(TypeError, match="got dict"):
        _extract_remote_kwargs(kwargs, default_state=None)
