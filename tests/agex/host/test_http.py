"""Tests for HTTP host implementation."""

import pytest

from agex.host import HTTP
from agex.host.http import RemoteExecutionError, RemoteTimeoutError


class TestURLValidation:
    """Tests for URL validation in HTTP host."""

    def test_valid_http_url(self):
        """Test that valid HTTP URLs are accepted."""
        host = HTTP(url="http://localhost:8000/execute")
        assert host.url == "http://localhost:8000/execute"

    def test_valid_https_url(self):
        """Test that valid HTTPS URLs are accepted."""
        host = HTTP(url="https://api.example.com/execute")
        assert host.url == "https://api.example.com/execute"

    def test_missing_scheme_rejected(self):
        """Test that URLs without scheme are rejected."""
        with pytest.raises(ValueError, match="must start with"):
            HTTP(url="example.com:8000/execute")

    def test_missing_host_rejected(self):
        """Test that URLs without host are rejected."""
        with pytest.raises(ValueError, match="missing host"):
            HTTP(url="http:///execute")

    def test_relative_url_rejected(self):
        """Test that relative URLs are rejected."""
        with pytest.raises(ValueError, match="must start with"):
            HTTP(url="/execute")


class TestHTTPHostConfig:
    """Tests for HTTP host configuration."""

    def test_default_timeout(self):
        """Test default timeout value."""
        host = HTTP(url="http://localhost:8000")
        assert host.timeout == 300.0

    def test_custom_timeout(self):
        """Test custom timeout value."""
        host = HTTP(url="http://localhost:8000", timeout=60.0)
        assert host.timeout == 60.0

    def test_default_retries(self):
        """Test default retries value."""
        host = HTTP(url="http://localhost:8000")
        assert host.retries == 0

    def test_custom_retries(self):
        """Test custom retries value."""
        host = HTTP(url="http://localhost:8000", retries=3)
        assert host.retries == 3


class TestRemoteErrors:
    """Tests for remote execution errors."""

    def test_remote_execution_error_str(self):
        """Test RemoteExecutionError string representation."""
        error = RemoteExecutionError("Task failed")
        assert str(error) == "Task failed"
        assert "Remote Traceback" not in str(error)

    def test_remote_execution_error_with_traceback(self):
        """Test RemoteExecutionError with traceback."""
        tb = (
            '  File "/server/app.py", line 42\n    raise ValueError("x")\nValueError: x'
        )
        error = RemoteExecutionError("Task failed", remote_traceback=tb)
        s = str(error)
        assert "Task failed" in s
        assert "Remote Traceback:" in s
        assert 'File "/server/app.py"' in s

    def test_remote_timeout_error(self):
        """Test RemoteTimeoutError is a TimeoutError."""
        error = RemoteTimeoutError("Timed out")
        assert isinstance(error, TimeoutError)


class TestLiveObjectValidation:
    """Tests for live object registration validation."""

    def test_http_host_rejects_live_objects(self):
        """HTTP host should reject live object registration."""
        from agex import Agent, connect_host

        agent = Agent(
            host=connect_host(provider="http", url="http://localhost:8000/execute")
        )

        class DummyObject:
            def method(self):
                return 42

        obj = DummyObject()

        with pytest.raises(
            ValueError, match="Cannot register live object.*remote host"
        ):
            agent.module(obj, name="live_obj")
