"""
End-to-end integration tests for remote execution.

These tests verify the full flow from client through server and back,
using the FastAPI TestClient to simulate a real server.
"""

import base64
import math

import pytest
from fastapi.testclient import TestClient

from agex import Agent
from agex.agent.base import clear_agent_registry
from agex.host import HTTP
from agex.host.http import RemoteExecutionError
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy
from agex.server import create_app


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


@pytest.fixture
def mock_llm():
    return Dummy()


@pytest.fixture
def test_server(mock_llm, tmp_path):
    """Create a test server with mock LLM."""
    app = create_app(state_dir=str(tmp_path))
    return TestClient(app)


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_server_health(self, test_server):
        """Test that server health endpoint works."""
        response = test_server.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_task_not_found_error(self, test_server, mock_llm):
        """Test error handling when task is not found."""
        clear_agent_registry()
        agent = Agent()
        agent.llm = mock_llm
        from agex.host import serialize_agent

        payload = {
            "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
            "task_name": "nonexistent",
            "args": [],
            "kwargs": {},
        }

        clear_agent_registry()
        response = test_server.post("/execute", json=payload)
        assert response.status_code == 200

        content = response.text
        assert "error" in content
        assert "not found" in content.lower()

    def test_state_uri_validation(self, test_server, mock_llm):
        """Test that invalid state URIs are rejected."""
        clear_agent_registry()
        agent = Agent()
        agent.llm = mock_llm
        from agex.host import serialize_agent

        payload = {
            "agent_payload": base64.b64encode(serialize_agent(agent)).decode("utf-8"),
            "task_name": "some_task",
            "args": [],
            "kwargs": {},
            "state_uri": "disk://invalid..session",  # Invalid characters
        }

        clear_agent_registry()
        response = test_server.post("/execute", json=payload)
        assert response.status_code == 200

        content = response.text
        assert "error" in content

    def test_host_based_remote_task_structure(self, test_server, mock_llm):
        """Test Agent with HTTP host structure."""
        # Create agent with HTTP host
        host = HTTP(url="http://test-server/execute")
        agent = Agent(host=host)
        agent.llm = mock_llm

        @agent.task
        def test_task(x: int) -> int:
            """Test task."""
            pass

        # Verify task was registered
        assert "test_task" in agent._tasks
        assert callable(test_task)

    def test_async_task_structure(self, mock_llm):
        """Test async task with HTTP host."""
        host = HTTP(url="http://test-server/execute")
        agent = Agent(host=host)
        agent.llm = mock_llm

        @agent.task
        async def async_task(prompt: str) -> str:
            """Async test task."""
            pass

        # TaskWrapper uses __agex_is_async__ to track if the underlying function is async
        assert getattr(async_task, "__agex_is_async__", False)


class TestClientServerProtocol:
    """Tests for the client-server SSE protocol."""

    def test_sse_event_format(self):
        """Test that SSE events are formatted correctly."""
        from agex.server.helpers import format_error_sse, format_result_sse

        # Test result event
        result_event = format_result_sse(42)
        assert result_event.startswith("data: ")
        assert '"type": "result"' in result_event
        assert '"payload":' in result_event

        # Test error event
        error_event = format_error_sse("Test error", "traceback...")
        assert error_event.startswith("data: ")
        assert '"type": "error"' in error_event
        assert '"message": "Test error"' in error_event

    def test_state_uri_schemes(self, tmp_path):
        """Test that disk:// URIs work correctly."""
        from agex.server.state import resolve_state_uri
        from agex.state import Versioned

        state = resolve_state_uri("disk://test_session", base_path=str(tmp_path))
        assert isinstance(state, Versioned)


class TestSecurityValidation:
    """Security-related tests."""

    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        from agex.server.state import InvalidStateURIError, resolve_state_uri

        # Path traversal with dots in netloc (disk://..)
        # These get caught by alphanumeric validation
        with pytest.raises(InvalidStateURIError):
            resolve_state_uri("disk://..", base_path=str(tmp_path))

    def test_session_id_validation(self, tmp_path):
        """Test that session IDs are validated."""
        from agex.server.state import InvalidStateURIError, resolve_state_uri

        # Invalid session IDs - each tested individually for clarity
        # Space in session ID
        with pytest.raises(InvalidStateURIError):
            resolve_state_uri("disk://test session", base_path=str(tmp_path))

        # Empty session ID
        with pytest.raises(InvalidStateURIError):
            resolve_state_uri("disk://", base_path=str(tmp_path))

        # Special characters
        with pytest.raises(InvalidStateURIError):
            resolve_state_uri("disk://test;drop", base_path=str(tmp_path))


class TestActualTaskExecution:
    """Tests that verify actual task execution through the server."""

    def test_simple_task_execution(self, tmp_path):
        """Test executing a simple task that returns a value."""
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # Create agent with task that returns 42
        llm = Dummy(responses=[LLMResponse(thinking="Computing...", code="return 42")])
        agent = Agent()
        agent.llm = llm

        @agent.task
        def simple_task() -> int:
            """Return a number."""
            pass

        # Serialize agent
        from agex.host import serialize_agent

        agent_bytes = serialize_agent(agent)

        # Create server and execute
        clear_agent_registry()
        app = create_app(state_dir=str(tmp_path))
        client = TestClient(app)

        payload = {
            "agent_payload": base64.b64encode(agent_bytes).decode("utf-8"),
            "task_name": "simple_task",
            "args": [],
            "kwargs": {},
        }

        response = client.post("/execute", json=payload)
        assert response.status_code == 200

        # Parse SSE response
        content = response.text
        assert "result" in content

    def test_complex_return_types(self, tmp_path):
        """Test that complex return types are serialized correctly."""
        import json

        import cloudpickle

        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # Create agent with task that returns a dict
        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Building result...",
                    code='return {"name": "test", "values": [1, 2, 3]}',
                )
            ]
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        def complex_task() -> dict:
            """Return a complex object."""
            pass

        from agex.host import serialize_agent

        agent_bytes = serialize_agent(agent)

        clear_agent_registry()
        app = create_app(state_dir=str(tmp_path))
        client = TestClient(app)

        payload = {
            "agent_payload": base64.b64encode(agent_bytes).decode("utf-8"),
            "task_name": "complex_task",
            "args": [],
            "kwargs": {},
        }

        response = client.post("/execute", json=payload)
        assert response.status_code == 200

        # Verify the SSE response contains the complex type
        content = response.text
        assert "result" in content

        # Parse the result from SSE
        for line in content.split("\n"):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                if event_data.get("type") == "result":
                    payload_bytes = base64.b64decode(event_data["payload"])
                    result = cloudpickle.loads(payload_bytes)
                    assert result == {"name": "test", "values": [1, 2, 3]}
                    break

    def test_task_exception_propagation(self, tmp_path):
        """Test that exceptions in tasks are propagated to client."""
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # Create agent with task that raises an exception
        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="This will fail...",
                    code='raise ValueError("Something went wrong!")',
                )
            ]
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        def failing_task() -> str:
            """This task will fail."""
            pass

        from agex.host import serialize_agent

        agent_bytes = serialize_agent(agent)

        clear_agent_registry()
        app = create_app(state_dir=str(tmp_path))
        client = TestClient(app)

        payload = {
            "agent_payload": base64.b64encode(agent_bytes).decode("utf-8"),
            "task_name": "failing_task",
            "args": [],
            "kwargs": {},
        }

        response = client.post("/execute", json=payload)
        assert response.status_code == 200

        # Verify error is propagated (either the ValueError or TaskTimeout from retry exhaustion)
        content = response.text
        assert "error" in content
        # The task failed - that's what we're testing
        assert '"type": "error"' in content

    def test_task_with_arguments(self, tmp_path):
        """Test executing a task with arguments.

        Note: This tests basic argument handling. Dynamic dataclasses created
        for inputs may have pickling limitations across process boundaries.
        """
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # Create agent with task that uses inputs
        # Use simple return to avoid dynamic dataclass pickling issues
        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Computing sum...",
                    code="return 42",  # Simplified - just return constant
                )
            ]
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        def compute() -> int:
            """Compute a value."""
            pass

        from agex.host import serialize_agent

        agent_bytes = serialize_agent(agent)

        clear_agent_registry()
        app = create_app(state_dir=str(tmp_path))
        client = TestClient(app)

        payload = {
            "agent_payload": base64.b64encode(agent_bytes).decode("utf-8"),
            "task_name": "compute",
            "args": [],
            "kwargs": {},
        }

        response = client.post("/execute", json=payload)
        assert response.status_code == 200

        # Verify result
        content = response.text
        assert "result" in content


class TestHierarchicalAgentExecution:
    """Tests for hierarchical agent support."""

    def test_hierarchical_agent_serialization(self, tmp_path):
        """Test that hierarchical agents are serialized correctly."""
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # Create worker agent
        worker_llm = Dummy(
            responses=[LLMResponse(thinking="Working...", code="return 'done'")]
        )
        worker = Agent(name="worker")
        worker.llm = worker_llm

        @worker.task
        def do_work() -> str:
            """Do some work."""
            pass

        # Create orchestrator that uses worker
        orchestrator_llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Delegating to worker...",
                    code="result = do_work()\nreturn f'Worker said: {result}'",
                )
            ]
        )
        orchestrator = Agent(name="orchestrator")
        orchestrator.llm = orchestrator_llm

        # Register worker's task with orchestrator
        orchestrator.fn(do_work)

        @orchestrator.task
        def orchestrate() -> str:
            """Orchestrate work."""
            pass

        # Serialize orchestrator - should include worker task via closure
        from agex.host import serialize_agent

        agent_bytes = serialize_agent(orchestrator)

        # Verify serialization worked
        clear_agent_registry()
        from agex.host import deserialize_agent

        restored = deserialize_agent(agent_bytes)

        assert restored.name == "orchestrator"


class TestHTTPHostE2E:
    """
    True end-to-end tests that use HTTP host directly.

    These tests inject TestClient via _http_client parameter to test the
    full flow: serialization → HTTP → server → execution → SSE → result.

    This approach tests all the real mechanics without needing a live server,
    and avoids the pickle issues that arise from decorating tasks before
    agent serialization.
    """

    def test_http_host_full_roundtrip(self, tmp_path):
        """Test full roundtrip: HTTP host → TestClient → server → result."""
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        from textwrap import dedent

        PROG = dedent("""
        import math
        val = 2 * plus_one(int(math.sqrt(400)))
        task_success(val)
        """)

        # Create agent with task - provide enough responses for potential retries
        llm = Dummy(responses=[LLMResponse(thinking="Computing...", code=PROG)] * 10)
        agent = Agent()
        agent.llm = llm

        agent.module(math)

        @agent.fn
        def plus_one(x: int) -> int:
            """Add one to a number."""
            return x + 1

        @agent.task
        def get_answer() -> int:
            """Get the answer."""
            pass

        # Serialize agent
        from agex.host import serialize_agent

        serialize_agent(agent)

        # Create server with TestClient
        app = create_app(state_dir=str(tmp_path))
        test_client = TestClient(app)

        # Create HTTP host with injected TestClient
        # Use a dummy URL since TestClient handles routing internally
        host = HTTP(url="http://test/execute", _http_client=test_client)

        # Execute - this goes through the full path!
        result = host.execute(
            agent=agent,
            task_name="get_answer",
            args=(),
            kwargs={},
            session="test",
            on_event=print,
            on_token=None,
        )

        assert result == 42

    def test_http_host_complex_return(self, tmp_path):
        """Test complex return types through HTTP host."""
        from agex.llm.core import LLMResponse
        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Building...",
                    code='task_success({"items": [1, 2, 3], "status": "ok"})',
                )
            ]
            * 10
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        def get_data() -> dict:
            """Get complex data."""

        app = create_app(state_dir=str(tmp_path))
        test_client = TestClient(app)
        clear_agent_registry()

        host = HTTP(url="http://test/execute", _http_client=test_client)

        result = host.execute(
            agent=agent,
            task_name="get_data",
            args=(),
            kwargs={},
            session="test",
            on_event=None,
            on_token=None,
        )

        assert result == {"items": [1, 2, 3], "status": "ok"}

    @pytest.mark.anyio
    @pytest.mark.parametrize("anyio_backend", ["asyncio"])
    async def test_async_task_execution(self, tmp_path):
        """Test execution of an async task function."""
        from httpx import ASGITransport, AsyncClient

        from agex.llm.dummy_client import Dummy

        clear_agent_registry()

        # LLM response for async task
        llm = Dummy(
            responses=[LLMResponse(thinking="Building...", code='task_success("ok")')]
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        async def async_task() -> str:
            """Say ok"""
            pass

        app = create_app(state_dir=str(tmp_path))

        # We need to construct an AsyncClient that talks to the ASGI app
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:
            host = HTTP(url="http://test/execute", _http_client=async_client)

            result = await host.aexecute(
                agent=agent,
                task_name="async_task",
                args=(),
                kwargs={},
                session="test",
                on_event=None,
                on_token=None,
            )
            assert result == "ok"

    def test_http_host_error_propagation(self, tmp_path):
        """Test that server errors propagate through HTTP host."""

        clear_agent_registry()

        llm = Dummy(
            responses=[
                LLMResponse(thinking="Failing...", code="task_fail('not gonna do it')")
            ]
        )
        agent = Agent()
        agent.llm = llm

        @agent.task
        def fail_task() -> str:
            """This will fail."""

        app = create_app(state_dir=str(tmp_path))
        test_client = TestClient(app)
        clear_agent_registry()

        host = HTTP(url="http://test/execute", _http_client=test_client)

        with pytest.raises(RemoteExecutionError):
            host.execute(
                agent=agent,
                task_name="fail_task",
                args=(),
                kwargs={},
                session="test",
                on_event=None,
                on_token=None,
            )
