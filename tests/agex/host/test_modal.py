"""Tests for Modal host implementation."""

from unittest.mock import MagicMock, patch

import pytest

from agex import Agent
from agex.host import connect_host


class TestConnectHost:
    """Test connect_host factory for Modal."""

    def test_connect_host_modal_requires_secrets(self):
        """Modal host requires 'secrets' parameter."""
        with pytest.raises(ValueError, match="Modal host requires 'secrets'"):
            connect_host(provider="modal", app="my-app")

    def test_connect_host_modal_basic(self):
        """Modal host can be created with app and secrets parameters."""
        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")
        assert host.app == "my-app"
        assert host.secrets == ["llm-keys"]  # Normalized to list
        assert host.gpu is None
        assert host.timeout == 300.0

    def test_connect_host_modal_full(self):
        """Modal host accepts all parameters."""
        host = connect_host(
            provider="modal",
            app="my-app",
            secrets=["llm-keys", "db-keys"],
            gpu="A10G",
            memory=4096,
            timeout=600.0,
        )
        assert host.app == "my-app"
        assert host.secrets == ["llm-keys", "db-keys"]
        assert host.gpu == "A10G"
        assert host.memory == 4096
        assert host.timeout == 600.0


class TestModalHostConfig:
    """Test Modal host configuration serialization."""

    def test_dump_config(self):
        """dump_config returns serializable configuration."""
        host = connect_host(
            provider="modal",
            app="my-app",
            secrets=["llm-keys"],
            gpu="A10G",
        )

        config = host.dump_config()

        assert config["provider"] == "modal"
        assert config["app"] == "my-app"
        assert config["secrets"] == ["llm-keys"]
        assert config["gpu"] == "A10G"

    def test_from_config(self):
        """Host.from_config can reconstruct Modal host."""
        from agex.host.base import Host

        config = {
            "provider": "modal",
            "app": "my-app",
            "secrets": ["llm-keys"],
            "timeout": 600.0,
        }

        host = Host.from_config(config)

        assert host.app == "my-app"
        assert host.timeout == 600.0


class TestModalHostValidation:
    """Test Modal host state validation."""

    def test_validate_state_ephemeral_ok(self):
        """Ephemeral state is always valid."""
        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")
        # Should not raise
        host.validate_state(None)

    def test_validate_state_unsupported_storage(self):
        """Modal host rejects unsupported storage types."""
        from agex.state.config import StateConfig

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")
        config = StateConfig(type="versioned", storage="redis")  # Not supported

        with pytest.raises(ValueError, match="supports storage='memory' or 'disk'"):
            host.validate_state(config)

    def test_validate_state_disk_requires_path(self):
        """ModalLocal requires path for disk storage."""
        from agex.host.modal import ModalLocal
        from agex.state.config import StateConfig

        host = ModalLocal()
        config = StateConfig(type="versioned", storage="disk")  # No path

        with pytest.raises(ValueError, match="requires 'path' parameter"):
            host.validate_state(config)

    def test_validate_state_live_rejected(self):
        """Live state is not supported on Modal."""
        from agex.host.modal import ModalLocal
        from agex.state.config import StateConfig

        host = ModalLocal()

        # Live with explicit storage should be rejected
        config_disk = StateConfig(type="live", storage="disk", path="test")
        with pytest.raises(ValueError, match="Live state is not supported on Modal"):
            host.validate_state(config_disk)

        # Live with memory should also be rejected
        config_mem = StateConfig(type="live", storage="memory")
        with pytest.raises(ValueError, match="Live state is not supported on Modal"):
            host.validate_state(config_mem)

    def test_validate_state_disk_ok(self):
        """Disk storage works when path is provided."""
        from agex.state.config import StateConfig

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")
        config = StateConfig(type="versioned", storage="disk", path="my-agent")

        # Should not raise (path is provided)
        host.validate_state(config)

    def test_validate_state_versioned_memory_rejected(self):
        """Versioned state with memory storage is rejected (memory resets between invocations)."""
        from agex.state.config import StateConfig

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")
        config = StateConfig(type="versioned", storage="memory")

        # Should raise - memory is reset between invocations on serverless
        with pytest.raises(
            ValueError, match="Versioned state with memory storage is not supported"
        ):
            host.validate_state(config)


class TestModalHostWarmup:
    """Test Modal host warmup."""

    def test_warmup_builds_function(self):
        """warmup() builds the Modal function with dependencies."""
        host = connect_host(provider="modal", app="my-app", secrets=["llm-keys"])

        from agex.host.dependencies import Dependencies

        deps = Dependencies(
            python_version="3.12",
            agex_version="0.1.0",
            packages=["numpy==1.24.0"],
        )

        # Mock _build_function and _modal_app to avoid actual Modal calls
        mock_app = MagicMock()
        host._modal_app = mock_app

        with patch.object(host, "_build_function") as mock_build:
            host.warmup(deps)

            # Verify _build_function was called with deps
            mock_build.assert_called_once_with(deps)
            # Verify deploy was called since detach=True by default
            mock_app.deploy.assert_called_once()

    def test_warmup_skips_if_already_built(self):
        """warmup() skips build if deps haven't changed."""
        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")

        from agex.host.dependencies import Dependencies

        deps = Dependencies(
            python_version="3.12",
            agex_version="0.1.0",
            packages=["numpy==1.24.0"],
        )

        # Simulate already built
        host._runner_fn = MagicMock()
        host._current_deps_id = deps.id
        mock_app = MagicMock()
        host._modal_app = mock_app

        with patch.object(host, "_build_function") as mock_build:
            host.warmup(deps)

            # Should NOT rebuild
            mock_build.assert_not_called()
            # But should still deploy
            mock_app.deploy.assert_called_once()


class TestModalHostExecute:
    """Test Modal host execute with mocked SDK."""

    def test_execute_serializes_agent(self):
        """execute() serializes agent and calls runner."""
        import modal

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")

        # Create a simple agent
        agent = Agent()

        @agent.task
        def test_task() -> str:
            """Test task."""
            pass

        # Mock the runner
        mock_runner = MagicMock()
        mock_runner.remote_gen.return_value = [
            {"type": "result", "data": "success"},
        ]

        # Mock the function lookup
        with patch.object(host, "_ensure_function"):
            with patch.object(modal.Function, "from_name", return_value=mock_runner):
                result = host.execute(
                    agent=agent,
                    task_name="test_task",
                    args=(),
                    kwargs={},
                    session="default",
                    on_event=None,
                    on_token=None,
                )

                assert result == "success"

                # Verify runner was called
                mock_runner.remote_gen.assert_called_once()
                call_kwargs = mock_runner.remote_gen.call_args[1]
                assert "agent_bytes" in call_kwargs
                assert call_kwargs["task_name"] == "test_task"
                assert call_kwargs["session"] == "default"

    def test_execute_processes_events(self):
        """execute() processes event stream."""
        import modal

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")

        agent = Agent()

        @agent.task
        def test_task() -> str:
            """Test task."""
            pass

        events = []
        tokens = []

        mock_runner = MagicMock()
        mock_runner.remote_gen.return_value = [
            {"type": "event", "data": {"event": "started"}},
            {"type": "token", "data": {"token": "Hello"}},
            {"type": "event", "data": {"event": "finished"}},
            {"type": "result", "data": "done"},
        ]

        with patch.object(host, "_ensure_function"):
            with patch.object(modal.Function, "from_name", return_value=mock_runner):
                result = host.execute(
                    agent=agent,
                    task_name="test_task",
                    args=(),
                    kwargs={},
                    session="default",
                    on_event=lambda e: events.append(e),
                    on_token=lambda t: tokens.append(t),
                )

                assert result == "done"
                assert len(events) == 2
                assert len(tokens) == 1

    def test_execute_handles_error(self):
        """execute() raises on error message."""
        import modal

        host = connect_host(provider="modal", app="my-app", secrets="llm-keys")

        agent = Agent()

        @agent.task
        def test_task() -> str:
            """Test task."""
            pass

        mock_runner = MagicMock()
        mock_runner.remote_gen.return_value = [
            {"type": "error", "data": "Task failed: something went wrong"},
        ]

        with patch.object(host, "_ensure_function"):
            with patch.object(modal.Function, "from_name", return_value=mock_runner):
                with pytest.raises(RuntimeError, match="Task failed"):
                    host.execute(
                        agent=agent,
                        task_name="test_task",
                        args=(),
                        kwargs={},
                        session="default",
                        on_event=None,
                        on_token=None,
                    )


class TestModalHostHierarchical:
    """Test hierarchical agent restrictions with Modal."""

    def test_nested_modal_host_rejected(self):
        """Sub-agents with Modal host are rejected when registered."""
        from agex.host import connect_host
        from agex.state import connect_state

        parent = Agent()
        child = Agent(
            host=connect_host(provider="modal", app="child-app", secrets="llm-keys"),
            state=connect_state(type="versioned", storage="disk", path="child"),
        )

        @child.task
        def child_task() -> str:
            """Child task."""
            ...  # Empty body required for @task

        # Registering a task from a Modal-hosted sub-agent should fail
        with pytest.raises(ValueError, match="sub-agents must use Local host"):
            parent.fn(child_task)

    def test_local_subagent_allowed(self):
        """Sub-agents with Local host are allowed."""
        from agex.host import Local
        from agex.state import connect_state

        parent = Agent()
        child = Agent(
            host=Local(),
            state=connect_state(type="versioned", storage="disk", path="child"),
        )

        @child.task
        def child_task() -> str:
            """Child task."""
            ...  # Empty body required for @task

        # This should work (no exception raised = success)
        parent.fn(child_task)


class TestLiveObjectValidation:
    """Test live object registration validation."""

    def test_modal_host_rejects_live_objects(self):
        """Modal host should reject live object registration."""
        agent = Agent(host=connect_host(provider="modal", app="test", secrets=["test"]))

        class DummyObject:
            def method(self):
                return 42

        obj = DummyObject()

        with pytest.raises(
            ValueError, match="Cannot register live object.*remote host"
        ):
            agent.module(obj, name="live_obj")
