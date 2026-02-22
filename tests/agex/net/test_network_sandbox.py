"""Tests for network access control."""

import socket

import pytest

from agex.net import SandboxError, allow_network, network_allowed
from agex.net.patch import (
    install_network_sandbox,
    is_network_sandbox_installed,
    uninstall_network_sandbox,
)


class TestNetworkContext:
    """Tests for network context variable."""

    def test_default_is_allowed(self):
        """Test that network is allowed by default (so asyncio/pytest work)."""
        assert network_allowed.get() is True

    def test_deny_network_context(self):
        """Test that deny_network context manager blocks access."""
        from agex.net import deny_network

        assert network_allowed.get() is True

        with deny_network():
            assert network_allowed.get() is False

        assert network_allowed.get() is True

    def test_allow_network_restores_in_deny_context(self):
        """Test that allow_network restores access within deny_network."""
        from agex.net import deny_network

        assert network_allowed.get() is True

        with deny_network():
            assert network_allowed.get() is False
            with allow_network():
                assert network_allowed.get() is True
            assert network_allowed.get() is False

        assert network_allowed.get() is True


class TestNetworkSandboxInstallation:
    """Tests for sandbox installation."""

    def setup_method(self):
        """Ensure sandbox is uninstalled before each test."""
        uninstall_network_sandbox()

    def teardown_method(self):
        """Reinstall sandbox after each test (for other tests)."""
        install_network_sandbox()

    def test_install_patches_socket_methods(self):
        """Test that installation patches socket methods."""
        from agex.net.socket import _patched_connect

        original_connect = socket.socket.connect
        install_network_sandbox()
        assert socket.socket.connect is _patched_connect
        uninstall_network_sandbox()
        assert socket.socket.connect is original_connect

    def test_uninstall_restores_socket_methods(self):
        """Test that uninstallation restores original socket methods."""
        original_connect = socket.socket.connect
        install_network_sandbox()
        uninstall_network_sandbox()
        assert socket.socket.connect is original_connect

    def test_install_is_idempotent(self):
        """Test that calling install multiple times is safe."""
        from agex.net.socket import _patched_connect

        install_network_sandbox()
        install_network_sandbox()  # Second call should be no-op
        assert socket.socket.connect is _patched_connect
        assert is_network_sandbox_installed()

    def test_uninstall_is_idempotent(self):
        """Test that calling uninstall multiple times is safe."""
        install_network_sandbox()
        uninstall_network_sandbox()
        uninstall_network_sandbox()  # Second call should be no-op
        assert not is_network_sandbox_installed()


class TestGatedSocketOperations:
    """Tests for gated socket operations."""

    def setup_method(self):
        """Ensure sandbox is installed."""
        install_network_sandbox()

    def test_connect_blocked_in_deny_context(self):
        """Test that connect is blocked in deny_network context."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with deny_network():
                with pytest.raises(SandboxError, match="connect.*blocked by sandbox"):
                    sock.connect(("localhost", 80))
        finally:
            sock.close()

    def test_connect_allowed_by_default(self):
        """Test that connect works by default (outside deny context)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)  # Use non-blocking to avoid actual connection
        try:
            # This should not raise SandboxError (default is allowed)
            try:
                sock.connect(("127.0.0.1", 1))  # Unlikely to be listening
            except SandboxError:
                pytest.fail("SandboxError should not be raised by default")
            except (ConnectionRefusedError, BlockingIOError, OSError):
                pass  # Expected - we just want to verify no SandboxError
        finally:
            sock.close()

    def test_connect_allowed_with_permission_in_deny_context(self):
        """Test that connect works with allow_network inside deny_network."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            with deny_network():
                with allow_network():
                    try:
                        sock.connect(("127.0.0.1", 1))
                    except SandboxError:
                        pytest.fail(
                            "SandboxError should not be raised with allow_network()"
                        )
                    except (ConnectionRefusedError, BlockingIOError, OSError):
                        pass
        finally:
            sock.close()

    def test_bind_blocked_in_deny_context(self):
        """Test that bind is blocked in deny_network context."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with deny_network():
                with pytest.raises(SandboxError, match="bind.*blocked by sandbox"):
                    sock.bind(("localhost", 0))
        finally:
            sock.close()

    def test_listen_blocked_in_deny_context(self):
        """Test that listen is blocked in deny_network context."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with deny_network():
                with pytest.raises(SandboxError, match="listen.*blocked by sandbox"):
                    sock.listen(1)
        finally:
            sock.close()

    def test_send_blocked_in_deny_context(self):
        """Test that send is blocked in deny_network context."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with deny_network():
                with pytest.raises(SandboxError, match="send.*blocked by sandbox"):
                    sock.send(b"test")
        finally:
            sock.close()

    def test_recv_blocked_in_deny_context(self):
        """Test that recv is blocked in deny_network context."""
        from agex.net import deny_network

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with deny_network():
                with pytest.raises(SandboxError, match="recv.*blocked by sandbox"):
                    sock.recv(1024)
        finally:
            sock.close()

    def test_getaddrinfo_blocked_in_deny_context(self):
        """Test that getaddrinfo (DNS) is blocked in deny_network context."""
        from agex.net import deny_network

        with deny_network():
            with pytest.raises(SandboxError, match="getaddrinfo.*blocked by sandbox"):
                socket.getaddrinfo("example.com", 80)

    def test_getaddrinfo_allowed_by_default(self):
        """Test that getaddrinfo works by default."""
        # Should not raise SandboxError (default is allowed)
        try:
            result = socket.getaddrinfo("localhost", 80)
            assert len(result) > 0
        except SandboxError:
            pytest.fail("SandboxError should not be raised by default")


class TestAgentNetworkIntegration:
    """Tests for Agent integration with network access via sandtrap bridge."""

    def test_execute_sandboxed_installs_sandbox(self):
        """Test that execute_sandboxed installs the network sandbox via sandtrap."""
        from gitkv import Live
        from sandtrap.net import patch as sandtrap_net_patch

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(name="net_install_test")

        state = Live()
        state["__event_log__"] = []

        # Running execute_sandboxed installs sandtrap's network patches
        execute_sandboxed("x = 1", agent, state, eval_timeout_seconds=5.0)

        assert sandtrap_net_patch._installed

    def test_registered_fn_with_network_access(self):
        """Test that functions registered with network_access=True can use network."""
        from gitkv import Live
        from sandtrap.net.context import network_allowed as sandtrap_network_allowed

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(name="net_fn_allowed_test")

        # Track whether network was accessible (via sandtrap's context var)
        network_was_allowed = []

        def check_network_access():
            """Check if network is currently allowed."""
            network_was_allowed.append(sandtrap_network_allowed.get())
            return sandtrap_network_allowed.get()

        agent.fn(check_network_access, network_access=True)

        state = Live()
        state["__event_log__"] = []
        execute_sandboxed(
            "result = check_network_access()",
            agent,
            state,
            eval_timeout_seconds=5.0,
        )

        assert network_was_allowed == [
            True
        ], "Network should be allowed inside registered function"

    def test_registered_fn_without_network_access(self):
        """Test that functions without network_access=True cannot use network."""
        from gitkv import Live
        from sandtrap.net.context import network_allowed as sandtrap_network_allowed

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(name="net_fn_denied_test")

        network_was_allowed = []

        def check_network_access():
            network_was_allowed.append(sandtrap_network_allowed.get())
            return sandtrap_network_allowed.get()

        agent.fn(check_network_access)  # No network_access=True

        state = Live()
        state["__event_log__"] = []
        execute_sandboxed(
            "result = check_network_access()",
            agent,
            state,
            eval_timeout_seconds=5.0,
        )

        assert network_was_allowed == [
            False
        ], "Network should be denied for normal function"

    def test_registered_cls_with_network_access(self):
        """Test that classes registered with network_access=True have network access."""
        from gitkv import Live
        from sandtrap.net.context import network_allowed as sandtrap_network_allowed

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(name="net_cls_test")

        class NetworkClient:
            def check_access(self):
                return sandtrap_network_allowed.get()

        agent.cls(NetworkClient, network_access=True)

        state = Live()
        state["__event_log__"] = []
        execute_sandboxed(
            "client = NetworkClient(); result = client.check_access()",
            agent,
            state,
            eval_timeout_seconds=5.0,
        )

        assert state.get("result") is True

    def test_registered_module_with_network_access(self):
        """Test that modules registered with network_access=True have network access."""
        from types import ModuleType

        from gitkv import Live
        from sandtrap.net.context import network_allowed as sandtrap_network_allowed

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(name="net_mod_test")

        # Create a test module with function that has correct __module__
        test_module = ModuleType("test_net")

        def check_access():
            return sandtrap_network_allowed.get()

        # Set __module__ so the policy lookup finds the namespace
        check_access.__module__ = "test_net"
        test_module.check_access = check_access

        agent.module(test_module, name="test_net", network_access=True)

        state = Live()
        state["__event_log__"] = []
        execute_sandboxed(
            "import test_net; result = test_net.check_access()",
            agent,
            state,
            eval_timeout_seconds=5.0,
        )

        assert state.get("result") is True
