"""Tests for resource limits module."""

import sys

import pytest

from agex.resource_limits import (
    ResourceLimits,
    apply_resource_limits,
    check_platform_support,
)


class TestPlatformSupport:
    """Tests for platform detection."""

    def test_check_platform_support(self):
        """Test platform support check returns boolean."""
        result = check_platform_support()
        assert isinstance(result, bool)

    def test_platform_support_matches_platform(self):
        """Test platform support matches sys.platform."""
        expected = sys.platform != "win32"
        assert check_platform_support() == expected


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default limits are None (unlimited)."""
        limits = ResourceLimits()
        assert limits.max_open_files is None

    def test_custom_limits(self):
        """Test custom limits are stored correctly."""
        limits = ResourceLimits(max_open_files=100)
        assert limits.max_open_files == 100


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestApplyResourceLimits:
    """Tests for apply_resource_limits context manager."""

    def test_no_limits_is_noop(self):
        """Test that no limits configured is a no-op."""
        limits = ResourceLimits()
        with apply_resource_limits(limits):
            x = [0] * 1000
            assert len(x) == 1000

    def test_file_limit_allows_small_count(self):
        """Test that opening few files succeeds within limit."""
        import os
        import tempfile

        limits = ResourceLimits(max_open_files=256)
        with apply_resource_limits(limits):
            files = []
            paths = []
            for _ in range(5):
                f = tempfile.NamedTemporaryFile(delete=False)
                files.append(f)
                paths.append(f.name)
            for f in files:
                f.close()
            for path in paths:
                os.unlink(path)

    def test_file_limit_blocks_excess_opens(self):
        """Test that opening too many files is blocked."""
        import tempfile

        low_limit = 32
        limits = ResourceLimits(max_open_files=low_limit)

        with pytest.raises(OSError, match="Too many open files"):
            with apply_resource_limits(limits):
                files = []
                try:
                    for i in range(100):
                        f = tempfile.NamedTemporaryFile(delete=True)
                        files.append(f)
                finally:
                    for f in files:
                        try:
                            f.close()
                        except Exception:
                            pass

    def test_file_limit_resets_after_context(self):
        """Test that file limit is restored after context."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_NOFILE)

        limits = ResourceLimits(max_open_files=256)
        with apply_resource_limits(limits):
            pass

        after = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert original == after


class TestWindowsWarning:
    """Tests for Windows platform warning."""

    def test_warns_on_windows_with_file_limits(self, monkeypatch):
        """Test that warning is issued for file limits on Windows."""
        monkeypatch.setattr("agex.resource_limits._PLATFORM_SUPPORTS_LIMITS", False)

        limits = ResourceLimits(max_open_files=100)
        with pytest.warns(RuntimeWarning, match="not supported on Windows"):
            with apply_resource_limits(limits):
                pass

    def test_no_warning_without_limits(self, monkeypatch):
        """Test that no warning is issued when no limits are configured."""
        monkeypatch.setattr("agex.resource_limits._PLATFORM_SUPPORTS_LIMITS", False)

        limits = ResourceLimits()
        with apply_resource_limits(limits):
            pass


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestAgentIntegration:
    """Tests for Agent integration with resource limits."""

    def test_agent_accepts_memory_limit(self):
        """Test that Agent accepts max_memory_mb parameter."""
        from agex import Agent, connect_llm

        agent = Agent(
            llm=connect_llm(provider="dummy"),
            max_memory_mb=500,
        )
        assert agent.max_memory_mb == 500

    def test_agent_accepts_file_limit(self):
        """Test that Agent accepts max_open_files parameter."""
        from agex import Agent, connect_llm

        agent = Agent(
            llm=connect_llm(provider="dummy"),
            max_open_files=256,
        )
        assert agent._resource_limits.max_open_files == 256

    def test_agent_accepts_both_limits(self):
        """Test that Agent accepts both limit parameters."""
        from agex import Agent, connect_llm

        agent = Agent(
            llm=connect_llm(provider="dummy"),
            max_memory_mb=500,
            max_open_files=256,
        )
        assert agent.max_memory_mb == 500
        assert agent._resource_limits.max_open_files == 256

    def test_agent_default_is_unlimited(self):
        """Test that Agent defaults to no limits."""
        from agex import Agent, connect_llm

        agent = Agent(llm=connect_llm(provider="dummy"))
        assert agent.max_memory_mb is None
        assert agent._resource_limits.max_open_files is None

    def test_clone_registrations_passes_limits(self):
        """Test that clone_registrations passes resource limits."""
        from agex import Agent, connect_llm, connect_state

        source = Agent(llm=connect_llm(provider="dummy"))

        cloned = Agent.clone_registrations(
            source,
            llm=connect_llm(provider="dummy"),
            state=connect_state(type="live", storage="memory"),
            max_memory_mb=100,
            max_open_files=50,
        )

        assert cloned.max_memory_mb == 100
        assert cloned._resource_limits.max_open_files == 50

    def test_memory_limit_passed_to_sandtrap_policy(self):
        """Test that max_memory_mb is passed through to sandtrap's Policy."""
        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge.policy import translate_policy

        clear_agent_registry()
        agent = Agent(
            name="mem_policy_test",
            max_memory_mb=500,
        )

        @agent.fn()
        def dummy():
            pass

        policy = translate_policy(agent)
        assert policy.memory_limit == 500
