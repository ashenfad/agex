"""Tests for resource limits module."""

import sys

import pytest

from agex.resource_limits import (
    ResourceLimits,
    _get_current_memory_bytes,
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
        assert limits.max_memory_mb is None
        assert limits.max_open_files is None

    def test_custom_limits(self):
        """Test custom limits are stored correctly."""
        limits = ResourceLimits(max_memory_mb=500, max_open_files=100)
        assert limits.max_memory_mb == 500
        assert limits.max_open_files == 100


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestMemoryMeasurement:
    """Tests for memory measurement."""

    def test_get_current_memory_returns_positive(self):
        """Test that current memory measurement returns positive value."""
        mem = _get_current_memory_bytes()
        assert mem > 0

    def test_memory_increases_with_allocation(self):
        """Test that memory measurement increases after allocation."""
        before = _get_current_memory_bytes()
        # Allocate ~10MB
        data = [0] * (10 * 1024 * 1024 // 8)
        after = _get_current_memory_bytes()
        # Memory should have increased (allow some variance)
        assert after >= before
        # Keep reference to prevent GC
        assert len(data) > 0


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestApplyResourceLimits:
    """Tests for apply_resource_limits context manager."""

    def test_no_limits_is_noop(self):
        """Test that no limits configured is a no-op."""
        limits = ResourceLimits()
        with apply_resource_limits(limits):
            # Should complete without error
            x = [0] * 1000
            assert len(x) == 1000

    def test_memory_limit_allows_small_allocation(self):
        """Test that small allocations succeed within limit."""
        limits = ResourceLimits(max_memory_mb=100)
        with apply_resource_limits(limits):
            # 1MB allocation should succeed
            x = [0] * (1024 * 1024 // 8)  # ~1MB of integers
            assert len(x) > 0

    def test_memory_limit_blocks_huge_allocation(self):
        """Test that huge allocations are blocked."""
        limits = ResourceLimits(max_memory_mb=50)
        with pytest.raises(MemoryError):
            with apply_resource_limits(limits):
                # Try to allocate ~1GB - should fail
                _ = [0] * (1024 * 1024 * 1024 // 8)

    def test_memory_limit_blocks_large_overflow(self):
        """Test that allocation well over limit is blocked."""
        limits = ResourceLimits(max_memory_mb=10)
        with pytest.raises(MemoryError):
            with apply_resource_limits(limits):
                # Try to allocate ~500MB when limit is 10MB - definitely exceeds
                _ = bytearray(500 * 1024 * 1024)

    def test_memory_limit_allows_just_under(self):
        """Test that allocation just under limit succeeds."""
        limits = ResourceLimits(max_memory_mb=50)
        with apply_resource_limits(limits):
            # Allocate ~20MB when limit is 50MB - should succeed
            data = bytearray(20 * 1024 * 1024)
            assert len(data) == 20 * 1024 * 1024

    def test_memory_limit_cumulative_allocation(self):
        """Test that cumulative allocations are limited."""
        limits = ResourceLimits(max_memory_mb=50)
        with pytest.raises(MemoryError):
            with apply_resource_limits(limits):
                # Allocate in chunks that together far exceed limit
                chunks = []
                for _ in range(10):
                    # Each chunk is ~100MB, total would be 1GB >> 50MB limit
                    chunks.append(bytearray(100 * 1024 * 1024))

    def test_memory_limit_resets_after_context(self):
        """Test that memory limit is restored after context."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)

        limits = ResourceLimits(max_memory_mb=100)
        with apply_resource_limits(limits):
            pass

        after = resource.getrlimit(resource.RLIMIT_AS)
        assert original == after

    def test_memory_limit_resets_after_memory_error(self):
        """Test that memory limit is restored even after MemoryError."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)

        limits = ResourceLimits(max_memory_mb=10)
        try:
            with apply_resource_limits(limits):
                # Allocate 500MB with 10MB limit - guaranteed to fail
                _ = bytearray(500 * 1024 * 1024)
        except MemoryError:
            pass

        after = resource.getrlimit(resource.RLIMIT_AS)
        assert original == after

    def test_file_limit_allows_small_count(self):
        """Test that opening few files succeeds within limit."""
        import os
        import tempfile

        # Use a high enough limit that doesn't interfere with pytest's file usage
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
        import resource
        import tempfile

        # Get current open file count estimate
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Set a very low limit - just above what pytest needs
        # We'll try to open more files than this allows
        low_limit = 32

        limits = ResourceLimits(max_open_files=low_limit)

        with pytest.raises(OSError, match="Too many open files"):
            with apply_resource_limits(limits):
                files = []
                try:
                    # Try to open way more files than limit allows
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

    def test_context_manager_cleans_up_on_exception(self):
        """Test that limits are restored even if exception is raised."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)

        limits = ResourceLimits(max_memory_mb=100)
        try:
            with apply_resource_limits(limits):
                raise ValueError("test error")
        except ValueError:
            pass

        after = resource.getrlimit(resource.RLIMIT_AS)
        assert original == after

    def test_nested_limits_restore_correctly(self):
        """Test that nested limit contexts restore correctly."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)

        outer_limits = ResourceLimits(max_memory_mb=200)
        inner_limits = ResourceLimits(max_memory_mb=100)

        with apply_resource_limits(outer_limits):
            mid = resource.getrlimit(resource.RLIMIT_AS)
            with apply_resource_limits(inner_limits):
                pass
            # After inner, should be back to outer's limit
            after_inner = resource.getrlimit(resource.RLIMIT_AS)
            assert after_inner == mid

        # After outer, should be back to original
        after = resource.getrlimit(resource.RLIMIT_AS)
        assert original == after


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestMemoryLimitEdgeCases:
    """Edge case tests for memory limits."""

    def test_large_limit_allows_large_allocation(self):
        """Test that large limit allows correspondingly large allocation."""
        limits = ResourceLimits(max_memory_mb=500)
        with apply_resource_limits(limits):
            # 100MB allocation should succeed with 500MB limit
            data = bytearray(100 * 1024 * 1024)
            assert len(data) == 100 * 1024 * 1024

    def test_allocation_fails_when_exceeds_limit(self):
        """Test that allocations exceeding the limit fail."""
        # Use a moderate limit where behavior is predictable
        limits = ResourceLimits(max_memory_mb=50)
        with pytest.raises(MemoryError):
            with apply_resource_limits(limits):
                # Try to allocate way more than limit
                _ = bytearray(500 * 1024 * 1024)


class TestWindowsWarning:
    """Tests for Windows platform warning."""

    def test_warns_on_windows_with_limits(self, monkeypatch):
        """Test that warning is issued on Windows when limits are configured."""
        monkeypatch.setattr("agex.resource_limits._PLATFORM_SUPPORTS_LIMITS", False)

        limits = ResourceLimits(max_memory_mb=500)
        with pytest.warns(RuntimeWarning, match="not supported on Windows"):
            with apply_resource_limits(limits):
                pass

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
        # Should not warn
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
        assert agent._resource_limits.max_memory_mb == 500

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
        assert agent._resource_limits.max_memory_mb == 500
        assert agent._resource_limits.max_open_files == 256

    def test_agent_default_is_unlimited(self):
        """Test that Agent defaults to no limits."""
        from agex import Agent, connect_llm

        agent = Agent(llm=connect_llm(provider="dummy"))
        assert agent._resource_limits.max_memory_mb is None
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

        assert cloned._resource_limits.max_memory_mb == 100
        assert cloned._resource_limits.max_open_files == 50

    def test_memory_error_propagates_through_eval(self):
        """Test that MemoryError from limit propagates through eval sandbox."""
        from gitkv import Live

        from agex.agent import Agent, clear_agent_registry
        from agex.eval.bridge import execute_sandboxed

        clear_agent_registry()
        agent = Agent(
            name="mem_limit_test",
            max_memory_mb=10,
        )

        @agent.fn()
        def allocate_huge():
            # Allocate 500MB - well beyond any 10MB headroom
            return bytearray(500 * 1024 * 1024)

        state = Live()
        state["__event_log__"] = []

        # The allocation should fail - sandtrap catches the MemoryError and
        # handle_result re-raises it directly
        with pytest.raises(MemoryError):
            with apply_resource_limits(agent._resource_limits):
                execute_sandboxed(
                    "result = allocate_huge()",
                    agent,
                    state,
                    eval_timeout_seconds=5.0,
                )


@pytest.mark.skipif(not check_platform_support(), reason="Unix only")
class TestDeltaBasedLimits:
    """Tests for delta-based memory limit behavior."""

    def test_delta_based_allows_existing_memory(self):
        """Test that delta-based limit doesn't count existing memory."""
        # Allocate some memory first
        existing = bytearray(50 * 1024 * 1024)  # 50MB

        # Set a 50MB limit - this should be 50MB of NEW allocations
        limits = ResourceLimits(max_memory_mb=50)
        with apply_resource_limits(limits):
            # Should be able to allocate 30MB more
            new_data = bytearray(30 * 1024 * 1024)
            assert len(new_data) == 30 * 1024 * 1024

        # Keep reference
        assert len(existing) > 0

    def test_delta_measures_at_context_entry(self):
        """Test that delta is measured when context is entered."""
        import resource

        limits = ResourceLimits(max_memory_mb=100)

        # Get limit before
        before_soft, _ = resource.getrlimit(resource.RLIMIT_AS)

        with apply_resource_limits(limits):
            # Get limit during
            during_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
            # Limit should be current memory + 100MB
            current_mem = _get_current_memory_bytes()
            expected = current_mem + 100 * 1024 * 1024
            # Allow some tolerance for memory fluctuation
            assert abs(during_soft - expected) < 10 * 1024 * 1024

    def test_sequential_contexts_get_fresh_headroom(self):
        """Test that each context entry gets fresh headroom measurement."""
        limits = ResourceLimits(max_memory_mb=50)

        # First context - allocate and hold
        data1 = None
        with apply_resource_limits(limits):
            data1 = bytearray(30 * 1024 * 1024)

        # Second context - should get fresh 50MB headroom from current state
        with apply_resource_limits(limits):
            data2 = bytearray(30 * 1024 * 1024)
            assert len(data2) == 30 * 1024 * 1024

        # Keep references
        assert data1 is not None
