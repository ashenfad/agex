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
        assert limits.max_memory_mb is None
        assert limits.max_open_files is None

    def test_custom_limits(self):
        """Test custom limits are stored correctly."""
        limits = ResourceLimits(max_memory_mb=500, max_open_files=100)
        assert limits.max_memory_mb == 500
        assert limits.max_open_files == 100


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

    def test_memory_limit_resets_after_context(self):
        """Test that memory limit is restored after context."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)

        limits = ResourceLimits(max_memory_mb=100)
        with apply_resource_limits(limits):
            pass

        after = resource.getrlimit(resource.RLIMIT_AS)
        assert original == after

    def test_file_limit_allows_small_count(self):
        """Test that opening few files succeeds within limit."""
        import os
        import tempfile

        # Use a high enough limit that doesn't interfere with pytest's file usage
        # Note: pytest and the test runner have many files open already
        limits = ResourceLimits(max_open_files=256)
        with apply_resource_limits(limits):
            # Open a few temp files - should succeed
            files = []
            paths = []
            for _ in range(5):
                f = tempfile.NamedTemporaryFile(delete=False)
                files.append(f)
                paths.append(f.name)
            for f in files:
                f.close()
            # Clean up temp files
            for path in paths:
                os.unlink(path)

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


class TestWindowsWarning:
    """Tests for Windows platform warning."""

    def test_warns_on_windows_with_limits(self, monkeypatch):
        """Test that warning is issued on Windows when limits are configured."""
        # Mock Windows platform
        monkeypatch.setattr("agex.resource_limits._PLATFORM_SUPPORTS_LIMITS", False)

        limits = ResourceLimits(max_memory_mb=500)
        with pytest.warns(RuntimeWarning, match="not supported on Windows"):
            with apply_resource_limits(limits):
                pass

    def test_no_warning_without_limits(self, monkeypatch):
        """Test that no warning is issued when no limits are configured."""
        # Mock Windows platform
        monkeypatch.setattr("agex.resource_limits._PLATFORM_SUPPORTS_LIMITS", False)

        limits = ResourceLimits()  # No limits
        # Should not warn
        with apply_resource_limits(limits):
            pass
