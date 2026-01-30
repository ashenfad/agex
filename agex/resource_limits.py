"""
Resource limits for in-process sandbox execution.

Provides defense-in-depth resource limiting using Unix RLIMIT on supported
platforms. This protects against catastrophic single-task resource exhaustion
(e.g., `[0] * 10**9`) without requiring containers.

Memory limits use a delta-based approach: the limit is set to current process
memory + configured headroom, so `max_memory_mb=500` means each task can
allocate up to 500MB of additional memory.

Platform support:
- Linux/macOS: Full support via resource.setrlimit
- Windows: No support (warns and continues)

For stronger isolation guarantees, use the Modal integration.
"""

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import psutil

# Unix only
if sys.platform != "win32":
    import resource
else:
    resource = None  # type: ignore


@dataclass
class ResourceLimits:
    """Resource limit configuration for task execution.

    Attributes:
        max_memory_mb: Maximum memory headroom for each task in megabytes.
            This is added to current process memory to get the absolute limit.
            None means unlimited.
        max_open_files: Maximum number of file descriptors the process can have.
            None means use system default.
    """

    max_memory_mb: int | None = None
    max_open_files: int | None = None


_PLATFORM_SUPPORTS_LIMITS = sys.platform != "win32"


def check_platform_support() -> bool:
    """Check if the current platform supports resource limits.

    Returns:
        True on Linux/macOS, False on Windows.
    """
    return _PLATFORM_SUPPORTS_LIMITS


def _get_current_memory_bytes() -> int:
    """Get current process virtual memory size in bytes.

    Uses psutil for cross-platform memory measurement.

    Returns:
        Current virtual memory size in bytes.
    """
    return psutil.Process().memory_info().vms


@contextmanager
def apply_resource_limits(limits: ResourceLimits) -> Iterator[None]:
    """Apply resource limits for the duration of the context.

    Memory limit uses delta-based headroom: measures current process memory
    and sets limit = current + max_memory_mb. This gives the task a budget
    for new allocations without counting existing process memory.

    On Unix: Sets RLIMIT_AS and RLIMIT_NOFILE, restores original limits on exit.
    On Windows: Warns and continues without limits.

    Args:
        limits: ResourceLimits configuration specifying memory and file limits.

    Yields:
        None. The context manager applies limits during the with block.

    Raises:
        MemoryError: If memory limit is exceeded during execution.
        OSError: If file descriptor limit is exceeded.

    Example:
        limits = ResourceLimits(max_memory_mb=500, max_open_files=100)
        with apply_resource_limits(limits):
            # Code here is limited to 500MB additional memory
            # and 100 file descriptors
            result = execute_user_code()

    Note:
        Limits are process-wide on Unix. For concurrent tasks in the same
        process, the limit applies to all tasks combined. Size your limits
        according to expected concurrency, or use Modal for isolated execution.
    """
    if not _PLATFORM_SUPPORTS_LIMITS:
        if limits.max_memory_mb is not None or limits.max_open_files is not None:
            warnings.warn(
                "Resource limits (max_memory_mb, max_open_files) are not supported "
                "on Windows. Consider using Modal integration for containerized "
                "execution with resource limits.",
                RuntimeWarning,
                stacklevel=2,
            )
        yield
        return

    old_limits: dict[int, tuple[int, int]] = {}

    try:
        # Set memory limit with delta-based headroom
        if limits.max_memory_mb is not None:
            current_bytes = _get_current_memory_bytes()
            headroom_bytes = limits.max_memory_mb * 1024 * 1024
            limit_bytes = current_bytes + headroom_bytes

            old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
            old_limits[resource.RLIMIT_AS] = (old_soft, old_hard)
            # Only set soft limit - can't raise hard limit
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, old_hard))

        # Set file descriptor limit
        if limits.max_open_files is not None:
            old_limits[resource.RLIMIT_NOFILE] = resource.getrlimit(
                resource.RLIMIT_NOFILE
            )
            # Don't exceed system hard limit
            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            soft = min(limits.max_open_files, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

        yield

    finally:
        # Restore original limits
        for limit_type, (soft, hard) in old_limits.items():
            try:
                resource.setrlimit(limit_type, (soft, hard))
            except (ValueError, OSError):
                # May fail if we're already over the old limit
                # This is expected - just continue
                pass
