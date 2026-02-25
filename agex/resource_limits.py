"""
Resource limits for in-process sandbox execution.

Provides file descriptor limiting using Unix RLIMIT_NOFILE on supported
platforms. Memory limits are handled by sandtrap's Policy.memory_limit.

Platform support:
- Linux/macOS: File descriptor limits via resource.setrlimit
- Windows: No support (warns and continues)
"""

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

# Unix only
if sys.platform != "win32":
    import resource
else:
    resource = None  # type: ignore


@dataclass
class ResourceLimits:
    """Resource limit configuration for task execution.

    Attributes:
        max_open_files: Maximum number of file descriptors the process can have.
            None means use system default.
    """

    max_open_files: int | None = None


_PLATFORM_SUPPORTS_LIMITS = sys.platform != "win32"


def check_platform_support() -> bool:
    """Check if the current platform supports resource limits.

    Returns:
        True on Linux/macOS, False on Windows.
    """
    return _PLATFORM_SUPPORTS_LIMITS


@contextmanager
def apply_resource_limits(limits: ResourceLimits) -> Iterator[None]:
    """Apply file descriptor limits for the duration of the context.

    On Unix: Sets RLIMIT_NOFILE, restores original limits on exit.
    On Windows: Warns and continues without limits.

    Args:
        limits: ResourceLimits configuration specifying file limits.

    Yields:
        None. The context manager applies limits during the with block.

    Raises:
        OSError: If file descriptor limit is exceeded.
    """
    if not _PLATFORM_SUPPORTS_LIMITS:
        if limits.max_open_files is not None:
            warnings.warn(
                "Resource limits (max_open_files) are not supported "
                "on Windows. Consider using Modal integration for containerized "
                "execution with resource limits.",
                RuntimeWarning,
                stacklevel=2,
            )
        yield
        return

    old_limits: dict[int, tuple[int, int]] = {}

    try:
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
                pass
