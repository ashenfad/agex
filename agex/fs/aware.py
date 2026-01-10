"""Agent-aware VFS wrapper that emits events for file changes.

This module provides AgentAwareVFS, a wrapper around VirtualFS that:
1. Emits FileEvent for external API calls (user uploads/deletions)
2. Provides the same interface as VirtualFS for seamless use
"""

from __future__ import annotations

from typing import Callable

from agex.fs.virtual import FileInfo, FileMetadata, VirtualFS
from agex.state.core import State


class AgentAwareVFS:
    """VFS wrapper that emits events for agent/user visibility.

    Used by agent.fs() to provide file access that automatically
    logs FileEvents when files are modified externally.

    Example:
        >>> fs = agent.fs()
        >>> fs.write("data.csv", b"content")  # Emits FileEvent
        >>> fs.read("data.csv")  # No event (reads don't emit)
        b'content'
    """

    def __init__(
        self,
        vfs: VirtualFS,
        state: State,
        agent_name: str,
        on_event: Callable | None = None,
    ):
        """Initialize wrapper.

        Args:
            vfs: The underlying VirtualFS instance.
            state: The state to log events to.
            agent_name: Name of the agent for event attribution.
            on_event: Optional callback for event notification.
        """
        self._vfs = vfs
        self._state = state
        self._agent_name = agent_name
        self._on_event = on_event

    def _emit_event(
        self,
        added: list[str] | None = None,
        modified: list[str] | None = None,
        removed: list[str] | None = None,
    ) -> None:
        """Emit a FileEvent for user-initiated changes."""
        from agex.agent.events import FileEvent
        from agex.state.log import add_event_to_log

        # Only emit if there are actual changes
        if not (added or modified or removed):
            return

        event = FileEvent(
            agent_name=self._agent_name,
            file_source="user",
            added=added or [],
            modified=modified or [],
            removed=removed or [],
        )
        add_event_to_log(self._state, event, on_event=self._on_event)

    # Write operations - emit events

    def _merge(self) -> None:
        """Merge commits to make changes visible to other sessions."""
        if hasattr(self._state, "merge"):
            self._state.merge()

    def write(self, path: str, content: bytes) -> None:
        """Write file and emit event atomically."""
        is_new = not self._vfs.exists(path)
        self._emit_event(
            added=[path] if is_new else [],
            modified=[path] if not is_new else [],
        )
        self._vfs.write(path, content)  # Snapshots file + event together
        self._merge()

    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files and emit single event atomically."""
        added = [p for p in files if not self._vfs.exists(p)]
        modified = [p for p in files if self._vfs.exists(p)]
        self._emit_event(added=added, modified=modified)
        self._vfs.write_many(files)  # Snapshots files + event together
        self._merge()

    def remove(self, path: str) -> None:
        """Remove file and emit event atomically."""
        if not self._vfs.exists(path):
            raise FileNotFoundError(path)
        self._emit_event(removed=[path])
        self._vfs.remove(path)  # Snapshots removal + event together
        self._merge()

    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files and emit single event atomically."""
        missing = [p for p in paths if not self._vfs.exists(p)]
        if missing:
            raise FileNotFoundError(f"Files not found: {', '.join(missing)}")
        self._emit_event(removed=paths)
        self._vfs.remove_many(paths)  # Snapshots removals + event together
        self._merge()

    def rename(self, src: str, dst: str) -> None:
        """Rename file and emit event atomically."""
        if not self._vfs.exists(src):
            raise FileNotFoundError(src)
        self._emit_event(removed=[src], added=[dst])
        self._vfs.rename(src, dst)  # Snapshots rename + event together
        self._merge()

    # Read-only operations - delegate without events

    def read(self, path: str) -> bytes:
        """Read file (no event)."""
        return self._vfs.read(path)

    def exists(self, path: str) -> bool:
        """Check if path exists (no event)."""
        return self._vfs.exists(path)

    def isfile(self, path: str) -> bool:
        """Check if path is a file (no event)."""
        return self._vfs.isfile(path)

    def isdir(self, path: str) -> bool:
        """Check if path is a directory (no event)."""
        return self._vfs.isdir(path)

    def list(self, path: str = "/") -> list[str]:
        """List directory (no event)."""
        return self._vfs.list(path)

    def list_detailed(self, path: str = "/") -> list[FileInfo]:
        """List directory with metadata (no event)."""
        return self._vfs.list_detailed(path)

    def stat(self, path: str) -> FileMetadata:
        """Get file metadata (no event)."""
        return self._vfs.stat(path)

    def getsize(self, path: str) -> int:
        """Get file size (no event)."""
        return self._vfs.getsize(path)

    def open(self, path: str, mode: str = "r"):
        """Open file (no event - writes happen at close)."""
        return self._vfs.open(path, mode)

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        """Create directory (no event - directories are implicit)."""
        self._vfs.mkdir(path, exist_ok)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory tree (no event - directories are implicit)."""
        self._vfs.makedirs(path, exist_ok)
