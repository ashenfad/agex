"""Agent-aware VFS wrapper that emits events for file changes.

This module provides AgentAwareFS, a wrapper around VirtualFS that:
1. Emits FileEvent for external API calls (user uploads/deletions)
2. Provides the same interface as VirtualFS for seamless use
"""

from __future__ import annotations

from typing import Callable

from agex.fs.base import FileSystem
from agex.fs.virtual import FileInfo, FileMetadata
from agex.state.core import State


class AgentAwareFS(FileSystem):
    """FS wrapper that emits events for agent/user visibility.

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
        fs: FileSystem,
        state: State,
        agent_name: str,
        on_event: Callable | None = None,
    ):
        """Initialize wrapper.

        Args:
            fs: The underlying FileSystem instance.
            state: The state to log events to.
            agent_name: Name of the agent for event attribution.
            on_event: Optional callback for event notification.
        """
        self._fs = fs
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
        is_new = not self._fs.exists(path)
        self._emit_event(
            added=[path] if is_new else [],
            modified=[path] if not is_new else [],
        )
        self._fs.write(path, content)  # Snapshots file + event together
        self._merge()

    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files and emit single event atomically."""
        added = [p for p in files if not self._fs.exists(p)]
        modified = [p for p in files if self._fs.exists(p)]
        self._emit_event(added=added, modified=modified)
        self._fs.write_many(files)  # Snapshots files + event together
        self._merge()

    def remove(self, path: str) -> None:
        """Remove file and emit event atomically."""
        if not self._fs.exists(path):
            raise FileNotFoundError(path)
        self._emit_event(removed=[path])
        self._fs.remove(path)  # Snapshots removal + event together
        self._merge()

    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files and emit single event atomically."""
        missing = [p for p in paths if not self._fs.exists(p)]
        if missing:
            raise FileNotFoundError(f"Files not found: {', '.join(missing)}")
        self._emit_event(removed=paths)
        self._fs.remove_many(paths)  # Snapshots removals + event together
        self._merge()

    def rename(self, src: str, dst: str) -> None:
        """Rename file and emit event atomically."""
        if not self._fs.exists(src):
            raise FileNotFoundError(src)
        self._emit_event(removed=[src], added=[dst])
        self._fs.rename(src, dst)  # Snapshots rename + event together
        self._merge()

    # Read-only operations - delegate without events

    def read(self, path: str) -> bytes:
        """Read file (no event)."""
        return self._fs.read(path)

    def exists(self, path: str) -> bool:
        """Check if path exists (no event)."""
        return self._fs.exists(path)

    def isfile(self, path: str) -> bool:
        """Check if path is a file (no event)."""
        return self._fs.isfile(path)

    def isdir(self, path: str) -> bool:
        """Check if path is a directory (no event)."""
        return self._fs.isdir(path)

    def islink(self, path: str) -> bool:
        """Check if path is a symbolic link (no event)."""
        # Note: Underlying FS may not support islink (like VirtualFS)
        # but we provide the method for interface consistency.
        if hasattr(self._fs, "islink"):
            return self._fs.islink(path)
        return False

    def lexists(self, path: str) -> bool:
        """Check if path exists (no event)."""
        if hasattr(self._fs, "lexists"):
            return self._fs.lexists(path)
        return self._fs.exists(path)

    def samefile(self, path1: str, path2: str) -> bool:
        """Check if two paths refer to the same file (no event)."""
        if hasattr(self._fs, "samefile"):
            return self._fs.samefile(path1, path2)
        # Fallback for FS that don't implement samefile
        return self._fs.exists(path1) and self._fs.exists(path2) and path1 == path2

    def realpath(self, path: str) -> str:
        """Return the canonical path (no event)."""
        if hasattr(self._fs, "realpath"):
            return self._fs.realpath(path)
        # Fallback for FS that don't implement realpath
        return path

    def list(self, path: str = "/", recursive: bool = False) -> list[str]:
        """List directory (no event)."""
        return self._fs.list(path, recursive=recursive)

    def list_detailed(self, path: str = "/", recursive: bool = False) -> list[FileInfo]:
        """List directory with metadata (no event)."""
        return self._fs.list_detailed(path, recursive=recursive)

    def stat(self, path: str) -> FileMetadata:
        """Get file metadata (no event)."""
        return self._fs.stat(path)

    def getsize(self, path: str) -> int:
        """Get file size (no event)."""
        return self._fs.getsize(path)

    def open(self, path: str, mode: str = "r"):
        """Open file (no event - writes happen at close)."""
        return self._fs.open(path, mode)

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        """Create directory (no event - directories are implicit)."""
        self._fs.mkdir(path, exist_ok)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory tree (no event - directories are implicit)."""
        self._fs.makedirs(path, exist_ok)

    def get_metadata_snapshot(self) -> dict[str, FileMetadata]:
        """Get snapshot of all file metadata."""
        return self._fs.get_metadata_snapshot()
