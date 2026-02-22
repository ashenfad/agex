"""Agent-aware FS wrapper that emits events for file changes.

Wraps any FileSystem and emits FileEvent on mutating operations
(write, remove, rename). Read operations delegate transparently.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Callable


class AgentAwareFS:
    """FS wrapper that emits events for agent/user visibility.

    Used by agent.fs() to provide file access that automatically
    logs FileEvents when files are modified externally.

    Write operations emit events; everything else delegates to the
    underlying FS via __getattr__.

    Example:
        >>> fs = agent.fs()
        >>> fs.write("data.csv", b"content")  # Emits FileEvent
        >>> fs.read("data.csv")  # No event (reads don't emit)
        b'content'
    """

    def __init__(
        self,
        fs: Any,
        state: MutableMapping[str, bytes],
        agent_name: str,
        on_event: Callable | None = None,
    ):
        self._fs = fs
        self._state = state
        self._agent_name = agent_name
        self._on_event = on_event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fs, name)

    def _emit_event(
        self,
        added: list[str] | None = None,
        modified: list[str] | None = None,
        removed: list[str] | None = None,
    ) -> None:
        """Emit a FileEvent for user-initiated changes."""
        from agex.agent.events import FileEvent
        from agex.state.log import add_event_to_log

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

    def write(self, path: str, content: bytes) -> None:
        """Write file and emit event."""
        is_new = not self._fs.exists(path)
        self._emit_event(
            added=[path] if is_new else [],
            modified=[path] if not is_new else [],
        )
        self._fs.write(path, content)

    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files and emit single event."""
        added = [p for p in files if not self._fs.exists(p)]
        modified = [p for p in files if self._fs.exists(p)]
        self._emit_event(added=added, modified=modified)
        self._fs.write_many(files)

    def remove(self, path: str) -> None:
        """Remove file and emit event."""
        if not self._fs.exists(path):
            raise FileNotFoundError(path)
        self._emit_event(removed=[path])
        self._fs.remove(path)

    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files and emit single event."""
        missing = [p for p in paths if not self._fs.exists(p)]
        if missing:
            raise FileNotFoundError(f"Files not found: {', '.join(missing)}")
        self._emit_event(removed=paths)
        self._fs.remove_many(paths)

    def rename(self, src: str, dst: str) -> None:
        """Rename file and emit event."""
        if not self._fs.exists(src):
            raise FileNotFoundError(src)
        self._emit_event(removed=[src], added=[dst])
        self._fs.rename(src, dst)
