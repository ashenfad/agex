"""Isolated filesystem with path restriction and optional tracking.

Provides real filesystem access restricted to a specific directory,
with optional file change tracking via FileEvents.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from agex.state import State

from .base import FileInfo, FileMetadata, FileSystem


class IsolatedFS(FileSystem):
    """FileSystem interface restricted to a root directory.

    All file operations are validated to ensure paths stay within the
    configured root directory. Optionally tracks file changes for FileEvents.

    Security features:
    - Rejects paths outside root directory
    - Handles symlinks securely (validates resolved paths)
    - Normalizes all path variations (../, ./, etc.)
    """

    METADATA_KEY = "__isolated_fs_metadata__"

    def __init__(self, root: str, state: State | None = None):
        """Initialize isolated filesystem.

        Args:
            root: Absolute path to root directory (must exist).
            state: Optional state for metadata tracking.

        Raises:
            ValueError: If root is not an absolute path or doesn't exist.
        """
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"Root must be absolute path: {root}")

        self.root = root_path.resolve()
        if not self.root.exists():
            raise ValueError(f"Root directory does not exist: {root}")
        if not self.root.is_dir():
            raise ValueError(f"Root must be a directory: {root}")

        self._state = state

    def _validate_path(self, path: str | Path) -> Path:
        """Validate and resolve path to ensure it's within root.

        Args:
            path: File path to validate (absolute or relative to root).

        Returns:
            Resolved absolute path within root.

        Raises:
            PermissionError: If path escapes root directory.
        """
        p = Path(path)

        # Reject absolute paths that don't start with root
        if p.is_absolute():
            try:
                # Check if it's already within root
                p.relative_to(self.root)
            except ValueError:
                raise PermissionError("Path outside root")

        # Resolve relative to root (handles .., symlinks, etc.)
        resolved = (self.root / p).resolve()

        # Final boundary check
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PermissionError("Path outside root")

        return resolved

    def _get_metadata(self) -> dict[str, FileMetadata]:
        """Get current metadata dictionary from state."""
        if self._state is None:
            return {}
        raw = self._state.get(self.METADATA_KEY)
        if raw is None:
            return {}
        return pickle.loads(raw)

    def _set_metadata(self, metadata: dict[str, FileMetadata]) -> None:
        """Store metadata dictionary in state."""
        if self._state is not None:
            self._state.set(self.METADATA_KEY, pickle.dumps(metadata))

    def _update_file_metadata(self, path: str, size: int) -> None:
        """Update metadata for a file after modification.

        Args:
            path: Normalized file path (relative to root).
            size: File size in bytes.
        """
        if self._state is None:
            return

        metadata = self._get_metadata()
        resolved = self._validate_path(path)
        rel_path = str(resolved.relative_to(self.root))

        # Get current timestamps
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        if rel_path not in metadata:
            # New file
            metadata[rel_path] = FileMetadata(
                size=size,
                created_at=now,
                modified_at=now,
            )
        else:
            # Existing file - preserve created_at
            metadata[rel_path] = FileMetadata(
                size=size,
                created_at=metadata[rel_path].created_at,
                modified_at=now,
            )

        self._set_metadata(metadata)

    def _remove_file_metadata(self, path: str) -> None:
        """Remove metadata for a deleted file.

        Args:
            path: File path that was deleted.
        """
        if self._state is None:
            return

        metadata = self._get_metadata()
        resolved = self._validate_path(path)
        rel_path = str(resolved.relative_to(self.root))

        if rel_path in metadata:
            del metadata[rel_path]
            self._set_metadata(metadata)

    def open(self, path: str, mode: str = "r", **kwargs: Any) -> Any:
        """Open a file within the isolated filesystem.

        Args:
            path: File path to open (relative to root).
            mode: File mode ('r', 'w', 'rb', 'wb', etc.).
            **kwargs: Additional arguments passed to open().

        Returns:
            File object.

        Raises:
            PermissionError: If path is outside root.
            FileNotFoundError: If file doesn't exist (read mode).
        """
        resolved = self._validate_path(path)

        # Open the file
        f = open(resolved, mode, **kwargs)

        # Track metadata for write/append modes
        if any(m in mode for m in ["w", "a", "+"]):
            # Register callback to update metadata on close
            original_close = f.close

            def tracked_close():
                original_close()
                if resolved.exists():
                    self._update_file_metadata(path, resolved.stat().st_size)

            f.close = tracked_close

        return f

    def read(self, path: str) -> bytes:
        """Read entire file as bytes.

        Args:
            path: File path to read.

        Returns:
            File contents as bytes.

        Raises:
            PermissionError: If path is outside root.
            FileNotFoundError: If file doesn't exist.
        """
        resolved = self._validate_path(path)
        return resolved.read_bytes()

    def write(self, path: str, content: bytes) -> None:
        """Write bytes to file, creating parent directories if needed.

        Args:
            path: File path to write.
            content: Bytes to write.

        Raises:
            PermissionError: If path is outside root.
        """
        resolved = self._validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)
        self._update_file_metadata(path, len(content))

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists within root.

        Raises:
            PermissionError: If path is outside root.
        """
        resolved = self._validate_path(path)
        return resolved.exists()

    def isfile(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path: Path to check.

        Returns:
            True if path is a file.

        Raises:
            PermissionError: If path is outside root.
        """
        resolved = self._validate_path(path)
        return resolved.is_file()

    def isdir(self, path: str) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a directory.

        Raises:
            PermissionError: If path is outside root.
        """
        resolved = self._validate_path(path)
        return resolved.is_dir()

    def listdir(self, path: str = ".") -> list[str]:
        """List directory contents.

        Args:
            path: Directory path to list (default: root).

        Returns:
            List of filenames in directory.

        Raises:
            PermissionError: If path is outside root.
            NotADirectoryError: If path is not a directory.
        """
        resolved = self._validate_path(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [p.name for p in resolved.iterdir()]

    def remove(self, path: str) -> None:
        """Remove a file.

        Args:
            path: File path to remove.

        Raises:
            PermissionError: If path is outside root.
            FileNotFoundError: If file doesn't exist.
            IsADirectoryError: If path is a directory.
        """
        resolved = self._validate_path(path)
        if resolved.is_dir():
            raise IsADirectoryError(f"Is a directory: {path}")
        resolved.unlink()
        self._remove_file_metadata(path)

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory.

        Args:
            path: Directory path to create.
            parents: Create parent directories if needed.
            exist_ok: Don't raise error if directory exists.

        Raises:
            PermissionError: If path is outside root.
            FileExistsError: If directory exists and exist_ok=False.
        """
        resolved = self._validate_path(path)
        resolved.mkdir(parents=parents, exist_ok=exist_ok)

    def rename(self, src: str, dst: str) -> None:
        """Rename/move a file or directory.

        Args:
            src: Source path.
            dst: Destination path.

        Raises:
            PermissionError: If either path is outside root.
            FileNotFoundError: If source doesn't exist.
        """
        src_resolved = self._validate_path(src)
        dst_resolved = self._validate_path(dst)

        # Track metadata change if it's a file
        if src_resolved.is_file():
            self._remove_file_metadata(src)
            size = src_resolved.stat().st_size
            src_resolved.rename(dst_resolved)
            self._update_file_metadata(dst, size)
        else:
            src_resolved.rename(dst_resolved)

    def stat(self, path: str) -> FileMetadata:
        """Get file metadata.

        Args:
            path: File path to stat.

        Returns:
            FileMetadata with size and timestamps.

        Raises:
            PermissionError: If path is outside root.
            FileNotFoundError: If file doesn't exist.
        """
        resolved = self._validate_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"No such file: {path}")

        stat_result = resolved.stat()
        from datetime import datetime, timezone

        return FileMetadata(
            size=stat_result.st_size,
            created_at=datetime.fromtimestamp(
                stat_result.st_ctime, tz=timezone.utc
            ).isoformat(),
            modified_at=datetime.fromtimestamp(
                stat_result.st_mtime, tz=timezone.utc
            ).isoformat(),
        )

    def get_metadata_snapshot(self) -> dict[str, FileMetadata]:
        """Get current metadata snapshot for all tracked files.

        Returns:
            Dictionary mapping file paths to metadata.
        """
        return self._get_metadata().copy()

    # VirtualFS-compatible aliases for AgentAwareFS

    def list(self, path: str = ".") -> list[str]:
        """List directory contents (alias for listdir)."""
        return self.listdir(path)

    def getsize(self, path: str) -> int:
        """Get file size in bytes."""
        return self.stat(path).size

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory tree (alias for mkdir with parents=True)."""
        self.mkdir(path, parents=True, exist_ok=exist_ok)

    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files at once.

        Args:
            files: Dictionary mapping file paths to content bytes.
        """
        for path, content in files.items():
            self.write(path, content)

    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files at once.

        Args:
            paths: List of file paths to remove.
        """
        for path in paths:
            self.remove(path)

    def list_detailed(self, path: str = ".") -> list[FileInfo]:
        """List directory with detailed file information.

        Args:
            path: Directory path to list.

        Returns:
            List of FileInfo objects with file info (name, size, timestamps).
        """
        resolved = self._validate_path(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        result = []
        for item in resolved.iterdir():
            rel_path = str(item.relative_to(self.root))
            if item.is_file():
                stat_info = item.stat()
                from datetime import datetime, timezone

                result.append(
                    FileInfo(
                        name=item.name,
                        path=rel_path,
                        is_dir=False,
                        size=stat_info.st_size,
                        created_at=datetime.fromtimestamp(
                            stat_info.st_ctime, tz=timezone.utc
                        ).isoformat(),
                        modified_at=datetime.fromtimestamp(
                            stat_info.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    )
                )
            else:
                result.append(
                    FileInfo(
                        name=item.name,
                        path=rel_path,
                        is_dir=True,
                        size=0,
                        created_at="",
                        modified_at="",
                    )
                )

        return result
