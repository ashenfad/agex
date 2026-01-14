"""State-backed virtual filesystem implementation.

Provides VirtualFS and VirtualFile classes for file operations backed by
agent state, enabling file persistence and versioning.
"""

from __future__ import annotations

import base64
import io
import os
import pickle
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agex.fs.context import vfs_defer_snapshots

if TYPE_CHECKING:
    from agex.state import State


from .base import FileInfo, FileMetadata, FileSystem


class VirtualFile:
    """File-like object that writes to state on close.

    Buffers content during write operations, then persists to state
    when the file is closed (either explicitly or via context manager).

    Attributes:
        path: The virtual filesystem path.
        mode: The file mode ('w', 'wb', 'a', 'ab').
    """

    def __init__(
        self, vfs: "VirtualFS", state: "State", key: str, path: str, mode: str
    ):
        """Initialize a writable virtual file.

        Args:
            vfs: The VirtualFS instance for metadata tracking.
            state: State backend for persistence.
            key: Encoded state key for this file.
            path: Original file path (for error messages).
            mode: File open mode.
        """
        self._vfs = vfs
        self._state = state
        self._key = key
        self._path = path
        self._mode = mode
        self._closed = False

        # Use BytesIO for binary, StringIO for text
        if "b" in mode:
            self._buffer: io.BytesIO | io.StringIO = io.BytesIO()
        else:
            self._buffer = io.StringIO()

        # For append mode, load existing content
        if "a" in mode:
            existing = state.get(key)
            if existing is not None:
                if "b" in mode:
                    self._buffer.write(existing)
                else:
                    self._buffer.write(existing.decode("utf-8"))

    def write(self, data: str | bytes) -> int:
        """Write data to the buffer.

        Args:
            data: Content to write (str for text mode, bytes for binary).

        Returns:
            Number of characters/bytes written.

        Raises:
            ValueError: If file is already closed.
        """
        if self._closed:
            raise ValueError(f"I/O operation on closed file: {self._path}")
        return self._buffer.write(data)  # type: ignore[arg-type]

    def read(self, size: int = -1) -> str | bytes:
        """Read is not supported for write-only files."""
        raise io.UnsupportedOperation("read")

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to a position in the buffer."""
        if self._closed:
            raise ValueError(f"I/O operation on closed file: {self._path}")
        return self._buffer.seek(offset, whence)

    def tell(self) -> int:
        """Return current position in the buffer."""
        if self._closed:
            raise ValueError(f"I/O operation on closed file: {self._path}")
        return self._buffer.tell()

    def flush(self) -> None:
        """Flush is a no-op (content persisted on close)."""
        pass

    def close(self) -> None:
        """Close the file and persist content to state with metadata tracking."""
        if self._closed:
            return

        content = self._buffer.getvalue()
        if isinstance(content, str):
            content = content.encode("utf-8")

        # Use VFS write to get proper metadata tracking
        # Check if we should defer snapshots (for agent execution)
        should_snapshot = not vfs_defer_snapshots.get()
        self._vfs.write(self._path, content, snapshot=should_snapshot)

        self._closed = True

    @property
    def closed(self) -> bool:
        """Return True if the file is closed."""
        return self._closed

    def __enter__(self) -> "VirtualFile":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class VirtualFS(FileSystem):
    """State-backed virtual filesystem with metadata tracking.

    Provides file operations backed by agent state. Each file is stored
    as a separate state key, enabling granular versioning with Versioned state.

    File metadata (size, creation time, modification time) is automatically
    tracked for all files and can be accessed via stat() or list_detailed().

    Files are stored as bytes. Text files are encoded as UTF-8.
    Directories are implicit (inferred from file paths, like S3).

    Example:
        >>> from agex.state import Live
        >>> state = Live()
        >>> vfs = VirtualFS(state)
        >>> vfs.write("data.csv", b"a,b,c\\n1,2,3")
        >>> vfs.read("data.csv")
        b'a,b,c\\n1,2,3'
        >>> vfs.list("/")
        ['data.csv']
        >>> meta = vfs.stat("data.csv")
        >>> print(f"Size: {meta.size} bytes, Created: {meta.created_at}")
    """

    PREFIX = "__vfs_"
    METADATA_KEY = "__vfs_metadata__"

    def __init__(self, state: "State"):
        """Initialize virtual filesystem backed by state.

        Args:
            state: State backend for file storage.
        """
        self._state = state

    def _now_iso(self) -> str:
        """Get current UTC timestamp as ISO 8601 string with milliseconds."""
        return datetime.now(timezone.utc).isoformat()

    def _get_metadata(self) -> dict[str, FileMetadata]:
        """Load metadata dict from state.

        Returns:
            Dict mapping normalized paths to FileMetadata objects.
        """
        metadata_bytes = self._state.get(self.METADATA_KEY)
        if metadata_bytes is None:
            return {}
        return pickle.loads(metadata_bytes)

    def _set_metadata(self, metadata: dict[str, FileMetadata]) -> None:
        """Save metadata dict to state.

        Args:
            metadata: Dict mapping normalized paths to FileMetadata objects.
        """
        self._state.set(self.METADATA_KEY, pickle.dumps(metadata))

    def _update_file_metadata(self, path: str, size: int, is_new: bool) -> None:
        """Update metadata for a file (create or modify).

        Args:
            path: Normalized file path.
            size: File size in bytes.
            is_new: True if this is a new file, False if modifying existing.
        """
        metadata = self._get_metadata()
        now = self._now_iso()

        if is_new:
            # Metadata keys must be normalized to match _encode_path
            path = self._normalize_path(path)

        if is_new or path not in metadata:
            # New file - set both created_at and modified_at
            metadata[path] = FileMetadata(
                size=size,
                created_at=now,
                modified_at=now,
            )
        else:
            # Existing file - preserve created_at, update modified_at and size
            metadata[path] = FileMetadata(
                size=size,
                created_at=metadata[path].created_at,
                modified_at=now,
            )

        self._set_metadata(metadata)

    def get_metadata_snapshot(self) -> dict[str, FileMetadata]:
        """Get a copy of current file metadata for change detection.

        Used to compare before/after agent turns to detect file changes.

        Returns:
            Copy of metadata dict (safe to modify).
        """
        return self._get_metadata().copy()

    def _normalize_path(self, path: str) -> str:
        """Normalize file path for consistent internal keys.

        Args:
            path: File path (e.g., "./data.csv").

        Returns:
            Normalized path (e.g., "data.csv").
        """
        # Normalize path to canonical form
        # This handles ./a.py vs a.py, and a/./b vs a/b
        path = os.path.normpath(path).replace("\\", "/")

        # Remove leading slashes, handle empty/root
        path = path.lstrip("/") or "/"
        return path

    def _encode_path(self, path: str) -> str:
        """Convert file path to state key.

        Uses base32 encoding for safe, reversible path encoding.

        Args:
            path: File path (e.g., "shared/data.csv").

        Returns:
            State key (e.g., "__vfs_ONQWIZI...").
        """
        path = self._normalize_path(path)
        encoded = base64.b32encode(path.encode()).decode().rstrip("=")
        return f"{self.PREFIX}{encoded}"

    def _decode_path(self, key: str) -> str:
        """Convert state key back to file path.

        Args:
            key: State key (e.g., "__vfs_ONQWIZI...").

        Returns:
            File path (e.g., "shared/data.csv").
        """
        encoded = key[len(self.PREFIX) :]
        # Add padding back
        padding = (8 - len(encoded) % 8) % 8
        encoded += "=" * padding
        return base64.b32decode(encoded).decode()

    def _is_vfs_key(self, key: str) -> bool:
        """Check if a state key is a VFS file."""
        return key.startswith(self.PREFIX)

    def open(
        self, path: str, mode: str = "r", **kwargs: object
    ) -> VirtualFile | io.BytesIO | io.StringIO:
        """Open a file, returning a file-like object.

        Args:
            path: File path to open.
            mode: File mode ('r', 'rb', 'w', 'wb', 'a', 'ab').
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            File-like object for reading or writing.

        Raises:
            FileNotFoundError: If reading a file that doesn't exist.
            ValueError: If mode is invalid.
        """
        key = self._encode_path(path)

        if "r" in mode and "w" not in mode and "a" not in mode and "x" not in mode:
            # Read mode
            content = self._state.get(key)
            if content is None:
                raise FileNotFoundError(path)

            if "b" in mode:
                return io.BytesIO(content)
            else:
                return io.StringIO(content.decode("utf-8"))

        elif "w" in mode or "a" in mode or "x" in mode:
            # Write, append, or exclusive creation mode
            if "x" in mode and self.exists(path):
                raise FileExistsError(f"[Errno 17] File exists: '{path}'")

            return VirtualFile(self, self._state, key, path, mode)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def read(self, path: str) -> bytes:
        """Read file contents as bytes.

        Args:
            path: File path to read.

        Returns:
            File contents as bytes.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        key = self._encode_path(path)
        content = self._state.get(key)
        if content is None:
            raise FileNotFoundError(path)
        return content

    def write(self, path: str, content: bytes, snapshot: bool = True) -> None:
        """Write bytes to a file.

        Args:
            path: File path to write.
            content: Content to write (must be bytes).
            snapshot: If True, create snapshot after write (for external API).
                     If False, defer snapshot (for agent code via patching).
        """
        if not isinstance(content, bytes):
            raise TypeError(f"Expected bytes, got {type(content).__name__}")
        key = self._encode_path(path)

        # Check if file exists to determine if this is new or modified
        is_new = key not in self._state

        # Write content
        self._state.set(key, content)

        # Update metadata
        self._update_file_metadata(path, len(content), is_new)

        # Snapshot if requested and state supports it
        if snapshot and hasattr(self._state, "snapshot"):
            self._state.snapshot()

    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files atomically.

        Creates a single snapshot (if using Versioned state) after writing
        all files, providing cleaner version history and better atomicity.

        Args:
            files: Mapping of file path to content (bytes).

        Raises:
            TypeError: If any content is not bytes.

        Example:
            >>> vfs.write_many({
            ...     "data/file1.txt": b"content1",
            ...     "data/file2.txt": b"content2",
            ... })
        """
        # Validate all first
        for path, content in files.items():
            if not isinstance(content, bytes):
                raise TypeError(
                    f"Expected bytes for '{path}', got {type(content).__name__}"
                )

        # Write all files and update metadata
        for path, content in files.items():
            key = self._encode_path(path)
            is_new = key not in self._state
            self._state.set(key, content)
            self._update_file_metadata(path, len(content), is_new)

        # Create single snapshot if using versioned state
        if hasattr(self._state, "snapshot"):
            self._state.snapshot()

    def list(self, path: str = "/") -> list[str]:
        """List directory contents.

        Returns immediate children of the directory (files and subdirectories).
        Directories are implicit (inferred from file paths).

        Args:
            path: Directory path to list.

        Returns:
            List of file/directory names in the directory.
        """
        # Normalize path
        path = self._normalize_path(path)

        # Adjust logic to match original list expectation (empty string for root)
        if path == "." or path == "/":
            path = ""
        else:
            path = path + "/"

        results: set[str] = set()
        for key in self._state.keys():
            # Skip metadata key
            if key == self.METADATA_KEY:
                continue

            if not self._is_vfs_key(key):
                continue

            file_path = self._decode_path(key)
            file_path = file_path.lstrip("/")

            if path and not file_path.startswith(path):
                continue

            # Get the remainder after the directory prefix
            remainder = file_path[len(path) :]
            if not remainder:
                continue

            # Get immediate child (first path component)
            if "/" in remainder:
                results.add(remainder.split("/")[0])  # Subdirectory
            else:
                results.add(remainder)  # File

        return sorted(results)

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists.

        For files, checks if the exact path exists.
        For directories, checks if any file has that path as prefix.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        # Check for exact file match
        key = self._encode_path(path)
        if key in self._state:
            return True

        # Check for directory (any file with this prefix)
        path = self._normalize_path(path)
        if path == "." or path == "/":
            return True

        prefix = path + "/"
        for k in self._state.keys():
            for k in self._state.keys():
                # Skip metadata key
                if k == self.METADATA_KEY:
                    continue

                if self._is_vfs_key(k):
                    file_path = self._decode_path(k).lstrip("/")
                    if file_path.startswith(prefix):
                        return True

        return False

    def isfile(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path: Path to check.

        Returns:
            True if path is a file, False otherwise.
        """
        key = self._encode_path(path)
        return key in self._state

    def isdir(self, path: str) -> bool:
        """Check if path is a directory.

        Directories are implicit (any path with files underneath).

        Args:
            path: Path to check.

        Returns:
            True if path is a directory, False otherwise.
        """
        # Root is always a directory
        path = self._normalize_path(path)
        if path == "." or path == "/":
            return True

        prefix = path + "/"
        for key in self._state.keys():
            # Skip metadata key
            if key == self.METADATA_KEY:
                continue

            if self._is_vfs_key(key):
                file_path = self._decode_path(key).lstrip("/")
                if file_path.startswith(prefix):
                    return True

        return False

    def getsize(self, path: str) -> int:
        """Get file size in bytes.

        Args:
            path: File path.

        Returns:
            Size in bytes.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        content = self.read(path)
        return len(content)

    def remove(self, path: str, snapshot: bool = True) -> None:
        """Remove a file.

        Args:
            path: File path to remove.
            snapshot: If True, create snapshot after remove (for external API).
                     If False, defer snapshot (for agent code via patching).

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        key = self._encode_path(path)
        removed = self._state.remove(key)
        if not removed:
            raise FileNotFoundError(path)

        # Remove from metadata
        path = self._normalize_path(path)
        metadata = self._get_metadata()
        metadata.pop(path, None)
        self._set_metadata(metadata)

        # Snapshot if requested and state supports it (after successful removal)
        if snapshot and hasattr(self._state, "snapshot"):
            self._state.snapshot()

    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files atomically.

        Creates a single snapshot (if using Versioned state) after removing
        all files, providing cleaner version history.

        Args:
            paths: List of file paths to remove.

        Raises:
            FileNotFoundError: If any file doesn't exist.

        Example:
            >>> vfs.remove_many(["temp/file1.txt", "temp/file2.txt"])
        """
        # Validate all exist first
        missing = [p for p in paths if not self.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"File{'s' if len(missing) > 1 else ''} not found: {', '.join(missing)}"
            )

        # Remove all files and metadata
        metadata = self._get_metadata()
        paths = [self._normalize_path(p) for p in paths]
        for path in paths:
            key = self._encode_path(path)
            self._state.remove(key)
            metadata.pop(path, None)
        self._set_metadata(metadata)

        # Create single snapshot if using versioned state
        if hasattr(self._state, "snapshot"):
            self._state.snapshot()

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        """Create a directory (no-op for virtual FS).

        Directories are implicit, so this does nothing.

        Args:
            path: Directory path.
            exist_ok: Ignored (always succeeds).
        """
        pass

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory tree (no-op for virtual FS).

        Directories are implicit, so this does nothing.

        Args:
            path: Directory path.
            exist_ok: Ignored (always succeeds).
        """
        pass

    def rename(self, src: str, dst: str, snapshot: bool = True) -> None:
        """Rename/move a file.

        Args:
            src: Source file path.
            dst: Destination file path.
            snapshot: If True, create snapshot after rename (for external API).
                     If False, defer snapshot (for agent code via patching).

        Raises:
            FileNotFoundError: If source doesn't exist.
        """
        content = self.read(src)

        # Preserve source file's metadata (especially created_at)
        metadata = self._get_metadata()
        src_meta = metadata.get(src)

        # Write destination (this will create new metadata)
        self.write(dst, content, snapshot=False)

        # Reload metadata (write() modified it)
        metadata = self._get_metadata()

        # If source had metadata, preserve its created_at
        if src_meta:
            dst_meta = metadata[dst]  # write() created this
            metadata[dst] = FileMetadata(
                size=dst_meta.size,
                created_at=src_meta.created_at,  # Preserve original
                modified_at=dst_meta.modified_at,  # Use current time
            )
            self._set_metadata(metadata)

        # Remove source (this removes source metadata too)
        self.remove(src, snapshot=False)

        # Snapshot once at the end if requested
        if snapshot and hasattr(self._state, "snapshot"):
            self._state.snapshot()

    def stat(self, path: str) -> FileMetadata:
        """Get metadata for a specific file.

        Args:
            path: File path.

        Returns:
            FileMetadata object with size and timestamps.

        Raises:
            FileNotFoundError: If file doesn't exist.

        Example:
            >>> meta = vfs.stat("data.csv")
            >>> print(f"Size: {meta.size} bytes")
            >>> print(f"Created: {meta.created_at}")
        """
        # Check file exists
        if not self.isfile(path):
            raise FileNotFoundError(path)

        path = self._normalize_path(path)
        metadata = self._get_metadata()
        return metadata[path]

    def list_detailed(self, path: str = "/") -> list[FileInfo]:
        """List directory contents with full file metadata.

        Returns FileInfo objects for each file and subdirectory with complete
        metadata (size, timestamps). Useful for UI file viewers.

        Args:
            path: Directory path to list (default: root).

        Returns:
            List of FileInfo objects sorted by name.

        Example:
            >>> files = vfs.list_detailed("/shared")
            >>> for f in files:
            ...     print(f"{f.name:20} {f.size:>10} {f.modified_at}")
        """
        # Get file list from existing list() method
        names = self.list(path)

        # Normalize path similar to list() methods to build correct full paths
        normalized_path = path.strip()
        if normalized_path == "." or normalized_path == "./":
            normalized_path = ""
        else:
            normalized_path = normalized_path.strip("/")

        # Load all metadata once
        all_metadata = self._get_metadata()

        # Build FileInfo objects
        result = []
        for name in names:
            # Construct full path
            if not normalized_path:
                full_path = name
            else:
                full_path = f"{normalized_path}/{name}"

            # Check if it's a directory
            is_dir = self.isdir(full_path)

            if is_dir:
                # Directory - no metadata, use dummy values
                result.append(
                    FileInfo(
                        name=name,
                        path=full_path,
                        size=0,
                        created_at="",
                        modified_at="",
                        is_dir=True,
                    )
                )
            else:
                # File - get metadata
                meta = all_metadata.get(full_path)
                if meta:
                    result.append(
                        FileInfo(
                            name=name,
                            path=full_path,
                            size=meta.size,
                            created_at=meta.created_at,
                            modified_at=meta.modified_at,
                            is_dir=False,
                        )
                    )
                else:
                    # File exists but has no metadata (shouldn't happen with new code)
                    # Create default metadata
                    content = self.read(full_path)
                    result.append(
                        FileInfo(
                            name=name,
                            path=full_path,
                            size=len(content),
                            created_at="",
                            modified_at="",
                            is_dir=False,
                        )
                    )

        return result
