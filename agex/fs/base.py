"""Base filesystem interface and dataclasses.

Defines the common interface for filesystem implementations (VirtualFS, IsolatedFS).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class FileMetadata:
    """Metadata for a single file or directory.

    Attributes:
        size: File size in bytes (0 for directories).
        created_at: ISO 8601 timestamp when file was created (UTC).
        modified_at: ISO 8601 timestamp when file was last modified (UTC).
        is_dir: True if this is a directory, False for files.
    """

    size: int
    created_at: str
    modified_at: str
    is_dir: bool = False


@dataclass
class FileInfo:
    """Complete file information for UI display.

    Attributes:
        name: File or directory name (basename).
        path: Full path to file or directory.
        size: File size in bytes (0 for directories).
        created_at: ISO 8601 timestamp when created (UTC).
        modified_at: ISO 8601 timestamp when last modified (UTC).
        is_dir: True if this is a directory, False if file.
    """

    name: str
    path: str
    size: int
    created_at: str
    modified_at: str
    is_dir: bool


class FileSystem(abc.ABC):
    """Abstract base class for filesystem implementations."""

    @abc.abstractmethod
    def open(self, path: str, mode: str = "r", **kwargs: Any) -> Any:
        """Open a file.

        Args:
            path: File path to open.
            mode: File mode.
            kwargs: Additional arguments.

        Returns:
            File-like object.
        """
        pass

    @abc.abstractmethod
    def read(self, path: str) -> bytes:
        """Read entire file as bytes."""
        pass

    @abc.abstractmethod
    def write(self, path: str, content: bytes, mode: str = "w") -> None:
        """Write bytes to file.

        Args:
            path: File path to write.
            content: Bytes to write.
            mode: Write mode ('w' for write/overwrite, 'a' for append).
        """
        pass

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abc.abstractmethod
    def isfile(self, path: str) -> bool:
        """Check if path is a file."""
        pass

    @abc.abstractmethod
    def isdir(self, path: str) -> bool:
        """Check if path is a directory."""
        pass

    @abc.abstractmethod
    def islink(self, path: str) -> bool:
        """Check if path is a symbolic link."""
        pass

    @abc.abstractmethod
    def lexists(self, path: str) -> bool:
        """Check if path exists (without following symlinks)."""
        pass

    @abc.abstractmethod
    def samefile(self, path1: str, path2: str) -> bool:
        """Check if two paths refer to the same file."""
        pass

    @abc.abstractmethod
    def realpath(self, path: str) -> str:
        """Return the canonical path."""
        pass

    @abc.abstractmethod
    def list(self, path: str = ".") -> list[str]:
        """List directory contents (filenames only)."""
        pass

    @abc.abstractmethod
    def list_detailed(self, path: str = ".") -> list[FileInfo]:
        """List directory contents with details."""
        pass

    @abc.abstractmethod
    def remove(self, path: str) -> None:
        """Remove a file."""
        pass

    @abc.abstractmethod
    def remove_many(self, paths: list[str]) -> None:
        """Remove multiple files."""
        pass

    @abc.abstractmethod
    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory."""
        pass

    @abc.abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory tree."""
        pass

    @abc.abstractmethod
    def rename(self, src: str, dst: str) -> None:
        """Rename/move a file or directory."""
        pass

    @abc.abstractmethod
    def stat(self, path: str) -> FileMetadata:
        """Get file metadata."""
        pass

    @abc.abstractmethod
    def getsize(self, path: str) -> int:
        """Get file size in bytes."""
        pass

    @abc.abstractmethod
    def write_many(self, files: dict[str, bytes]) -> None:
        """Write multiple files at once."""
        pass

    @abc.abstractmethod
    def get_metadata_snapshot(self) -> dict[str, FileMetadata]:
        """Get snapshot of all file metadata."""
        pass
