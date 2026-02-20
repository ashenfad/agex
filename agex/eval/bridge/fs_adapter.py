"""
Filesystem adapter: wraps agex's FileSystem to satisfy sblite's FileSystem ABC.

agex's FileSystem is a superset of sblite's protocol. This adapter delegates
all calls and handles the stat() return type difference (agex returns
FileMetadata, sblite expects os.stat_result-like).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from sblite.fs.protocol import FileSystem as SbliteFileSystem

if TYPE_CHECKING:
    from agex.fs.base import FileSystem as AgexFileSystem


class _StatResult:
    """Minimal os.stat_result-like object built from agex FileMetadata."""

    def __init__(self, meta: Any) -> None:
        self.st_size = meta.size
        self.st_mode = 0o100644 if not getattr(meta, "is_dir", False) else 0o040755
        self.st_ino = 0
        self.st_dev = 0
        self.st_nlink = 1
        self.st_uid = os.getuid() if hasattr(os, "getuid") else 0
        self.st_gid = os.getgid() if hasattr(os, "getgid") else 0

        # Parse ISO timestamps to epoch seconds
        import time
        from datetime import datetime

        def _ts(iso_str: str) -> float:
            try:
                return datetime.fromisoformat(iso_str).timestamp()
            except Exception:
                return time.time()

        created = _ts(meta.created_at)
        modified = _ts(meta.modified_at)
        self.st_atime = modified
        self.st_mtime = modified
        self.st_ctime = created


class SbliteFS(SbliteFileSystem):
    """Adapter wrapping agex FileSystem for sblite's sandbox."""

    def __init__(self, agex_fs: "AgexFileSystem") -> None:
        self._fs = agex_fs

    def open(self, path: str, mode: str = "r", **kwargs: Any) -> Any:
        return self._fs.open(path, mode, **kwargs)

    def stat(self, path: str) -> Any:
        meta = self._fs.stat(path)
        return _StatResult(meta)

    def listdir(self, path: str) -> list[str]:
        return self._fs.list(path)

    def exists(self, path: str) -> bool:
        return self._fs.exists(path)

    def isfile(self, path: str) -> bool:
        return self._fs.isfile(path)

    def isdir(self, path: str) -> bool:
        return self._fs.isdir(path)

    def mkdir(
        self,
        path: str,
        mode: int = 0o777,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        self._fs.mkdir(path, parents=parents, exist_ok=exist_ok)

    def makedirs(self, path: str, mode: int = 0o777, *, exist_ok: bool = False) -> None:
        self._fs.makedirs(path, exist_ok=exist_ok)

    def remove(self, path: str) -> None:
        self._fs.remove(path)

    def rename(self, src: str, dst: str) -> None:
        self._fs.rename(src, dst)

    def getcwd(self) -> str:
        return self._fs.getcwd()

    def chdir(self, path: str) -> None:
        self._fs.chdir(path)
