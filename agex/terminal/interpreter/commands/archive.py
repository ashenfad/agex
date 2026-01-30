"""
Archive commands for the terminal interpreter.

Supports tar, gzip, gunzip, zip, and unzip operations.
"""

import gzip as gzip_module
import io
import os
import tarfile
import zipfile
from typing import TextIO

from agex.fs.base import FileSystem
from agex.terminal.interpreter.datatypes import TerminalError

from .filesystem import CommandArgParser


def gzip(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Compress files using gzip."""
    parser = CommandArgParser(prog="gzip", add_help=False)
    parser.add_argument("-d", "--decompress", action="store_true", help="Decompress")
    parser.add_argument("-k", "--keep", action="store_true", help="Keep original files")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"gzip: unknown arguments: {unknown}")

    if not parsed.files:
        raise TerminalError("gzip: no files specified")

    for path in parsed.files:
        try:
            if parsed.decompress:
                # Decompress
                if not path.endswith(".gz"):
                    raise TerminalError(f"gzip: {path}: unknown suffix -- ignored")

                content = fs.read(path)
                try:
                    result = gzip_module.decompress(content)
                except Exception as e:
                    raise TerminalError(f"gzip: {path}: {e}")

                out_path = path[:-3]  # Remove .gz suffix
                if fs.exists(out_path) and not parsed.force:
                    raise TerminalError(
                        f"gzip: {out_path} already exists; use -f to overwrite"
                    )

                fs.write(out_path, result)
                if not parsed.keep:
                    fs.remove(path)
            else:
                # Compress
                if path.endswith(".gz"):
                    raise TerminalError(
                        f"gzip: {path} already has .gz suffix -- unchanged"
                    )

                content = fs.read(path)
                result = gzip_module.compress(content)

                out_path = path + ".gz"
                if fs.exists(out_path) and not parsed.force:
                    raise TerminalError(
                        f"gzip: {out_path} already exists; use -f to overwrite"
                    )

                fs.write(out_path, result)
                if not parsed.keep:
                    fs.remove(path)

        except FileNotFoundError:
            raise TerminalError(f"gzip: {path}: No such file or directory")
        except IsADirectoryError:
            raise TerminalError(f"gzip: {path}: Is a directory")


def gunzip(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Decompress gzip files. Equivalent to gzip -d."""
    # gunzip is just gzip with -d prepended
    gzip(["-d"] + args, stdin, stdout, fs)


def tar(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Create or extract tar archives."""
    parser = CommandArgParser(prog="tar", add_help=False)
    parser.add_argument("-c", "--create", action="store_true", help="Create archive")
    parser.add_argument("-x", "--extract", action="store_true", help="Extract archive")
    parser.add_argument(
        "-t", "--list", action="store_true", help="List archive contents"
    )
    parser.add_argument("-f", "--file", type=str, help="Archive file")
    parser.add_argument(
        "-z", "--gzip", action="store_true", help="Use gzip compression"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-C", "--directory", type=str, help="Change to directory")
    parser.add_argument("files", nargs="*")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"tar: unknown arguments: {unknown}")

    # Validate mode - exactly one of -c, -x, -t required
    modes = [parsed.create, parsed.extract, parsed.list]
    if sum(modes) != 1:
        raise TerminalError("tar: exactly one of -c, -x, -t must be specified")

    if not parsed.file:
        raise TerminalError("tar: -f option is required")

    archive_path = parsed.file
    target_dir = parsed.directory or fs.getcwd()

    if parsed.create:
        _tar_create(archive_path, parsed.files, parsed.gzip, parsed.verbose, stdout, fs)
    elif parsed.extract:
        _tar_extract(archive_path, target_dir, parsed.gzip, parsed.verbose, stdout, fs)
    elif parsed.list:
        _tar_list(archive_path, parsed.gzip, stdout, fs)


def _tar_create(
    archive_path: str,
    files: list[str],
    use_gzip: bool,
    verbose: bool,
    stdout: TextIO,
    fs: FileSystem,
) -> None:
    """Create a tar archive."""
    if not files:
        raise TerminalError("tar: no files specified for archive")

    # Create tar in memory
    buffer = io.BytesIO()
    mode = "w:gz" if use_gzip else "w"

    try:
        with tarfile.open(fileobj=buffer, mode=mode) as tf:
            for file_path in files:
                _add_to_tar(tf, file_path, file_path, verbose, stdout, fs)
    except Exception as e:
        raise TerminalError(f"tar: error creating archive: {e}")

    # Write archive to filesystem
    fs.write(archive_path, buffer.getvalue())


def _add_to_tar(
    tf: tarfile.TarFile,
    file_path: str,
    arcname: str,
    verbose: bool,
    stdout: TextIO,
    fs: FileSystem,
) -> None:
    """Recursively add a file or directory to a tar archive."""
    if not fs.exists(file_path):
        raise TerminalError(f"tar: {file_path}: No such file or directory")

    if fs.isdir(file_path):
        # Add directory entry
        info = tarfile.TarInfo(name=arcname.rstrip("/") + "/")
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
        if verbose:
            stdout.write(f"{arcname}/\n")

        # Recursively add contents
        for name in fs.list(file_path):
            child_path = os.path.join(file_path, name)
            child_arcname = os.path.join(arcname, name)
            _add_to_tar(tf, child_path, child_arcname, verbose, stdout, fs)
    else:
        # Add file
        content = fs.read(file_path)
        info = tarfile.TarInfo(name=arcname)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
        if verbose:
            stdout.write(f"{arcname}\n")


def _tar_extract(
    archive_path: str,
    target_dir: str,
    use_gzip: bool,
    verbose: bool,
    stdout: TextIO,
    fs: FileSystem,
) -> None:
    """Extract a tar archive."""
    try:
        content = fs.read(archive_path)
    except FileNotFoundError:
        raise TerminalError(f"tar: {archive_path}: No such file or directory")

    buffer = io.BytesIO(content)
    mode = "r:gz" if use_gzip else "r:*"  # r:* auto-detects compression

    try:
        with tarfile.open(fileobj=buffer, mode=mode) as tf:
            for member in tf.getmembers():
                # Security: prevent path traversal with ..
                if ".." in member.name:
                    raise TerminalError(
                        f"tar: {member.name}: path traversal detected, skipping"
                    )

                # Strip leading slashes (like real tar)
                safe_name = member.name.lstrip("/")
                if not safe_name:
                    continue

                out_path = os.path.join(target_dir, safe_name)

                if member.isdir():
                    fs.makedirs(out_path, exist_ok=True)
                elif member.isfile():
                    # Ensure parent directory exists
                    parent = os.path.dirname(out_path)
                    if parent:
                        fs.makedirs(parent, exist_ok=True)

                    # Extract file content
                    extracted = tf.extractfile(member)
                    if extracted:
                        fs.write(out_path, extracted.read())

                if verbose:
                    stdout.write(f"{member.name}\n")
    except tarfile.TarError as e:
        raise TerminalError(f"tar: error reading archive: {e}")


def _tar_list(
    archive_path: str, use_gzip: bool, stdout: TextIO, fs: FileSystem
) -> None:
    """List contents of a tar archive."""
    try:
        content = fs.read(archive_path)
    except FileNotFoundError:
        raise TerminalError(f"tar: {archive_path}: No such file or directory")

    buffer = io.BytesIO(content)
    mode = "r:gz" if use_gzip else "r:*"

    try:
        with tarfile.open(fileobj=buffer, mode=mode) as tf:
            for member in tf.getmembers():
                stdout.write(f"{member.name}\n")
    except tarfile.TarError as e:
        raise TerminalError(f"tar: error reading archive: {e}")


def zip_cmd(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Create zip archives."""
    parser = CommandArgParser(prog="zip", add_help=False)
    parser.add_argument(
        "-r", "--recurse-paths", action="store_true", help="Recurse into directories"
    )
    parser.add_argument("zipfile", help="Archive file to create")
    parser.add_argument("files", nargs="*", help="Files to add")

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"zip: unknown arguments: {unknown}")

    if not parsed.files:
        raise TerminalError("zip: no files specified")

    archive_path = parsed.zipfile
    if not archive_path.endswith(".zip"):
        archive_path += ".zip"

    # Create zip in memory
    buffer = io.BytesIO()

    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in parsed.files:
                _add_to_zip(zf, file_path, file_path, parsed.recurse_paths, fs)
    except Exception as e:
        raise TerminalError(f"zip: error creating archive: {e}")

    # Write archive to filesystem
    fs.write(archive_path, buffer.getvalue())


def _add_to_zip(
    zf: zipfile.ZipFile,
    file_path: str,
    arcname: str,
    recursive: bool,
    fs: FileSystem,
) -> None:
    """Add a file or directory to a zip archive."""
    if not fs.exists(file_path):
        raise TerminalError(f"zip: {file_path}: No such file or directory")

    if fs.isdir(file_path):
        if not recursive:
            raise TerminalError(f"zip: {file_path}: is a directory (use -r to include)")

        # Add directory entry (trailing slash indicates directory)
        zf.writestr(arcname.rstrip("/") + "/", "")

        # Recursively add contents
        for name in fs.list(file_path):
            child_path = os.path.join(file_path, name)
            child_arcname = os.path.join(arcname, name)
            _add_to_zip(zf, child_path, child_arcname, recursive, fs)
    else:
        # Add file
        content = fs.read(file_path)
        zf.writestr(arcname, content)


def unzip(args: list[str], stdin: TextIO, stdout: TextIO, fs: FileSystem) -> None:
    """Extract zip archives."""
    parser = CommandArgParser(prog="unzip", add_help=False)
    parser.add_argument(
        "-l", "--list", action="store_true", help="List archive contents"
    )
    parser.add_argument("-d", "--directory", type=str, help="Extract to directory")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite files without prompting",
    )
    parser.add_argument("zipfile", help="Archive file to extract")
    parser.add_argument(
        "files", nargs="*", help="Specific files to extract (default: all)"
    )

    parsed, unknown = parser.parse_known_args(args)
    if unknown:
        raise TerminalError(f"unzip: unknown arguments: {unknown}")

    archive_path = parsed.zipfile
    target_dir = parsed.directory or fs.getcwd()

    try:
        content = fs.read(archive_path)
    except FileNotFoundError:
        raise TerminalError(f"unzip: cannot find {archive_path}")

    buffer = io.BytesIO(content)

    try:
        with zipfile.ZipFile(buffer, "r") as zf:
            if parsed.list:
                # List mode
                stdout.write(f"Archive:  {archive_path}\n")
                stdout.write("  Length      Name\n")
                stdout.write("---------  ----\n")
                total_size = 0
                for info in zf.infolist():
                    stdout.write(f"{info.file_size:9}  {info.filename}\n")
                    total_size += info.file_size
                stdout.write("---------  ----\n")
                stdout.write(f"{total_size:9}  {len(zf.infolist())} files\n")
            else:
                # Extract mode
                members_to_extract = parsed.files if parsed.files else None

                for info in zf.infolist():
                    # Filter by requested files if specified
                    if members_to_extract and info.filename not in members_to_extract:
                        continue

                    # Security: prevent path traversal with ..
                    if ".." in info.filename:
                        raise TerminalError(
                            f"unzip: {info.filename}: path traversal detected, skipping"
                        )

                    # Strip leading slashes (like real unzip)
                    safe_name = info.filename.lstrip("/")
                    if not safe_name:
                        continue

                    out_path = os.path.join(target_dir, safe_name)

                    if info.is_dir():
                        fs.makedirs(out_path, exist_ok=True)
                    else:
                        # Ensure parent directory exists
                        parent = os.path.dirname(out_path)
                        if parent:
                            fs.makedirs(parent, exist_ok=True)

                        # Check for existing file
                        if fs.exists(out_path) and not parsed.overwrite:
                            stdout.write(f"  skipping: {safe_name} (already exists)\n")
                            continue

                        # Extract file content
                        fs.write(out_path, zf.read(info.filename))
                        stdout.write(f"  inflating: {safe_name}\n")

    except zipfile.BadZipFile:
        raise TerminalError(f"unzip: {archive_path}: not a valid zip file")
    except Exception as e:
        raise TerminalError(f"unzip: error extracting archive: {e}")
