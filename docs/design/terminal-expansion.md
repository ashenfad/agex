# Terminal Interpreter Expansion - Design Doc

## Overview

This document outlines the implementation plan for expanding the agex terminal interpreter with additional commands commonly used in file exploration and text processing workflows.

**New commands:** `wc`, `sort`, `uniq`, `diff`, `xargs`, `tee`, `cut`

**Completing existing:** `cp -r`, `rm -r`

---

## Command Specifications

### 1. `wc` - Word/Line/Character Count

**Complexity: Low**

**Flags:**
- `-l` - count lines
- `-w` - count words
- `-c` - count bytes
- `-m` - count characters (same as -c for UTF-8)

**Behavior:**
- No flags: show lines, words, bytes (like `-lwc`)
- Multiple files: show per-file counts + total
- Stdin: read from pipe if no files

**Output format:**
```
    42    156   1234 file.txt
   100    500   4000 other.txt
   142    656   5234 total
```

**Implementation:**
```python
def wc(args, stdin, stdout, fs):
    # Parse -l, -w, -c, -m flags
    # For each file (or stdin):
    #   content = read file
    #   lines = content.count('\n')
    #   words = len(content.split())
    #   bytes = len(content.encode('utf-8'))
    # Format output with right-justified counts
```

**Edge cases:**
- Empty file: `0 0 0 file.txt`
- Binary files: count bytes, words may be meaningless
- No trailing newline: still count content as lines? (match GNU wc behavior)

---

### 2. `sort` - Sort Lines

**Complexity: Medium**

**Flags:**
- `-r` - reverse order
- `-n` - numeric sort
- `-u` - unique (remove duplicates after sort)
- `-f` - case-insensitive (fold)
- `-k N` - sort by field N (1-indexed)
- `-t CHAR` - field delimiter (default: whitespace)

**Behavior:**
- Reads all input into memory, sorts, outputs
- Stdin or files

**Implementation:**
```python
def sort(args, stdin, stdout, fs):
    # Parse flags
    # Read all lines from stdin or files
    # Build sort key function based on -k, -n, -f
    # sorted(lines, key=key_func, reverse=parsed.r)
    # If -u: deduplicate (preserve first occurrence)
    # Write output
```

**Key function for -k:**
```python
def make_key(line, field_num, delimiter, numeric, fold):
    fields = line.split(delimiter) if delimiter else line.split()
    if field_num <= len(fields):
        val = fields[field_num - 1]
    else:
        val = ""
    if fold:
        val = val.lower()
    if numeric:
        try:
            return (0, float(val))
        except ValueError:
            return (1, val)  # Non-numeric sorts after numeric
    return val
```

**Edge cases:**
- `-k` field beyond available fields: treat as empty string
- `-n` on non-numeric: sort as string (GNU behavior: non-numeric < numeric)
- Stable sort: preserve order of equal elements

---

### 3. `uniq` - Report/Filter Unique Lines

**Complexity: Low**

**Flags:**
- `-c` - prefix lines with count
- `-d` - only print duplicates
- `-u` - only print unique (non-duplicated)
- `-i` - case-insensitive comparison

**Behavior:**
- Operates on **adjacent** lines only (expects sorted input for global uniqueness)
- Stdin or single file

**Implementation:**
```python
def uniq(args, stdin, stdout, fs):
    # Parse flags
    # Read lines
    # Group adjacent identical lines
    # For each group:
    #   count = len(group)
    #   if -d and count == 1: skip
    #   if -u and count > 1: skip
    #   if -c: output f"{count:7d} {line}"
    #   else: output line
```

**Edge cases:**
- Empty input: no output
- Single line: output it (unless -d)
- `-i` comparison but preserve original case in output

---

### 4. `diff` - Compare Files

**Complexity: Medium-High**

**Flags:**
- `-u` - unified format (default)
- `-c` - context format
- `-q` - brief (just report if different)
- `-y` - side-by-side (complex, skip for now)
- `-B` - ignore blank lines
- `-w` - ignore whitespace

**Behavior:**
- Two file arguments required
- Output diff to stdout
- Exit status: 0 = same, 1 = different (not used in our model)

**Implementation using difflib:**
```python
import difflib

def diff(args, stdin, stdout, fs):
    # Parse flags
    # Read both files
    file1_lines = fs.read(path1).decode().splitlines(keepends=True)
    file2_lines = fs.read(path2).decode().splitlines(keepends=True)

    if parsed.q:  # Brief
        if file1_lines != file2_lines:
            stdout.write(f"Files {path1} and {path2} differ\n")
        return

    if parsed.c:  # Context format
        diff = difflib.context_diff(
            file1_lines, file2_lines,
            fromfile=path1, tofile=path2
        )
    else:  # Unified (default)
        diff = difflib.unified_diff(
            file1_lines, file2_lines,
            fromfile=path1, tofile=path2
        )

    stdout.writelines(diff)
```

**Edge cases:**
- One or both files missing: error
- Binary files: warn and skip? Or treat as text?
- Large files: difflib handles reasonably but memory-bound
- Directory diff: not supported (would be `diff -r`, skip for now)

---

### 5. `xargs` - Build and Execute Commands

**Complexity: Medium**

**Flags:**
- `-I REPLACE` - replace string in command (e.g., `-I {} cmd {} arg`)
- `-n NUM` - max arguments per command
- `-0` - null-delimited input (for filenames with spaces)
- `-t` - print command before executing (trace)

**Behavior:**
- Read items from stdin (whitespace or newline delimited, or null with -0)
- Execute command with items as arguments
- Default command: `echo`

**Implementation:**
```python
def xargs(args, stdin, stdout, fs):
    # Parse flags, extract command and its base args
    # Read stdin, split into items
    # If -I: for each item, substitute and execute once
    # Else: batch items by -n (or all), execute command with items appended

    # Execute means: look up command in BUILTINS, call it
    # Capture output, write to our stdout
```

**Example flows:**
```bash
# Find .py files and grep them
find . -name "*.py" | xargs grep "TODO"

# With -I for placement
ls *.txt | xargs -I {} cp {} backup/{}
```

**Complexity notes:**
- Need to invoke other commands from within xargs
- Share the same fs, stdin/stdout plumbing
- `-I {}` requires string substitution in args

**Implementation detail:**
```python
def _execute_command(name, args, stdin_content, fs):
    """Helper to execute a builtin command and capture output."""
    if name not in BUILTINS:
        raise TerminalError(f"{name}: command not found")

    cmd_stdin = io.StringIO(stdin_content)
    cmd_stdout = io.StringIO()
    BUILTINS[name](args, cmd_stdin, cmd_stdout, fs)
    return cmd_stdout.getvalue()
```

**Edge cases:**
- Empty stdin: no commands executed
- Command not found: error
- Quoted items in stdin: respect quotes? (complex, maybe skip)
- Newlines in filenames with -0: must use null delimiter

---

### 6. `tee` - Duplicate Output

**Complexity: Low**

**Flags:**
- `-a` - append to files instead of overwrite

**Behavior:**
- Read stdin
- Write to all specified files AND stdout
- Useful for: `cmd | tee log.txt | next_cmd`

**Implementation:**
```python
def tee(args, stdin, stdout, fs):
    parser = CommandArgParser(prog="tee", add_help=False)
    parser.add_argument("-a", "--append", action="store_true")
    parser.add_argument("files", nargs="*")

    parsed, _ = parser.parse_known_args(args)

    content = stdin.read()

    # Write to stdout
    stdout.write(content)

    # Write to each file
    mode = "a" if parsed.append else "w"
    for path in parsed.files:
        fs.write(path, content.encode("utf-8"), mode=mode)
```

**Edge cases:**
- No files: just pass through (acts like cat)
- File write error: report but continue? Or fail?

---

### 7. `cut` - Extract Fields/Columns

**Complexity: Medium**

**Flags:**
- `-d DELIM` - field delimiter (default: TAB)
- `-f FIELDS` - select fields (1-indexed)
- `-c CHARS` - select character positions
- `-b BYTES` - select byte positions (same as -c for ASCII)
- `--complement` - invert selection

**Field specification:**
- `N` - single field
- `N-M` - range inclusive
- `N-` - from N to end
- `-M` - from start to M
- `N,M,O` - multiple specific fields

**Implementation:**
```python
def cut(args, stdin, stdout, fs):
    # Parse flags
    # Parse field/char spec into a selector function

    for line in input_lines:
        if parsed.f:  # Field mode
            fields = line.split(parsed.d) if parsed.d else line.split('\t')
            selected = select_fields(fields, parsed.f)
            stdout.write(parsed.d.join(selected) + "\n")
        elif parsed.c:  # Character mode
            selected = select_chars(line.rstrip('\n'), parsed.c)
            stdout.write(selected + "\n")
```

**Field selector parsing:**
```python
def parse_field_spec(spec):
    """Parse '1,3-5,7-' into a function that selects from a list."""
    ranges = []
    for part in spec.split(','):
        if '-' in part:
            start, end = part.split('-', 1)
            start = int(start) if start else 1
            end = int(end) if end else None  # None means "to end"
            ranges.append((start, end))
        else:
            n = int(part)
            ranges.append((n, n))

    def selector(items):
        result = []
        for start, end in ranges:
            if end is None:
                result.extend(items[start-1:])
            else:
                result.extend(items[start-1:end])
        return result

    return selector
```

**Edge cases:**
- Field beyond available: skip silently (GNU behavior)
- `-d ''` (empty delimiter): error
- Multiple delimiters: each delimiter creates empty field

---

### 8. `cp -r` - Recursive Copy (Complete)

**Complexity: Medium**

**Current state:** Raises "not fully implemented"

**Implementation:**
```python
def _copy_recursive(src, dst, fs):
    """Recursively copy src directory to dst."""
    # Create destination directory
    if not fs.exists(dst):
        fs.mkdir(dst)

    items = fs.list_detailed(src, recursive=False)
    for item in items:
        src_path = item.path
        # Compute relative path from src
        rel_path = src_path[len(src):].lstrip('/')
        dst_path = f"{dst}/{rel_path}"

        if item.is_dir:
            _copy_recursive(src_path, dst_path, fs)
        else:
            content = fs.read(src_path)
            fs.write(dst_path, content)
```

**Edge cases:**
- Copying into self: detect and error
- Destination exists: overwrite files, merge directories?
- Symlinks: VFS probably doesn't have these, skip
- Permissions: VFS doesn't track, skip

---

### 9. `rm -r` - Recursive Remove (Complete)

**Complexity: Medium**

**Current state:** Raises "not fully implemented"

**Implementation:**
```python
def _remove_recursive(path, fs):
    """Recursively remove directory and contents."""
    items = fs.list_detailed(path, recursive=False)

    for item in items:
        if item.is_dir:
            _remove_recursive(item.path, fs)
        else:
            fs.remove(item.path)

    # Remove now-empty directory
    fs.rmdir(path)
```

**Note:** VFS has `rmdir()` - confirmed at `agex/fs/virtual.py:872`.

**Edge cases:**
- Remove current directory: error? Or allow?
- Remove root: probably error
- Partial failure: stop or continue?

---

## File Organization

```
agex/terminal/interpreter/commands/
├── filesystem.py   # existing + cp -r, rm -r completion
├── io.py           # existing + tee
├── search.py       # existing (grep, find)
├── text.py         # NEW: wc, sort, uniq, cut
├── diff.py         # NEW: diff (uses difflib)
└── meta.py         # NEW: xargs
```

**Rationale:**
- `text.py` groups line-oriented text processing
- `diff.py` separate because it's more complex and self-contained
- `meta.py` for commands that invoke other commands (just xargs for now)

---

## Implementation Order

Recommended order by value/complexity ratio:

| Order | Command | Complexity | Value | Notes |
|-------|---------|------------|-------|-------|
| 1 | `wc` | Low | High | "How big?" is constant need |
| 2 | `tee` | Low | Medium | Simple, enables logging |
| 3 | `uniq` | Low | Medium | Pairs with sort |
| 4 | `sort` | Medium | High | Core for analysis |
| 5 | `cut` | Medium | Medium | Field extraction |
| 6 | `cp -r` | Medium | Medium | Complete existing |
| 7 | `rm -r` | Medium | Medium | Complete existing |
| 8 | `diff` | Medium-High | High | Very useful for agents |
| 9 | `xargs` | Medium | High | Pipeline power, but complex |

---

## Testing Strategy

Each command should have tests for:

1. **Basic functionality** - happy path
2. **Flag combinations** - common flag combos
3. **Stdin vs files** - both input modes
4. **Pipeline integration** - works with `|`
5. **Edge cases** - empty input, missing files, etc.
6. **Error handling** - appropriate TerminalError messages

**Example test structure:**
```python
class TestWc:
    def test_line_count(self, fs):
        fs.write("test.txt", b"line1\nline2\nline3\n")
        result = execute("wc -l test.txt", fs)
        assert "3" in result

    def test_stdin(self, fs):
        result = execute("echo 'a b c' | wc -w", fs)
        assert "3" in result

    def test_multiple_files(self, fs):
        fs.write("a.txt", b"one\n")
        fs.write("b.txt", b"two\nthree\n")
        result = execute("wc -l a.txt b.txt", fs)
        assert "total" in result
```

---

## Design Decisions

1. **GNU-ish behavior** - Match GNU semantics (more flags, more forgiving) for agent familiarity.

2. **Error handling** - Match bash default: commands continue processing all arguments, reporting errors but not stopping. Script-level `set -e` controls whether pipeline stops on command failure.

3. **Binary file handling** - Match GNU behavior:
   - `grep`/`diff`: Detect binary files and handle specially (suppress matches or report "files differ")
   - `wc`: Process normally without special handling (byte counts are valid)

4. **xargs quoting** - Support simple cases only. Full shell quoting is complex; defer to phase 2 if needed.

5. **Performance** - In-memory processing is acceptable for VFS scale. Large external files can be streamed if needed later.

6. **diff default format** - Unified diff (`-u`) as default, most useful for agents.

7. **Field edge cases (cut/sort)** - Match real GNU semantics:
   - Field beyond available: treat as empty string
   - `-k` without delimiter: whitespace-separated
   - Multiple consecutive delimiters create empty fields

8. **VFS support** - Confirmed: VFS has `rmdir()` at `agex/fs/virtual.py:872`.

---

## Estimated Effort

| Command | Lines of Code | Time Estimate |
|---------|---------------|---------------|
| wc | ~50 | 1 hour |
| tee | ~25 | 30 min |
| uniq | ~50 | 1 hour |
| sort | ~80 | 2 hours |
| cut | ~80 | 2 hours |
| cp -r | ~40 | 1 hour |
| rm -r | ~30 | 1 hour |
| diff | ~60 | 1.5 hours |
| xargs | ~100 | 3 hours |
| **Total** | ~515 | ~13 hours |

Plus testing: ~1x implementation time = **~26 hours total**

---

## Alternatives Considered

1. **Python builtins instead** - `ls()`, `grep()` as Python functions
   - Rejected: Terminal syntax is familiar, pipelines are natural

2. **Subprocess to real bash** - Just shell out
   - Rejected: Breaks sandboxing, VFS isolation

3. **Minimal set only** - Just wc, sort, uniq
   - Possible: Could defer xargs, diff if time-constrained

4. **Use external libraries** - e.g., `sh` package
   - Rejected: Still need VFS integration, not simpler
