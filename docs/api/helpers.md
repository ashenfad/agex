# Registration Helpers

To make it easier to get started with common libraries, `agex` provides a set of registration helpers. These helpers are simple functions that pre-register popular libraries like NumPy, pandas, and the Python standard library with sensible defaults.

## Philosophy

The core idea behind these helpers is to:

1.  **Save Time**: Provide a one-line way to give agents access to powerful, well-known libraries.
2.  **Use Sensible Defaults**: The helpers register modules with `visibility="low"`, which means they are available to the agent but don't clutter the agent's limited context window with detailed documentation. LLMs are already extensively trained on these libraries, so they don't need to see the full docstrings.
3.  **Demonstrate Best Practices**: The helpers serve as a practical example of how to use `agent.module` and `agent.cls` to create your own registration patterns for your internal tools and libraries.
4.  **Promote Security**: They exclude potentially dangerous functions (like `os.system` or file I/O operations in data libraries) that are not suitable for a sandboxed agent environment.

## Usage

To use a helper, simply import it and pass your agent instance to it.

```python
from agex import Agent
from agex.helpers import register_numpy, register_pandas, register_stdlib

# Create an agent
data_analyst = Agent(name="data_analyst")

# Register libraries with one line each
register_numpy(data_analyst)
register_pandas(data_analyst)

# The agent now has access to these libraries
@data_analyst.task
def analyze(data: list) -> float: # type: ignore[return-value]
    """Calculate the mean of the data using pandas."""
    pass
```


## Available Helpers

### `register_stdlib(agent)`

Registers a curated list of safe and useful modules from the Python standard library.

-   **Mathematical**: `math`, `random` (with state-setting functions like `seed()` excluded), `statistics`, `decimal`, `fractions`, `time`.
-   **Utilities**: `collections`, `itertools`, `datetime` (including its classes), `calendar`, `uuid`.
-   **Text Processing**: `re`, `string`, `textwrap`.
-   **Data Encoding**: `base64`, `hashlib`, `zoneinfo`.
-   **Debugging**: `traceback` (safe formatting functions only).
-   **Types**: `typing`.
-   All modules are registered with `visibility="low"`.

### `register_numpy(agent)`

Registers the `numpy` library for numerical computing.

-   Registers the core `numpy` module and sub-modules.
-   Excludes potentially unsafe functions like `load`, `save`, and file I/O operations.
-   All modules are registered with `visibility="low"`.

### `register_pandas(agent)`

Registers the `pandas` library for data analysis.

-   Registers the core `pandas` module sub-modules.
-   Excludes all `read_*` and `to_*` functions to prevent file system access.
-   All modules and classes are registered with `visibility="low"`.

### `register_plotly(agent)`

Registers the `plotly` library for creating interactive visualizations.

-   Registers `plotly` and sub-modules.
-   Excludes functions related to writing files or showing plots directly (e.g., `write_image`, `show`), as these are side-effects that should be handled by the user's code, not the agent's.
-   All modules and classes are registered with `visibility="low"`.

### `register_io(agent)`

Registers IO-related modules for file operations. Used with VFS and isolated filesystems.

-   **File objects**: `io.BytesIO`, `io.StringIO`, `io.TextIOWrapper` (plus the internal `_io` types for real file objects)
-   **OS operations**: `os.listdir`, `os.walk`, `os.remove`, `os.mkdir`, `os.makedirs`, `os.rename`, `os.stat`
-   **Path utilities**: `os.path.exists`, `os.path.isfile`, `os.path.isdir`, `os.path.join`, `os.path.basename`, `os.path.dirname`, `os.path.splitext`
-   **Data formats**: `json`, `csv`, `pickle`, `pathlib`
-   **Pattern matching**: `glob`, `fnmatch`
-   **Compressed / archive IO**: `gzip`, `zipfile`, `tarfile`
-   **Built-in**: `open()`

> [!NOTE]
> When an agent is configured with `fs=connect_fs(...)`, `register_io()` is called automatically during task execution. You typically don't need to call it manually.

```python
from agex import Agent
from agex.helpers import register_io

# Manual registration (not usually needed with connect_fs)
agent = Agent()
register_io(agent)
```

### `register_matplotlib(agent)`

Registers `matplotlib` for headless plotting in agex deployments.

Beyond the usual module exposure, this helper handles two sandbox-environment quirks that matplotlib's lazy initialization runs into:

1. **Backend selection.** Forces the non-interactive `Agg` backend before any pyplot import — agex deployments are headless (Pyodide, subprocess workers, kernel-isolated containers), and matplotlib's default backend probing fails in those environments.
2. **Font cache warm-up.** matplotlib builds its `fontlist.json` cache the first time `font_manager` resolves a glyph (i.e. the first `savefig` with text). The build acquires a lock file inside matplotlib's package directory — a write the sandbox blocks. Pre-warming at registration time, *before* any sandboxed code runs, lets the lock write succeed and populates the cache for later sandboxed `savefig` calls to read.

```python
from agex.helpers import register_matplotlib

register_matplotlib(agent)
```

Must be called from outside the sandbox (i.e. before any task execution). The other helpers in this package have the same implicit contract; matplotlib is the first case where violating it has a visible failure mode.

## Terminal Command Helpers

These helpers register entries that surface inside `terminal_action` rather than `python_action`. See [Registration](registration.md) for the underlying `.terminal()` API.

### `register_git(agent)`

Registers the `git` skill and `git` terminal command, giving agents version control over their VFS workspace files.

```python
from agex.git_cli import register_git

register_git(agent)
```

Enables `git log`, `git commit -m`, `git diff`, `git branch`, `git checkout`, `git reset --hard`, `git show`, and `git merge` from `terminal_action` blocks, backed by kvgit. The skill (`/skills/git/SKILL.md`) is mounted alongside the command for agents that want detailed reference.

Key behaviors:

-   **All file writes are automatically tracked** — there is no staging area or `git add`.
-   **History is virtualized** — only agent-tagged commits (those with a message) appear in `git log`. System commits from the framework are filtered out.
-   **`git reset --hard`** restores files without moving kvgit's real HEAD, preserving session state.

> Note: `register_git` lives at `agex.git_cli`, not `agex.helpers`. The other helpers above sit alongside the popular Python libraries they wrap; git's pairs more naturally with the kvgit-backed state machinery so the import path tracks that.
