# Registration Methods

Agent registration methods allow you to expose functions, classes, modules, terminal commands, and skills to your agents. These methods control what capabilities agents have access to and how they're presented in the agent's context.

Registration happens on [Agent](agent.md) instances - create an agent first, then register capabilities using these methods.

## `.fn()` - Function Registration

Register individual functions as agent capabilities.

```python
agent.fn(
    func: Callable | None = None,
    *,
    name: str | None = None,
    visibility: Literal["high", "medium", "low"] = "high",
    docstring: str | None = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable | None` | `None` | Function to register (filled automatically when used as decorator) |
| `name` | `str | None` | `None` | Override the function name in the agent environment |
| `visibility` | `Literal["high", "medium", "low"]` | `"high"` | How prominently to show this function in agent context |
| `docstring` | `str | None` | `None` | Override the function's docstring for the agent |
| `host_fs_access` | `bool` | `False` | Allow this function to access the host filesystem even when VFS/IsolatedFS is active (see [FileSystem docs](fs.md#host-filesystem-access)) |
| `network_access` | `bool` | `False` | Allow this function to make network connections (see [Security - Network Access](../concepts/security.md#network-access-control)) |
| `scope` | `str | None` | `None` | Gate this function behind a named capability *scope*: locked by default, available only in sessions that have been granted the scope. See [Scoped Capabilities](#scoped-capabilities). |

### Visibility Levels

| Level | What Agent Sees | Best For |
|-------|----------------|----------|
| `"high"` | Function signature + full docstring | Custom functions or complex APIs where detailed guidance is needed. |
| `"medium"` | Function signature only | Familiar APIs where the agent only needs a reminder of the function's name and parameters. |
| `"low"` | Available for use but not shown in context | Common libraries (e.g., `numpy`, `pandas`) that the LLM is already trained on. Saves context space. |

> **Tip**: For libraries registered with `visibility="low"`, consider pairing them with a [skill](fs.md#skills). Use `agent.skill()` to register documentation that the agent reads on-demand. This gives you the best of both worlds: minimal context overhead from low-visibility registration, with detailed usage guidance available when the agent needs it.
>
> ```python
> from importlib.resources import files
>
> agent.module(my_lib, visibility="low", recursive=True)
> agent.skill(files("my_lib") / "skills" / "my_lib" / "SKILL.md")
> ```

### Usage Patterns

#### As a Decorator

```python
from agex import Agent

agent = Agent()

@agent.fn
def calculate_square_root(x: float) -> float:
    """Calculate the square root of a number."""
    return x ** 0.5

@agent.fn(visibility="medium")
def helper_function(data: list) -> int:
    """Process data and return count."""
    return len(data)
```

#### Direct Registration

```python
import math

# Register existing functions
agent.fn(math.sin)
agent.fn(math.cos, visibility="low")
agent.fn(len, name="count_items")
```

#### Custom Docstrings

Useful when the original docstring is too technical or verbose for agents:

```python
@agent.fn(docstring="Add two numbers together quickly")
def add(a: float, b: float) -> float:
    """
    Performs mathematical addition of two floating-point numbers.
    
    This function implements the standard IEEE 754 floating-point
    addition operation with proper handling of edge cases...
    """
    return a + b
```

## `.cls()` - Class Registration

Register classes, giving agents access to their attributes and methods.

```python
# Type alias for include/exclude patterns
Pattern = str | list[str] | Callable[[str], bool]

agent.cls(
    cls: type,
    *,
    include: Pattern = "*",
    exclude: Pattern = "_*",
    visibility: Literal["high", "medium", "low"] = "high",
    constructable: bool = True,
    configure: dict[str, MemberSpec] | None = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cls` | `type` | | Class to register |
| `include` | `Pattern` | `"*"` | Pattern for members to include |
| `exclude` | `Pattern` | `"_*"` | Pattern for members to exclude |
| `visibility` | `Literal["high", "medium", "low"]` | `"high"` | How prominently to show this class |
| `constructable` | `bool` | `True` | Whether agents can create instances |
| `configure` | `dict[str, MemberSpec] | None` | `None` | Per-member configuration overrides |
| `host_fs_access` | `bool` | `False` | Allow this class and its methods to access the host filesystem (see [FileSystem docs](fs.md#host-filesystem-access)) |
| `network_access` | `bool` | `False` | Allow this class and its methods to make network connections (see [Security - Network Access](../concepts/security.md#network-access-control)) |
| `scope` | `str | None` | `None` | Gate this class behind a named capability *scope* (see [Scoped Capabilities](#scoped-capabilities)). |

### Usage Patterns

#### As a Decorator

Use the decorator pattern for classes you are defining in your own code. This is the most common pattern for exposing your application's data structures to an agent.

```python
from dataclasses import dataclass

@agent.cls
@dataclass
class User:
    name: str
    email: str
```

#### Direct Registration

Use the direct call pattern to register classes that are imported from external libraries, such as `pandas` or even the Python standard library.

```python
import pandas as pd

# Register the pandas DataFrame class with specific methods
agent.cls(
    pd.DataFrame,
    include=["head", "tail", "describe", "info"],
    visibility="medium"
)
```

### Include/Exclude Patterns

Pattern types work the same for both `.cls()` and `.module()` registration:

- **String (Glob)**: `"get_*"`, `"*"` - matches names using shell-style wildcards
- **List of Strings**: `["name", "email"]` - explicit member names or glob patterns  
- **Predicate Function**: `lambda name: not name.startswith('_')` - custom logic

## Per-Member Configuration (`configure` parameter)

Both `.cls()` and `.module()` support fine-grained per-member configuration using `MemberSpec`:

```python
from agex import MemberSpec

# For classes
agent.cls(
    DatabaseService,
    configure={
        "connect": MemberSpec(visibility="high"),     # Promote method
        "config_path": MemberSpec(visibility="low"),  # Demote attribute
        "admin_reset": MemberSpec(visibility="low"),  # Hide dangerous method
    }
)

# For modules (supports dot notation for class members)
agent.module(
    math,
    configure={
        "sin": MemberSpec(visibility="high"),                    # Promote function
        "SomeClass.method": MemberSpec(visibility="low"),        # Configure class member
    }
)
```

**MemberSpec Properties:**

- `visibility`: Override visibility for this specific member
- `docstring`: Custom docstring for the agent (for functions/methods)  
- `constructable`: Whether class can be instantiated (for classes in modules)

## `.module()` - Module Registration

Register functions, classes, and constants from entire modules.

```python
agent.module(
    module: ModuleType,
    *,
    name: str | None = None,
    include: Pattern = "*",
    exclude: Pattern = ["_*", "*._*"],
    visibility: Literal["high", "medium", "low"] = "medium",
    configure: dict[str, MemberSpec] | None = None,
    recursive: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `ModuleType` | | Module or object to register |
| `name` | `str | None` | `None` | Name in agent environment (required for non-modules) |
| `include` | `Pattern` | `"*"` | Pattern for members to include |
| `exclude` | `Pattern` | `["_*", "*._*"]` | Pattern for members to exclude |
| `visibility` | `Literal["high", "medium", "low"]` | `"medium"` | Default visibility for registered items |
| `configure` | `dict[str, MemberSpec] | None` | `None` | Per-member configuration overrides |
| `recursive` | `bool` | `False` | If `True`, recursively register all sub-modules of the given module. |
| `host_fs_access` | `bool` | `False` | Allow all functions/classes in this module to access the host filesystem (see [FileSystem docs](fs.md#host-filesystem-access)) |
| `network_access` | `bool` | `False` | Allow all functions/classes in this module to make network connections (see [Security - Network Access](../concepts/security.md#network-access-control)) |
| `scope` | `str | None` | `None` | Gate this whole module/instance behind a named capability *scope* (see [Scoped Capabilities](#scoped-capabilities)). |

### A Note on Instance Registration

While `.module()` is typically used for Python modules, it can also register the methods of a class *instance*. This is done to maintain a consistent API based on a key design principle:

*   `agent.fn()` registers a single callable.
*   `agent.module()` registers a namespace containing multiple callables.

From this perspective, an instance (a collection of methods) is treated similarly to a module (a collection of functions).

However, because instances do not have an intrinsic `__name__` attribute like modules do, you **must** provide the `name` parameter when registering an instance. This gives the agent a handle to refer to the object in its code.

```python
# Registering an instance requires the 'name' parameter
db_connection = sqlite3.connect(":memory:")
agent.module(db_connection, name="db", include=["execute", "commit"])
```

### Recursive Registration

For large libraries with many sub-modules (like `pandas` or `numpy`), registering each component individually is tedious. By setting `recursive=True`, `agex` will automatically discover and register all public sub-modules within a package.

This is the recommended way to register large, trusted libraries. It uses the same `include`, `exclude`, and `visibility` settings for all discovered sub-modules.

```python
import pandas as pd

# Automatically register all of pandas, excluding file I/O methods
agent.module(
    pd,
    recursive=True,
    visibility="low",
    exclude=["_*", "*._*", "read_*", "*.to_*"]
)
```

> **Note**: The `recursive` option is only valid for modules, not for class instances.

### Recursive Modules and `view(agent)`

With `recursive=True`, agents can resolve nested members at runtime (e.g., `routing.shortest_path`). However, `view(agent)` only shows top-level members and explicitly configured dotted members — it does not enumerate entire subpackages.

To make key nested functions visible in `view(agent)`, promote them via `configure`:

```python
import osmnx as ox
from agex import Agent, MemberSpec

agent = Agent()
agent.module(
    ox,
    visibility="low",
    recursive=True,
    configure={
        "geocoder.geocode": MemberSpec(visibility="high"),
        "routing.shortest_path": MemberSpec(visibility="high"),
    },
)
```

Alternatively, register submodules directly: `agent.module(ox.routing, include=["shortest_path"], visibility="high")`

### Usage Examples

```python
import math, random, sqlite3
import numpy as np
import pandas as pd

# Standard library - broad registration with low visibility
agent.module(math, visibility="low")
agent.module(random, include=["choice", "randint", "shuffle"])

# Third-party libraries  
agent.module(np, include="*", exclude=["_*", "test*"], visibility="low")
agent.module(pd, include=["DataFrame", "Series", "read_csv"])

# Class member targeting with dot notation
agent.module(requests, include=["Session", "Session.get", "Session.post"])

# Instance registration (requires name parameter)
db = sqlite3.connect("data.db")
agent.module(db, name="db", include=["execute", "commit", "close"])
```


## `.skill()` - Skill Registration

Register documentation that teaches agents how to use specific libraries or accomplish specific tasks. Skills are mounted read-only at `/skills/<name>/` and listed in the agent's system message.

A skill can be a single file or a directory containing `SKILL.md` and any number of sibling documents (e.g. type references, examples, tutorials).

```python
agent.skill(source: bytes | Path-like)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `bytes \| Path-like` | Skill content as raw bytes, a file path (`Path` / `importlib.resources` Traversable), or a directory path containing `SKILL.md` |

### How It Works

1. **Registration**: Call `agent.skill()` one or more times to collect skills
2. **Mounting**: At task execution, skills are mounted as a read-only overlay at `/skills/` — no VFS writes, no state commits
3. **System Message**: Skill names and descriptions (from YAML frontmatter in `SKILL.md`) are listed in the system message
4. **On-Demand**: The agent reads skill content via `cat /skills/<name>/SKILL.md` (or any sibling file) only when needed

### SKILL.md Format

Each skill requires a `SKILL.md` file with optional YAML frontmatter:

```markdown
---
name: my-library
description: Short description of what this skill covers
---

# my-library

Detailed instructions, examples, and patterns for the agent...
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Display name (defaults to parent directory name or filename stem) |
| `description` | No | One-line description shown in the skill listing |

### Usage Examples

```python
from pathlib import Path
from importlib.resources import files
from agex import Agent

agent = Agent()

# Directory skill — SKILL.md + sibling docs all mounted together
agent.skill(files("my_dsl") / "skills" / "my-dsl")
agent.skill(Path("./skills/my-dsl/"))

# Single file (mounted as SKILL.md)
agent.skill(files("calgebra") / "skills" / "calgebra" / "SKILL.md")
agent.skill(Path("./my-custom-skill.md"))

# From raw bytes (useful for dynamic/generated skills)
agent.skill(b"""---
name: my-tool
description: How to use my-tool effectively
---

# my-tool

Call `my_tool.run()` with a config dict...
""")
```

When registering a directory, all files are mounted under `/skills/<name>/`. For example, a directory containing `SKILL.md`, `types.md`, and `examples.md` becomes:

```
/skills/my-dsl/
├── SKILL.md
├── types.md
└── examples.md
```

The directory must contain a `SKILL.md` at its root. Dotfiles and dotdirectories (e.g. `.git`, `.DS_Store`) are automatically excluded.

### Pairing Skills with Low-Visibility Modules

Skills work especially well alongside `visibility="low"` registrations. The module is available but hidden from context, while the skill provides detailed guidance on-demand:

```python
import calgebra
from importlib.resources import files

# Register with low visibility (agent knows it, but no context overhead)
agent.module(calgebra, visibility="low", recursive=True)

# Pair with a skill (detailed docs, read on-demand)
agent.skill(files("calgebra") / "skills" / "calgebra" / "SKILL.md")
```

## `.terminal()` - Terminal Command Registration

Register a custom shell command for use inside `terminal_action`. Hosts use this when a capability is more naturally CLI-shaped than library-shaped — compilers, formatters, archive tools, anything an agent has seen as a command-line invocation in training.

```python
agent.terminal(
    handler: Callable | None = None,
    *,
    name: str | None = None,
    visibility: Literal["high", "medium", "low"] = "high",
    docstring: str | None = None,
)
```

The handler receives a `TerminalContext` (args, stdin, stdout, fs) per invocation and returns `None` (success, exit code 0) or a `CommandResult` (with `exit_code` / `stderr` set). Both shapes compose cleanly with termish's pipeline machinery.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handler` | `Callable | None` | `None` | Command handler (filled automatically when used as decorator) |
| `name` | `str | None` | `None` | Override the command name in the agent's terminal. Defaults to `handler.__name__` |
| `visibility` | `Literal["high", "medium", "low"]` | `"high"` | How prominently the command is surfaced in the agent's primer |
| `docstring` | `str | None` | `None` | Override `handler.__doc__` for the primer description |

### Visibility Levels

| Level | Behavior | Best For |
|-------|----------|----------|
| `"high"` | Command name + docstring shown in the primer | Novel host-specific commands the agent should know about |
| `"medium"` | Command name only | Tells the agent the command exists without spending tokens on a description |
| `"low"` | Not surfaced in the primer; the command still works | Commands the agent already knows from training (`git`, `esbuild`, `tsc`) — pair with a skill markdown file for in-depth reference |

The command is callable from `terminal_action` regardless of visibility — the setting only controls primer placement, not availability.

### Reserved Names

The string `"python"` is reserved for agex's internal bridge to nested `python_action` execution and cannot be registered. All other names — including termish builtins (`ls`, `cat`, `grep`, `find`, `tar`, `gzip`, `jq`, etc.) — are *overridable* per termish's contract. User registrations are last-wins among themselves.

```python
agent.terminal(my_handler, name="ls")    # ✅ overrides termish's `ls`
agent.terminal(my_handler, name="python") # ❌ ValueError — reserved
```

### Usage Patterns

#### As a Decorator

```python
from agex.terminal import TerminalContext, CommandResult

@agent.terminal
def greet(ctx: TerminalContext) -> CommandResult | None:
    """Print a greeting."""
    name = ctx.args[0] if ctx.args else "world"
    ctx.stdout.write(f"hello {name}\n")
    return None  # exit 0
```

#### Decorator With Options

```python
@agent.terminal(visibility="low", name="esbuild")
def my_esbuild_handler(ctx):
    """Bundle JS source files. Run `esbuild --help` for options."""
    ...
```

#### Direct Registration

```python
def my_handler(ctx):
    pass

agent.terminal(my_handler, name="custom-cmd")
```

#### Returning Errors

Use `ctx.fail(message, exit_code=N)` to build a `CommandResult` with a non-zero exit:

```python
@agent.terminal
def deploy(ctx):
    """Deploy the current build."""
    if not ctx.args:
        return ctx.fail("deploy: missing target. Run `deploy --help`.")
    target = ctx.args[0]
    if target not in ("staging", "production"):
        return ctx.fail(f"deploy: unknown target '{target}'", exit_code=2)
    ...
```

Termish raises a `TerminalError` from non-zero exits, which surfaces to the agent as a regular tracebacked error — same agex idiom as letting Python exceptions surface.

### Composing With Termish Builtins

Custom commands compose with termish's built-in pipeline operators (`|`, `>`, `>>`, `<`) and 30+ builtins (ls, cat, grep, etc.):

```python
@agent.terminal
def emit_data(ctx):
    """Emit some lines."""
    ctx.stdout.write("alpha\nbeta\ngamma\n")
```

The agent can then use it in pipelines naturally:

```
emit_data | grep beta | wc -l
```

### Discoverability Patterns

For commands at `visibility="low"` (i.e., agents already know the CLI from training), pair with a skill markdown for in-depth reference:

```python
agent.terminal(esbuild_handler, visibility="low", docstring="Bundle JS source files.")
agent.skill(files("my_pkg") / "skills" / "esbuild" / "SKILL.md")
```

Agents can also probe with `<command> --help` — handler authors should support it explicitly:

```python
@agent.terminal
def my_cmd(ctx):
    """Short summary for the primer."""
    if not ctx.args or ctx.args[0] in ("--help", "-h"):
        ctx.stdout.write("Usage: my_cmd <input> [--flag]\n...")
        return None
    ...
```

## Bundled Terminal Commands

agex ships two terminal commands out of the box — one always available, one opt-in.

### Python Scripts (always available)

Agents can run `python file.py` from `terminal_action` blocks. Scripts execute in a sandtrap sandbox with the agent's full policy (registered modules, VFS) but in a fresh namespace — no REPL state and no `task_*` bindings. The agent completes tasks by importing from scripts in a `python_action` block.

`"python"` is the only reserved terminal-command name (see above) — the bridge to nested `python_action` execution depends on it.

### Git CLI (opt-in)

Register the git skill to give agents version control over their workspace files:

```python
from agex.git_cli import register_git

register_git(agent)
```

This enables `git log`, `git commit -m`, `git diff`, `git branch`, `git checkout`, `git reset --hard`, `git show`, and `git merge` from `terminal_action` blocks, backed by kvgit. The `register_git` helper mounts the git SKILL.md *and* registers the `git` command via an internal factory mechanism (`_terminal_command_factory`) for handlers needing per-action runtime context (Staged state, VFS internals).

The internal factory API is currently used only by `register_git`. It will be promoted to public (`agent.terminal_factory`) when other downstream cases emerge — for now, public registrations should use `.terminal()` and reach for runtime values via closures over the agent at registration time when needed.

Key behaviors:

- **All file writes are automatically tracked** — there is no staging area or `git add`.
- **History is virtualized** — only agent-tagged commits (those with a message) appear in `git log`. System commits from the framework are filtered out.
- **`git reset --hard`** restores files without moving kvgit's real HEAD, preserving session state.

## Scoped Capabilities

Any registration can be tagged with a `scope=` — a named capability that is
**locked by default** and only available in sessions that have been *granted*
that scope. This is the gating half of human-in-the-loop permissions; the
agent requests a locked scope at runtime and the host decides (see
[Task — Requesting permission](task.md#requesting-permission-scopes)).

```python
agent.fn(send_mail, scope="email")
agent.fn(read_inbox, scope="email")    # same scope → a bundle: one grant unlocks both
agent.module(requests, scope="net")
agent.fn(clean_data)                   # no scope → always available (unchanged behavior)
```

- A `scope` names a **capability bundle**: registrations sharing a scope are
  unlocked together by a single grant. Granularity is whatever you register
  (a whole module, a single function), composing with `include`/`exclude`.
- **Unscoped registrations are always available** — fully backward compatible.
- Scoped registrations are **locked by default**. In a session without the
  grant, using one raises a `ScopeRequired` error that names the scope; the
  agent reacts by requesting it.

### Discovering declared scopes

```python
agent.scope_names   # -> {"email", "net"}  — the scopes this agent declares
```

`agent.scope_names` is static (derived from registrations), distinct from
`scopes(state).list()` which reports the scopes currently *granted* in a given
session. Granting/revoking is done with the `scopes(state)` accessor — see
[State](state.md) — and the request/resume flow in [Task](task.md#requesting-permission-scopes).

> **v1 constraint:** a scoped agent is *top-level only* — registering a scoped
> agent's task as another agent's `.fn()` raises, because scoped capabilities
> in a sub-agent aren't supported yet.

## Next Steps

- **Agent Creation**: See [Agent](agent.md) for Agent class documentation
- **Task Definition**: See [Task](task.md) for defining agent behavior using `@agent.task`
- **State Management**: See [State](state.md) for persistent memory in agent tasks
- **Debugging**: See [View](view.md) for inspecting registered capabilities