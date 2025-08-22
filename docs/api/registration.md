# Registration Methods

Agent registration methods allow you to expose functions, classes, and modules to your agents. These methods control what capabilities agents have access to and how they're presented in the agent's context.

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

### Visibility Levels

| Level | What Agent Sees | Best For |
|-------|----------------|----------|
| `"high"` | Function signature + full docstring | Custom functions or complex APIs where detailed guidance is needed. |
| `"medium"` | Function signature only | Familiar APIs where the agent only needs a reminder of the function's name and parameters. |
| `"low"` | Available for use but not shown in context | Common libraries (e.g., `numpy`, `pandas`) that the LLM is already trained on. Saves context space. |

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

### Viewing Recursive Modules in `view(agent)`

When you register a large package with `recursive=True`, the agent can resolve nested members at runtime (e.g., `routing.shortest_path`). However, the human-readable `view(agent)` intentionally does not enumerate entire subpackages to avoid importing large trees and overwhelming the context.

What `view(agent)` shows for recursive modules:

- Top-level members of the root module
- Explicitly configured dotted members (via `configure`) that have `visibility` set to `"medium"` or `"high"`

What it does not show:

- Arbitrary nested submodules discovered by recursion
- Dotted members that are only listed in `include` without a corresponding `configure` entry

Workarounds (pick one):

- Promote specific dotted members with `configure` so they render in `view(agent)`:

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
        "routing.route_to_gdf": MemberSpec(visibility="high"),
    },
)
# view(agent) will now list those dotted entries under the module
```

- Register submodules directly when you want their members listed without dotted names:

```python
agent.module(ox.routing, include=["shortest_path", "route_to_gdf"], visibility="high")
```

Notes:

- `full=True` in `view(agent, full=True)` lifts visibility gating but does not force deep submodule enumeration.
- This design keeps the view concise and avoids expensive imports; recursive registration still works fully at runtime.

### About Pattern Matching vs. Concrete Listings

Include/exclude patterns (globs like `"foo.bar*"` or lists of names) control what is allowed and how visibility is applied, but they do not automatically expand into concrete member listings in the rendered context.

- The agent’s rendered context (and `view(agent)`) shows:
  - Members discovered via introspection at that level (filtered by include/exclude), and
  - Explicit entries in `configure` (including dotted names), when their visibility is medium/high.

- It does not enumerate wildcard matches under dotted paths. For example:

```python
agent.module(
    some_pkg,
    recursive=True,
    include=["foo.bar*"],
    configure={"foo.bar1": MemberSpec(visibility="high")},
)

# The context will allow calls under foo.bar*, but only foo.bar1 is explicitly shown.
# If you want foo.bar2 to be visible, list it too:
# configure={"foo.bar1": MemberSpec(visibility="high"), "foo.bar2": MemberSpec(visibility="high")}
```

Tip: For a browsable listing in `view(agent)`, explicitly promote key dotted members via `configure`, or register the submodule directly and include its members at that level.

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


## Next Steps

- **Agent Creation**: See [Agent](agent.md) for Agent class documentation
- **Task Definition**: See [Task](task.md) for defining agent behavior using `@agent.task`
- **State Management**: See [State](state.md) for persistent memory in agent tasks
- **Debugging**: See [View](view.md) for inspecting registered capabilities 