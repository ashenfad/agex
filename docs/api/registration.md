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
| `func` | `Callable \| None` | `None` | Function to register (filled automatically when used as decorator) |
| `name` | `str \| None` | `None` | Override the function name in the agent environment |
| `visibility` | `Literal["high", "medium", "low"]` | `"high"` | How prominently to show this function in agent context |
| `docstring` | `str \| None` | `None` | Override the function's docstring for the agent |

### Visibility Levels

| Level | What Agent Sees |
|-------|----------------|
| `"high"` | Function signature + full docstring |
| `"medium"` | Function signature only |
| `"low"` | Available for use but not shown in context |

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
| `configure` | `dict[str, MemberSpec] \| None` | `None` | Per-member configuration overrides |

### Include/Exclude Patterns

Pattern types work the same for both `.cls()` and `.module()` registration:

- **String (Glob)**: `"get_*"`, `"*"` - matches names using shell-style wildcards
- **List of Strings**: `["name", "email"]` - explicit member names or glob patterns  
- **Predicate Function**: `lambda name: not name.startswith('_')` - custom logic

```python
# Glob patterns
agent.cls(MyClass, include="get_*", exclude=["delete", "remove_*"])

# Explicit lists  
agent.cls(User, include=["name", "email", "get_profile"])

# Custom logic
agent.cls(MyClass, include=lambda n: not n.startswith('_') and 'delete' not in n.lower())
```

## Per-Member Configuration (`configure` parameter)

Both `.cls()` and `.module()` support fine-grained per-member configuration using `MemberSpec`:

```python
from agex.agent.datatypes import MemberSpec

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

### Usage Examples

```python
from dataclasses import dataclass
import pandas as pd

# Decorator pattern for classes you're defining
@agent.cls
@dataclass
class User:
    name: str
    email: str


# Custom registration with options
@agent.cls(include=["name", "email"])
@dataclass
class RestrictedUser:
    name: str
    email: str
    _id: int
    admin_token: str  # Not exposed to agent

# Direct call pattern for external libraries
agent.cls(
    pd.DataFrame, 
    include=["head", "tail", "describe", "info"],
    exclude=["to_*"],
    visibility="medium"
)
```

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
    configure: dict[str, MemberSpec] | None = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `ModuleType` | | Module or object to register |
| `name` | `str \| None` | `None` | Name in agent environment (required for non-modules) |
| `include` | `Pattern` | `"*"` | Pattern for members to include |
| `exclude` | `Pattern` | `["_*", "*._*"]` | Pattern for members to exclude |
| `visibility` | `Literal["high", "medium", "low"]` | `"medium"` | Default visibility for registered items |
| `configure` | `dict[str, MemberSpec] \| None` | `None` | Per-member configuration overrides |



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