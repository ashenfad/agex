# Nearly Python: Understanding Agent Code Constraints

agex agents generate and execute code in a secure sandbox powered by [sandtrap](https://github.com/ashenfad/sandtrap). The sandbox uses AST rewriting to compile agent code into restricted bytecode that runs within your process. The result looks and feels like Python — but with some important differences.

This guide helps you understand what constraints agents face when writing code, so you can design better integrations and understand agent behavior.

**State Choice Affects Constraints**: Some limitations depend on whether you use live state (default, no persistence) or persistent state (remembers variables between task calls). Live state is more flexible but doesn't persist memory; persistent state has more constraints but enables complex multi-step workflows.

!!! important "Imports: Registered or VFS-Resident"

    Agent-generated code may use `import` statements. These only succeed for:
    1. **Registered Modules**: Libraries explicitly exposed via `agent.module(...)`.
    2. **Workspace Modules**: Python files the agent has created in the Virtual Filesystem (VFS).

    Within registered modules, only whitelisted members are visible. For workspace modules, all members are available.

    **Example:**
    ```python
    import pandas as pd              # OK if `pandas` was registered
    import helpers.utils             # OK if agent created `helpers/utils.py`
    import os                        # Fails if `os` was not registered
    ```

## What Works (Agent-Generated Code)

Most Python features work exactly as you'd expect when agents generate code:

- **Basic operations**: arithmetic, string manipulation, list/dict operations
- **Control flow**: `if/else`, `for/while` loops, `match/case` (Python 3.10+), function calls
- **Built-in functions**: `print()`, `len()`, `range()`, `enumerate()`, `sorted()`, etc.
- **Classes**: full `class` definitions with inheritance, methods, and dunders
- **Generators**: `yield` and `yield from`
- **Closures**: nested functions with `nonlocal` and `global`
- **Exception handling**: `try/except/finally/else` with 50+ built-in exception types
- **Decorators**: function and class decorators
- **Comprehensions**: list, dict, set, and generator comprehensions
- **Registered capabilities**: anything you've exposed via `agent.module()` or `agent.fn()`
- **Function definitions**: agents can define helper functions within their code
- **Variable assignment**: storing values in variables works normally

```python
# Agent-generated code like this works perfectly
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
for i, num in enumerate(numbers):
    if num % 2 == 0:
        print(f"Even number at index {i}: {num}")
```

## What's Different (Sandbox Restrictions)

### Three-Argument `type()`
**Blocked**: Dynamic class creation via `type('Name', bases, dict)` is rejected. Single-argument `type(obj)` works normally for inspection.

```python
# ✅ Allowed: type inspection
t = type(42)           # <class 'int'>
isinstance(42, t)      # True

# ❌ Blocked: dynamic class creation
MyClass = type('MyClass', (object,), {'x': 1})  # TypeError
```

**Why?**: Dynamic class creation bypasses the AST rewriter's validation of class definitions, which is needed to enforce attribute access control on instances.

### Format String Traversal
**Blocked**: `.format()` and `.format_map()` reject attribute and item traversal in field names. F-strings are unaffected (they go through proper AST validation).

```python
# ✅ Allowed: simple positional/keyword formatting
"Hello {name}".format(name="World")
"{0} + {1}".format(1, 2)

# ❌ Blocked: attribute traversal via format string
"{obj.__class__}".format(obj=x)  # AttributeError

# ✅ Secure alternative: f-strings use AST-level attribute gating
f"{obj.attr}"
```

**Why?**: Format string attribute traversal is a classic Python sandbox escape technique.

### Bare `except:` Clause
**Rewritten**: A bare `except:` is automatically rewritten to `except Exception:`. This prevents agent code from swallowing control exceptions (`KeyboardInterrupt`, sandbox timeout/cancellation signals, etc.).

```python
# What the agent writes:
try:
    risky()
except:
    pass

# What actually runs:
try:
    risky()
except Exception:
    pass
```

### Unavailable Names
The following names are not available in agent code (raise `NameError`):

- **Control exceptions**: `BaseException`, `KeyboardInterrupt`, `GeneratorExit`, `SystemExit`
- **Dangerous builtins**: `exec`, `eval`, `compile`
- **Introspection**: `dir()`, `help()`, `globals()`

`locals()` is available but returns a filtered copy (sandbox internals excluded).

### Wildcard Imports
**Rejected**: `from module import *` is blocked at parse time. Agents must import specific names.

### Private Attribute Access
**Blocked by default**: Attributes starting with `_` are not accessible on registered objects unless explicitly included in the registration's `include` pattern.

```python
# ❌ Blocked by default
obj._internal_method()
obj.__dict__

# ✅ If explicitly included in registration
agent.cls(MyClass, include=["_special_method"])
```

### Unpicklable Objects — Automatic Handling

**When using versioned state, unpicklable objects (like database cursors, file handles, network connections) are automatically detected and handled gracefully.**

#### It Just Works for Single-Turn Use

```python
# ✅ Natural Python - works perfectly in single turn
cursor = db.cursor()
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()
# Single-turn use - perfect! No special handling needed.
```

#### Cross-Turn Use Gets Helpful Guidance

If you try to reference an unpicklable variable from a previous turn, you'll get a clear error with solutions:

```python
# Turn 1
cursor = db.cursor()
data = cursor.fetchone()

# Turn 2 - agent tries to reuse cursor
cursor.execute("SELECT ...")  # Error on variable reference!

# UnpicklableVariableError: Variable 'cursor' (sqlite3.Cursor) is not available.
# It was not persisted from a previous execution because it is unpicklable.
#
# Solutions:
#   1. Recreate it: cursor = db.cursor()
#   2. Chain operations: results = db.cursor().fetchall()
#   3. Use this variable only within a single turn
```

**Impact**: Write natural Python for unpicklable objects. Single-turn use is friction-free. Multi-turn reuse gives clear, actionable guidance.

**Best practices:**
- Chain operations for one-off queries: `results = db.cursor().fetchall()`
- Recreate resources at the start of each turn: `cursor = db.cursor()`

### Summary of State Modes

| State Mode | Unpicklable Objects | Memory Between Calls |
| :--- | :--- | :--- |
| **Default (No State)** | Allowed (no persistence) | No |
| **`Live` State** | Allowed (in-memory) | Yes (in-process) |
| **Versioned State** | Auto-handled via markers | Yes (persistent) |

### Object Identity Between Executions
**Objects are reconstructed**: Between task executions (when using versioned state), objects are serialized and deserialized. This breaks object identity (`id()`) and shared references between separate task runs.

```python
# During a single task execution, identity works normally:
my_list = [1, 2, 3]
shared_ref = my_list
shared_ref.append(4)
print(my_list)  # [1, 2, 3, 4]

# But identity is not preserved across task executions.
# Task 1 creates my_list with id=1000. It is saved to state.
# Task 2 loads my_list from state. It is now a new object with id=2000.
```

**Impact**: Objects that rely on `is` checks or `id()` for identity across multiple task executions may behave unexpectedly. Use value-based comparisons instead.

### Function Closures Across Turns
**Captured variables are "frozen" on save**: When using versioned state, closures work and persist, but any variables they capture from their enclosing scope are "frozen" with their current values when the task completes.

A closure will not see subsequent changes to a captured variable in a later task execution.

```python
# Assume the agent executes this code in its first task run:
factor = 2
def multiplier(x):
    return x * factor

# `multiplier` is now "frozen" with `factor=2`.

# --- End of first task ---

# In a second task run, the agent changes factor:
factor = 10
# But `multiplier` still uses the value it was frozen with.
result = multiplier(5)  # Returns 10, not 50.
```

**Impact**: This can lead to unexpected behavior if you assume closures will always see the latest version of their captured variables across different task runs. This is inherent to the serialization-based state model.

## Async Architecture

agex supports async task execution while keeping the agent sandbox synchronous. This is achieved through a three-layer architecture:

### Layer 1: Task Execution (async optional)

Your tasks can be defined as sync or async:

```python
# Sync task
@agent.task
def my_task(data: str) -> dict:  # type: ignore[return-value]
    """Process data."""
    pass

# Async task
@agent.task
async def my_async_task(data: str) -> dict:  # type: ignore[return-value]
    """Process data asynchronously."""
    pass
```

Async tasks use async LLM client methods internally, allowing non-blocking I/O in async codebases.

### Layer 2: Transparent Bridging

Async functions registered via `@agent.fn` are bridged transparently. Agents call them like sync functions:

```python
# You register an async function
@agent.fn
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Agent sees and calls it as sync:
# data = fetch_data("https://api.example.com")
```

The framework automatically awaits async results using `run_coroutine_threadsafe()`, so agents receive resolved values without needing async syntax.

> [!IMPORTANT]
> Async bridging only works in async task context. If an agent calls an async function from a **sync** task, it will see a clear error: *"'fn_name' is an async function and cannot be called from a sync task."* The agent can then use a synchronous alternative or report the limitation. Use async tasks when you need to call async registered functions.

### Layer 3: Sandbox Execution (always sync)

Agent-generated code runs in a synchronous sandbox. This is intentional:

- **Simpler reasoning**: Sync code is easier for LLMs to generate correctly
- **Safer execution**: No async race conditions or deadlock risks
- **Predictable behavior**: Sequential execution is easier to debug

The separation means you get async benefits at the framework level (non-blocking I/O, compatibility with FastAPI/asyncio) without exposing async complexity to agents.

## Resource Limits

agex can enforce resource limits via [sandtrap](https://github.com/ashenfad/sandtrap) to prevent runaway code from exhausting system resources.

### Memory Limits

```python
agent = Agent(max_memory_mb=500)  # 500MB headroom per task
```

Memory limits use two layers: `RLIMIT_AS` (kernel-enforced on Linux) and checkpoint-based detection (Linux + macOS). If agent code exceeds the limit, it raises `MemoryError` (wrapped in `EvalError`). The agent receives the error and can adjust its approach — for example, processing data in chunks.

### File Descriptor Limits

```python
agent = Agent(max_open_files=256)
```

Prevents agents from opening too many files simultaneously.

### VFS Size Limits

```python
agent = Agent(
    fs=connect_fs(type="virtual", max_size_mb=100),
)
```

Limits total storage in the Virtual FileSystem:

```python
# Agent tries to write beyond the limit
with open("huge.bin", "wb") as f:
    f.write(b"x" * (200 * 1024 * 1024))  # 200MB
# Raises OSError: VFS size limit exceeded
```

### Platform Note

Memory and file descriptor limits require Unix (Linux/macOS). On Windows, these limits are not enforced. VFS size limits work on all platforms.

## Why These Restrictions?

These constraints exist for important reasons:

- **Security**: Prevents agents from accessing dangerous Python features (attribute traversal, dynamic class creation, control exception swallowing)
- **Serialization**: Enables memory and rollback by ensuring all persistent state can be saved
- **Sandboxing**: Ensures agent code cannot escape the execution environment
- **Resource Protection**: Prevents runaway code from exhausting memory, files, or storage

**Note**: With `Live` state, serialization constraints don't apply since no state is persisted between task calls. Choose versioned state when you need agents to remember variables across multiple task executions.
