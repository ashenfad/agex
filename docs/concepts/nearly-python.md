# Nearly Python: Understanding Agent Code Constraints

agex agents generate and execute code in a secure sandbox that looks and feels like Python—but with some important differences. This guide helps you understand what constraints agents face when writing code, so you can design better integrations and understand agent behavior.

To that effect, all code samples in this document are for the sandboxed Python-esque DSL that agex agents generate.

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
- **Control flow**: `if/else`, `for/while` loops, function calls
- **Built-in functions**: `print()`, `len()`, `range()`, `enumerate()`, etc.
- **Function decorators**: full support for standard library (e.g., `@functools.lru_cache`) and custom decorators
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

## What's Different (Agent Code Limitations)

### Class Decorators
**Not supported**: Agents cannot use decorators on class definitions (except the special `@dataclass` placeholder).

```python
# ❌ Agents cannot use class decorators
def my_decorator(cls):
    cls.added_method = lambda: "hi"
    return cls

@my_decorator
class MyClass:
    pass

# ✅ Exception: @dataclass works (special sandbox implementation)
@dataclass
class Point:
    x: int
    y: int

# ✅ Function decorators work perfectly
@functools.lru_cache
def expensive_function(n):
    return n * 2
```

**Why?**: Standard Python class decorators expect real `type` objects, but the sandbox creates `AgexClass` objects (custom data structures). Function decorators work because they operate on callable objects, which the sandbox can provide.

**Impact**: Libraries that require class decorators (like some ORMs or validation frameworks) won't work. Use alternative patterns or provide pre-decorated classes.

**Future**: Would require major refactor to create real Python types instead of `AgexClass` objects.

### Async/Await
**Not supported**: Agents cannot generate async code; the sandbox is synchronous-only.

```python
# ❌ Agents cannot generate async code
async def fetch_data():
    await some_async_call()

# ✅ Register synchronous equivalents instead
def fetch_data():
    return requests.get("https://api.example.com")
```

**Impact**: Async libraries won't work. Provide synchronous wrappers or use libraries with sync APIs.

> [!TIP]
> **Registered async functions work transparently.** If you register an async function via `@agent.fn`, the framework automatically bridges async results. Agents call them like regular sync functions and receive resolved values—no special handling needed.

**Future**: Unlikely to change - async support would require major architectural changes.

### Exception Handling
**Limited**: Agents can only catch specific built-in exceptions (`ValueError`, `TypeError`, `KeyError`, etc.).

```python
# ✅ Agents can catch built-in exceptions
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except KeyError as e:
    print(f"Key error: {e}")

# ❌ Agents cannot catch custom exceptions
try:
    result = operation()
except CustomException:  # Won't catch properly
    pass
```

**Impact**: Libraries that rely on custom exceptions may not handle errors gracefully. Convert custom exceptions to standard ones in wrapper functions.

**Future**: Likely to be added - registering custom exceptions should be straightforward to implement.

### Generators and Yield
**Not supported**: Agents cannot generate code that uses `yield` or `yield from` to create generators.

```python
# ❌ Agents cannot create generators
def my_generator():
    yield 1
    yield 2
    yield 3

def delegating_generator():
    yield from range(5)

# ✅ Agents can return lists or other data structures instead
def my_list_function():
    return [1, 2, 3]

def delegating_list_function():
    return list(range(5))
```

**Impact**: Libraries that expect generator objects won't work. Provide list-returning alternatives or materialize generators before registering them.

**Future**: Unlikely to change - would require significant architectural changes.

### Global Variables
**Not supported**: Agents cannot use the `global` statement to modify global variables.

```python
# ❌ Agents cannot use global statement
counter = 0

def increment():
    global counter  # Not supported
    counter += 1

# ✅ Agents can use return values or mutable containers
def increment_counter(current_count):
    return current_count + 1

# Or use registered state management
def increment_with_state():
    # Assuming you've registered state management functions
    current = get_counter()
    set_counter(current + 1)
```

**Impact**: Functions that modify global state won't work. Use explicit state management or pass state as parameters.

**Future**: Unlikely to change - would break the sandbox security model.

### Unpicklable Objects - Automatic Handling

**When using `Versioned` state, unpicklable objects (like database cursors, file handles, network connections) are automatically detected and handled gracefully.**

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
| **Default (No State)** | ✅ Allowed (no persistence) | No |
| **`Live` State** | ✅ Allowed (in-memory) | Yes (in-process) |
| **`Versioned` State** | ✅ Auto-handled via markers | Yes (persistent) |

### Object Identity Between Executions
**Objects are reconstructed**: Between task executions (when using `Versioned` state), objects are serialized and deserialized. This breaks object identity (`id()`) and shared references between separate task runs.

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

**Impact**: Objects that rely on `is` checks or `id()` for identity across multiple task executions may behave unexpectedly. Use explicit state management and value-based comparisons instead.

**Future**: This is an inherent aspect of serialization-based persistence and is unlikely to change.

### Function Closures
**Captured variables are "frozen" on save**: When using `Versioned` state, closures work and persist, but any variables they capture from their enclosing scope are "frozen" with their current values when the task completes.

A closure will not see subsequent changes to a captured variable in a later task execution.

```python
# This example demonstrates the "freezing" behavior across two task runs.
# Assume the agent executes this code in its first task run:
factor = 2
def multiplier(x):
    # This closure captures the `factor` variable
    return x * factor
    
# The agent saves `multiplier` and `factor` to its state.
# `multiplier` is now "frozen" with `factor=2`.

# --- End of first task ---

# Now, assume the agent executes this code in a second task run:
# It loads `multiplier` and `factor` from its state.
factor = 10
# Even though `factor` is now 10 in the agent's state, the
# `multiplier` function is still using the value it was frozen with.
result = multiplier(5) # This will return 10, not 50.
```

**No `nonlocal` support**: The `nonlocal` statement is not supported. To modify state, use mutable containers (like a single-element list `[0]`) or have functions return the new state.

**Impact**: This can lead to unexpected behavior if you assume closures will always see the latest version of their captured variables across different task runs.

**Future**: The freezing behavior is inherent to the state model and is unlikely to change.

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

Beyond language restrictions, agex can enforce resource limits to prevent runaway code from exhausting system resources.

### Memory Limits

```python
agent = Agent(max_memory_mb=500)  # 500MB headroom per task
```

If agent code attempts to allocate more memory than allowed:

```python
# Agent tries to allocate 1GB with 500MB limit
data = bytearray(1024 * 1024 * 1024)
# Raises MemoryError (wrapped in EvalError)
```

The agent receives an error and can adjust its approach—for example, processing data in chunks.

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

## Why These Limitations?

These constraints exist for important reasons:

- **Security**: Prevents agents from accessing dangerous Python features
- **Serialization**: Enables memory and rollback by ensuring all persistent state can be saved
- **Sandboxing**: Ensures agent code cannot escape the execution environment
- **Resource Protection**: Prevents runaway code from exhausting memory, files, or storage

**Note**: With `Live` state, serialization constraints don't apply since no state is persisted between task calls. Choose `Versioned` state when you need agents to remember variables across multiple task executions.
