# Nearly Python: Understanding Agent Code Constraints

agex agents generate and execute code in a secure sandbox that looks and feels like Python—but with some important differences. This guide helps you understand what constraints agents face when writing code, so you can design better integrations and understand agent behavior.

**State Choice Affects Constraints**: Some limitations depend on whether you use live state (default, no persistence) or persistent state (remembers variables between task calls). Live state is more flexible but doesn't persist memory; persistent state has more constraints but enables complex multi-step workflows.

## What Works (Agent-Generated Code)

Most Python features work exactly as you'd expect when agents generate code:

- **Basic operations**: arithmetic, string manipulation, list/dict operations
- **Control flow**: `if/else`, `for/while` loops, function calls
- **Built-in functions**: `print()`, `len()`, `range()`, `enumerate()`, etc.
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

### Class Inheritance
**Cannot create subclasses**: Agents cannot generate code that defines new classes with inheritance.

```python
# ❌ Agents cannot create new subclasses
class MyList(list):
    def special_method(self):
        return "custom"

# ✅ Agents can use existing classes with inheritance
existing_list = MySpecialList()  # If MySpecialList inherits from list
existing_list.append(1)          # Works fine

# ✅ Agents can create composition-based classes
class MyList:
    def __init__(self):
        self.items = []
    
    def special_method(self):
        return "custom"
```

**Impact**: Agents cannot implement abstract classes or create new subclasses. Design APIs to provide concrete implementations rather than requiring agents to subclass.

**Future**: Unlikely to change - would require significant architectural changes.

### Decorators  
**Cannot use @ syntax**: Agents cannot generate code that uses the `@decorator` syntax.

```python
# ❌ Agents cannot use @ syntax
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def my_function():
    pass

# ✅ Agents can call decorators as functions
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

def my_function():
    pass

my_function = my_decorator(my_function)  # Works fine
```

**Impact**: Decorator-heavy libraries work fine if they can be called as functions. The `@` syntax is just syntactic sugar.

**Future**: Likely to be added - the syntax is straightforward to implement.

### `with` Statements for Temporary Scope
**Special behavior for unpicklable objects**: When using `Versioned` (persistent) state, the `with` statement has a special purpose: it allows an agent to temporarily work with an unpicklable object that will not be saved to the agent's permanent memory.

Any variable created inside a `with` block is "ephemeral"—it exists only for the duration of that block and is discarded afterward. This provides a safe way to interact with stateful, unpicklable resources like database connections.

```python
# Assume `db_connection` is an unpicklable database connection
# object that has been registered with the agent.

# ❌ This code will FAIL with Versioned state because `cursor` is unpicklable.
# cursor = db_connection.cursor()
# cursor.execute("SELECT * FROM users")
# results = cursor.fetchall()

# ✅ This is the correct pattern.
# The cursor is used and discarded within the temporary scope.
with db_connection.cursor() as cursor:
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()

# The `results` variable (a list of tuples) IS picklable and can be
# stored in the agent's state or returned. The `cursor` object is gone.
```

**Impact**: This is the primary pattern for using context managers that yield unpicklable resources (like database cursors) when working with `Versioned` state. It allows agents to perform stateful operations without violating the pickling requirement for persistent memory.

**Future**: This is a core feature of the sandbox and is unlikely to change.

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

### Unpicklable Objects
**Your choice of state object determines whether agents can assign unpicklable objects to variables.**

This is a critical concept when working with stateful resources like database connections, file handles, or network sockets.

#### Mode 1: No State (Default)
When you call a task with no `state` parameter, the execution is self-contained. Agents can freely use unpicklable objects, but nothing is remembered between calls.

```python
# ✅ Works perfectly fine
def process_query(query: str):
    cursor = db.execute(query) # Assigning cursor is okay
    return cursor.fetchall()
```
**Use for:** Simple, single-shot tasks that don't require memory.

#### Mode 2: Live State
When you pass a `Live` state object (`state=Live()`), agents gain in-process memory and can still freely assign unpicklable objects to variables. This is the ideal mode for multi-step workflows involving live resources.

```python
# ✅ Live state allows assigning unpicklable objects
def multi_step_db_work(queries: list[str]):
    # The `state` parameter is added automatically by the @agent.task decorator
    for query in queries:
        cursor = db.execute(query) # Storing cursor in state is okay
        # ... do more work ...
```
**Use for:** Multi-step workflows that need to remember stateful, unpicklable objects like database cursors or file handles.

#### Mode 3: Versioned State
When you pass a `Versioned` state object (`state=Versioned()`), state is persisted and versioned, which requires all stored objects to be picklable.

```python
# ❌ With Versioned state, agents cannot assign unpicklable objects
cursor = db.execute("SELECT * FROM users")  # ERROR: Cannot assign cursor to a variable
result = cursor.fetchall()

# ✅ Must chain operations immediately
result = db.execute("SELECT * FROM users").fetchall()
```
**Use for:** Production workflows requiring persistence, rollback capabilities, and multi-agent coordination.

### Summary of Approaches

| State Mode | Unpicklable Objects | Memory Between Calls |
| :--- | :--- | :--- |
| **Default (No State)** | ✅ Allowed | No |
| **`Live` State** | ✅ Allowed | Yes (in-process) |
| **`Versioned` State** | ❌ Not Allowed | Yes (persistent) |

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

## Why These Limitations?

These constraints exist for important reasons:

- **Security**: Prevents agents from accessing dangerous Python features
- **Serialization**: Enables memory and rollback by ensuring all persistent state can be saved
- **Sandboxing**: Ensures agent code cannot escape the execution environment

**Note**: With `Live` state, serialization constraints don't apply since no state is persisted between task calls. Choose `Versioned` state when you need agents to remember variables across multiple task executions.
