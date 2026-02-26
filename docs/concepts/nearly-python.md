# Nearly Python: Sandbox Restrictions

agex agents generate and execute code in a secure sandbox powered by [sandtrap](https://github.com/ashenfad/sandtrap). The sandbox uses AST rewriting to compile agent code into restricted bytecode. By default this runs in-process, but can be configured for subprocess or kernel-level isolation (see [Sandbox Isolation](#sandbox-isolation)). The result looks and feels like Python — but with some important differences.

!!! important "Imports: Registered or VFS-Resident"

    Agent-generated code may use `import` statements. These only succeed for:
    1. **Registered Modules**: Libraries explicitly exposed via `agent.module(...)`.
    2. **Workspace Modules**: Python files the agent has created in the Virtual Filesystem (VFS).

    Within registered modules, only whitelisted members are visible. For workspace modules, all members are available.

    ```python
    import pandas as pd              # OK if `pandas` was registered
    import helpers.utils             # OK if agent created `helpers/utils.py`
    import os                        # Fails if `os` was not registered
    ```

## What Works

Most Python features work exactly as you'd expect:

- **Basic operations**: arithmetic, string manipulation, list/dict operations
- **Control flow**: `if/else`, `for/while` loops, `match/case` (Python 3.10+), function calls
- **Built-in functions**: `print()`, `len()`, `range()`, `enumerate()`, `sorted()`, etc.
- **Classes**: full `class` definitions with inheritance, methods, and dunders
- **Generators**: `yield` and `yield from`
- **Closures**: nested functions with `nonlocal` and `global`
- **Exception handling**: `try/except/finally/else` with 50+ built-in exception types
- **Decorators**: function and class decorators
- **Comprehensions**: list, dict, set, and generator comprehensions

## What's Different

### Three-Argument `type()`
**Blocked**: Dynamic class creation via `type('Name', bases, dict)` is rejected. Single-argument `type(obj)` works normally for inspection.

```python
# ✅ Allowed: type inspection
t = type(42)           # <class 'int'>
isinstance(42, t)      # True

# ❌ Blocked: dynamic class creation
MyClass = type('MyClass', (object,), {'x': 1})  # TypeError
```

### Format String Traversal
**Blocked**: `.format()` and `.format_map()` reject attribute and item traversal in field names. F-strings are unaffected (they go through proper AST validation).

```python
# ✅ Allowed: simple positional/keyword formatting
"Hello {name}".format(name="World")

# ❌ Blocked: attribute traversal via format string
"{obj.__class__}".format(obj=x)  # AttributeError

# ✅ Secure alternative: use f-strings
f"{obj.attr}"
```

### Bare `except:` Clause
**Rewritten**: A bare `except:` is automatically rewritten to `except Exception:`. This prevents agent code from swallowing control exceptions (`KeyboardInterrupt`, sandbox timeout/cancellation signals, etc.).

### Unavailable Names
The following names raise `NameError`:

- **Control exceptions**: `BaseException`, `KeyboardInterrupt`, `GeneratorExit`, `SystemExit`
- **Dangerous builtins**: `exec`, `eval`, `compile`
- **Introspection**: `globals()`

`locals()` is available but returns a filtered copy (sandbox internals excluded). Python's built-in `dir()` and `help()` are available.

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

## Async Bridging

Agent-generated code always runs synchronously — no `async`/`await` syntax. However, async functions registered via `@agent.fn` are bridged transparently: agents call them like sync functions and the framework awaits the result automatically. See [Task - Async Execution](../api/task.md#async-execution) for details.

## Versioned State Serialization

When using versioned state, agent variables are serialized between task executions. This has some implications for object identity, closures, and unpicklable objects. See [State - Serialization Behavior](../api/state.md#serialization-behavior) for details.

## Resource Limits

agex can enforce memory, file descriptor, and VFS size limits via [sandtrap](https://github.com/ashenfad/sandtrap). See [Agent - Resource Limits](../api/agent.md#resource-limits) for configuration.

## Sandbox Isolation

By default, agent code runs in-process. For crash protection or kernel-level security, set the `isolation` parameter. See [Security - Sandbox Isolation](security.md#sandbox-isolation) and [Agent - Sandbox Isolation](../api/agent.md#sandbox-isolation).
