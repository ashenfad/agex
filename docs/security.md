# Security Model

`agex` provides a secure Python execution environment for AI agents through a comprehensive multi-layer security strategy.

## Core Security Strategy

The `agex` sandbox uses a **whitelist-based security model** with these key components:

- **AST-level validation**: All Python code is parsed and validated before execution to block dangerous language features.
- **Attribute access control**: Only explicitly whitelisted attributes and methods are accessible.
- **Data Isolation**: All data is serialized at the boundary of an agent task. Agents never get a direct reference to host objects, preventing accidental or malicious mutation.
- **Type system isolation**: Safe type placeholders prevent access to dangerous methods on type objects.
- **Import restrictions**: Module imports are controlled through explicit registration.

## Python Language Restrictions

The framework modifies standard Python behavior in specific areas to maintain security:

### String Formatting (.format() method)

Python's `.format()` method is intercepted to prevent attribute access attacks. Malicious code can use format strings to introspect and access arbitrary attributes of objects, a common technique for escaping sandboxes.

```python
# ✅ Allowed: Simple key-based formatting
"Hello {name}".format(name="World")

# ❌ Blocked: Attribute access via format string
"{obj.attr}".format(obj=obj)  # SecurityError: Format string attribute access not allowed

# ✅ Secure alternative: f-strings use proper AST validation
f"{obj.attr}"
```

### Type System (`type()` builtin)

The `type()` builtin is overridden to return safe placeholder objects. This blocks access to dangerous, low-level methods on type objects while preserving legitimate functionality like `isinstance()` checks.

The primary goal is to prevent access to `object.__subclasses__()`, which can be used to traverse the entire class hierarchy of a Python application, find sensitive classes (like `os._wrap_close`), and achieve arbitrary code execution.

```python
# ✅ Safe operations:
type(42)(123)              # Constructor calls
isinstance(42, type(42))   # Type checking  

# ❌ Blocked operations that allow sandbox escapes:
type(42).__subclasses__    # Class hierarchy introspection
type(42).__mro__           # Method resolution order
```

### Introspection Functions

The built-in `dir()` and `help()` functions are overridden to only show the attributes and methods that have been explicitly whitelisted for the agent. This allows for useful introspection without leaking access to sensitive internal methods or private attributes (those prefixed with `_`).

For a complete overview of all sandbox limitations, see our [Nearly Python guide](./nearly-python.md).
