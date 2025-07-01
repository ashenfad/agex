# Security Model

`agex` provides a secure Python execution environment for AI agents through a comprehensive multi-layer security strategy.

## Core Security Strategy

The `agex` sandbox uses a **whitelist-based security model** with these key components:

- **AST-level validation**: All Python code is parsed and validated before execution
- **Attribute access control**: Only explicitly whitelisted attributes/methods are accessible  
- **Type system isolation**: Safe type placeholders prevent access to dangerous type methods
- **Function registration**: Only pre-registered functions are callable
- **Import restrictions**: Module imports are controlled through explicit registration
- **Serialization boundaries**: All data is serialized into/out of agent tasks, preventing accidental mutations to host objects

## Python Language Restrictions

The framework modifies standard Python behavior in specific areas to maintain security:

### String Formatting (.format() method)

Python's `.format()` method is intercepted at the AST level to prevent attribute access attacks that could bypass sandbox restrictions.

```python
# ✅ Allowed:
"Hello {name}".format(name="World")

# ❌ Blocked:
"{obj.attr}".format(obj=obj)  # SecurityError: Format string attribute access not allowed

# ✅ Secure alternative:
f"{obj.attr}"  # Uses proper AST validation
```

### Type System (type() builtin)

The `type()` builtin returns safe placeholder objects that block access to dangerous type methods while preserving legitimate functionality.

```python
# ✅ Safe operations:
type(42)(123)              # Constructor calls
isinstance(42, type(42))   # Type checking  
isinstance(x, (int, str))  # Tuple type checking

# ❌ Blocked operations:
type(42).__subclasses__    # Class hierarchy introspection
type(42).__bases__         # Type system access
type(42).__mro__           # Method resolution order
```

### Introspection Functions

Limited information exposure through introspection functions like `dir()` and `help()` is intentional - these provide the minimum information needed for agent functionality while blocking dangerous operations.
