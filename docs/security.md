# Security Documentation

## Overview

`agex` provides a secure Python execution environment for AI agents. This document outlines the security measures built into the framework.

## Security Model

The `agex` sandbox uses a **whitelist-based security model**:

- **AST-level validation**: All Python code is parsed and validated before execution
- **Attribute access control**: Only explicitly whitelisted attributes/methods are accessible
- **Type system isolation**: Safe type placeholders prevent access to dangerous type methods
- **Function registration**: Only pre-registered functions are callable
- **Import restrictions**: Module imports are controlled through explicit registration

## Security Protections

### 🛡️ String Format Security

**Protection Level**: CRITICAL

The framework includes comprehensive protection against string format injection attacks. Python's `.format()` method is intercepted at the AST level to prevent sandbox bypass.

**Threat Model**: String formatting could potentially bypass sandbox restrictions by performing attribute resolution at the C level, outside of AST validation.

**Protection Mechanism**: AST-level interception in `CallEvaluator.visit_Call()`:
- Detects all `.format()` calls on strings
- Uses custom `SandboxFormatter` that blocks dotted field access
- Provides clear error messages directing users to f-strings
- Maintains compatibility with simple format strings

**Allowed vs. Blocked Operations**:
```python
# ✅ Allowed:
"Hello {name}".format(name="World")

# ❌ Blocked:
"{obj.attr}".format(obj=obj)  # SecurityError: Format string attribute access not allowed

# ✅ Secure alternative:
f"{obj.attr}"  # Uses proper AST validation
```

### 🛡️ Type System Security

**Protection Level**: HIGH

The framework provides robust protection against type-based attacks through a dual-layer approach:

**Protection Mechanisms**:
- `_TicTypePlaceholder` objects have zero whitelisted attributes
- All dangerous methods blocked: `__subclasses__`, `__bases__`, `__mro__`
- Constructor delegation works safely for legitimate use cases
- Defense-in-depth: Both placeholder design AND sandbox rules protect

**Allowed Operations**:
```python
# ✅ Safe type operations:
type(42)(123)              # Constructor calls
isinstance(42, type(42))   # Type checking
isinstance(x, (int, str))  # Tuple type checking

# ❌ Blocked dangerous operations:
type(42).__subclasses__    # Classic SSTI attack vector
type(42).__bases__         # Type hierarchy access
type(42)._wrapped_type     # Direct access to real type
```

### 🛡️ Core AST Evaluation

**Protection Level**: HIGH

- Comprehensive whitelist-based attribute access
- Safe handling of user-defined classes and dataclasses
- Proper import isolation and module registration

### ℹ️ Information Disclosure (By Design)

**Risk Level**: LOW

Limited information exposure through introspection functions:
- `dir()` output shows available attributes (by design for usability)
- `help()` function documentation (necessary for agent functionality)
- `repr()` and `str()` on placeholder objects (minimal exposure)

**Design Decision**: Information disclosed is intentionally limited to what's needed for agent functionality.

## Security Best Practices

### For Framework Users

1. **Use F-strings Instead of .format()**:
   ```python
   # ✅ Secure:
   f"Hello {user.name}"
   
   # ❌ Blocked:
   "Hello {user.name}".format(user=user)
   ```

2. **Register Minimal Necessary Attributes**:
   ```python
   # Only expose what's needed
   agent.cls(MyClass, include=['safe_method'], exclude=['_private'])
   ```

3. **Validate User Inputs**:
   ```python
   # Always validate data before processing
   if not isinstance(user_input, expected_type):
       raise ValueError("Invalid input type")
   ```

### For Framework Developers

1. **Default to Whitelist-Only**:
   - Never add attributes to whitelists without security review
   - Prefer explicit inclusion over exclusion patterns

2. **AST-Level Validation**:
   - Intercept potentially dangerous operations at AST level
   - Validate all dynamic operations before execution

3. **Defense in Depth**:
   - Multiple security layers (AST + attribute access + type placeholders)
   - Fail-safe defaults (block by default, allow explicitly)

## Security Testing

### Automated Tests

The framework includes comprehensive security tests:

- **String format security tests**: `tests/agex/eval/test_string_format_security.py`
- **Sandbox escape tests**: `tests/agex/eval/test_sandbox_scenarios.py`
- **Type system security**: Verified through integration tests

### Manual Security Review Checklist

When adding new features:

- [ ] All new attributes added to whitelists are reviewed
- [ ] Dynamic operations go through AST validation
- [ ] No direct access to Python internals
- [ ] Error messages don't leak sensitive information
- [ ] New functionality includes security tests

## Known Limitations

1. **Performance Impact**: Security checks add computational overhead
2. **Feature Restrictions**: Some Python features are intentionally disabled for security
3. **Debug Information**: Limited stack traces to prevent information leakage

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** create a public issue
2. Contact the maintainers directly
3. Provide minimal reproduction case
4. Include impact assessment
