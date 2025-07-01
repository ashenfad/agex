# Top-Level API Design (`@agent.task`)

This document outlines the design for the high-level `@agent.task` decorator, which provides a developer-friendly way to define and interact with agent-driven tasks.

## Core Concepts

The primary goal is to make defining an agent task feel as natural as defining a Python function. The developer provides a function signature, docstring for documentation, and optionally a primer for agent implementation instructions. The agent handles the implementation.

### Decorator Behavior

1.  **Function Replacement:** The `@agent.task` decorator does not merely wrap the decorated function; it completely **replaces** it.
2.  **Implementation Constraint:** The body of a function decorated with `@agent.task` must be empty (i.e., contain only `pass`). The decorator will raise an exception if it finds any other statements. The agent, not the developer, is responsible for the implementation.
3.  **Signature and Documentation:** The decorator captures the function definition:
    *   Its name and argument signature (parameter names, types, defaults)
    *   Its return type annotation
    *   Its docstring for caller documentation
    *   An optional primer parameter for agent-specific implementation instructions
4.  **Agent Loop Trigger:** The new function that replaces the original one will be responsible for triggering the agent's internal "think-act" loop to produce a result that matches the requested return type.
5.  **State Management:** The replacement function will have a new, optional parameter added to its signature: `state: Versioned | None = None`.
    *   If a `Versioned` object is provided, the agent will use it for long-term memory across multiple calls.
    *   If no state object is provided (the default), the agent will operate in a single-shot, ephemeral mode for that call.

### Two Decorator Patterns

The `@agent.task` decorator supports two usage patterns:

#### **Naked Decorator (Docstring as Instructions)**
```python
@agent.task
def generate_data(prompt: str) -> list[np.ndarray]:
    """Generate numpy arrays based on the prompt description."""
    pass
```
The function's docstring serves both as agent instructions and caller documentation.

#### **Parameterized Decorator (Separate Primer and Docstring)**
```python
@agent.task("Use numpy and random modules to create realistic synthetic data")
def generate_data(prompt: str) -> list[np.ndarray]:
    """
    Generate synthetic numpy arrays from a text description.
    
    Args:
        prompt: Description of the data to generate
        
    Returns:
        List of numpy arrays representing the generated data
    """
    pass
```
The primer provides agent-specific implementation guidance, while the docstring remains caller-focused.

## API Design Philosophy: Definable vs. Configurable

The agent registration API is built on a core principle that makes its syntax consistent and predictable:

*   **Definable** methods are for registering new functions (`def`) and classes (`class`) as they are being defined. These methods (`.fn`, `.cls`) support the decorator pattern (`@agent.fn`) and therefore use a "decorator factory" syntax: `agent.fn(my_func)`. This separates configuration from application.

*   **Configurable** methods are for exposing parts of *already existing* objects, like imported modules. These methods (`.module`, and in the future, `.method` and `.attr`) are direct function calls: `agent.module(math, ...)`.

This distinction means the syntax itself tells you what kind of operation you are performing.

## Registering Capabilities (`.fn`, `.cls`, `.module`)

Before an agent can execute a task, it needs capabilities: functions it can call and data structures it can understand. We register these using methods on the `Agent` instance. This design allows for multiple, independent agents with different skill sets.

### Visibility Control

A core challenge is managing the LLM's context window. We don't want to overwhelm the agent's prompt with docs for every single available function. To solve this, all registration methods have a `visibility` parameter. The meaning of each level depends on the type of object being registered:

| Visibility | For a Function (`my_func`)                                 | For a Constant (`pi`)                              |
| :--- | :--- | :--- |
| **`high`**   | Shows signature + docstring                                | Shows name + value                                 |
| **`medium`** | Shows signature only                                       | Shows name + type                                  |
| **`low`**    | Available, but not shown                                   | Available, but not shown                           |

### Registering Functions (`.fn`)

The `.fn()` method registers a function as a tool the agent can use. It can be used as a decorator or a direct function call. It also allows for overriding the function's docstring, which is useful when a function needs a different prompt for an agent than its standard documentation.

```python
agent = Agent()

# As a decorator, overriding the docstring
@agent.fn(docstring="A simple tool that returns a boolean.")
def my_tool(arg: str) -> bool:
    """This is the original, detailed docstring for human developers."""
    return True

# As a direct call with low visibility
import math
# Note the double parentheses: the first call configures, the second applies.
agent.fn(visibility="low")(math.sqrt)
```

### Registering Classes (`.cls`)

The `.cls()` method registers a class, giving the agent access to its attributes and methods. This is where fine-grained control becomes essential.

#### Pattern Logic

The `include` and `exclude` parameters use flexible patterns to specify what members to expose:

1.  **String (Glob Pattern):** A single string is treated as a glob pattern (e.g., `"get_*"`). `"*"` selects all members.
2.  **List/Set of Strings:** A collection of glob patterns. A member is included if it matches any pattern in the list.
3.  **Callable Predicate:** A function that takes the member name and returns `True` or `False`. This allows for arbitrary custom logic.

Members are included if they match the `include` pattern AND do not match the `exclude` pattern.

#### Example Usage

```python
from tic.agent import Agent

agent = Agent()

@dataclass
class UserProfile:
    user_id: str
    _internal_key: str
    name: str

    def get_info(self):
        pass
    
    def _validate_internal(self):
        pass

# Register the class, exposing all non-private members
# with medium visibility to save context.
agent.cls(UserProfile, include="*", exclude="_*", visibility="medium")

# This is equivalent to:
# agent.cls(UserProfile, include=["user_id", "name", "get_info"], ...)

# Or use a custom predicate:
# agent.cls(UserProfile, include=lambda n: not n.startswith('_'), ...)
```

### Registering Modules (`.module`)

For exposing entire libraries, `agent.module()` is the preferred "power tool." It allows you to register functions, classes, and constants from a module using include/exclude patterns.

By default, `include="*"` and `exclude=["_*", "*._*"]`, which safely exposes the public API while excluding private members at both the top level and within classes.

```python
import math
import pandas as pd

# Default: Exposes public members, excludes privates
agent.module(math, name="math")

# Custom patterns: Only expose specific functions
agent.module(math, name="math", include=["sqrt", "sin", "cos", "pi"])

# Include everything (even privates) except specific patterns
agent.module(pandas, name="pd", include="*", exclude=["_*", "*.test_*"])

# Exclude class private methods but allow top-level privates
agent.module(my_module, include="*", exclude="*._*")
```

#### Class Method Patterns

For modules containing classes, you can use dotted notation to target specific class members:

```python
# Include only specific class methods
agent.module(requests, include=["Session", "Session.get", "Session.post"])

# Include all public methods but exclude specific ones
agent.module(my_lib, include="*", exclude=["_*", "MyClass.dangerous_method"])
```

If you need more fine-grained control over individual members (e.g., making one class non-constructable), you should use the more specific methods like `.cls()` and `.fn()` after the initial bulk registration.

### Overriding

A key principle is that **more specific registrations override more general ones**. This allows for powerful workflows where you can bulk-register a class or module with low visibility, and then "promote" specific, important members to high visibility using a more specific registration method.

For example, `agent.module(..., visibility="low")` can be partially overridden by a future direct call like `agent.fn(module.some_func, visibility="high")`.

## Developer Experience Example

From the developer's perspective, defining and using an agent task should be as simple as:

```python
from agex import Agent, Versioned

# 1. Instantiate the agent
my_agent = Agent(primer="You are a helpful assistant.")

# 2. Define a task - multiple patterns supported:

# Option A: Naked decorator (docstring serves both agent and caller)
@my_agent.task
def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a random integer between the given values, inclusive."""
    pass

# Option B: Separate primer and docstring (when they need to differ)
@my_agent.task("Use Python's random module with proper seeding for reproducibility")
def advanced_random(min_val: int, max_val: int) -> int:
    """
    Generate a random integer within a specified range.
    
    Args:
        min_val: The minimum value (inclusive)
        max_val: The maximum value (inclusive)
        
    Returns:
        A random integer between min_val and max_val
    """
    pass

# 3. Use the agent-powered function directly
# (Agent runs in single-shot mode)
random_num = generate_random_number(1, 100)
print(f"The agent generated: {random_num}")

# Or, use it with persistent state
agent_state = Versioned()
another_num = generate_random_number(1, 100, state=agent_state)

```

This design abstracts away the complexity of the evaluation loop, state management, and prompting, providing a clean and intuitive API. 