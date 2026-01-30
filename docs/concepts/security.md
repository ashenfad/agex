# Security Model

`agex` provides a secure Python execution environment for AI agents through a comprehensive multi-layer security strategy.

### Philosophy: A Walled Garden, Not a Fortress

The security model of `agex` is best understood as a **"walled garden,"** not an impenetrable fortress. Its primary goal is to provide a robust safety net that prevents common and foreseeable errors—both accidental and malicious—within its intended use case. It is designed to stop an agent from inadvertently deleting files, accessing sensitive data, or otherwise impacting the host system in unintended ways.

This approach is a deliberate trade-off. While a fully isolated Virtual Machine (VM) offers a higher degree of theoretical security, it comes at a cost to ease of integration.

`agex` is significantly safer than frameworks that rely on direct `exec()` calls, but is not intended to be a substitute for a VM when running fully untrusted code from unknown sources.

## Core Security Strategy

The `agex` sandbox uses a **whitelist-based security model** with these key components:

- **AST-level validation**: All Python code is parsed and validated by the execution engine to block dangerous language features.
- **Attribute access control**: Only explicitly whitelisted attributes and methods are accessible.
- **Data Isolation**: When using `Versioned` state, all data is serialized at the boundary of an agent task. Agents never get a direct reference to host objects, preventing accidental or malicious mutation.
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

The built-in `dir()` and `help()` functions are overridden to show only the attributes and methods that have been explicitly whitelisted for the agent. This allows for useful introspection without leaking access to sensitive internal methods or private attributes (those prefixed with `_`).

For a complete overview of all sandbox limitations, see our [Nearly Python guide](./nearly-python.md).

## Resource Limits

Beyond code validation, agex provides defense-in-depth resource limiting to protect against catastrophic resource exhaustion (e.g., `[0] * 10**9` or infinite file creation):

### Memory Limits

```python
agent = Agent(max_memory_mb=500)  # 500MB headroom per task
```

Memory limits use a **delta-based approach**: the limit is set to current process memory + configured headroom. This gives each task a budget for new allocations without counting existing process memory.

When exceeded, Python raises `MemoryError`, which the agent sees as an `EvalError` with the underlying `MemoryError` in the cause chain.

### File Descriptor Limits

```python
agent = Agent(max_open_files=256)  # Max 256 open files
```

Prevents agents from exhausting system file descriptors through excessive file operations.

### VFS Size Limits

```python
agent = Agent(
    fs=connect_fs(type="virtual", max_size_mb=100),  # 100MB total VFS size
)
```

Limits the total size of all files in the Virtual FileSystem, preventing unbounded storage consumption.

### Platform Support

Resource limits use Unix `RLIMIT_AS` and `RLIMIT_NOFILE`, supported on Linux and macOS. On Windows, limits are not enforced (a warning is issued).

### Process-Level Limits

Memory and file descriptor limits are process-wide. For concurrent tasks, size limits according to expected concurrency. For per-task isolation, use the Modal integration which provides containerized execution.

See [Agent Resource Limits](../api/agent.md#resource-limits) and [VFS Size Limits](../api/fs.md#size-limits) for configuration details.

## Network Access Control

By default, agent code cannot make network connections. This prevents agents from exfiltrating data, making unauthorized API calls, or accessing internal services.

### How It Works

Network access is controlled at the socket level. During agent code evaluation, all network socket operations (connect, send, recv, bind, listen, DNS lookups) are blocked by default. This affects all networking code, including HTTP libraries like `requests`, `httpx`, and `aiohttp`.

```python
# Agent code trying to make HTTP requests will fail:
import requests
response = requests.get("https://example.com")  # SandboxError: Network access denied
```

### Granting Network Access

To allow specific functions, classes, or modules to use the network, register them with `network_access=True`:

```python
import requests

# Allow this specific function to make network calls
@agent.fn(network_access=True)
def fetch_weather(city: str) -> dict:
    """Fetch weather data for a city."""
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()

# Or register an HTTP client class
agent.cls(requests.Session, network_access=True)

# Or an entire module
agent.module(requests, network_access=True)
```

### Practical Example: pandas

Many libraries have functions that work with both local files and URLs. For example, pandas' `read_csv` can load data from disk or from the network:

```python
import pandas as pd

# Register pandas WITHOUT network_access
agent.module(pd, visibility="low", recursive=True)
```

With this registration, agent code can read local files but not URLs:

```python
# ✅ Works - reads from VFS or local disk
df = pd.read_csv("data/sales.csv")

# ❌ Blocked - attempts network connection
df = pd.read_csv("https://example.com/data.csv")  # SandboxError
```

This gives agents full pandas functionality for data analysis while preventing them from loading arbitrary data from the internet or exfiltrating data to external servers.

If you want to allow network access for pandas (e.g., for fetching public datasets), register it with `network_access=True`:

```python
agent.module(pd, visibility="low", recursive=True, network_access=True)
```

### What Gets Blocked

The sandbox intercepts these socket operations when network access is denied:

| Operation | Purpose |
|-----------|---------|
| `connect` / `connect_ex` | Outbound connections |
| `bind` / `listen` / `accept` | Server sockets |
| `send` / `sendall` / `sendto` / `sendfile` | Sending data |
| `recv` / `recvfrom` / `recv_into` / `recvfrom_into` | Receiving data |
| `getaddrinfo` | DNS resolution |

### What's Allowed

Local inter-process communication (Unix domain sockets, `AF_UNIX`) is always allowed. This ensures Python's internal mechanisms (like asyncio's event loop) continue to work correctly. Only actual network sockets (`AF_INET`, `AF_INET6`) are gated.

### Error Handling

When network access is denied, a `SandboxError` is raised with a helpful message:

```
SandboxError: Network access denied: connect() blocked by sandbox.
Register function with network_access=True to allow network operations.
```

See [Registration Methods](../api/registration.md) for details on the `network_access` parameter.
