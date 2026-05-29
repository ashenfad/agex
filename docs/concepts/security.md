# Security Model

`agex` provides a secure Python execution environment for AI agents through a comprehensive multi-layer security strategy.

### Philosophy: Defense in Depth

The security model of `agex` is **layered**. At the base, AST-level validation and attribute access control provide a robust safety net that prevents common and foreseeable errors—both accidental and malicious. On top of that, **process and kernel isolation** provide stronger guarantees for production and high-security environments.

| Isolation Level | What It Adds |
|----------------|-------------|
| `"none"` (default) | In-process AST sandbox. Prevents common errors, blocks dangerous builtins and imports. Good for local development and trusted agents. |
| `"process"` | Subprocess execution. A crash, OOM, or segfault in agent code won't take down the host process. |
| `"kernel"` | Subprocess + OS-level restrictions (seccomp/Landlock on Linux, Seatbelt on macOS). Filesystem, syscall, and network access are locked down by the kernel. |

With `isolation="none"`, agex is significantly safer than frameworks that rely on direct `exec()` calls, but is not intended to be a substitute for a VM when running fully untrusted code. For stronger guarantees, use `isolation="process"` or `"kernel"`.

## Core Security Strategy

The `agex` sandbox uses a **whitelist-based security model** with these key components:

- **AST-level validation**: All Python code is parsed and validated by the execution engine to block dangerous language features.
- **Attribute access control**: Only explicitly whitelisted attributes and methods are accessible.
- **Data Isolation**: When using versioned state, all data is serialized at the boundary of an agent task. Agents never get a direct reference to host objects, preventing accidental or malicious mutation.
- **Dynamic class creation blocked**: Three-argument `type()` is rejected, preventing agents from creating classes that bypass AST-level validation.
- **Dangerous builtins removed**: `exec`, `eval`, `compile`, `globals`, and control exceptions are not available in agent code.
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

Single-argument `type(obj)` works normally for type inspection. Three-argument `type('Name', bases, dict)` — which dynamically creates new classes — is blocked. This prevents agents from creating classes that bypass the AST rewriter's validation of class definitions, which is needed to enforce attribute access control on instances.

```python
# ✅ Allowed: type inspection
t = type(42)           # <class 'int'>
isinstance(42, t)      # True

# ❌ Blocked: dynamic class creation
MyClass = type('MyClass', (object,), {'x': 1})  # TypeError
```

### Unavailable Builtins

The following names are not available in agent code (raise `NameError`):

- **Dangerous builtins**: `exec`, `eval`, `compile`
- **Introspection**: `globals()`
- **Control exceptions**: `BaseException`, `KeyboardInterrupt`, `GeneratorExit`, `SystemExit`

`locals()` is available but returns a filtered copy (sandbox internals excluded). Python's built-in `dir()` and `help()` are available.

For a complete overview of all sandbox limitations, see our [Nearly Python guide](./nearly-python.md).

## Resource Limits

Beyond code validation, agex provides defense-in-depth resource limiting to protect against catastrophic resource exhaustion (e.g., `[0] * 10**9` or infinite file creation):

### Memory Limits

```python
agent = Agent(max_memory_mb=500)  # 500MB headroom per task
```

Memory limits are enforced by [sandtrap](https://github.com/ashenfad/sandtrap)'s sandbox. On Linux, `RLIMIT_AS` provides kernel-enforced virtual address space caps. On macOS, enforcement is checkpoint-based (fires at loop iterations, function entries, and builtin calls). When exceeded, the agent sees a `MemoryError` wrapped in `EvalError`.

### File Descriptor Limits

```python
agent = Agent(max_open_files=256)  # Max 256 open files
```

Prevents agents from exhausting system file descriptors through excessive file operations. Uses `RLIMIT_NOFILE` on Linux/macOS.

### VFS Size Limits

```python
agent = Agent(
    fs=connect_fs(type="virtual", max_size_mb=100),  # 100MB total VFS size
)
```

Limits the total size of all files in the Virtual FileSystem, preventing unbounded storage consumption.

### Platform Support

Memory limits are handled by [sandtrap](https://github.com/ashenfad/sandtrap) (Linux: kernel-enforced, macOS: checkpoint-based, Windows: no-op). File descriptor limits use `RLIMIT_NOFILE` on Linux/macOS. On Windows, resource limits are not enforced (a warning is issued).

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

## Capability Scopes (Human-in-the-Loop)

Network access is one fixed gate. **Scopes** generalize the idea: any
registration can be tagged with a named `scope=`, making it **locked by
default** and available only in sessions that have been *granted* it.

This shifts the capability model. Normally an agent's surface is **static** —
it either can or can't do a thing, decided at registration. With scopes the
effective surface is a function of **`(agent, session)`**, and — crucially —
the agent can **request** access at runtime rather than the host hard-coding
every gate:

- **Least privilege by default.** A scoped capability is simply absent from a
  session's effective policy until granted — so using it fails exactly as any
  unregistered name would. There's no new enforcement boundary; gating rides
  the same sandbox you already trust (the locked capability is replaced by a
  stand-in that raises `ScopeRequired`).
- **The agent asks; the human decides.** The agent ends its turn with
  `task_request_permission(scope, reason)`, which suspends the task. The host
  sees the request, decides, and resumes. The agent **cannot grant itself** a
  scope: grants live in host-private session state the sandboxed code can't
  read or write.
- **Grants are per-`(agent, session)` and durable.** A request is committed to
  the event log, so it can be surfaced and resolved later — even after a
  restart, on versioned state — making genuine "approve it next week"
  human-in-the-loop flows possible, not just synchronous prompts.

This is the security-relevant half (conditional, least-privilege capability
access); the control-flow mechanics (the suspend/resume cycle) are covered in
[Task — Requesting Permission](../api/task.md#requesting-permission-scopes),
and declaring scopes in [Registration — Scoped Capabilities](../api/registration.md#scoped-capabilities).

> Scoped capabilities are **top-level only** in v1 — an agent with scopes
> can't be composed as a sub-agent, since scope-needs in a sub-agent aren't
> resolved yet.

## Sandbox Isolation

By default, agent code runs in-process (`isolation="none"`), relying on AST-level validation for security. For stronger guarantees, agex supports subprocess and kernel-level isolation via [sandtrap](https://github.com/ashenfad/sandtrap).

### Process Isolation

```python
agent = Agent(isolation="process")
```

Agent code runs in a forked subprocess. If the code crashes, segfaults, or triggers an OOM, the host process is unaffected. The sandbox communicates results back via IPC.

All core functionality works cross-process: `task_success` / `task_fail` / `task_clarify`, `print()`, `view_image()`, registered functions/classes/modules, and state synchronization. Values must be picklable to cross the process boundary.

### Kernel Isolation

```python
agent = Agent(
    isolation="kernel",
    fs=connect_fs(type="isolated", root="/tmp/sandbox"),
)
```

Adds OS-level restrictions on top of process isolation:

- **Linux**: seccomp (syscall filtering) + Landlock (filesystem restriction)
- **macOS**: Seatbelt (sandbox profiles)

When paired with an `IsolatedFS`, kernel-level filesystem restriction locks access to the IsolatedFS root directory. With a `VirtualFS` or no filesystem, the kernel blocks all host filesystem access.

### Choosing an Isolation Level

| Scenario | Recommended Level |
|----------|------------------|
| Local development, notebooks | `"none"` |
| Production server, multiple users | `"process"` |
| Running untrusted code, public-facing | `"kernel"` |

See [Agent - Sandbox Isolation](../api/agent.md#sandbox-isolation) for configuration details.
