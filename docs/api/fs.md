# FileSystem Configuration

The `connect_fs()` factory function configures agent filesystem access, built on [monkeyfs](https://github.com/ashenfad/monkeyfs). **By default, all agents are configured with a Virtual Filesystem (VFS)**, allowing them to create and manage persistent workspace modules. Use `connect_fs()` to change this configuration to:
- **Isolated filesystem**: Real filesystem access restricted to a specific directory on the host
- **Manual/Unrestricted**: Disable automatic sandboxed filesystem configuration by passing `fs=None` to the agent constructor. In this mode, no default IO capabilities are provided, but agents may have unrestricted host access if risky modules (like `os` or `pathlib`) are manually registered.

> [!NOTE]
> When `fs=connect_fs(...)` is configured, `register_io()` is automatically applied during task execution, giving the agent access to file operations (`open()`, `os.listdir()`, etc.) that are routed through VFS or validated against the isolated root.

## `connect_fs()` API

```python
from agex import connect_fs

# Virtual filesystem (in-memory, state-backed)
fs_config = connect_fs(type="virtual")

# Virtual filesystem with size limit
fs_config = connect_fs(type="virtual", max_size_mb=100)

# Isolated filesystem (real filesystem, path-restricted)
fs_config = connect_fs(
    type="isolated",
    root="/path/to/workspace",  # Required: must exist
    per_session=True,           # Optional: create session subdirectories
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"virtual"` | FileSystem type: `"virtual"` or `"isolated"` |
| `max_size_mb` | `int \| None` | `None` | (virtual only) Maximum total size in MB. `None` means unlimited. |
| `root` | `str` | — | (isolated only) Absolute path to root directory. Must exist. |
| `per_session` | `bool` | `False` | (isolated only) Whether to create session-specific subdirectories |

## Virtual FileSystem (VFS)

The Virtual FileSystem provides a secure, state-backed filesystem for agents. Files exist only in memory/state, not on the host filesystem.

### Key Features

- **State-Backed**: Files stored as keys in agent state
- **Versioned**: If using `versioned` state, file changes are checkpointed and can be rolled back
- **Secure**: No access to host filesystem
- **Standard Library Support**: `open()`, `os.listdir()`, `os.path.exists()`, `os.stat()`, etc.
- **Metadata**: Tracks creation time, modification time, and size
- **Events**: Emits `FileEvent` when files are added, modified, or removed

### Usage

```python
from agex import Agent, connect_state

# VirtualFS is enabled by default
agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/agent-state"),
)
```

### Size Limits

Limit the total size of all files in the VFS to prevent unbounded storage consumption:

```python
agent = Agent(
    fs=connect_fs(type="virtual", max_size_mb=100),  # 100MB limit
)
```

When the limit is exceeded, file operations raise `OSError`:

```python
# Agent tries to write beyond the limit
with open("large.bin", "wb") as f:
    f.write(b"x" * (200 * 1024 * 1024))  # 200MB
# Raises OSError: VFS size limit exceeded
```

**Behavior:**

- The limit applies to the **total size** of all files combined
- Overwriting a file accounts for the size difference (shrinking a file frees space)
- Deleting files frees space for new writes
- `write_many()` is atomic: if the combined size would exceed the limit, no files are written

```python
# Example: manage space by removing old files
fs = agent.fs()
fs.remove("old_data.bin")  # Free up space
fs.write("new_data.bin", new_content)  # Now succeeds
```

## Isolated FileSystem

The Isolated FileSystem provides restricted access to a real directory on the host filesystem. All operations are validated to stay within the root boundary.

### Key Features

- **Real Files**: Operates on actual filesystem (not in-memory)
- **Path Restriction**: All paths validated to stay within root directory
- **Security**: Prevents path traversal attacks (`../`, symlinks pointing outside)
- **Standard Library Support**: Same Python file operations as VFS

### Usage

```python
from agex import Agent, connect_fs

agent = Agent(
    fs=connect_fs(
        type="isolated",
        root="/path/to/project",
    ),
)
```

### Per-Session Isolation

By default, all sessions share the same root directory. Use `per_session=True` to automatically create session-specific subdirectories:

```python
# Enable per-session isolation
agent = Agent(
    fs=connect_fs(
        type="isolated",
        root="/data",
        per_session=True,
    ),
)

# Each session gets its own subdirectory
fs1 = agent.fs(session="user_123")  # Works in /data/user_123/
fs2 = agent.fs(session="user_456")  # Works in /data/user_456/

# Sessions are completely isolated
fs1.write("config.txt", b"user 123 settings")
fs2.write("config.txt", b"user 456 settings")
```

**Use case**: Multi-tenant applications where each user/session needs isolated file storage.

### Security

The isolated filesystem protects against common filesystem attacks:

- **Path Traversal**: `../../../etc/passwd` → blocked
- **Absolute Paths**: `/etc/passwd` → blocked  
- **Symlink Escapes**: Symlinks pointing outside root → blocked
- **Generic Errors**: No information leakage about host filesystem

```python
# Agent tries to escape - gets PermissionError
with open("../../../etc/passwd") as f:  # ❌ Blocked
    pass

# Agent works within root - works fine
with open("data/input.txt") as f:  # ✅ Works
    data = f.read()
```

### Modal Volumes

Isolated filesystem works seamlessly with Modal volumes:

```python
# On Modal
volume = modal.Volume.from_name("my-data")

@app.function(volumes={"/vol": volume})
def run_agent():
    agent = Agent(
        fs=connect_fs(type="isolated", root="/vol/workspace"),
    )
    # Agent reads/writes to volume
```

## Choosing Between VFS and Isolated

| Use Case | Recommended |
|----------|-------------|
| Ephemeral workspaces | `virtual` (default) |
| Testing/development | `virtual` (default) |
| Processing user uploads (temporary) | `virtual` (default) |
| State-backed persistence (with versioning) | `virtual` (default) |
| Accessing existing project files | `isolated` |
| Working with real files/directories | `isolated` |
| Per-session isolation in multi-tenant apps | `isolated` with `per_session=True` |
| Modal/cloud deployments with volumes | `isolated` |

**Note**: Both can persist - VFS persists through state (e.g., `storage="disk"`), isolated persists through real filesystem.

## External File Access (User Side)

Use `agent.fs(session)` to manage files from outside the agent:

```python
fs = agent.fs(session="default")

# Write a file (upload)
fs.write("data/sales.csv", b"date,amount\n2024-01-01,100")

# List files
print(fs.list("data/"))  # ['sales.csv']

# Read a file
content = fs.read("data/sales.csv")
```

This works identically for both `virtual` and `isolated` filesystem types.

## Agent File Access (Task Side)

Inside a task, agents use standard Python file operations:

```python
@agent.task
def analyze_sales():
    import os

    if os.path.exists("data/sales.csv"):
        with open("data/sales.csv", "r") as f:
            data = f.read()
        
        with open("report.txt", "w") as f:
            f.write(f"Analyzed {len(data)} bytes")
```

Operations are automatically routed to VFS or validated against isolated root.

## Events

File changes emit `FileEvent` to the event log:

- **VFS**: Always emits events
- **Isolated**: Emits events via metadata snapshot comparison

```python
from agex import events

for event in events(agent.state()):
    if event.type == "file":
        print(f"Source: {event.file_source}")  # "user" or "agent"
        print(f"Added: {event.added}")
        print(f"Modified: {event.modified}")
        print(f"Removed: {event.removed}")
```

## Host Filesystem Access

By default, when VirtualFS is configured, agents cannot access the host filesystem. With IsolatedFS, agents can only access a restricted directory. However, some trusted libraries (e.g., for OAuth credentials, API keys) may need unrestricted host access. The `host_fs_access` parameter allows whitelisting specific functions, classes, or modules to bypass these restrictions.

> [!WARNING]
> **Security**: Enabling `host_fs_access` grants unrestricted filesystem access to the registered item. Only use this for trusted libraries.

### Function-Level Access

Grant host filesystem access to a specific function. This example shows registering a function that needs to read configuration files:

```python
from agex import Agent, connect_fs
import json

agent = Agent(fs=connect_fs(type="virtual"))

def load_api_config() -> dict:
    """Load API configuration from host filesystem."""
    with open("/etc/myapp/api_config.json", 'r') as f:
        return json.load(f)

# Register with host_fs_access=True
agent.fn(load_api_config, host_fs_access=True)

@agent.task
def call_external_api(query: str) -> dict:  # type: ignore[return-value]
    """Call external API using configuration from host filesystem."""
    pass
```

### Class-Level Access

Grant host filesystem access to a class and all its methods. This example uses the Google Calendar library which needs OAuth credentials:

```python
from gcsa.google_calendar import GoogleCalendar

# Register Google Calendar with host filesystem access
# (needs to read OAuth credentials from ~/.credentials/)
agent.cls(GoogleCalendar, host_fs_access=True)

@agent.task
def check_schedule(date: str) -> list:  # type: ignore[return-value]
    """Check calendar schedule for the given date."""
    pass
```

### Module-Level Access

Grant host filesystem access to all functions/classes in a module. This example uses boto3 which needs AWS credentials:

```python
import boto3

# Register boto3 with host filesystem access
# (needs to read ~/.aws/credentials)
agent.module(boto3, host_fs_access=True)

@agent.task
def upload_file(filename: str, bucket: str) -> bool:  # type: ignore[return-value]
    """Upload a file from VFS to S3."""
    pass
```

### How It Works

During a call to a function/method with `host_fs_access=True`:
1. Filesystem patching is temporarily suspended
2. The function executes with normal filesystem access
3. Patching resumes after the call completes

Agent code and other registered functions remain isolated:

```python
agent.fn(load_api_config, host_fs_access=True)

@agent.task
def use_api() -> dict:  # type: ignore[return-value]
    """Call API using loaded configuration."""
    pass

# ✅ Agent can call load_api_config() which reads from host
# ❌ Agent's own os.open() calls are still blocked
```

### Per-Method Override

For classes, you can override `host_fs_access` per method:

```python
from agex import MemberSpec

agent.cls(
    SomeClass,
    host_fs_access=True,  # Default for all methods
    configure={
        "safe_method": MemberSpec(host_fs_access=False),  # Override
    }
)
```

### Best Practices

- **Minimize Scope**: Only enable `host_fs_access` for specific functions that need it
- **Trust**: Only whitelist libraries you trust completely

### Security Considerations

**Argument Injection**: Agents can pass arbitrary paths to whitelisted functions:

```python
agent.fn(read_file, host_fs_access=True)

# Agent could do this:
result = read_file("/etc/passwd")  # ⚠️ Not prevented
```

**Exception Message Leaks**: Error messages might reveal host filesystem structure.

**Recommended Mitigation**: Wrap sensitive functions with path validation:

```python
def read_credentials(filename: str) -> str:
    """Read credentials file (validates path)."""
    allowed_dir = "/app/credentials/"
    full_path = os.path.join(allowed_dir, filename)
    
    # Validate path stays within allowed directory
    if not os.path.realpath(full_path).startswith(allowed_dir):
        raise ValueError("Invalid credentials path")
    
    with open(full_path, 'r') as f:
        return f.read()

agent.fn(read_credentials, host_fs_access=True)
```

See [Registration docs](registration.md) for more details on the `host_fs_access` parameter.

## Next Steps

- **[Agent](agent.md)**: Configure agents
- **[State](state.md)**: Files are stored in state (VFS)
- **[Registration](registration.md)**: Register functions/classes/modules with `host_fs_access`
