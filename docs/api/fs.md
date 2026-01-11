# FileSystem Configuration

The `connect_fs()` factory function configures agent filesystem access. By default, agents with IO capabilities (e.g., via `register_io()`) have unrestricted access to the host filesystem. Use `connect_fs()` to restrict access to:
- **Virtual filesystem (VFS)**: In-memory filesystem backed by agent state
- **Isolated filesystem**: Real filesystem access restricted to a specific directory

> [!NOTE]
> When `fs=connect_fs(...)` is configured, `register_io()` is automatically applied during task execution, giving the agent access to file operations (`open()`, `os.listdir()`, etc.) that are routed through VFS or validated against the isolated root.

## `connect_fs()` API

```python
from agex import connect_fs

# Virtual filesystem (in-memory, state-backed)
fs_config = connect_fs(type="virtual")

# Isolated filesystem (real filesystem, path-restricted)
fs_config = connect_fs(
    type="isolated",
    root="/path/to/workspace",  # Required: must exist
    tracking=True,              # Optional: emit FileEvents
    per_session=True,           # Optional: create session subdirectories
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"virtual"` | FileSystem type: `"virtual"` or `"isolated"` |
| `root` | `str` | — | (isolated only) Absolute path to root directory. Must exist. |
| `tracking` | `bool` | `False` | (isolated only) Whether to track file changes for FileEvents |
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
from agex import Agent, connect_fs, connect_state

agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/agent-state"),
    fs=connect_fs(type="virtual"),
)
```

## Isolated FileSystem

The Isolated FileSystem provides restricted access to a real directory on the host filesystem. All operations are validated to stay within the root boundary.

### Key Features

- **Real Files**: Operates on actual filesystem (not in-memory)
- **Path Restriction**: All paths validated to stay within root directory
- **Security**: Prevents path traversal attacks (`../`, symlinks pointing outside)
- **Optional Tracking**: Enable `tracking=True` to emit `FileEvent` for changes
- **Standard Library Support**: Same Python file operations as VFS

### Usage

```python
from agex import Agent, connect_fs

agent = Agent(
    fs=connect_fs(
        type="isolated",
        root="/path/to/project",
        tracking=True,  # Optional: emit FileEvents
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
| Ephemeral workspaces | `virtual` |
| Testing/development | `virtual` |
| Processing user uploads (temporary) | `virtual` |
| State-backed persistence (with versioning) | `virtual` |
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
- **Isolated**: Emits events when `tracking=True`

```python
from agex import events

for event in events(agent.state()):
    if event.type == "file":
        print(f"Source: {event.file_source}")  # "user" or "agent"
        print(f"Added: {event.added}")
        print(f"Modified: {event.modified}")
        print(f"Removed: {event.removed}")
```

## Next Steps

- **[Agent](agent.md)**: Configure agents
- **[State](state.md)**: Files are stored in state (VFS)
