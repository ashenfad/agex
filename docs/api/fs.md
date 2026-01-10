# Filesystem Configuration

The `connect_fs()` factory function configures agent filesystem access. By default, agents have no filesystem access. You can enable a virtual filesystem (VFS) backed by agent state.

## `connect_fs()` API

```python
from agex import connect_fs

fs_config = connect_fs(
    type: Literal["virtual"] = "virtual",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"virtual"` | Filesystem type: `"virtual"` (in-memory/state-backed). Future versions may support `"real"` for sandboxed local access. |

## Virtual Filesystem (VFS)

The Virtual Filesystem provides a secure, state-backed filesystem for agents. It allows agents to perform standard Python file operations (`open()`, `os.listdir()`, etc.) that are routed to the agent's state instead of the host's real filesystem.

### Key Features

*   **State-Backed**: Files are stored as keys in the agent's state (e.g., in memory or on disk, depending on `connect_state`).
*   **Versioned**: If using `versioned` state, file changes are checkpointed and can be rolled back along with variable changes.
*   **Secure**: Agents cannot access the host's real filesystem, preventing accidental or malicious damage.
*   **Standard Library Support**: Common filesystem operations (`open()`, `os.listdir()`, `os.path.exists()`, `os.stat()`, etc.) are automatically patched to work with VFS.
*   **Metadata**: Tracks creation time, modification time, and size for all files.
*   **Events**: Emits `FileEvent` when files are added, modified, or removed (by user or agent).

## Usage

### 1. Configure the Agent

Enable the VFS by passing `fs=connect_fs()` to the Agent constructor:

```python
from agex import Agent, connect_fs, connect_state

agent = Agent(
    state=connect_state(type="versioned", storage="disk", path="/tmp/agent-state"),
    fs=connect_fs(type="virtual"),
)
```

### 2. External File Access (User Side)

Use the `agent.fs(session)` accessor to manage files from outside the agent (e.g., uploading files from a UI):

```python
# Get VFS for the default session
fs = agent.fs(session="default")

# Write a file (upload)
fs.write("data/sales.csv", b"date,amount\n2024-01-01,100")

# List files
print(fs.list("data/"))  # ['sales.csv']

# Read a file
content = fs.read("data/sales.csv")
```

### 3. Agent File Access (Task Side)

Inside a task, the agent can use standard Python file operations. These are automatically intercepted and routed to the VFS:

```python
@agent.task
def analyze_sales():
    # Standard Python file operations work transparently
    import os

    if os.path.exists("data/sales.csv"):
        with open("data/sales.csv", "r") as f:
            data = f.read()
        
        # os.stat() is also supported for metadata access
        stat_info = os.stat("data/sales.csv")
        file_size = stat_info.st_size
        
        # Write output
        with open("report.txt", "w") as f:
            f.write(f"Analyzed {len(data)} bytes (size: {file_size})")
```

## Agent Integration

### `agent.fs(session)` API

The accessor returned by `agent.fs()` provides a high-level API for VFS manipulation:

```python
class AgentAwareVFS:
    def write(self, path: str, content: bytes) -> None: ...
    def read(self, path: str) -> bytes: ...
    def list(self, path: str = "/") -> list[str]: ...
    def remove(self, path: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    
    # Metadata
    def stat(self, path: str) -> FileMetadata: ...
    def list_detailed(self, path: str = "/") -> list[FileInfo]: ...
```

### Metadata

The VFS tracks metadata for every file. You can access this via `stat()` or `list_detailed()`:

```python
files = agent.fs().list_detailed("data/")
for f in files:
    print(f"{f.name} ({f.size} bytes) - Modified: {f.modified_at}")
```

### Events

File changes emit `FileEvent` to the event log. This provides visibility into what files were created or changed during a task or via external upload.

*   **User Uploads**: Emitted immediately when calling `fs.write()`.
*   **Agent Changes**: Batched and emitted at the end of the agent's turn.

```python
from agex import events

for event in events(agent.state()):
    if event.type == "file":
        print(f"File changes: {event.added}, {event.modified}")
```

## Next Steps

- **[Agent](agent.md)**: Configure agents
- **[State](state.md)**: Files are stored in state
