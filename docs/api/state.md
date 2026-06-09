# State Configuration

The `connect_state()` factory function configures the agent's durable workspace and how it's stored.  State holds the things the agent's work *deliberately* persists:

- **Event log** — the agent's record of what it has done.  Surfaces as the rendered conversation history on the next turn.
- **Virtual filesystem** — files written under `/scratch/`, `helpers/`, etc.  See [FileSystem](fs.md).
- **`cache`** — an explicit per-session dict the agent can write to (`cache["model"] = fitted`) for Python objects it wants to remember.

State does **not** hold the agent's local variables, imported modules, or definitions inside a `python_action` — each emission runs as a fresh Python script, and cross-action continuity goes through the three channels above by design.

## `connect_state()` API

```python
from agex import connect_state

state_config = connect_state(
    type: Literal["versioned", "live", "resolver"] = "versioned",
    storage: Literal["memory", "disk"] = "memory",
    path: str | None = None,  # Required for disk storage
    init: Callable[[], dict] | dict | None = None,  # Initialize state vars
    resolver: StateResolver | None = None,  # Required for type="resolver"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"versioned"` | State type: `"versioned"` (with checkpointing), `"live"` (in-memory only), or `"resolver"` (bring-your-own, see [Custom State Resolvers](#custom-state-resolvers)) |
| `storage` | `str` | `"memory"` | Storage backend: `"memory"` or `"disk"` |
| `path` | `str \| None` | `None` | Path for disk storage (required when `storage="disk"`) |
| `init` | `Callable \| dict \| None` | `None` | Initialize state variables on first session creation |
| `resolver` | `StateResolver \| None` | `None` | Per-session state lookup (required and only valid when `type="resolver"`) |

## State Types

### No State (Default)

When no state is configured, each task call is independent:

```python
agent = Agent(primer="You are helpful.")

@agent.task
def analyze(data: str) -> dict:
    """Analyze data."""
    pass

# Each call starts fresh - no memory
result1 = analyze("first")
result2 = analyze("second")  # No memory of first call
```

### Versioned State (Recommended for Persistence)

Provides checkpointing, rollback, and cross-process persistence.  The agent's event log, VFS, and `cache` are all stored in kvgit; each turn commits a new snapshot.

```python
from agex import Agent, connect_state

agent = Agent(
    primer="You are helpful.",
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def build_analysis(data: str) -> dict:
    """Build cumulative analysis."""
    pass

# Across tasks, the agent sees:
#  - its own prior actions (rendered from the event log)
#  - any cache entries it wrote (e.g. cache["analysis"] = ...)
#  - any files it left in the VFS
result1 = build_analysis("first")
result2 = build_analysis("second")  # Sees result1 via event log + cache
```

### Live State (In-Process Only)

A faster, ephemeral backend.  Same shape as versioned (event log, VFS, cache), but stored in plain in-memory dicts — no kvgit, no checkpointing, no cross-process persistence.

```python
agent = Agent(
    primer="You are helpful.",
    state=connect_state(type="live"),
)
```

**Use for:** development, testing, agents whose `cache` and VFS only need to survive within a single process.  Each task call still sees the same Live state when the same agent instance is reused, but everything is lost when the process ends.

## Session Management

The `session` parameter isolates state between different users or conversations. You can also access state directly via [`agent.state(session)`](agent.md#statesession-str--default).

```python
agent = Agent(
    state=connect_state(type="versioned", storage="memory"),
)

@agent.task
def chat(message: str) -> str:
    """Chat with the user."""
    pass

# Different sessions have isolated event logs, VFS, and cache
chat("Hello", session="user_alice")  # Alice's conversation
chat("Hello", session="user_bob")    # Bob's conversation (separate)

# Same session shares state — the agent sees prior turns via the
# event log, plus anything it wrote to cache or the VFS.
chat("Remember this", session="alice")
chat("What did I say?", session="alice")  # Sees Alice's history
```

### Capability Grants (scopes)

When an agent registers [scoped capabilities](registration.md#scoped-capabilities), the scopes granted to a session live in that session's state. Manage them with the `scopes(state)` accessor — mirroring `events(state)` / `view(state)`:

```python
from agex import scopes

s = scopes(agent.state("alice"))
s.grant("email")        # grant for this session
s.has("email")          # -> True
s.list()                # -> {"email"}
s.revoke("email")
```

Grants live in a host-private, versioned key: they commit with the session's state, roll back with it, and are invisible to (and unwritable by) the agent's own code. A standalone `grant`/`revoke` commits immediately. The usual path, though, is the [request/resume flow](task.md#requesting-permission-scopes), where the agent asks and the host grants while resuming.

### Default Session

If you don't specify a session, the default session `"default"` is used:

```python
# These are equivalent
chat("Hello")
chat("Hello", session="default")
```

### Session Ids

Session ids are embedded into storage namespaces (disk paths, IndexedDB
names), so the Local host validates them: ids must match
`[A-Za-z0-9_-][A-Za-z0-9_.-]*` — typical shapes like `"default"` or
`"chat-3f2a"` pass; anything with path separators, leading dots, or
whitespace is rejected with a `ValueError`. If session ids flow from
untrusted input (e.g. the HTTP server), this is what stops a crafted id
from escaping the state directory.

### Custom State Resolvers

The built-in storages give every session its own substrate — a separate
diskcache directory, a separate IndexedDB name. That isolation is usually
what you want, but it can't express *one shared substrate with a kvgit
branch per session*: many working trees over one repo, where sessions
fork cheaply and concurrent writers reconcile through kvgit's CAS +
three-way merge instead of last-write-wins.

For those shapes, bring your own resolver — an object with a
`resolve(session)` method and a `versioned` flag (the
`agex.state.StateResolver` protocol). The host delegates session lookup
to it entirely; the resolver owns caching, initialization, and codec
choice. Build kvgit-backed states with `agex.state.staged_state`, which
applies agex's encoder/decoder (required for chunked numpy/pandas values
to round-trip):

```python
from agex import Agent, connect_state
from agex.state import assert_safe_session, staged_state
from agex.state.kv import Disk

store = Disk("/srv/agent/shared")     # one substrate...

class BranchResolver:
    versioned = True
    def __init__(self):
        self._cache = {}
    def resolve(self, session):
        assert_safe_session(session)
        if session not in self._cache:
            self._cache[session] = staged_state(store, branch=session)
        return self._cache[session]  # ...a branch per session

agent = Agent(state=connect_state(type="resolver", resolver=BranchResolver()))
```

Caching is the resolver's contract to honor: returning the same instance
per session keeps the task loop, `agent.fs()`, and `agent.state()` on one
staging area. Return fresh instances deliberately if you want independent
working trees over the same branch (e.g. optimistic concurrency between a
background loop and an interactive channel in one process).

Resolvers are **Local host only** — they're live in-process objects and
can't be serialized to the HTTP or Modal hosts (those reject resolver
configs with a clear error).

> [!NOTE]
> This mirrors agex-ts's `StateResolver`, added for agex-studio's
> concurrent-sessions work, where each studio session is a branch over
> one shared store.

## Storage Options

### Memory Storage (Default)

Fast, in-process storage. Lost when process ends:

```python
state = connect_state(type="versioned", storage="memory")
```

**Use for:** Development, testing, single-process applications.

> [!NOTE]
> On Modal, `memory` storage uses Modal Dict (not in-process memory) with a 7-day TTL on inactive keys. Dict names are auto-generated from the agent's fingerprint. See [Host - Modal](host.md#modalserverless-execution) for details.

> [!WARNING]
> **Modal Sub-Agent Limitation**: When the parent agent uses Modal host, sub-agents cannot yet have persistent state. Sub-agents must use ephemeral state (no `state=` parameter).

### Disk Storage

Persistent storage that survives restarts:

```python
state = connect_state(type="versioned", storage="disk", path="/var/agex/state")
```

**Use for:** Production, remote execution, long-running workflows.

## State Initialization

The `init` parameter lets you populate state variables when a session is first created. This is useful for loading data that should be mutable within state (e.g., calendars, datasets, config objects).

```python
from agex import Agent, connect_state

def load_initial_data():
    """Called once per new session."""
    return {
        "calendar": load_calendar("events.ics"),
        "config": {"theme": "dark", "locale": "en"},
    }

agent = Agent(
    primer="You manage my calendar.",
    state=connect_state(
        type="versioned",
        storage="disk",
        path="/tmp/agex/calendar",
        init=load_initial_data,  # Callable - deferred until first session
    ),
)
```

**How it works:**
1. On first access to a session, `init()` is called (or dict is used directly)
2. Each key-value pair is set in state
3. A sentinel (`__agex_init__`) marks the session as initialized
4. For versioned state, a snapshot is committed
5. Subsequent calls skip init (sentinel detected)

> [!TIP]
> Use a callable for lazy initialization - it defers loading until the session is actually created, avoiding work at agent definition time.

## Features of Versioned State

### Automatic Checkpointing

Every agent iteration creates a snapshot. You can inspect historical states or revert the agent to a previous point in time.

**Inspecting History (Read-Only)**
Use `checkout()` to get a read-only view of the state at a specific commit:

```python
from agex import events, view

# Get events after a task run
all_events = events(resolved_state)

# Inspect state as it was when an action occurred
action = all_events[0]
historical = resolved_state.checkout(action.commit_hash)
print(view(historical, focus="full"))
```

**Reverting State (Destructive)**
Use `reset_to()` to move the agent's HEAD to a specific commit. This can orphan commits that become unreachable from the new HEAD (which are cleaned up when branches are deleted).

```python
# Reset to a previous successful outcome
resolved_state.reset_to(success_event.commit_hash)

# The agent continues from this point as if later actions never happened
```

> [!TIP]
> Use `state.initial_commit` to get the hash of the very first commit (the empty root state). This is useful for resetting the agent completely.

### Concurrent Task Handling

Versioned state handles concurrent execution safely via the `on_conflict` parameter on tasks:

```python
@agent.task(on_conflict="retry")  # Default: retry on conflict
def interactive_task(query: str) -> str:
    pass

@agent.task(on_conflict="abandon")  # Background: abandon on conflict
def background_task() -> None:
    pass
```

See [Task - Concurrency Control](task.md#concurrency-control) for details.

### Persisted values: picklability and identity

Local variables defined inside a `python_action` are turn-local; they're never serialized.  The agent only persists values it deliberately writes to one of the durable channels:

- `cache[k] = v` — validated for picklability at write time using stdlib `pickle.HIGHEST_PROTOCOL` (the protocol the state codec uses underneath).  Unpicklable values raise `CacheError` immediately so the agent can choose a different representation rather than silently markering on a later read.  The cache holds **data**: functions and classes the agent defines in an action are plain (unpicklable) objects and can't be cached — cache their results, and carry reusable *code* in `helpers/` (source, re-imported on demand).
- VFS file writes — file contents are bytes/strings, which pickle trivially.
- Init-supplied values (see below) — go through the state codec on first session creation.

Two things to know about objects that round-trip:

**Object identity is not preserved.**  Anything pulled back out of the cache or `init` was reconstructed from serialized bytes, so `id()` changes and `is`-comparisons across reads fail.  Use value-based comparisons (`==`) instead.

**Closures capture "frozen" values.**  When a closure is cached, the variables it captured are frozen at their current values.  A cached `multiplier` defined when `factor = 2` will keep returning `x * 2` even if a later action defines a new `factor = 10`:

```python
# Action 1: define and cache a closure
factor = 2
def multiplier(x):
    return x * factor
cache["multiplier"] = multiplier  # captured with factor=2

# Action 2 (later task): retrieve and call
multiplier = cache["multiplier"]
multiplier(5)  # Returns 10 — factor stays at 2 inside the closure
```

> [!NOTE]
> Resources like database cursors, file handles, and network connections are not picklable.  Inside a single `python_action` they work normally; they just can't be cached.  Either re-acquire them at the top of each action (`cursor = db.cursor()`), chain the operation in a single line (`results = db.cursor().fetchall()`), or have the host expose them through a registered function so the live object stays in the host process.

## Advanced: Direct State Construction

> [!NOTE]
> Most users should use `connect_state()` for configuration. Direct construction is for advanced use cases requiring fine-grained control or custom storage backends.

For advanced use cases, you can construct state objects directly using types from [kvgit](https://github.com/ashenfad/kvgit):

### Direct Construction

```python
from agex import Staged, Live
from agex.state.kv import Disk, Memory
from kvgit import Versioned

# Equivalent to connect_state(type="versioned", storage="disk", path="/path")
state = Staged(Versioned(Disk("/path/to/storage")))

# Equivalent to connect_state(type="versioned", storage="memory")
state = Staged(Versioned(Memory()))

# Equivalent to connect_state(type="live")
state = Live()
```

### Custom Storage Backends

Implement the `KVStore` interface to use custom storage backends (Redis, S3, etc.):

```python
from agex.state.kv import KVStore

class RedisStore(KVStore):
    def get(self, key: str) -> bytes | None:
        """Retrieve value for key, or None if not found."""
        ...
    
    def set(self, key: str, value: bytes) -> None:
        """Store value for key."""
        ...
    
    def cas(self, key: str, value: bytes, expected: bytes | None) -> bool:
        """Compare-and-swap: set value only if current value matches expected."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete key."""
        ...
    
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        ...

# Use with versioned state
from agex import Staged
from kvgit import Versioned
state = Staged(Versioned(RedisStore(host="localhost", port=6379)))
```

## Quick Reference

| Configuration | Memory | Persistence | Use Case |
|--------------|--------|-------------|----------|
| No state | None | None | Simple, one-off tasks |
| `connect_state(type="live")` | In-process | None | Unpicklable objects |
| `connect_state(type="versioned", storage="memory")` | In-process | Checkpoints | Development, testing |
| `connect_state(type="versioned", storage="disk", path="...")` | Disk | Full | Production |

## Next Steps

- **[Agent](agent.md)**: Configure agents with state
- **[Task](task.md)**: Session parameter and concurrency control
- **[Host](host.md)**: State requirements for remote execution
