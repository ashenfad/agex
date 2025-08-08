# DB (Raw SQLite)

Let agents operate directly on `sqlite3.Connection` and `Cursor` — no wrappers. The agent learns a safe pattern for persistent state: chain `.execute(...).fetch*()` to avoid storing unpicklable cursors.

## Setup: register a live connection

```python
import sqlite3
from typing import Any
from agex import Agent

# Create an architected DB agent with guidance (see primer idea below)
db = Agent(name="db_agent")

# In-memory database for the demo
conn = sqlite3.connect(":memory:")

# Register instance methods just like a module; name is required for instances
db.module(conn, name="db", include=["execute", "executemany", "commit"])

# Also register the Cursor class methods used to collect results
import sqlite3 as _sqlite

db.cls(_sqlite.Cursor, include=["fetchone", "fetchall", "fetchmany"])
```

## Define tasks (agent implements them)

```python
@db.task
def update_db(prompt: str):  # type: ignore[return-value]
    """Update the database based on a natural language description."""
    pass

@db.task
def query_db(prompt: str) -> Any:  # type: ignore[return-value]
    """Query the database and return structured results."""
    pass
```

## Use with persistent state

```python
from agex import Versioned

state = Versioned()

# Create schema and populate rows
update_db("Create a 'users' table with columns: id, name, email, age", state=state)
update_db("Add 10 users to the users table", state=state)

# Ask questions in natural language
oldest = query_db("Who is the oldest user?", state=state)
print(oldest)
# {'id': 10, 'name': 'User10', 'email': 'user10@example.com', 'age': 30}

# You can also inspect the real DB in host code
print(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
# 10
```

## The safe pattern (why it works)
- With `Versioned` state, unpicklable objects (like cursors) cannot be saved.
- The agent chains `.execute(...).fetch*()` so only picklable results persist.
- For transactional writes, the agent uses `with db as connection: ...` (ephemeral scope inside the sandbox) to commit safely.

—

Source: https://github.com/ashenfad/agex/blob/main/examples/db.py
Primer text: https://github.com/ashenfad/agex/blob/main/examples/db_primer.py
