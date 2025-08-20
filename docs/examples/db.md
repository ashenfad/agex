# DB (Raw SQLite)

Agents can work directly with a live stateful objects like an `sqlite3.Connection` while still peristing their compute environment.

Create an agent and give it access to a live object:

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

Define task fns (agent implements at call time):

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

Ask the agent to create a a table and then follow up with questions about it:

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

Working with live objects like a database connection while also using [`Versioned`](../api/state.md) state is one of `agex`'s most compelling features. The linked primer file demonstrates how to currently coach the agent on the correct patterns for this. Making this guidance fully automatic is our plan for [Contextual Primers](../roadmap/primers.md).

—

Source: [https://github.com/ashenfad/agex/blob/main/examples/db.py](https://github.com/ashenfad/agex/blob/main/examples/db.py)

Primer: [https://github.com/ashenfad/agex/blob/main/examples/db_primer.py](https://github.com/ashenfad/agex/blob/main/examples/db_primer.py)
