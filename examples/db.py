"""
Raw SQLite API Integration

Agent works directly with sqlite3.Connection and Cursor objects - no wrapper
classes needed. Demonstrates complex method chaining (db.execute().fetchall())
and stateful object management with live database connections.
"""

import sqlite3
from typing import Any

from agex import Agent, Versioned
from examples.db_primer import PRIMER

db = Agent(name="db_agent", primer=PRIMER)

# create an in-memory database and register the connection with the agent
conn = sqlite3.connect(":memory:")
db.module(
    conn,  # we register instance methods just like we do for module fns
    name="db",  # name is required when registering instance methods
    include=["execute", "execute_many", "commit"],
)

# also register the Cursor class for gathering results
db.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])


@db.task
def update_db(prompt: str):  # type: ignore[return-value]
    """Update the database based on a natural language description."""
    pass


@db.task
def query_db(prompt: str) -> Any:  # type: ignore[return-value]
    """Query the database based on a natural language description and return results."""
    pass


def example():
    state = Versioned()
    update_db("Create a 'users' table with columns: id, name, email, age", state=state)
    update_db("Add 10 users to the users table", state=state)

    oldest = query_db("Who is the oldest user?", state=state)
    print(f"Oldest user: {oldest}")

    # see the results directly
    print(conn.execute("SELECT * FROM users").fetchall())
