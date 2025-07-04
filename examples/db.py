"""
Raw SQLite API Integration

Agent works directly with sqlite3.Connection and Cursor objects - no wrapper
classes needed. Demonstrates complex method chaining (db.execute().fetchall())
and stateful object management with live database connections.

Note: This example was tested with gpt-4.1-nano to demonstrate that
even smaller LLMs can effectively use the agex framework.
"""

import sqlite3
from typing import Any

from db_primer import PRIMER

from agex import Agent, Versioned

db = Agent(name="db_agent", primer=PRIMER)

# create an in-memory database and register the connection with the agent
conn = sqlite3.connect(":memory:")
db.module(
    conn,  # we register instance methods just like we do for module fns
    name="db",  # name is required when registering instance methods
    include=["execute", "executemany", "commit"],
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


def main():
    state = Versioned()
    update_db("Create a 'users' table with columns: id, name, email, age", state=state)
    update_db("Add 10 users to the users table", state=state)

    oldest = query_db("Who is the oldest user?", state=state)
    print(f"Oldest user: {oldest}")
    # Oldest user: {'id': 10, 'name': 'User10', 'email': 'user10@example.com', 'age': 30}

    # see the results directly
    print(conn.execute("SELECT * FROM users").fetchall())
    # [(1, 'User1', 'user1@example.com', 21), (2, 'User2', 'user2@example.com', 22), ...]


if __name__ == "__main__":
    # Run with: python examples/db.py OR python -m examples.db
    main()
