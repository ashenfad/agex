"""
Raw SQLite API Integration

Agent works directly with sqlite3.Connection and Cursor objects - no wrapper
classes needed. Demonstrates stateful object management with live database
connections.

https://asciinema.org/a/LM0phpZWktTueeenfOuZBIp5r
"""

import sqlite3
from typing import Any

from db_primer import PRIMER

from agex import Agent, Versioned, connect_llm, pprint_tokens

db = Agent(
    name="db_agent",
    primer=PRIMER,
    llm=connect_llm(provider="anthropic", model="claude-haiku-4-5"),
)

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
    print("\nPROMPT:", "Create a 'users' table with columns: id, name, email, age")
    update_db(
        "Create a 'users' table with columns: id, name, email, age",
        state=state,
        on_token=pprint_tokens,
    )
    print("\nPROMPT:", "Add 10 users to the users table")
    update_db("Add 10 users to the users table", state=state, on_token=pprint_tokens)

    print("\nPROMPT:", "Who is the oldest user?")
    oldest = query_db("Who is the oldest user?", state=state, on_token=pprint_tokens)
    print(f"Oldest user: {oldest}")
    # Oldest user: {'id': 10, 'name': 'User10', 'email': 'user10@example.com', 'age': 30}

    # see the results directly
    print("\nFull users table:")
    print(conn.execute("SELECT * FROM users").fetchall())
    # [(1, 'User1', 'user1@example.com', 21), (2, 'User2', 'user2@example.com', 22), ...]


if __name__ == "__main__":
    # Run with: python examples/db.py OR python -m examples.db
    main()
