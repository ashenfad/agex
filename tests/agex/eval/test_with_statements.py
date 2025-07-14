"""
End-to-end tests for with statement support with live objects.

This test suite demonstrates the unique capability of using Python's 'with' statement
directly with live, unpickleable objects in agent code - a feature no other agent
framework currently offers.
"""

import sqlite3
import tempfile
from pathlib import Path

from agex import Agent
from agex.eval.user_errors import AgexValueError
from agex.llm.dummy_client import DummyLLMClient, LLMResponse


class DatabaseManager:
    """A database manager with context manager support for testing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None
        self.transaction_active = False

    def connect(self):
        """Connect to the database."""
        self.connection = sqlite3.connect(self.db_path)
        return self.connection

    def __enter__(self):
        """Context manager entry - start a transaction."""
        if not self.connection:
            self.connect()
        assert self.connection is not None  # For type checker
        self.connection.execute("BEGIN")
        self.transaction_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - commit or rollback."""
        if self.transaction_active and self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.transaction_active = False
        return False  # Don't suppress exceptions

    def execute(self, sql, params=None):
        """Execute a SQL statement."""
        if not self.connection:
            self.connect()
        assert self.connection is not None  # For type checker
        return self.connection.execute(sql, params or [])

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


def test_basic_with_statement():
    """Test basic with statement functionality with a simple context manager."""

    class SimpleContext:
        def __init__(self):
            self.entered = False
            self.exited = False
            self.value = "test_value"

        def __enter__(self):
            self.entered = True
            return self.value

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    # Create agent and register context manager
    agent = Agent()
    ctx = SimpleContext()
    agent.module(ctx, name="ctx", include=["__enter__", "__exit__"])

    # Configure dummy LLM
    agent.llm_client = DummyLLMClient(
        [
            LLMResponse(
                thinking="I'll use a with statement to test the context manager.",
                code="""
with ctx as value:
    print(f"Inside context, got: {value}")
    result = f"processed_{value}"

print("Outside context")
task_success(result)
""",
            )
        ]
    )

    @agent.task("Test basic with statement functionality")
    def test_task() -> str:  # type: ignore[return-value]
        pass

    result = test_task()

    assert result == "processed_test_value"
    assert ctx.entered is True
    assert ctx.exited is True


def test_database_with_statement():
    """Test with statement with SQLite database operations."""

    # Create temporary database
    db_path = Path(tempfile.mktemp(suffix=".db"))

    try:
        # Set up database
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        """
        )
        conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            ("Alice", "alice@example.com"),
        )
        conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)", ("Bob", "bob@example.com")
        )
        conn.commit()
        conn.close()

        # Create database manager
        db_manager = DatabaseManager(str(db_path))

        # Create agent and register database manager
        agent = Agent()
        agent.module(
            db_manager, name="db", include=["execute", "__enter__", "__exit__"]
        )
        agent.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])

        # Configure dummy LLM
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll use a with statement for safe database operations.",
                    code="""
# Query existing users
initial_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
print(f"Initial user count: {initial_count}")

# Use with statement for transactional operations
with db as transaction:
    # Add a new user
    db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Charlie", "charlie@example.com"))
    
    # Verify the addition
    new_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"Count after insert: {new_count}")
    
    # Get the new user's details
    user_data = db.execute("SELECT name, email FROM users WHERE name = ?", ("Charlie",)).fetchone()
    result = {"name": user_data[0], "email": user_data[1], "total_users": new_count}

print("Transaction completed successfully")
task_success(result)
""",
                )
            ]
        )

        @agent.task("Perform database operations using with statement")
        def db_task() -> dict:  # type: ignore[return-value]
            pass

        result = db_task()

        assert result["name"] == "Charlie"
        assert result["email"] == "charlie@example.com"
        assert result["total_users"] == 3

        # Verify the data was actually committed
        conn = sqlite3.connect(str(db_path))
        final_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        charlie_exists = conn.execute(
            "SELECT COUNT(*) FROM users WHERE name = ?", ("Charlie",)
        ).fetchone()[0]
        conn.close()

        assert final_count == 3
        assert charlie_exists == 1

    finally:
        if db_path.exists():
            db_path.unlink()


def test_with_statement_exception_handling():
    """Test with statement exception handling and rollback."""

    # Create temporary database
    db_path = Path(tempfile.mktemp(suffix=".db"))

    try:
        # Set up database
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """
        )
        conn.execute("INSERT INTO products (name) VALUES (?)", ("Widget",))
        conn.commit()
        conn.close()

        # Create database manager
        db_manager = DatabaseManager(str(db_path))

        # Create agent and register database manager
        agent = Agent()
        agent.module(
            db_manager,
            name="db",
            include=["execute", "__enter__", "__exit__"],
            exception_mappings={sqlite3.IntegrityError: AgexValueError},
        )
        agent.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])

        # Configure dummy LLM
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll test exception handling with database rollback using a simpler constraint violation.",
                    code="""
# Get initial state
initial_count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
print(f"Initial product count: {initial_count}")

try:
    with db as transaction:
        # This should work
        db.execute("INSERT INTO products (name) VALUES (?)", ("Gadget",))
        
        # This should fail due to UNIQUE constraint (duplicate name)
        db.execute("INSERT INTO products (name) VALUES (?)", ("Widget",))
        
except Exception as e:
    print(f"Caught expected exception: {e}")
    
    # Check that rollback worked
    final_count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"Final product count after rollback: {final_count}")
    
    result = {
        "initial_count": initial_count,
        "final_count": final_count,
        "rollback_successful": initial_count == final_count
    }

task_success(result)
""",
                )
            ]
        )

        @agent.task("Test exception handling with database rollback")
        def rollback_task() -> dict:  # type: ignore[return-value]
            pass

        result = rollback_task()

        assert result["initial_count"] == 1
        assert result["final_count"] == 1
        assert result["rollback_successful"] is True

        # Verify rollback actually worked in the database
        conn = sqlite3.connect(str(db_path))
        final_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        gadget_exists = conn.execute(
            "SELECT COUNT(*) FROM products WHERE name = ?", ("Gadget",)
        ).fetchone()[0]
        conn.close()

        assert final_count == 1  # Only original product should remain
        assert gadget_exists == 0  # Gadget should have been rolled back

    finally:
        if db_path.exists():
            db_path.unlink()


def test_nested_with_statements():
    """Test nested with statements for complex operations."""

    class ResourceManager:
        def __init__(self, name):
            self.name = name
            self.acquired = False
            self.released = False

        def __enter__(self):
            self.acquired = True
            return f"resource_{self.name}"

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.released = True
            return False

    # Create agent and register multiple resource managers
    agent = Agent()
    resource1 = ResourceManager("A")
    resource2 = ResourceManager("B")

    agent.module(resource1, name="res1", include=["__enter__", "__exit__"])
    agent.module(resource2, name="res2", include=["__enter__", "__exit__"])

    # Configure dummy LLM
    agent.llm_client = DummyLLMClient(
        [
            LLMResponse(
                thinking="I'll test nested with statements for resource management.",
                code="""
results = []

with res1 as r1:
    results.append(f"Acquired {r1}")
    
    with res2 as r2:
        results.append(f"Acquired {r2}")
        results.append("Both resources active")
    
    results.append(f"Released {r2}, still have {r1}")

results.append("All resources released")
task_success(results)
""",
            )
        ]
    )

    @agent.task("Test nested with statements")
    def nested_task() -> list:  # type: ignore[return-value]
        pass

    result = nested_task()

    expected = [
        "Acquired resource_A",
        "Acquired resource_B",
        "Both resources active",
        "Released resource_B, still have resource_A",
        "All resources released",
    ]

    assert result == expected
    assert resource1.acquired is True
    assert resource1.released is True
    assert resource2.acquired is True
    assert resource2.released is True


def test_with_statement_raw_sqlite_connection():
    """Test with statement using raw SQLite connection directly."""

    # Create temporary database
    db_path = Path(tempfile.mktemp(suffix=".db"))

    try:
        # Set up database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO items (value) VALUES (?)", ("initial",))
        conn.commit()
        conn.close()

        # Create new connection for the test
        conn = sqlite3.connect(str(db_path))

        # Create agent and register raw connection
        agent = Agent()
        agent.module(
            conn,
            name="conn",
            include=["execute", "commit", "rollback", "__enter__", "__exit__"],
        )
        agent.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])

        # Configure dummy LLM
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll use the raw SQLite connection with a with statement.",
                    code="""
# SQLite connections support the context manager protocol
# They automatically commit on success or rollback on exception

initial_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
print(f"Initial items: {initial_count}")

with conn:
    # Insert multiple items in a transaction
    conn.execute("INSERT INTO items (value) VALUES (?)", ("item1",))
    conn.execute("INSERT INTO items (value) VALUES (?)", ("item2",))
    conn.execute("INSERT INTO items (value) VALUES (?)", ("item3",))
    
    # Count items in transaction
    temp_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"Items in transaction: {temp_count}")

# Transaction should be committed automatically
final_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
print(f"Final items: {final_count}")

result = {
    "initial": initial_count,
    "final": final_count,
    "added": final_count - initial_count
}

task_success(result)
""",
                )
            ]
        )

        @agent.task("Test raw SQLite connection with with statement")
        def raw_conn_task() -> dict:  # type: ignore[return-value]
            pass

        result = raw_conn_task()

        assert result["initial"] == 1
        assert result["final"] == 4
        assert result["added"] == 3

        # Verify data was committed
        final_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        conn.close()

        assert final_count == 4

    finally:
        if db_path.exists():
            db_path.unlink()


def test_transient_variables():
    """Test the new transient variable system for unpickleable objects."""

    # Create temporary database
    db_path = Path(tempfile.mktemp(suffix=".db"))

    try:
        # Set up database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO items (value) VALUES (?)", ("item1",))
        conn.execute("INSERT INTO items (value) VALUES (?)", ("item2",))
        conn.execute("INSERT INTO items (value) VALUES (?)", ("item3",))
        conn.commit()
        conn.close()

        # Create new connection for test
        conn = sqlite3.connect(str(db_path))

        # Create agent and register raw connection
        agent = Agent()
        agent.module(conn, name="conn", include=["execute"])
        agent.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])

        # Configure dummy LLM
        agent.llm_client = DummyLLMClient(
            [
                LLMResponse(
                    thinking="I'll test the new transient variable system that allows unpickleable cursors.",
                    code="""
# This pattern previously failed due to pickle safety, but now works!
results = []

# Test 1: Basic transient cursor usage
with conn.execute("SELECT COUNT(*) FROM items") as cursor:
    count = cursor.fetchone()[0]
    results.append(f"Count: {count}")

# Test 2: Multiple transient operations
with conn.execute("SELECT value FROM items ORDER BY value") as cursor:
    items = cursor.fetchall()
    results.append(f"Items: {[item[0] for item in items]}")

# Test 3: Verify variables defined inside are accessible outside
final_count = count  # This should work
results.append(f"Final count: {final_count}")

task_success(results)
""",
                )
            ]
        )

        @agent.task("Test transient variables with unpickleable cursors")
        def transient_task() -> list:  # type: ignore[return-value]
            pass

        result = transient_task()

        expected_items = ["item1", "item2", "item3"]
        assert result[0] == "Count: 3"
        assert result[1] == f"Items: {expected_items}"
        assert result[2] == "Final count: 3"

        conn.close()

    finally:
        if db_path.exists():
            db_path.unlink()


def test_sqlite_context_manager_method_access():
    """
    Test the specific scenario that was failing in db_direct.py:
    Using 'with db as connection:' and then calling 'connection.execute()'.

    This test ensures that when a registered SQLite connection is used as a context manager,
    the returned object still provides access to the registered methods like 'execute'.
    """

    # Create in-memory database
    conn = sqlite3.connect(":memory:")

    # Create agent and register connection (matching db_direct.py setup)
    agent = Agent()
    agent.module(
        conn,
        name="db",
        include=["execute", "commit", "rollback", "close"],
    )
    agent.cls(sqlite3.Cursor, include=["fetchone", "fetchall", "fetchmany"])

    # Configure agent to use the exact pattern that was failing
    agent.llm_client = DummyLLMClient(
        [
            LLMResponse(
                thinking="I'll recreate the exact failing scenario from db_direct.py",
                code="""
# This is the exact pattern that was failing before the fix:
# "Error at line 11, col 4: 'Connection' object has no attribute 'execute'"

# Parse the instruction (simulating the original failing case)
sql_create_table = '''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER
);'''

# Execute the SQL statement using a with block for safety
with db as connection:
    connection.execute(sql_create_table)

# Commit the changes
db.commit()

# Exit with success
task_success("Created 'users' table with columns: id, name, email, age")
""",
            )
        ]
    )

    @agent.task("Recreate the exact failing scenario from db_direct.py")
    def failing_scenario_task():  # type: ignore[return-value]
        """Test the exact pattern that was failing: with db as connection: connection.execute()"""
        pass

    # This should now work without the "'Connection' object has no attribute 'execute'" error
    result = failing_scenario_task()

    assert result == "Created 'users' table with columns: id, name, email, age"

    # Verify the table was actually created
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
    )
    tables = cursor.fetchall()
    assert len(tables) == 1
    assert tables[0][0] == "users"

    # Verify table structure
    cursor = conn.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    assert "id" in column_names
    assert "name" in column_names
    assert "email" in column_names
    assert "age" in column_names

    conn.close()


if __name__ == "__main__":
    # Run tests individually for debugging
    test_basic_with_statement()
    print("✅ Basic with statement test passed")

    test_database_with_statement()
    print("✅ Database with statement test passed")

    test_with_statement_exception_handling()
    print("✅ Exception handling test passed")

    test_nested_with_statements()
    print("✅ Nested with statements test passed")

    test_with_statement_raw_sqlite_connection()
    print("✅ Raw SQLite connection test passed")

    test_transient_variables()
    print("✅ Transient variables test passed")

    test_sqlite_context_manager_method_access()
    print("✅ SQLite context manager method access test passed")

    print("\n🎉 All with statement tests passed successfully!")
    print("This demonstrates a unique capability no other agent framework offers!")
    print("✨ NEW: Transient variables allow `with obj as var:` for any object!")
