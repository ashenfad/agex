from agex import Agent
from agex.agent.datatypes import MemberSpec
from agex.eval.core import evaluate_program
from agex.state.kv import Memory
from agex.state.namespaced import Namespaced
from agex.state.versioned import Versioned


class MockDatabaseConnection:
    """A mock database connection for testing builtin functions."""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection_id = "conn_123"

    def query(self, table: str, record_id: int) -> str:
        """Runs a query on the mock database."""
        return f"Result from {table}[{record_id}]"

    def connect(self):
        """Establishes a connection."""
        return True

    def _internal_method(self):
        """An internal method that should be excluded by default."""
        return "internal"


def test_dir_on_registered_object():
    """Test that dir() works on registered objects and shows the correct attributes."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        configure={
            "query": MemberSpec(visibility="high"),
            "connection_id": MemberSpec(visibility="high"),
        },
    )

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test dir() on the registered object
    code = "dir_result = dir(db)"
    evaluate_program(code, agent, exec_state, 30.0)

    # Check stdout for the printed result
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 1
    dir_result = stdout[0][0]  # PrintTuple content

    # Should contain the registered methods and properties
    assert "query" in dir_result
    assert "connection_id" in dir_result
    assert "connect" in dir_result  # Default visibility should include this
    # Should not contain private methods (excluded by default)
    assert "_internal_method" not in dir_result


def test_hasattr_on_registered_object():
    """Test that hasattr() works correctly on registered objects."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db, name="db", exclude=["_*", "*._*", "connect"]
    )  # Exclude connect method and private methods

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test hasattr() on the registered object
    code = """
has_query = hasattr(db, "query")
has_connection_id = hasattr(db, "connection_id")
has_connect = hasattr(db, "connect")  # This should be False (excluded)
has_internal = hasattr(db, "_internal_method")  # This should be False (private)
has_nonexistent = hasattr(db, "nonexistent_attr")  # This should be False
"""
    evaluate_program(code, agent, exec_state, 30.0)

    # Check the results
    assert exec_state.get("has_query") is True
    assert exec_state.get("has_connection_id") is True
    assert exec_state.get("has_connect") is False  # Excluded
    assert exec_state.get("has_internal") is False  # Private
    assert exec_state.get("has_nonexistent") is False


def test_help_on_registered_object():
    """Test that help() works on registered objects and shows useful information."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        configure={
            "query": MemberSpec(
                visibility="high", docstring="Execute a database query"
            ),
            "connection_id": MemberSpec(
                visibility="high", docstring="The connection identifier"
            ),
        },
    )

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test help() on the registered object
    code = "help(db)"
    evaluate_program(code, agent, exec_state, 30.0)

    # Check stdout for the help output
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 1
    help_text = stdout[0][0]  # PrintTuple content

    # Should contain object help header
    assert "Help on object db:" in help_text

    # Should contain methods section
    assert "METHODS" in help_text
    assert "query - Execute a database query" in help_text

    # Should contain properties section
    assert "PROPERTIES" in help_text
    assert "connection_id - The connection identifier" in help_text


def test_help_general_includes_objects():
    """Test that help() with no arguments includes registered objects."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    # Register some items
    @agent.fn
    def my_function():
        pass

    @agent.cls
    class MyClass:
        pass

    agent.module(db, name="db")

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test general help()
    code = "help()"
    evaluate_program(code, agent, exec_state, 30.0)

    # Check stdout for the help output
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 1
    help_text = stdout[0][0]  # PrintTuple content

    # Should contain all sections
    assert "Available items:" in help_text
    assert "Functions:" in help_text
    assert "- my_function" in help_text
    assert "Classes:" in help_text
    assert "- MyClass" in help_text
    assert "Objects:" in help_text
    assert "- db" in help_text


def test_dir_no_args_includes_object_names():
    """Test that dir() with no arguments includes registered object names in scope."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(db, name="db")

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Add the object to scope (this would normally happen via name resolution)
    exec_state.set("db", agent.object_registry["db"])

    # Test dir() with no arguments
    code = """
x = 1
y = "hello"
dir()
"""
    evaluate_program(code, agent, exec_state, 30.0)

    # Check stdout for the dir output
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 1
    dir_result = stdout[0][0]  # PrintTuple content

    # Should include variables and the registered object name
    assert "x" in dir_result
    assert "y" in dir_result
    assert "db" in dir_result


def test_help_on_object_with_no_docstrings():
    """Test help() on an object where methods/properties have no docstrings."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    # Register without custom docstrings
    agent.module(db, name="db")

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test help() on the registered object
    code = "help(db)"
    evaluate_program(code, agent, exec_state, 30.0)

    # Check stdout for the help output
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 1
    help_text = stdout[0][0]  # PrintTuple content

    # Should contain object help header
    assert "Help on object db:" in help_text

    # Should contain methods section with method names but no custom docstrings
    assert "METHODS" in help_text
    assert "query" in help_text
    assert "connect" in help_text

    # Should contain properties section
    assert "PROPERTIES" in help_text
    assert "connection_id" in help_text
    assert "db_name" in help_text


def test_object_behaves_like_module():
    """Test that registered objects behave similarly to modules in terms of introspection."""
    import math

    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    # Register both a module and an object
    agent.module(math, include=["sin", "cos", "pi"])
    agent.module(db, name="db")

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test that both can be introspected similarly
    code = """
# Test dir() on both
import math
dir(math)
dir(db)

# Test hasattr() on both
has_math_sin = hasattr(math, "sin")
has_math_nonexistent = hasattr(math, "nonexistent")
has_db_query = hasattr(db, "query")
has_db_nonexistent = hasattr(db, "nonexistent")

# Test help() on both
help(math)
help(db)
"""
    evaluate_program(code, agent, exec_state, 30.0)

    # Check that both work similarly
    stdout = exec_state.get("__stdout__")
    assert len(stdout) == 4  # 2 dir() calls + 2 help() calls

    # dir() results
    math_dir = stdout[0][0]
    db_dir = stdout[1][0]
    assert "sin" in math_dir
    assert "cos" in math_dir
    assert "pi" in math_dir
    assert "query" in db_dir
    assert "connection_id" in db_dir

    # hasattr() results
    assert exec_state.get("has_math_sin") is True
    assert exec_state.get("has_math_nonexistent") is False
    assert exec_state.get("has_db_query") is True
    assert exec_state.get("has_db_nonexistent") is False

    # help() results
    math_help = stdout[2][0]
    db_help = stdout[3][0]
    assert "Help on module math:" in math_help
    assert "Help on object db:" in db_help
