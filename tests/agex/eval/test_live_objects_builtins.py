"""
Tests for live object attribute assignment, deletion, and security.

This covers the new functionality added for BoundInstanceObject setattr/delattr
and the improved class registration that automatically detects instance attributes.
"""

import pytest

from agex import Agent
from agex.agent.datatypes import MemberSpec
from agex.eval.core import evaluate_program
from agex.eval.user_errors import AgexAttributeError
from agex.state import Ephemeral
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


class MockLiveObject:
    """Mock object for live object registration testing."""

    def __init__(self):
        self.public_attr = "public_value"
        self.private_attr = "private_value"
        self.numeric_attr = 42

    def public_method(self):
        return "public_method_result"

    def private_method(self):
        return "private_method_result"


class MockInheritedAttributes:
    """Mock class with inherited attributes for MRO testing."""

    def __init__(self, inherited_attr1, inherited_attr2):
        self.inherited_attr1 = inherited_attr1
        self.inherited_attr2 = inherited_attr2


class MockChildClass(MockInheritedAttributes):
    """Mock child class that inherits attributes."""

    def __init__(self, inherited_attr1, inherited_attr2, child_attr):
        super().__init__(inherited_attr1, inherited_attr2)
        self.child_attr = child_attr


def test_live_object_attribute_assignment():
    """Test that live objects allow attribute assignment for registered properties."""
    agent = Agent()
    test_obj = MockLiveObject()

    # Register with specific properties
    agent.module(test_obj, name="test_obj", include=["public_attr", "numeric_attr"])

    state = Ephemeral()

    # Test setting allowed attribute
    evaluate_program('test_obj.public_attr = "modified"', agent, state)
    assert test_obj.public_attr == "modified"

    # Test setting another allowed attribute
    evaluate_program("test_obj.numeric_attr = 100", agent, state)
    assert test_obj.numeric_attr == 100

    # Test reading back the modified values
    evaluate_program("result1 = test_obj.public_attr", agent, state)
    evaluate_program("result2 = test_obj.numeric_attr", agent, state)

    assert state.get("result1") == "modified"
    assert state.get("result2") == 100


def test_live_object_attribute_assignment_blocked():
    """Test that live objects block assignment to unregistered properties."""
    agent = Agent()
    test_obj = MockLiveObject()

    # Register with limited properties
    agent.module(test_obj, name="test_obj", include=["public_attr"])

    state = Ephemeral()

    # Test that setting unregistered attribute is blocked
    with pytest.raises(AgexAttributeError) as exc_info:
        evaluate_program('test_obj.private_attr = "blocked"', agent, state)

    assert "no registered property 'private_attr'" in str(exc_info.value)
    # Verify the original value wasn't changed
    assert test_obj.private_attr == "private_value"


def test_live_object_attribute_deletion():
    """Test that live objects allow attribute deletion for registered properties."""
    agent = Agent()
    test_obj = MockLiveObject()

    # Register with specific properties
    agent.module(test_obj, name="test_obj", include=["public_attr"])

    state = Ephemeral()

    # Verify attribute exists initially
    assert hasattr(test_obj, "public_attr")

    # Test deleting allowed attribute
    evaluate_program("del test_obj.public_attr", agent, state)

    # Verify attribute was deleted
    assert not hasattr(test_obj, "public_attr")


def test_live_object_attribute_deletion_blocked():
    """Test that live objects block deletion of unregistered properties."""
    agent = Agent()
    test_obj = MockLiveObject()

    # Register with limited properties
    agent.module(test_obj, name="test_obj", include=["public_attr"])

    state = Ephemeral()

    # Test that deleting unregistered attribute is blocked
    with pytest.raises(AgexAttributeError) as exc_info:
        evaluate_program("del test_obj.private_attr", agent, state)

    assert "no registered property 'private_attr'" in str(exc_info.value)
    # Verify the attribute still exists
    assert test_obj.private_attr == "private_value"


def test_automatic_instance_attribute_detection():
    """Test that class registration automatically detects instance attributes from MRO."""
    agent = Agent()

    # Register class with default wildcard pattern
    agent.cls(MockChildClass)

    # Check what attributes were detected
    spec = agent.cls_registry_by_type[MockChildClass]
    attrs = set(spec.attrs.keys())

    # Should include attributes from both parent and child classes
    assert "inherited_attr1" in attrs
    assert "inherited_attr2" in attrs
    assert "child_attr" in attrs


def test_explicit_include_overrides_automatic_detection():
    """Test that explicit include patterns override automatic attribute detection."""
    agent = Agent()

    # Register class with explicit include list
    agent.cls(MockChildClass, include=["child_attr"])

    # Check what attributes were registered
    spec = agent.cls_registry_by_type[MockChildClass]
    attrs = list(spec.attrs.keys())

    # Should only include explicitly listed attribute
    assert attrs == ["child_attr"]
    assert "inherited_attr1" not in attrs
    assert "inherited_attr2" not in attrs


def test_class_instance_respects_registration_limits():
    """Test that class instances respect registration limitations."""
    agent = Agent()

    # Register class with limited attributes
    agent.cls(MockChildClass, include=["child_attr"])

    state = Ephemeral()

    # Create instance
    evaluate_program('obj = MockChildClass("val1", "val2", "val3")', agent, state)

    # Test accessing allowed attribute
    evaluate_program("result1 = obj.child_attr", agent, state)
    assert state.get("result1") == "val3"

    # Test that accessing blocked attribute fails
    with pytest.raises(AgexAttributeError) as exc_info:
        evaluate_program("result2 = obj.inherited_attr1", agent, state)

    assert "object has no attribute 'inherited_attr1'" in str(exc_info.value)


def test_agent_self_registration_works():
    """Test the original dogfood scenario - agent registering itself."""
    agent = Agent(primer="initial_primer")

    # Register Agent class (should auto-detect primer and other attributes)
    agent.cls(Agent)

    # Register agent instance as live object
    agent.module(agent, name="agent")

    state = Ephemeral()

    # Test assignment to primer
    evaluate_program('agent.primer = "modified_primer"', agent, state)
    assert agent.primer == "modified_primer"

    # Test reading primer back
    evaluate_program("result = agent.primer", agent, state)
    assert state.get("result") == "modified_primer"

    # Test accessing other auto-detected attributes
    evaluate_program("timeout = agent.timeout_seconds", agent, state)
    assert state.get("timeout") == agent.timeout_seconds


def test_live_object_security_vs_class_security():
    """Test the difference in security between live objects and class instances."""
    agent = Agent()

    # Register Agent class with limited attributes
    agent.cls(Agent, include=["primer"])

    # Register specific agent instance as live object with all attributes
    test_agent = Agent(primer="test")
    agent.module(test_agent, name="live_agent")

    state = Ephemeral()

    # Live object should allow access to all registered properties
    evaluate_program("result1 = live_agent.timeout_seconds", agent, state)
    assert state.get("result1") == test_agent.timeout_seconds

    # But if we create a new Agent instance through the evaluator,
    # it should respect the class registration limits
    evaluate_program("new_agent = Agent()", agent, state)

    # Test that the new instance respects class registration limits
    # Should be able to access registered attribute
    evaluate_program("primer_value = new_agent.primer", agent, state)
    assert state.get("primer_value") is None  # Default primer value

    # Should NOT be able to access unregistered attribute
    with pytest.raises(AgexAttributeError) as exc_info:
        evaluate_program("timeout_value = new_agent.timeout_seconds", agent, state)

    assert "object has no attribute 'timeout_seconds'" in str(exc_info.value)


def test_live_object_method_access_unchanged():
    """Test that method access for live objects still works correctly."""
    agent = Agent()
    test_obj = MockLiveObject()

    # Register with methods and properties
    agent.module(test_obj, name="test_obj", include=["public_method", "public_attr"])

    state = Ephemeral()

    # Test method call
    evaluate_program("result = test_obj.public_method()", agent, state)
    assert state.get("result") == "public_method_result"

    # Test property access
    evaluate_program("attr_value = test_obj.public_attr", agent, state)
    assert state.get("attr_value") == "public_value"

    # Test blocked method access
    with pytest.raises(AgexAttributeError) as exc_info:
        evaluate_program("blocked = test_obj.private_method()", agent, state)

    assert "object has no attribute 'private_method'" in str(exc_info.value)
