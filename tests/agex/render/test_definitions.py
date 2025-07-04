from types import ModuleType

import pytest

from agex.agent import Agent, MemberSpec
from agex.render.definitions import render_definitions


def test_render_simple_function():
    agent = Agent()

    def my_func(a: int, b: str = "hello") -> bool:
        """This is a test function."""
        return True

    agent.fn(my_func)

    output = render_definitions(agent)
    expected = """
def my_func(a: int, b: str = 'hello') -> bool:
    \"\"\"
    This is a test function.
    \"\"\"
""".strip()
    assert output.strip() == expected


def test_render_function_with_no_docstring():
    agent = Agent()

    @agent.fn
    def no_doc(x, y):
        return x + y

    output = render_definitions(agent)
    expected = """
def no_doc(x, y):
    ...
""".strip()
    assert output.strip() == expected


def test_render_high_visibility_class():
    agent = Agent()

    class MyTestClass:
        my_attr: int = 1

        def __init__(self, val: int):
            pass

        def do_something(self):
            """Does a thing."""
            pass

        def _private(self):
            pass

    agent.cls(
        MyTestClass,
        visibility="high",
        include=["my_attr", "do_something", "__init__"],
        configure={"do_something": MemberSpec(visibility="high")},
    )

    output = render_definitions(agent)
    expected = """
# Available classes (use directly, no import needed):

class MyTestClass:
    def __init__(self, val: int):
        ...
    my_attr: int
    def do_something(self):
        \"\"\"
        Does a thing.
        \"\"\"
""".strip()
    assert output.strip() == expected


def test_render_medium_visibility_class():
    agent = Agent()

    @agent.cls(visibility="medium")
    class MediumVis:
        def __init__(self, val: int):
            pass

        def should_be_rendered(self):
            pass

    output = render_definitions(agent)
    expected = """
# Available classes (use directly, no import needed):

class MediumVis:
    def __init__(self, val: int):
        ...
    def should_be_rendered(self):
        ...
""".strip()
    assert output.strip() == expected


def test_render_medium_vis_class_with_init_override():
    agent = Agent()

    class MyClass:
        def __init__(self, val: int):
            """This is the original init docstring."""
            pass

        def other_method(self):
            pass

    agent.cls(
        MyClass,
        visibility="medium",
        include=["__init__"],
        configure={
            "__init__": MemberSpec(docstring="This is an overridden docstring.")
        },
    )

    output = render_definitions(agent)
    expected = """
# Available classes (use directly, no import needed):

class MyClass:
    def __init__(self, val: int):
        ...
""".strip()

    assert output.strip() == expected
    assert "other_method" not in output


@pytest.fixture
def dummy_module():
    mod = ModuleType("test_render_mod")

    def func_high():
        """High-vis func doc."""
        pass

    setattr(mod, "func_high", func_high)

    def func_med():
        pass

    setattr(mod, "func_med", func_med)

    class ClsHigh:
        """High-vis cls doc."""

        def __init__(self):
            pass

        def meth_high(self):
            """High-vis meth doc."""
            pass

        def meth_med(self):
            pass

    setattr(mod, "ClsHigh", ClsHigh)

    class ClsMed:
        """Med-vis cls doc."""

        def __init__(self):
            pass

        def meth_high_promote(self):
            """Promoting meth doc."""
            pass

    setattr(mod, "ClsMed", ClsMed)

    class ClsLow:
        def __init__(self):
            pass

        def meth_high_promote(self):
            """Promoting meth doc."""
            pass

    setattr(mod, "ClsLow", ClsLow)

    return mod


def test_render_high_vis_module(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="high",
        include=["func_high", "func_med", "ClsHigh", "ClsHigh.*"],
        exclude=["*._*"],
        configure={
            "func_high": MemberSpec(visibility="high"),
            "func_med": MemberSpec(visibility="medium"),
            "ClsHigh": MemberSpec(visibility="high"),
            "ClsHigh.meth_high": MemberSpec(visibility="high"),
            "ClsHigh.meth_med": MemberSpec(visibility="medium"),
        },
    )
    output = render_definitions(agent)
    expected = '''
# Available modules (import before using):

module my_mod:
    class ClsHigh:
        def __init__(self):
            ...
        def meth_high(self):
            """
            High-vis meth doc.
            """
        def meth_med(self):
            ...
    def func_high():
        """
        High-vis func doc.
        """
    def func_med():
        ...
'''.strip()
    assert output.strip() == expected


def test_render_medium_vis_module(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="medium",
        include=["func_high", "ClsHigh", "ClsHigh.meth_high"],
        configure={
            "func_high": MemberSpec(visibility="high"),
            "ClsHigh": MemberSpec(visibility="high"),
            "ClsHigh.meth_high": MemberSpec(visibility="high"),
        },
    )
    output = render_definitions(agent)
    expected = '''
# Available modules (import before using):

module my_mod:
    class ClsHigh:
        def __init__(self):
            ...
        def meth_high(self):
            """
            High-vis meth doc.
            """
    def func_high():
        """
        High-vis func doc.
        """
'''.strip()
    # Note: func_med (medium-vis) is not rendered because the container is medium-vis.
    assert output.strip() == expected


def test_render_low_vis_module_is_empty(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="low",
        include=["func_med"],
        configure={"func_med": MemberSpec(visibility="medium")},
    )
    output = render_definitions(agent)
    expected = """# Available modules (import before using):

module my_mod:
    ..."""
    assert output.strip() == expected


def test_render_low_vis_module_promoted_by_high_vis_func(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="low",
        include=["func_high"],
        configure={"func_high": MemberSpec(visibility="high")},
    )
    output = render_definitions(agent)
    expected = '''
# Available modules (import before using):

module my_mod:
    def func_high():
        """
        High-vis func doc.
        """
'''.strip()
    assert output.strip() == expected


def test_render_low_vis_module_promoted_by_class(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="low",
        include=["ClsHigh", "ClsHigh.meth_high"],
        configure={
            "ClsHigh": MemberSpec(visibility="high", constructable=True),
            "ClsHigh.meth_high": MemberSpec(visibility="high"),
        },
    )
    output = render_definitions(agent)
    expected = '''
# Available modules (import before using):

module my_mod:
    class ClsHigh:
        def __init__(self):
            ...
        def meth_high(self):
            """
            High-vis meth doc.
            """
'''.strip()
    assert output.strip() == expected


def test_render_low_vis_module_promoted_by_method(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="low",
        include=["ClsMed", "ClsMed.meth_high_promote"],
        configure={
            "ClsMed": MemberSpec(visibility="medium", constructable=True),
            "ClsMed.meth_high_promote": MemberSpec(visibility="high"),
        },
    )
    output = render_definitions(agent)
    # The module is promoted to medium.
    # ClsMed is medium-vis, but it is also promoted, so it gets rendered.
    # Inside ClsMed, only high-vis members are rendered.
    expected = '''
# Available modules (import before using):

module my_mod:
    class ClsMed:
        def __init__(self):
            ...
        def meth_high_promote(self):
            """
            Promoting meth doc.
            """
'''.strip()
    assert output.strip() == expected


def test_low_vis_class_in_low_vis_module_promoted_by_method(dummy_module):
    agent = Agent()
    agent.module(
        dummy_module,
        name="my_mod",
        visibility="low",
        include=["ClsLow", "ClsLow.meth_high_promote"],
        configure={
            "ClsLow": MemberSpec(visibility="low", constructable=True),
            "ClsLow.meth_high_promote": MemberSpec(visibility="high"),
        },
    )
    output = render_definitions(agent)
    # Module promoted to medium.
    # ClsLow is promoted to medium.
    # It should be rendered because it's promoted.
    expected = '''
# Available modules (import before using):

module my_mod:
    class ClsLow:
        def __init__(self):
            ...
        def meth_high_promote(self):
            """
            Promoting meth doc.
            """
'''.strip()
    assert output.strip() == expected


class MockDatabaseConnection:
    """A mock database connection for testing object rendering."""

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


def test_render_object_basic():
    """Test basic object rendering with default visibility."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(db, name="db")

    rendered = render_definitions(agent)

    # Should contain the object definition
    assert "object db:" in rendered
    # Should contain public methods
    assert "def query(" in rendered
    assert "def connect(" in rendered
    # Should contain properties
    assert "connection_id: ..." in rendered
    assert "db_name: ..." in rendered
    # Should not contain private methods (excluded by default)
    assert "_internal_method" not in rendered


def test_render_object_with_high_visibility():
    """Test object rendering with high visibility shows docstrings."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        configure={
            "query": MemberSpec(visibility="high", docstring="Custom query docstring"),
            "connection_id": MemberSpec(
                visibility="high", docstring="The connection identifier"
            ),
        },
    )

    rendered = render_definitions(agent)

    # Should show docstrings for high-visibility members
    assert "Custom query docstring" in rendered
    assert "The connection identifier" in rendered
    # Should still have the method signatures
    assert "def query(" in rendered
    assert "connection_id: ..." in rendered


def test_render_object_with_medium_visibility():
    """Test object rendering with medium visibility hides docstrings."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        configure={
            "query": MemberSpec(visibility="medium", docstring="This should be hidden"),
            "connect": MemberSpec(visibility="medium"),
        },
    )

    rendered = render_definitions(agent)

    # Should show method signatures but not docstrings
    assert "def query(" in rendered
    assert "def connect(" in rendered
    assert "This should be hidden" not in rendered


def test_render_object_with_low_visibility():
    """Test object rendering with low visibility."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(db, name="db", visibility="low")

    rendered = render_definitions(agent)

    # Low visibility objects should be rendered with ellipsis (showing they exist but hiding contents)
    assert "object db:" in rendered
    assert "    ..." in rendered
    # Should not show any methods or properties
    assert "def query(" not in rendered
    assert "connection_id: ..." not in rendered


def test_render_object_promotion():
    """Test that low-visibility objects get promoted when they have high-visibility members."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        visibility="low",  # Object is low visibility
        configure={
            "query": MemberSpec(visibility="high", docstring="Important query method"),
        },
    )

    rendered = render_definitions(agent)

    # Should be promoted and rendered because it has a high-visibility member
    assert "object db:" in rendered
    assert "def query(" in rendered
    assert "Important query method" in rendered


def test_render_object_full_mode():
    """Test object rendering in full mode shows everything."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(
        db,
        name="db",
        visibility="low",
        configure={
            "query": MemberSpec(visibility="low", docstring="Low visibility method"),
        },
    )

    rendered = render_definitions(agent, full=True)

    # Full mode should show everything regardless of visibility
    assert "object db:" in rendered
    assert "def query(" in rendered
    assert "Low visibility method" in rendered


def test_render_object_with_exclude():
    """Test that excluded methods are not rendered."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    agent.module(db, name="db", exclude=["query"])

    rendered = render_definitions(agent)

    # Should contain the object but not the excluded method
    assert "object db:" in rendered
    assert "def connect(" in rendered
    assert "def query(" not in rendered


def test_render_mixed_registrations():
    """Test rendering when agent has functions, classes, modules, and objects."""
    import math

    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    # Register a function
    @agent.fn
    def my_function():
        """A test function."""
        pass

    # Register a class
    @agent.cls
    class MyClass:
        def method(self):
            pass

    # Register a module
    agent.module(math, include=["sin", "cos"])

    # Register an object
    agent.module(db, name="db")

    rendered = render_definitions(agent)

    # Should contain all types
    assert "def my_function(" in rendered
    assert "class MyClass:" in rendered
    assert "module math:" in rendered
    assert "object db:" in rendered


def test_render_object_empty():
    """Test rendering an object with no exposed methods or properties."""
    agent = Agent(primer="Test agent.")
    db = MockDatabaseConnection("test_db")

    # Exclude everything
    agent.module(db, name="db", exclude=["*"])

    rendered = render_definitions(agent)

    # Should show the object but with ellipsis
    assert "object db:" in rendered
    assert "    ..." in rendered


def test_render_class_with_init_type_hints():
    """Test that attributes defined in __init__ with type annotations are rendered with type hints."""
    agent = Agent()

    class TestClassWithInitTypes:
        def __init__(self, name: str, age: int, email: str | None = None):
            self.name = name
            self.age = age
            self.email = email
            self.count = 0  # No type annotation

    agent.cls(TestClassWithInitTypes)

    rendered = render_definitions(agent)

    # Should include type hints for parameters that become attributes
    assert "name: str" in rendered
    assert "age: int" in rendered
    assert "email: str | None" in rendered
    assert "count" in rendered  # No type hint for this one
    assert "count:" not in rendered  # Should not have a colon if no type hint
