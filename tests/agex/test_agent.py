import math
from dataclasses import dataclass
from types import ModuleType

import pytest

from agex.agent import Agent, MemberSpec, RegisteredClass
from tests.agex import test_module


def test_agent_fn_registration_decorator():
    agent = Agent()

    @agent.fn()
    def my_func():
        """My docstring"""
        return 1

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "high"
    assert reg.docstring == "My docstring"


def test_agent_fn_registration_decorator_with_args():
    agent = Agent()

    @agent.fn(visibility="low", docstring="New doc")
    def my_func():
        """Original doc"""
        return 1

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "low"
    assert reg.docstring == "New doc"


def test_agent_fn_registration_functional():
    agent = Agent()

    def my_func():
        return 1

    agent.fn(visibility="medium")(my_func)

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "medium"
    assert reg.docstring is None


def test_agent_fn_registration_functional_builtin():
    agent = Agent()
    agent.fn()(math.sqrt)  # Test the decorator factory style
    assert "sqrt" in agent.fn_registry
    reg = agent.fn_registry["sqrt"]
    assert reg.fn == math.sqrt


def test_agent_fn_registration_direct_call():
    agent = Agent()
    agent.fn(math.sqrt)  # Test the direct call style
    assert "sqrt" in agent.fn_registry
    reg = agent.fn_registry["sqrt"]
    assert reg.fn == math.sqrt


def test_agent_fn_registration_with_name_alias():
    agent = Agent()

    def original_function_name():
        return "aliased"

    agent.fn(original_function_name, name="alias")

    assert "alias" in agent.fn_registry
    assert "original_function_name" not in agent.fn_registry
    reg = agent.fn_registry["alias"]
    assert reg.fn() == "aliased"


def test_registering_reserved_name_fails():
    """Tests that registering a reserved name raises a ValueError."""
    agent = Agent()

    def my_fn():
        pass

    class MyClass:
        pass

    dummy_module = ModuleType("dummy")

    with pytest.raises(ValueError, match="is reserved"):
        agent.fn(my_fn, name="dataclass")

    with pytest.raises(ValueError, match="is reserved"):
        agent.cls(MyClass, name="dataclass")

    with pytest.raises(ValueError, match="is reserved"):
        agent.module(dummy_module, name="dataclasses")


def test_agent_cls_registration_defaults():
    agent = Agent()

    @agent.cls
    @dataclass
    class MyData:
        x: int
        y: str
        _z: float

        def do_stuff(self):
            pass

        def _do_private_stuff(self):
            pass

    assert "MyData" in agent.cls_registry
    reg = agent.cls_registry["MyData"]
    assert reg.cls == MyData
    assert reg.visibility == "high"
    assert reg.constructable is True
    assert set(reg.attrs.keys()) == {"x", "y"}  # _z is excluded
    assert reg.attrs["x"].visibility == "high"
    assert set(reg.methods.keys()) == {
        "do_stuff",
        "__init__",
    }  # _do_private_stuff is excluded


def test_agent_cls_registration_selectors():
    agent = Agent()

    class MyClass:
        x: int
        _y: str

        def do_stuff(self):
            pass

        def _do_private_stuff(self):
            pass

    # Use as a decorator factory
    agent.cls(
        MyClass,
        visibility="medium",
        constructable=False,
        include=["x", "do_stuff", "_y"],
        exclude=None,  # Include everything from the list
    )

    assert "MyClass" in agent.cls_registry
    reg = agent.cls_registry["MyClass"]
    assert reg.cls == MyClass
    assert reg.visibility == "medium"
    assert reg.constructable is False
    assert set(reg.attrs.keys()) == {"x", "_y"}
    assert set(reg.methods.keys()) == {
        "do_stuff"
    }  # constructable=False removed __init__


def test_agent_cls_registration_configure_and_exclude():
    agent = Agent()

    class MyService:
        config_path = "/etc/service.conf"
        name: str = "default_name"
        _internal_id = "xyz-123"

        def __init__(self):
            pass

        def critical_op(self):
            pass

        def regular_op(self):
            pass

        def _private_op(self):
            pass

    agent.cls(
        MyService,
        visibility="medium",  # Default for selected
        include="*",  # Explicitly include everything to start
        exclude=["regular_op", "_*"],  # Exclude one public and all private
        configure={
            "critical_op": MemberSpec(visibility="high"),
            "config_path": MemberSpec(visibility="low"),
        },
    )

    assert "MyService" in agent.cls_registry
    reg = agent.cls_registry["MyService"]

    # Check methods: critical_op was included and configured, regular_op was excluded.
    # __init__ is included because constructable=True by default.
    assert set(reg.methods.keys()) == {"critical_op", "__init__"}
    assert reg.methods["critical_op"].visibility == "high"
    assert "_private_op" not in reg.methods

    # Check attrs: name and config_path included, _internal_id excluded.
    assert set(reg.attrs.keys()) == {"config_path", "name"}
    assert reg.attrs["config_path"].visibility == "low"
    assert reg.attrs["name"].visibility == "medium"


def test_agent_module_registration():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",
        include=["public_fn", "PI", "PublicClass", "PublicClass.*"],
        exclude=["*.secret_*", "*._*"],
        configure={
            "PI": MemberSpec(visibility="high"),
            "PublicClass": MemberSpec(constructable=True),  # Ensure it's constructable
            "PublicClass.public_method": MemberSpec(visibility="high"),
        },
    )

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    # Check top-level visibilities
    assert reg.visibility == "low"

    # Check top-level consts and fns
    assert reg.consts["PI"].visibility == "high"
    assert reg.fns["public_fn"].visibility == "low"

    # Check class and its nested method
    cls_reg = reg.classes["PublicClass"]
    assert cls_reg.constructable is True
    assert "__init__" in cls_reg.methods
    assert "public_method" in cls_reg.methods
    assert cls_reg.methods["public_method"].visibility == "high"


def test_agent_module_registration_defaults():
    agent = Agent()
    agent.module(test_module, name="test_module")

    assert "test_module" in agent.importable_modules
    reg = agent.importable_modules["test_module"]

    assert reg.name == "test_module"
    assert set(reg.fns.keys()) == {"public_fn"}
    assert set(reg.consts.keys()) == {"PI"}
    assert "PublicClass" in reg.classes
    assert "_PrivateClass" not in reg.classes

    public_class_reg = reg.classes["PublicClass"]
    assert isinstance(public_class_reg, RegisteredClass)
    assert public_class_reg.constructable is True
    assert set(public_class_reg.methods.keys()) == {"public_method", "__init__"}


def test_agent_module_with_configure():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",  # default for selected items
        include=["*"],
        exclude=["_*", "*._*"],  # Exclude both top-level and class privates
        configure={
            "PI": MemberSpec(visibility="high"),
            "PublicClass": MemberSpec(constructable=False),
            "PublicClass.public_method": MemberSpec(visibility="high"),
            "public_fn": MemberSpec(visibility="medium"),
        },
    )

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    # Check that the module itself has the base visibility
    assert reg.visibility == "low"

    # Check that a selected function has the configured visibility
    assert "public_fn" in reg.fns
    assert reg.fns["public_fn"].visibility == "medium"

    # Check that a constant's visibility can be configured
    assert "PI" in reg.consts
    assert reg.consts["PI"].visibility == "high"

    # Check that a class's method can be configured and it is not constructable
    assert "PublicClass" in reg.classes
    pub_cls = reg.classes["PublicClass"]
    assert pub_cls.constructable is False
    assert set(pub_cls.methods.keys()) == {"public_method"}  # No __init__
    assert pub_cls.methods["public_method"].visibility == "high"


def test_agent_cls_no_parens():
    """Tests that the @agent.cls decorator works without parentheses."""
    agent = Agent()

    @agent.cls
    @dataclass
    class SimpleData:
        value: str

    assert "SimpleData" in agent.cls_registry
    reg = agent.cls_registry["SimpleData"]
    assert reg.cls == SimpleData
    assert reg.visibility == "high"  # Check default visibility
    assert set(reg.attrs.keys()) == {"value"}  # Check default attr selection


def test_agent_cls_direct_call():
    agent = Agent()

    class MyClass:
        pass

    agent.cls(MyClass, visibility="low")

    assert "MyClass" in agent.cls_registry
    reg = agent.cls_registry["MyClass"]
    assert reg.cls == MyClass
    assert reg.visibility == "low"


def test_agent_cls_with_name_alias():
    agent = Agent()

    class OriginalClassName:
        pass

    agent.cls(OriginalClassName, name="AliasClass")

    assert "AliasClass" in agent.cls_registry
    assert "OriginalClassName" not in agent.cls_registry

    reg = agent.cls_registry["AliasClass"]
    assert reg.cls == OriginalClassName


# =============================================================================
# Decorator Validation Tests
# =============================================================================


def test_task_decorator_single():
    """Test that a single task decorator works correctly."""
    agent = Agent()

    @agent.task("Implement a simple function")
    def simple_task():
        """A simple task function."""
        pass

    # Check that task metadata is set correctly
    assert hasattr(simple_task, "__is_agent_task__")
    assert simple_task.__is_agent_task__ is True
    assert hasattr(simple_task, "__agent_task_owner__")
    assert simple_task.__agent_task_owner__ is agent


def test_fn_decorator_multiple():
    """Test that multiple fn decorators on the same function are allowed."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.fn(docstring="First agent function")
    @agent2.fn(docstring="Second agent function")
    def shared_function():
        """A function shared across agents."""
        return "shared"

    # Check that fn metadata is set correctly
    assert hasattr(shared_function, "__is_agent_fn__")
    assert shared_function.__is_agent_fn__ is True
    assert hasattr(shared_function, "__agent_fn_owners__")
    assert len(shared_function.__agent_fn_owners__) == 2
    assert agent1 in shared_function.__agent_fn_owners__
    assert agent2 in shared_function.__agent_fn_owners__


def test_task_decorator_multiple_not_allowed():
    """Test that multiple task decorators on the same function are not allowed."""
    agent1 = Agent()
    agent2 = Agent()

    # First task decorator should work
    @agent1.task("First task implementation")
    def multi_task_attempt():
        pass

    # Second task decorator should fail
    with pytest.raises(ValueError, match="already has a task decorator"):
        agent2.task("Second task implementation")(multi_task_attempt)


def test_decorator_order_wrong_not_allowed():
    """Test that wrong decorator order (task before fn) is not allowed."""
    agent1 = Agent()
    agent2 = Agent()

    with pytest.raises(ValueError, match="Invalid decorator order"):

        @agent1.task("This should fail")  # Task applied first (inner)
        @agent2.fn(
            docstring="This comes after"
        )  # Fn applied second (outer) - WRONG ORDER
        def wrong_order_example():
            pass


def test_decorator_order_correct_allowed():
    """Test that correct decorator order (fn before task) is allowed."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.fn(docstring="Outer fn decorator")  # Fn applied first (outer) - CORRECT
    @agent2.task("Inner task decorator")  # Task applied second (inner) - CORRECT
    def correct_order_example():
        """Function with correct dual decorator order."""
        pass

    # Check that both decorators were applied
    assert hasattr(correct_order_example, "__is_agent_task__")
    assert correct_order_example.__is_agent_task__ is True
    assert hasattr(correct_order_example, "__agent_task_owner__")
    assert correct_order_example.__agent_task_owner__ is agent2


def test_fn_decorator_builtin_functions():
    """Test that fn decorator works with built-in functions without errors."""
    agent = Agent()

    # This should not raise any AttributeError about setting __agent_fn_owners__
    registered_sqrt = agent.fn(docstring="Built-in square root")(math.sqrt)

    # Should be the same function object
    assert registered_sqrt is math.sqrt

    # Should be registered in the agent
    assert "sqrt" in agent.fn_registry
    assert agent.fn_registry["sqrt"].fn is math.sqrt


def test_task_decorator_validation_error_messages():
    """Test that validation error messages are clear and helpful."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.task("First task")
    def test_function():
        pass

    # Test multiple task decorator error message
    with pytest.raises(ValueError) as exc_info:
        agent2.task("Second task")(test_function)

    error_msg = str(exc_info.value)
    assert "already has a task decorator" in error_msg
    assert "Multi-agent tasks are not supported" in error_msg

    # Test wrong order error message
    with pytest.raises(ValueError) as exc_info:

        @agent1.task("Should fail")
        @agent2.fn(docstring="Wrong order")
        def wrong_order():
            pass

    error_msg = str(exc_info.value)
    assert "Invalid decorator order" in error_msg
    assert "@agent.fn() must be applied AFTER @agent.task()" in error_msg
    assert "Correct order:" in error_msg
