import math
from dataclasses import dataclass
from types import ModuleType

import pytest

from tests.tic import test_module
from tic.agent import Agent, MemberSpec, RegisteredClass


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
        exclude=["_*"],
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
