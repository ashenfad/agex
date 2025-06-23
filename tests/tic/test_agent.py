import math
from dataclasses import dataclass

from tests.tic import test_module
from tic.agent import Agent, MemberSpec, RegisteredClass, Select


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
    agent.fn()(math.sqrt)
    assert "sqrt" in agent.fn_registry
    reg = agent.fn_registry["sqrt"]
    assert reg.fn == math.sqrt


def test_agent_cls_registration_dataclass_defaults():
    agent = Agent()

    @agent.cls()
    @dataclass
    class MyData:
        x: int
        y: str

        def do_stuff(self):
            pass

    assert "MyData" in agent.cls_registry
    reg = agent.cls_registry["MyData"]
    assert reg.cls == MyData
    assert reg.visibility == "high"
    assert reg.constructable is True
    assert set(reg.attrs.keys()) == {"x", "y"}
    assert reg.attrs["x"].visibility == "high"
    assert not reg.methods  # Methods are not selected by default


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
        visibility="medium",
        constructable=False,
        attrs=Select.non_private,
        methods=["do_stuff"],
    )(MyClass)

    assert "MyClass" in agent.cls_registry
    reg = agent.cls_registry["MyClass"]
    assert reg.cls == MyClass
    assert reg.visibility == "medium"
    assert reg.constructable is False
    assert reg.attrs["x"].visibility == "medium"
    assert set(reg.attrs.keys()) == {"x"}
    assert reg.methods["do_stuff"].visibility == "medium"
    assert set(reg.methods.keys()) == {"do_stuff"}


def test_agent_cls_registration_overrides():
    agent = Agent()

    class MyService:
        config_path = "/etc/service.conf"
        name: str = "default_name"
        _internal_id = "xyz-123"

        def critical_op(self):
            pass

        def regular_op(self):
            pass

        def _private_op(self):
            pass

    agent.cls(
        MyService,
        methods=["regular_op"],
        attrs=["name"],
        visibility="medium",  # Default for selected
        overrides={
            "critical_op": MemberSpec(visibility="high"),  # Override and include
            "config_path": MemberSpec(visibility="low"),  # Override and include
            "_private_op": MemberSpec(visibility="low"),  # Should not be included
        },
    )

    assert "MyService" in agent.cls_registry
    reg = agent.cls_registry["MyService"]

    # Check methods: regular_op was selected, critical_op was added by override
    assert set(reg.methods.keys()) == {"critical_op", "regular_op"}
    assert reg.methods["critical_op"].visibility == "high"
    assert reg.methods["regular_op"].visibility == "medium"
    assert "_private_op" not in reg.methods

    # Check attrs: name was selected, config_path was added by override
    assert set(reg.attrs.keys()) == {"config_path", "name"}
    assert reg.attrs["config_path"].visibility == "low"
    assert reg.attrs["name"].visibility == "medium"
    assert "_internal_id" not in reg.attrs


def test_agent_module_registration():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",
        fns=["public_fn"],
        classes=["PublicClass"],
        class_methods=["public_method"],
        overrides={
            "PI": MemberSpec(visibility="high"),
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
    assert reg.classes["PublicClass"].methods["public_method"].visibility == "high"


def test_agent_module_registration_defaults():
    agent = Agent()
    agent.module(test_module)  # name should be inferred

    assert "tests.tic.test_module" in agent.importable_modules
    reg = agent.importable_modules["tests.tic.test_module"]

    assert reg.name == "tests.tic.test_module"
    assert set(reg.fns.keys()) == {"public_fn"}
    assert set(reg.consts.keys()) == {"PI"}
    assert "PublicClass" in reg.classes
    assert "_PrivateClass" not in reg.classes

    public_class_reg = reg.classes["PublicClass"]
    assert isinstance(public_class_reg, RegisteredClass)
    assert set(public_class_reg.methods.keys()) == {"public_method"}


def test_agent_module_with_overrides():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",  # default for selected items
        fns=["public_fn"],
        classes=["PublicClass"],
        class_methods=["public_method"],
        overrides={
            "PI": MemberSpec(visibility="high"),
            "PublicClass.public_method": MemberSpec(visibility="high"),
        },
    )

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    # Check that the module itself has the base visibility
    assert reg.visibility == "low"

    # Check that a selected function has the module's visibility
    assert "public_fn" in reg.fns
    assert reg.fns["public_fn"].visibility == "low"

    # Check that a constant's visibility can be overridden
    assert "PI" in reg.consts
    assert reg.consts["PI"].visibility == "high"

    # Check that a class's method can be overridden
    assert "PublicClass" in reg.classes
    pub_cls = reg.classes["PublicClass"]
    assert "public_method" in pub_cls.methods
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
