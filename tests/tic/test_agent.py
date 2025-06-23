import math
from dataclasses import dataclass

from tests.tic import test_module
from tic.agent import Agent, RegisteredClass, Select


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
    assert reg.allowed_attrs == {"x", "y"}
    assert reg.allowed_methods == set()  # Methods are not selected by default


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
    assert reg.allowed_attrs == {"x"}
    assert reg.allowed_methods == {"do_stuff"}


def test_agent_module_registration():
    agent = Agent()

    agent.module(
        math,
        name="math",
        visibility="low",
        fns=["sqrt", "is*"],
        consts=["pi", "e"],
    )

    assert "math" in agent.importable_modules
    reg = agent.importable_modules["math"]
    assert reg.module == math
    assert reg.name == "math"
    assert reg.visibility == "low"
    assert reg.fns == {"sqrt", "isclose", "isfinite", "isinf", "isnan", "isqrt"}
    assert reg.consts == {"pi", "e"}


def test_agent_module_registration_defaults():
    agent = Agent()
    agent.module(test_module)  # name should be inferred

    assert "tests.tic.test_module" in agent.importable_modules
    reg = agent.importable_modules["tests.tic.test_module"]

    assert reg.name == "tests.tic.test_module"
    assert reg.fns == {"public_fn"}
    assert reg.consts == {"PI"}
    assert "PublicClass" in reg.classes
    assert "_PrivateClass" not in reg.classes

    public_class_reg = reg.classes["PublicClass"]
    assert isinstance(public_class_reg, RegisteredClass)
    assert public_class_reg.allowed_methods == {"public_method"}


def test_agent_module_registration_custom():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",
        fns="*",  # select all fns
        consts=None,  # select no consts
        classes=["Public*"],
        class_methods=None,  # select no methods for found classes
    )

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    assert reg.visibility == "low"
    assert reg.fns == {"public_fn", "_private_fn"}
    assert reg.consts == set()
    assert "PublicClass" in reg.classes
    assert reg.classes["PublicClass"].allowed_methods == set()
