from types import ModuleType

import pytest

from tic.agent import Agent, MemberSpec
from tic.render.definitions import render_definitions


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

        def should_not_be_rendered(self):
            pass

    output = render_definitions(agent)
    expected = """
class MediumVis:
    def __init__(self, val: int):
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
    assert output.strip() == "module my_mod:\n    ..."


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
