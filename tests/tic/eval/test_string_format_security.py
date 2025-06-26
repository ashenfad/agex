"""
Tests for string formatting security vulnerabilities.

This module tests whether string formatting methods like .format(), f-strings,
and other template mechanisms can be used to bypass the sandbox and access
non-registered attributes or dangerous functionality.
"""

import pytest

from tic.agent import Agent
from tic.eval.error import EvalError
from tic.eval.user_errors import TicAttributeError

from .helpers import eval_and_get_state


class SecurityTestObject:
    """A test class with various attributes for security testing."""

    def __init__(self):
        self.public_attr = "public"
        self._private_attr = "private"
        self.__secret_attr = "secret"

    def public_method(self):
        return "public_method_result"

    def _private_method(self):
        return "private_method_result"


def test_format_blocks_attribute_access():
    """Test that .format() blocks all attribute access, preventing security vulnerabilities."""
    agent = Agent()

    # Register some attributes
    agent.cls(SecurityTestObject, include=["public_attr", "public_method"])

    # Simple format strings without attribute access should work
    program_simple = """
name = "World"
result = "Hello {name}".format(name=name)
"""
    state = eval_and_get_state(program_simple, agent)
    assert state.get("result") == "Hello World"

    # But ANY attribute access in format strings should be blocked (even registered ones)
    # This prevents the security vulnerability entirely
    programs_blocked = [
        "obj = SecurityTestObject(); result = '{obj.public_attr}'.format(obj=obj)",  # Even registered attrs
        "obj = SecurityTestObject(); result = '{obj._private_attr}'.format(obj=obj)",  # Private attrs
        "obj = SecurityTestObject(); result = '{obj.__secret_attr}'.format(obj=obj)",  # Secret attrs
        "obj = SecurityTestObject(); result = '{obj._private_method}'.format(obj=obj)",  # Methods
    ]

    for program in programs_blocked:
        with pytest.raises(
            EvalError, match="Format string attribute access .* is not allowed"
        ):
            eval_and_get_state(program, agent)


def test_f_strings_respect_sandbox():
    """Test that f-strings properly respect sandbox attribute registration."""
    agent = Agent()

    # Only register the public_attr
    agent.cls(SecurityTestObject, include=["public_attr"])

    # This should work - accessing registered attribute
    program_good = """
obj = SecurityTestObject()
result = f"Value: {obj.public_attr}"
"""
    state = eval_and_get_state(program_good, agent)
    assert state.get("result") == "Value: public"

    # This should fail - accessing unregistered attribute
    program_bad = """
obj = SecurityTestObject()
result = f"Value: {obj._private_attr}"
"""
    with pytest.raises(
        TicAttributeError, match="object has no attribute '_private_attr'"
    ):
        eval_and_get_state(program_bad, agent)


def test_format_blocks_dunder_access():
    """Test that format strings block access to dangerous dunder methods."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_attr"])

    dangerous_programs = [
        "obj = SecurityTestObject(); result = '{obj.__class__}'.format(obj=obj)",
        "obj = SecurityTestObject(); result = '{obj.__dict__}'.format(obj=obj)",
        "obj = SecurityTestObject(); result = '{obj.__module__}'.format(obj=obj)",
        "obj = SecurityTestObject(); result = '{obj.__class__.__bases__}'.format(obj=obj)",
    ]

    for program in dangerous_programs:
        with pytest.raises(
            EvalError, match="Format string attribute access .* is not allowed"
        ):
            eval_and_get_state(program, agent)


def test_format_blocks_builtin_type_access():
    """Test that format strings block access to builtin type internals."""
    programs = [
        "result = '{0.__subclasses__}'.format(int)",
        "result = '{0.__bases__}'.format(str)",
        "result = '{0.__mro__}'.format(list)",
        "result = '{0.__dict__}'.format(dict)",
    ]

    for program in programs:
        with pytest.raises(
            EvalError, match="Format string attribute access .* is not allowed"
        ):
            eval_and_get_state(program)


def test_percent_formatting_security():
    """Test that % formatting is also secure."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_attr"])

    # This should work
    program_good = """
obj = SecurityTestObject()
result = "Value: %(attr)s" % {"attr": obj.public_attr}
"""
    state = eval_and_get_state(program_good, agent)
    assert state.get("result") == "Value: public"

    # Test that we can't trick % formatting
    # (Note: % formatting is generally safer since it doesn't support attribute access)


def test_template_string_security():
    """Test string.Template security if it's available."""
    # Test if string module can be accessed and if Template is vulnerable
    program = """
import string
template = string.Template("Hello $name")
result = template.substitute(name="World")
"""

    # This might fail if string module isn't registered
    try:
        state = eval_and_get_state(program)
        assert state.get("result") == "Hello World"
    except EvalError:
        # Expected if string module isn't registered
        pass


def test_format_spec_security():
    """Test that format specifications with attribute access are blocked."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_attr"])

    # Format specs with attribute access should be blocked
    program = """
obj = SecurityTestObject()
result = "{obj.public_attr:>10}".format(obj=obj)
"""
    with pytest.raises(
        EvalError, match="Format string attribute access .* is not allowed"
    ):
        eval_and_get_state(program, agent)

    # But simple format specs should work
    program_simple = """
value = "hello"
result = "{value:>10}".format(value=value)
"""
    state = eval_and_get_state(program_simple, agent)
    assert state.get("result") == "     hello"


def test_format_blocks_nested_attribute_access():
    """Test that nested attribute access through formatting is blocked."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_attr"])

    # Try to access __class__ through public_attr - should be blocked
    program = """
obj = SecurityTestObject()
result = "{obj.public_attr.__class__}".format(obj=obj)
"""
    with pytest.raises(
        EvalError, match="Format string attribute access .* is not allowed"
    ):
        eval_and_get_state(program, agent)


def test_format_blocks_method_calls():
    """Test that method calls in format strings are blocked entirely."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_method"])

    # ALL method calls should be blocked - even registered ones
    programs_blocked = [
        "obj = SecurityTestObject(); result = '{obj.public_method()}'.format(obj=obj)",  # Even registered
        "obj = SecurityTestObject(); result = '{obj._private_method()}'.format(obj=obj)",  # Private
    ]

    for program in programs_blocked:
        with pytest.raises(
            EvalError, match="Format string attribute access .* is not allowed"
        ):
            eval_and_get_state(program, agent)


def test_format_allows_simple_expressions():
    """Test that format strings allow simple variable access but block attribute access."""
    agent = Agent()
    agent.cls(SecurityTestObject, include=["public_attr"])

    # Simple expressions without attribute access should work
    program_simple = """
num = 42
text = "hello"
result = "{num} and {text}".format(num=num, text=text)
"""
    state = eval_and_get_state(program_simple, agent)
    assert state.get("result") == "42 and hello"

    # But expressions with attribute access should be blocked
    program_complex = """
obj = SecurityTestObject()
result = "{len(obj.public_attr)}".format(obj=obj)
"""
    with pytest.raises(
        EvalError, match="Format string attribute access .* is not allowed"
    ):
        eval_and_get_state(program_complex, agent)


if __name__ == "__main__":
    pytest.main([__file__])
