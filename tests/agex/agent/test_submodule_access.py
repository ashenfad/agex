"""Test for parent.submodule attribute access in policy resolution."""

import os

from agex.agent.policy.policy import AgentPolicy


def test_submodules_auto_injected_on_registration():
    """Test that registering os.path after os auto-injects submodule reference."""
    policy = AgentPolicy()

    # Register parent first
    policy.register_module(module=os, include=["listdir", "remove"])

    # Register submodule
    policy.register_module(module=os.path, include=["exists", "isfile"])

    # Check that os namespace now has path in submodules
    os_ns = policy.namespaces.get("os")
    assert os_ns is not None
    assert "path" in os_ns.submodules
    # On Unix, os.path is actually posixpath
    assert os_ns.submodules["path"] in ["posixpath", "ntpath"]


def test_submodule_registration_order_independent():
    """Test that registration order doesn't matter."""
    # Register submodule before parent
    policy = AgentPolicy()
    policy.register_module(module=os.path, include=["exists"])
    policy.register_module(module=os, include=["listdir"])

    # Should still inject submodule
    os_ns = policy.namespaces.get("os")
    assert os_ns is not None
    assert "path" in os_ns.submodules


def test_external_module_aliasing():
    """Test resolution of attributes through externally aliased modules.

    This covers cases like plotly.express.colors.qualitative which is actually
    _plotly_utils.colors.qualitative. The attribute chain traversal should work
    even when importlib.import_module fails for the synthetic path.
    """
    import sys
    from types import ModuleType

    # Create mock packages to simulate plotly's aliasing behavior:
    # mock_pkg.express.colors.qualitative -> _mock_external.colors.qualitative
    _mock_external = ModuleType("_mock_external")
    _mock_external_colors = ModuleType("_mock_external.colors")
    _mock_external_colors_qualitative = ModuleType("_mock_external.colors.qualitative")
    _mock_external_colors_qualitative.Safe = ["#000", "#FFF", "#F00"]

    _mock_external.colors = _mock_external_colors
    _mock_external_colors.qualitative = _mock_external_colors_qualitative

    mock_pkg = ModuleType("mock_pkg")
    mock_pkg_express = ModuleType("mock_pkg.express")
    mock_pkg.express = mock_pkg_express
    # This is the aliasing - colors points to external package
    mock_pkg_express.colors = _mock_external_colors

    # Register in sys.modules
    original_modules = {}
    for name in [
        "_mock_external",
        "_mock_external.colors",
        "_mock_external.colors.qualitative",
        "mock_pkg",
        "mock_pkg.express",
    ]:
        if name in sys.modules:
            original_modules[name] = sys.modules[name]

    sys.modules["_mock_external"] = _mock_external
    sys.modules["_mock_external.colors"] = _mock_external_colors
    sys.modules["_mock_external.colors.qualitative"] = _mock_external_colors_qualitative
    sys.modules["mock_pkg"] = mock_pkg
    sys.modules["mock_pkg.express"] = mock_pkg_express
    # NOTE: mock_pkg.express.colors is NOT registered as a module path
    # because it's actually _mock_external.colors

    try:
        from agex.agent.policy.datatypes import Namespace, ResolutionScope
        from agex.agent.policy.resolve import resolve_member

        # Create namespace for mock_pkg with recursive=True and include="*"
        mock_ns = Namespace(
            name="mock_pkg",
            kind="module",
            module=mock_pkg,
            include="*",
            recursive=True,
        )

        scope = ResolutionScope(
            namespaces={"mock_pkg": mock_ns},
            module_index={"mock_pkg": [mock_ns]},
        )

        # This is the key test: accessing express.colors.qualitative.Safe
        # should work even though importlib.import_module("mock_pkg.express.colors.qualitative")
        # would fail (because the real module is _mock_external.colors.qualitative)
        result = resolve_member(mock_ns, "express.colors.qualitative.Safe", scope)

        assert result is not None, "Resolution should succeed for aliased module paths"
        from agex.agent.policy.datatypes import ResolvedObj

        assert isinstance(
            result, ResolvedObj
        ), f"Expected ResolvedObj, got {type(result)}"
        assert result.value == ["#000", "#FFF", "#F00"]

    finally:
        # Clean up sys.modules
        for name in [
            "_mock_external",
            "_mock_external.colors",
            "_mock_external.colors.qualitative",
            "mock_pkg",
            "mock_pkg.express",
        ]:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]


def test_recursive_path_respects_exclude_on_leaf():
    """Test that include/exclude patterns are applied to the leaf member in recursive paths."""
    import sys
    from types import ModuleType

    # Create mock package with both public and private attributes
    mock_pkg = ModuleType("mock_pkg_exclude")
    mock_pkg_sub = ModuleType("mock_pkg_exclude.sub")
    mock_pkg.sub = mock_pkg_sub
    mock_pkg_sub.public_value = "allowed"
    mock_pkg_sub._private_value = "should be blocked"

    # Register in sys.modules
    original_modules = {}
    for name in ["mock_pkg_exclude", "mock_pkg_exclude.sub"]:
        if name in sys.modules:
            original_modules[name] = sys.modules[name]

    sys.modules["mock_pkg_exclude"] = mock_pkg
    sys.modules["mock_pkg_exclude.sub"] = mock_pkg_sub

    try:
        from agex.agent.policy.datatypes import Namespace, ResolutionScope
        from agex.agent.policy.resolve import resolve_member

        # Create namespace with default exclude pattern for private names
        mock_ns = Namespace(
            name="mock_pkg_exclude",
            kind="module",
            module=mock_pkg,
            include="*",
            exclude="_*",  # Exclude private names
            recursive=True,
        )

        scope = ResolutionScope(
            namespaces={"mock_pkg_exclude": mock_ns},
            module_index={"mock_pkg_exclude": [mock_ns]},
        )

        # Public attribute should be accessible
        result = resolve_member(mock_ns, "sub.public_value", scope)
        assert result is not None, "Public attribute should be accessible"
        from agex.agent.policy.datatypes import ResolvedObj

        assert isinstance(result, ResolvedObj)
        assert result.value == "allowed"

        # Private attribute should be blocked
        result = resolve_member(mock_ns, "sub._private_value", scope)
        assert result is None, "Private attribute should be blocked by exclude pattern"

    finally:
        # Clean up sys.modules
        for name in ["mock_pkg_exclude", "mock_pkg_exclude.sub"]:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]


def test_external_module_aliasing_class_access():
    """Test resolution of class attributes through externally aliased modules."""
    import sys
    from types import ModuleType

    # Create mock packages with a class in the aliased module
    _mock_external = ModuleType("_mock_external")
    _mock_external_utils = ModuleType("_mock_external.utils")

    class MockColor:
        def __init__(self, value):
            self.value = value

        @classmethod
        def from_hex(cls, hex_str):
            return cls(hex_str)

    _mock_external_utils.Color = MockColor
    _mock_external.utils = _mock_external_utils

    mock_pkg = ModuleType("mock_pkg2")
    mock_pkg_graphics = ModuleType("mock_pkg2.graphics")
    mock_pkg.graphics = mock_pkg_graphics
    # Aliasing: graphics.utils points to external package
    mock_pkg_graphics.utils = _mock_external_utils

    # Register in sys.modules
    original_modules = {}
    for name in [
        "_mock_external",
        "_mock_external.utils",
        "mock_pkg2",
        "mock_pkg2.graphics",
    ]:
        if name in sys.modules:
            original_modules[name] = sys.modules[name]

    sys.modules["_mock_external"] = _mock_external
    sys.modules["_mock_external.utils"] = _mock_external_utils
    sys.modules["mock_pkg2"] = mock_pkg
    sys.modules["mock_pkg2.graphics"] = mock_pkg_graphics

    try:
        from agex.agent.policy.datatypes import (
            Namespace,
            ResolutionScope,
            ResolvedClass,
        )
        from agex.agent.policy.resolve import resolve_member

        mock_ns = Namespace(
            name="mock_pkg2",
            kind="module",
            module=mock_pkg,
            include="*",
            recursive=True,
        )

        scope = ResolutionScope(
            namespaces={"mock_pkg2": mock_ns},
            module_index={"mock_pkg2": [mock_ns]},
        )

        # Access class through aliased path
        result = resolve_member(mock_ns, "graphics.utils.Color", scope)

        assert (
            result is not None
        ), "Resolution should succeed for class in aliased module"
        assert isinstance(
            result, ResolvedClass
        ), f"Expected ResolvedClass, got {type(result)}"
        assert result.cls is MockColor

    finally:
        # Clean up sys.modules
        for name in [
            "_mock_external",
            "_mock_external.utils",
            "mock_pkg2",
            "mock_pkg2.graphics",
        ]:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]


if __name__ == "__main__":
    test_submodules_auto_injected_on_registration()
    print("✓ test_submodules_auto_injected_on_registration passed")

    test_submodule_registration_order_independent()
    print("✓ test_submodule_registration_order_independent passed")

    test_external_module_aliasing()
    print("✓ test_external_module_aliasing passed")

    test_recursive_path_respects_exclude_on_leaf()
    print("✓ test_recursive_path_respects_exclude_on_leaf passed")

    test_external_module_aliasing_class_access()
    print("✓ test_external_module_aliasing_class_access passed")

    print("\nAll tests passed!")
